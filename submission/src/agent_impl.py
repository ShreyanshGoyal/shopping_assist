"""The conversational shopping agent.

The agent keeps a structured request — a product type, a set of attributes, and a
set of rejections — and searches against that rather than against the raw
transcript. See `frame.py` for why that distinction is the whole design.

Retrieval, ranking and questioning are offline by construction. The extraction
step is the one place a language model can help, and it is a swappable component:
with no credential the agent runs the lexical extractor and still works, which
matters because the organiser reserves the right to score with network access
disabled.

Slate sizing is the other half of the strategy. Padding a shortlist with weak
guesses buys a low-ranked conversion, which scores far worse than the same
conversion one turn later at the top of the list.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

from . import parsing, ranking
from .catalog import Catalog
from .extract import LexicalExtractor, ModelExtractor
from .frame import resolve_category
from .inquiry import InquiryPlanner
from .listwise import ListwiseReranker
from .policy import ProbePolicy
from .state import SessionState

# Turn by which the agent must stop narrowing and present a full shortlist,
# regardless of how much it still hopes to learn.
COMMIT_TURN = 6
# Withholding a shortlist is a bet that another turn promotes the right product
# to the top. Widening early instead — as soon as the leader's score plateaus —
# was tried and measured worse on both benchmarks (-0.034 on the LLM customer,
# MRR -0.079): converting sooner locks in a poor rank, and rank is worth roughly
# six times a turn. Kept behind AGENT_STALL=1 so the experiment is reproducible.
STALL_GAIN = 0.05
STALL_TURNS = 2
STALL_ENABLED = os.environ.get("AGENT_STALL", "0") == "1"
# Score lead over the runner-up at which the top candidate is worth showing
# alone rather than buried in a padded list.
CONFIDENT_MARGIN = 0.20
# Candidates considered when deciding which question would narrow the field most.
PLANNING_POOL = 200
# Consult the listwise reranker only when retrieval has no clear leader. Above
# this margin the ranking is already decided and the call is wasted — on tokens,
# on latency, and on the risk of overturning a correct answer.
LISTWISE_MARGIN = 0.20
# A generated question is narrow, so it extracts less per turn than the open
# probe when its answer fails to discriminate — measured at -0.055 when asked
# every turn. Reserve it for sessions that are visibly failing.
#
# "The open probe has run dry" cannot be detected by counting constraints here:
# a language-model customer always says something, and the extractor always turns
# it into a requirement, so the count grows every turn even when the turn taught
# nothing. Confidence stalling is the observable form of the same condition — the
# leading candidate's score no longer improving means the requirements arriving
# are not separating anything.
RESCUE_TURN = 4
# Measured at -0.061 on the LLM customer, worse than asking every turn: the gate
# kept 2 of 6 rescues while blocking only 3 of 8 casualties. Off by default.
RESCUE_ENABLED = os.environ.get("AGENT_RESCUE", "0") == "1"
# A category named outright in the opening turn is stronger evidence than
# anything an extractor infers, so it seeds the type slot above their reach.
STATED_CATEGORY_CONFIDENCE = 2.0


class ShoppingAgent:
    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        strategy: str | None = None,
        extractor: str | None = None,
    ) -> None:
        self.catalog = Catalog(catalog_path)
        # The dense track is used when its dependencies and prebuilt index are
        # present, and silently skipped when they are not. The lexical paths are
        # the baseline, never a fallback bolted on afterwards.
        if os.environ.get("AGENT_DENSE", "1") != "0":
            self.catalog.attach_dense()
        # Cross-encoder reranking is off by default: measured worse on both
        # benchmarks (see NOTES.md). Kept reproducible behind AGENT_RERANK=1.
        if os.environ.get("AGENT_RERANK", "0") == "1":
            self.catalog.attach_reranker()
        self.planner = InquiryPlanner(self.catalog)
        self.policy = ProbePolicy(
            strategy or os.environ.get("AGENT_STRATEGY", "adaptive"), planner=self.planner
        )
        self.client = None
        self.extractor = self._build_extractor(extractor or os.environ.get("AGENT_EXTRACTOR", "lexical"))
        self.listwise = None
        if self.client is not None and os.environ.get("AGENT_LISTWISE", "0") == "1":
            self.listwise = ListwiseReranker(
                self.client, self.catalog, wide=os.environ.get("AGENT_LISTWISE_WIDE", "0") == "1"
            )
        self._sessions: dict[str, SessionState] = {}
        self._reported_prompt = 0
        self._reported_completion = 0

    def _build_extractor(self, choice: str):
        lexical = LexicalExtractor(self.catalog)
        if choice not in ("model", "auto"):
            return lexical
        try:
            from .llm import make_client

            self.client = make_client(
                os.environ.get("AGENT_MODEL", "gemini-3.5-flash-lite"),
                os.environ.get("AGENT_BACKEND", "auto"),
                max_output_tokens=300,
            )
            return ModelExtractor(self.catalog, self.client, lexical)
        except Exception as error:
            # No credential, no network, no problem: the offline path is the
            # baseline, not a degraded mode bolted on afterwards. This degrades
            # even when the model extractor was explicitly requested — raising
            # here would fail every session in a graded run whose environment
            # happens to set the variable, and the rules count an exception as a
            # miss. The warning keeps the downgrade visible rather than silent.
            warnings.warn(
                f"model extractor unavailable ({type(error).__name__}); "
                "running the offline lexical path",
                RuntimeWarning,
                stacklevel=2,
            )
            return lexical

    @property
    def extractor_name(self) -> str:
        return getattr(self.extractor, "name", "lexical")

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._sessions[session_id] = SessionState(session_id, user_profile or {})

    @staticmethod
    def _confidence_stalled(state: SessionState) -> bool:
        """Has the leading candidate's score stopped improving?"""
        history = state.top_scores
        if len(history) <= STALL_TURNS:
            return False
        recent, earlier = history[-1], history[-1 - STALL_TURNS]
        if earlier <= 0:
            return False
        return (recent - earlier) / earlier < STALL_GAIN

    def _slate_width(self, state: SessionState, scored: list[tuple[float, str]], top_k: int) -> int:
        if not scored:
            return 0
        if state.turn >= COMMIT_TURN or not self.policy.information_left(state):
            return top_k
        if STALL_ENABLED and self._confidence_stalled(state):
            return top_k
        top_k = min(top_k, len(scored))
        if len(scored) == 1:
            return top_k
        if state.last_slate and scored[0][1] == state.last_slate[0]:
            # Nothing learned has changed our mind. Repeating one product the
            # customer already declined wastes the turn; widen instead.
            return top_k
        best, runner_up = scored[0][0], scored[1][0]
        if best > 0 and (best - runner_up) / best >= CONFIDENT_MARGIN:
            # Clear favourite: showing the rest costs nothing, it still ranks first.
            return top_k
        return 1

    def _consult_listwise(self, state: SessionState, scored: list[tuple[float, str]],
                          recommendations: list[str]):
        """Ask the model to settle the order and, if useful, supply a question.

        Only when retrieval has not already decided. Returns the ranking and an
        optional question; both fall through untouched when the call is skipped.
        """
        from .listwise import Verdict

        if self.listwise is None or len(recommendations) < 2 or len(scored) < 2:
            return Verdict(order=recommendations)
        best, runner_up = scored[0][0], scored[1][0]
        if best > 0 and (best - runner_up) / best >= LISTWISE_MARGIN:
            self.listwise.skipped += 1
            return Verdict(order=recommendations)
        return self.listwise.reorder(state.raw_messages, recommendations)

    def _usage(self) -> dict:
        """Tokens spent on *this* turn.

        The harness sums whatever each turn reports, so returning the client's
        running totals double-counts every earlier turn and grows quadratically —
        a 200-session run reported 254 million tokens against an actual few
        hundred thousand. Report the delta since the last call.
        """
        if self.client is None:
            return {"prompt_tokens": 0, "completion_tokens": 0}
        usage = self.client.usage
        prompt = usage.prompt_tokens - self._reported_prompt
        completion = usage.completion_tokens - self._reported_completion
        self._reported_prompt = usage.prompt_tokens
        self._reported_completion = usage.completion_tokens
        return {"prompt_tokens": max(prompt, 0), "completion_tokens": max(completion, 0)}

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            # Defensive: never fail a session over a missing reset.
            state = SessionState(session_id, {})
            self._sessions[session_id] = state

        before = len(state.constraints)
        state.note_utterance(parsing.parse(user_message, turn), turn, raw=user_message or "")

        # A category stated outright outranks anything inferred. If it names a
        # catalog node exactly, take it verbatim — resolving it fuzzily can land
        # on a neighbouring node and throw away a perfect signal.
        if state.category and state.frame.type_category is None:
            stated = state.category.lower()
            category = stated if stated in self.catalog.by_category else resolve_category(self.catalog, state.category)[0]
            if category:
                state.frame.propose_type(state.category, category, STATED_CATEGORY_CONFIDENCE)

        self.extractor.update(state.frame, user_message or "", turn)

        if turn > 1 and state.asked:
            # A question that taught us nothing counts against the current
            # strategy, whichever kind it was — that is the signal to switch.
            if len(state.constraints) > before:
                state.barren_asks = 0
            else:
                state.barren_asks += 1

        scored = ranking.rank_scored(self.catalog, state, PLANNING_POOL)
        state.top_scores.append(scored[0][0] if scored else 0.0)
        # Settle the order over the full shortlist first, then decide how much of
        # it to show. Reordering after truncation would leave the reranker with
        # nothing to do on the turns where the agent shows a single product — which
        # are exactly the turns where picking the right one matters most.
        verdict = self._consult_listwise(state, scored, [asin for _, asin in scored[:top_k]])
        shortlist = verdict.order
        width = self._slate_width(state, scored, top_k)
        if width == 1 and shortlist and scored and shortlist[0] != scored[0][1]:
            # The reranker overruled retrieval's leader. On a single-slot slate
            # that hides the leader entirely, and a wrong promotion costs a whole
            # turn. Showing both costs one of them rank 2 instead — the same
            # rank-versus-turn arithmetic that justified withholding in the first
            # place, applied to disagreement rather than uncertainty.
            width = 2
        recommendations = shortlist[:width]
        state.note_shown(recommendations, turn)
        state.last_slate = recommendations
        pool = [self.catalog.products[asin] for _, asin in scored if asin in self.catalog.products]
        attribute, question = self.policy.next_question(state, pool)

        # Rescue gate. A generated question is added only when the session is
        # visibly failing, and it is *appended* to the open probe rather than
        # replacing it, so the turn keeps the probe's breadth and gains a
        # discriminator. `ask_attribute` stays whatever the policy chose — the
        # scripted simulator reads only that field and never the prose, so the
        # wording is free there by construction.
        if (
            RESCUE_ENABLED
            and verdict.question
            and turn >= RESCUE_TURN
            and self._confidence_stalled(state)
        ):
            follow_up = verdict.question.strip().rstrip("?")
            if follow_up[:1].isupper() and not follow_up.split(" ")[0].isupper():
                follow_up = follow_up[0].lower() + follow_up[1:]
            question = f"{question.rstrip('?')} — for instance, {follow_up}?"
            state.rescued_turns += 1

        return {
            "message": question,
            "ask_attribute": attribute,
            "recommendations": [{"parent_asin": asin} for asin in recommendations],
            "usage": self._usage(),
        }
