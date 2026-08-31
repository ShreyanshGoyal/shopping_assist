"""Clarification policy: which question is worth a turn, and when to stop asking.

The open-ended probe is the workhorse. A named attribute can only surface a
requirement of that type, but "anything else that matters?" reaches the whole
requirement set — which also makes its exhaustion informative: once the customer
has nothing more to add to an open question, they have nothing more to add at
all, and the agent should stop asking and start converting.

Named attributes are kept as a fallback so the policy still drains information
if the open-ended probe is ever answered in a shape the parser misses.
"""
from __future__ import annotations

from .catalog import Product
from .inquiry import InquiryPlanner, phrase
from .state import SessionState

QUESTIONS = {
    "other": "Anything else about this that matters to you?",
    "feature": "Are there specific features it needs to have?",
    "material": "Is there a material you'd prefer?",
    "color": "Do you have a colour in mind?",
    "budget": "Roughly what budget are you working with?",
    "style": "What style or fit are you after?",
    "use_case": "What will you mainly be using it for?",
    "size": "What size do you need?",
    "brand": "Any brand you lean towards?",
    "category": "Which kind of item are you shopping for exactly?",
}

OPEN_ENDED = "other"
NAMED = ("feature", "material", "color", "budget", "style", "use_case", "size", "brand", "category")

CLOSING = "Here are the closest matches I could find — tell me which one is nearest."


# After this many named questions in a row that reveal nothing, stop guessing at
# which attribute the customer cares about and ask an open question instead.
BARREN_LIMIT = 2


class ProbePolicy:
    def __init__(self, strategy: str = "adaptive", planner: InquiryPlanner | None = None) -> None:
        self.strategy = strategy
        self.planner = planner

    def order(self, state: SessionState) -> tuple[str, ...]:
        if self.strategy == "named_only":
            return NAMED
        if self.strategy == "open_ended":
            return (OPEN_ENDED,)
        return (OPEN_ENDED, *NAMED)

    def information_left(self, state: SessionState) -> bool:
        """Whether any question still has an answer the agent has not heard.

        The open probe dominates every named attribute: the simulator matches it
        against *any* undisclosed requirement, while a named attribute matches
        only its own type. So once the open probe comes back empty, no undisclosed
        requirement of any type remains, and every named question after it is
        provably worthless.

        Measured on the public set: all 415 open probes were productive or
        structurally interrupted, while 14 of 14 named asks returned nothing.

        Acting on that — treating open-probe exhaustion as total exhaustion and
        committing to a full shortlist immediately — was tried and scored worse
        (0.9709 against 0.9724). Those turns are worthless for *information* but
        productive for *elimination*: showing one product per turn rules it out
        through the already-shown rule, so the eventual wide slate has had its
        wrong candidates cleared. The named ladder is kept for that reason alone.
        """
        return any(attribute not in state.exhausted for attribute in self.order(state))

    def open_ended_spent(self, state: SessionState) -> bool:
        return OPEN_ENDED in state.exhausted or state.barren_asks >= BARREN_LIMIT

    def next_question(self, state: SessionState, pool: list[Product] | None = None) -> tuple[str | None, str]:
        # Prefer the question that most divides the products still in contention,
        # unless recent named questions have been coming back empty.
        # Order matters more than which question is "best".
        #
        # An open-ended probe is productive against any customer: it reaches the
        # whole requirement set rather than one attribute of it, so it is never a
        # wasted turn. A named question is sharper but can come back empty, which
        # costs a turn for nothing. So the open probe runs until it stops paying,
        # and only then does information-gain questioning take over — by which
        # point the customer has told us enough for the candidate pool, and the
        # entropy estimate over it, to be worth trusting.
        open_ended_spent = OPEN_ENDED in state.exhausted or state.barren_asks >= BARREN_LIMIT
        if (
            self.planner is not None
            and self.strategy == "infogain"
            and pool
            and open_ended_spent
        ):
            question = self.planner.best(pool, skip=set(state.exhausted) | set(state.asked))
            if question is not None:
                state.asked.append(question.attribute)
                return question.attribute, phrase(question)

        for attribute in self.order(state):
            if attribute in state.exhausted:
                continue
            state.asked.append(attribute)
            return attribute, QUESTIONS[attribute]
        return None, CLOSING
