"""Listwise reranking of the shortlist by a language model.

The cross-encoder attempt failed for two diagnosed reasons: it was handed a frame
description rather than anything resembling natural language, and it had no access
to the domain evidence the retrieval score already carries. This has neither
defect. It reads the shopper's own words verbatim, and it is given bounded
authority over an ordering that retrieval has already decided.

Bounded is the operative word. The model is consulted only when retrieval is
genuinely undecided — when the leading candidate has no clear margin — and its
answer promotes a candidate rather than replacing the ranking. When a verbatim
quote has already identified the product, or the leader is well ahead, the call is
skipped entirely: no tokens, no latency, no risk to a ranking that was already
right.

Optional on the same terms as everything else: no client, no network, or a
malformed reply all leave the retrieval order untouched.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

MAX_CANDIDATES = 10      # candidates shown to the model when selecting is off
WIDE_CANDIDATES = 30     # candidates shown when selecting from a deeper pool
MAX_UTTERANCES = 6

ATTRIBUTES = (
    "material", "color", "size", "style", "brand",
    "budget", "feature", "use_case", "category", "other",
)

PROMPT = """A shopper is looking for one specific product. These are their own \
words, in order:

{utterances}

Here are the candidates a search system returned.

{candidates}

Do two things.

First, decide which single candidate the shopper is most likely asking for. Judge \
by what they actually asked for, not by which product sounds most appealing. If \
several fit equally, prefer the one matching their most specific requirement.

Second, decide whether a question would help. If no candidate clearly matches \
everything the shopper has said, ask ONE short question about the concrete \
property that best separates the candidates still in contention — the difference \
that would actually rule some of them out. Use plain shopper language, never \
catalog jargon, and never ask about something they have already told you. If one \
candidate already matches everything stated, return an empty question.

Reply with only a JSON object, no other text:
{{"best": <candidate number>, "runner_up": <candidate number>, \
"question": "<one short question, or empty string>", \
"question_attribute": "<one of: material, color, size, style, brand, budget, \
feature, use_case, category, other>"}}"""

# Selecting from a deeper pool is a strictly harder task, so it is asked as an
# extension of the one that works rather than a replacement: name the best pick
# first, exactly as before, then nominate the rest of the shortlist. A malformed
# or missing shortlist therefore degrades to the proven behaviour instead of
# losing it.
WIDE_PROMPT = """A shopper is looking for one specific product. These are their \
own words, in order:

{utterances}

Here are {count} candidates a search system returned, roughly in its order of \
confidence. Its ordering is unreliable below the first few.

{candidates}

Judge by what the shopper actually asked for, not by which product sounds most \
appealing. If several fit equally, prefer the one matching their most specific \
requirement.

Reply with only a JSON object, no other text:
{{"best": <candidate number>, "runner_up": <candidate number>, \
"shortlist": [<the 10 candidate numbers worth showing, best first>]}}"""


def describe(product) -> str:
    bullets = sorted(product.quotable, key=len, reverse=True)[:2]
    parts = [product.title[:110]]
    if product.coarse_category:
        parts.append(f"[{product.coarse_category}]")
    parts.extend(b[:90] for b in bullets)
    return " | ".join(parts)


@dataclass
class Verdict:
    """What the turn-planner call decided."""
    order: list[str] = field(default_factory=list)
    question: str = ""
    attribute: str | None = None


@dataclass
class ListwiseReranker:
    client: object
    catalog: object
    wide: bool = False
    failures: int = 0
    max_failures: int = 3
    calls: int = 0
    skipped: int = 0
    rescued: int = 0     # times a pick came from beyond retrieval's top 10
    questions: int = 0   # times the model chose to ask something

    def reorder(self, utterances: list[str], ranked: list[str]) -> Verdict:
        """Promote the model's pick and optionally supply a question.

        Returns the ranking unchanged, and no question, on any failure.
        """
        if self.failures >= self.max_failures or len(ranked) < 2:
            return Verdict(order=ranked)
        depth = WIDE_CANDIDATES if self.wide else MAX_CANDIDATES
        candidates = ranked[:depth]
        products = [self.catalog.products.get(a) for a in candidates]
        if any(p is None for p in products):
            return Verdict(order=ranked)

        spoken = "\n".join(f'  "{u}"' for u in utterances[-MAX_UTTERANCES:] if u.strip())
        listing = "\n".join(f"{i}. {describe(p)}" for i, p in enumerate(products, 1))
        prompt = (
            WIDE_PROMPT.format(utterances=spoken, candidates=listing, count=len(candidates))
            if self.wide
            else PROMPT.format(utterances=spoken, candidates=listing)
        )

        try:
            raw = self.client.generate(prompt, [{"role": "user", "text": "Which one?"}])
            payload = _parse(raw)
        except Exception:
            self.failures += 1
            return Verdict(order=ranked)
        if payload is None:
            self.failures += 1
            return Verdict(order=ranked)

        self.calls += 1
        order: list[str] = []
        for key in ("best", "runner_up"):
            index = payload.get(key)
            if isinstance(index, int) and 1 <= index <= len(candidates):
                asin = candidates[index - 1]
                if asin not in order:
                    order.append(asin)
                    if index > MAX_CANDIDATES:
                        self.rescued += 1
        # The shortlist is advisory and additive: it fills the slate behind the
        # picks, and anything malformed is simply skipped.
        for index in payload.get("shortlist") or []:
            if isinstance(index, int) and 1 <= index <= len(candidates):
                asin = candidates[index - 1]
                if asin not in order:
                    order.append(asin)
        if not order:
            return Verdict(order=ranked)

        # The scripted simulator acts only on `ask_attribute` and ignores the prose
        # entirely, so an attribute outside the allowed set would silently become
        # "other" there and could mislead here. Validate it rather than trust it.
        question = payload.get("question")
        question = question.strip() if isinstance(question, str) else ""
        attribute = payload.get("question_attribute")
        if not isinstance(attribute, str) or attribute not in ATTRIBUTES:
            attribute = "other"
        if question:
            self.questions += 1

        return Verdict(
            order=order + [a for a in ranked if a not in order],
            question=question,
            attribute=attribute if question else None,
        )


def _parse(raw: str) -> dict | None:
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text, flags=re.S)
    match = re.search(r"\{.*?\}", text, re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
