"""Understanding what the customer just said.

Two layers, deliberately:

1. Template matching for the phrasings the simulator produces today.
2. A generic fallback that strips conversational lead-ins and keeps the payload.

The fallback matters: the specification warns that the organiser may paraphrase
customer turns, and states that paraphrasing "cannot decide correctness". An
agent that only pattern-matches today's exact strings would silently collapse.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .catalog import clean_text

BUYING = "buying"
BROWSING = "browsing"
OVERRIDE = "intent_override"
BOUNDARY = "boundary"
UNKNOWN = "unknown"

# Opening turns.
_OPEN_BUYING = re.compile(r"^\s*i'?m looking for (?P<cat>.+?)[.,]\s*a key requirement is:\s*(?P<c>.+?)\.?\s*$", re.I)
_OPEN_BROWSING = re.compile(r"^\s*i'?m looking for (?P<cat>.+?),\s*but i'?m still exploring\.?\s*$", re.I)
_OPEN_WITH_PREF = re.compile(r"^\s*i'?m looking for (?P<cat>.+?)\.\s*(?P<c>.+?)\s*$", re.I)
_OPEN_BARE = re.compile(r"^\s*i'?m looking for (?P<cat>.+?)\.?\s*$", re.I)

# Follow-up turns.
_DISCLOSURE = re.compile(r"what matters is:\s*(?P<c>.+?)\.?\s*$", re.I)
_OVERRIDE = re.compile(r"ignore my earlier preference.*?what i need is:\s*(?P<c>.+?)\.?\s*$", re.I)
_NO_PREFERENCE = re.compile(r"i don'?t have an additional preference for (?P<attr>[\w-]+)", re.I)
_NO_OPINION = re.compile(r"i don'?t have a preference for (?P<attr>[\w-]+).*use your judg", re.I)
_ASK_HARDER = re.compile(r"ask me about one specific attribute", re.I)

# Generic fallback: anything after a "here is what I want:" style lead-in.
_PAYLOAD = re.compile(r"(?:requirement is|what matters is|what i need is|need is|prefer|want)\s*:?\s*(?P<c>.+)$", re.I)


@dataclass(slots=True)
class Utterance:
    """Structured reading of one customer message."""
    category: str | None = None
    constraints: list[str] = field(default_factory=list)
    scenario_hint: str = UNKNOWN
    exhausted_attribute: str | None = None   # asked, and the customer has nothing more
    refused_attribute: str | None = None     # asked, but the customer declined to answer
    is_override: bool = False
    asked_for_specificity: bool = False


def _split_constraints(payload: str) -> list[str]:
    """Split a disclosure into individual requirements.

    Requirements are joined with '; ', but a quoted product bullet can itself
    contain a semicolon, so the unsplit payload is kept as a candidate too and
    scored on its own merits.
    """
    payload = payload.strip().rstrip(".")
    parts = [clean_text(part) for part in payload.split(";")]
    parts = [part for part in parts if part]
    whole = clean_text(payload)
    if whole and whole not in parts:
        parts.append(whole)
    return parts


def parse(message: str, turn: int) -> Utterance:
    text = (message or "").strip()
    result = Utterance()
    if not text:
        return result

    if _ASK_HARDER.search(text):
        result.asked_for_specificity = True
        return result

    match = _NO_OPINION.search(text)
    if match:
        # Boundary sessions stonewall the first question only; the attribute
        # itself is still worth asking again later.
        result.refused_attribute = match.group("attr").lower()
        result.scenario_hint = BOUNDARY
        return result

    match = _NO_PREFERENCE.search(text)
    if match:
        result.exhausted_attribute = match.group("attr").lower()
        return result

    match = _OVERRIDE.search(text)
    if match:
        result.is_override = True
        result.scenario_hint = OVERRIDE
        result.constraints = _split_constraints(match.group("c"))
        return result

    if turn == 1:
        match = _OPEN_BUYING.match(text)
        if match:
            result.scenario_hint = BUYING
            result.category = clean_text(match.group("cat"))
            result.constraints = _split_constraints(match.group("c"))
            return result
        match = _OPEN_BROWSING.match(text)
        if match:
            result.scenario_hint = BROWSING
            result.category = clean_text(match.group("cat"))
            return result
        match = _OPEN_WITH_PREF.match(text)
        if match:
            # A stated preference that is not flagged as a hard requirement:
            # characteristic of a session that will later be overridden.
            result.scenario_hint = OVERRIDE
            result.category = clean_text(match.group("cat"))
            result.constraints = _split_constraints(match.group("c"))
            return result
        match = _OPEN_BARE.match(text)
        if match:
            result.category = clean_text(match.group("cat"))
            return result

    match = _DISCLOSURE.search(text)
    if match:
        result.constraints = _split_constraints(match.group("c"))
        return result

    # Unrecognised phrasing: salvage the payload rather than discarding the turn.
    match = _PAYLOAD.search(text)
    if match:
        result.constraints = _split_constraints(match.group("c"))
    elif turn > 1:
        salvaged = clean_text(text)
        if salvaged:
            result.constraints = [salvaged]
    return result
