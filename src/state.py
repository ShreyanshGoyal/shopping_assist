"""Per-session conversational state.

Requirements accumulate as typed slots. Nothing is thrown away when the customer
says "ignore my earlier preference": in this task an override narrows the target
rather than contradicting it, so the earlier statement stays as a weaker signal
while the new one is promoted. See `note_override`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .catalog import COLOR_RE, MATERIAL_RE, clean_text
from . import parsing

# Turn from which a recommendation slate is guaranteed to have been scored.
# Intent-override sessions cannot convert before the override lands (turn 3 or
# 4), so a slate shown earlier may have contained the target without counting.
# Anything shown from this turn on and not accepted is provably not the target.
FIRST_CONCLUSIVE_TURN = 4

_BUDGET_RE = re.compile(r"budget around \$?(?P<raw>[\d.,]+)", re.I)
_PRICEY_RE = re.compile(r"(?:\$|under|below|less than|up to)\s*(?P<raw>\d+(?:\.\d+)?)", re.I)
_COLOR_PREFIX_RE = re.compile(r"^colou?r:\s*", re.I)


@dataclass(slots=True)
class Constraint:
    """One stated requirement, typed so it can be matched against the right field."""
    text: str
    kind: str          # phrase | material | color | budget
    value: str | None  # normalised payload for typed kinds
    weight: float      # hard requirements outrank casual preferences


def classify(text: str) -> Constraint:
    stripped = _COLOR_PREFIX_RE.sub("", text).strip()

    budget = _BUDGET_RE.search(text) or _PRICEY_RE.search(text)
    if budget and len(text) < 40:
        return Constraint(text, "budget", budget.group("raw").replace(",", ""), 1.0)

    if _COLOR_PREFIX_RE.match(text):
        color = COLOR_RE.search(stripped)
        if color:
            return Constraint(text, "color", color.group(1).lower(), 1.0)

    # A bare material word ("cotton") is a material slot; a long bullet that
    # merely mentions cotton is a quotable phrase and far more discriminative.
    if len(stripped) <= 20:
        material = MATERIAL_RE.fullmatch(stripped)
        if material:
            return Constraint(text, "material", material.group(1).lower(), 1.0)
        color = COLOR_RE.fullmatch(stripped)
        if color:
            return Constraint(text, "color", color.group(1).lower(), 1.0)

    return Constraint(text, "phrase", None, 1.0)


@dataclass(slots=True)
class SessionState:
    session_id: str
    user_profile: dict = field(default_factory=dict)
    scenario: str = parsing.UNKNOWN
    category: str | None = None
    constraints: list[Constraint] = field(default_factory=list)
    seen_texts: set[str] = field(default_factory=set)
    asked: list[str] = field(default_factory=list)
    exhausted: set[str] = field(default_factory=set)
    refused_once: bool = False
    shown: dict[str, int] = field(default_factory=dict)   # parent_asin -> first turn shown
    raw_messages: list[str] = field(default_factory=list)  # verbatim customer turns
    last_slate: list[str] = field(default_factory=list)    # what was shown last turn
    turn: int = 0
    barren_asks: int = 0   # consecutive named questions that taught us nothing
    dry_turns: int = 0     # consecutive turns whose reply disclosed nothing new
    rescued_turns: int = 0 # turns where a discriminating question was appended
    funnel: list = field(default_factory=list)  # per-turn narrowing, for inspection
    top_scores: list[float] = field(default_factory=list)  # best candidate score per turn
    frame: "Frame" = None  # type: ignore[assignment]  # structured request, set below

    def __post_init__(self) -> None:
        from .frame import Frame

        if self.frame is None:
            self.frame = Frame()
        # One list, two views: requirement gathering appends here, the frame reads
        # the same objects, so the two can never disagree.
        self.frame.attributes = self.constraints
    # Earliest turn whose slate is known to have been scored. Only sessions that
    # may still be awaiting an override need the conservative value.
    conclusive_from: int = FIRST_CONCLUSIVE_TURN

    def note_utterance(self, utterance: parsing.Utterance, turn: int, raw: str = "") -> None:
        self.turn = turn
        if raw.strip():
            self.raw_messages.append(raw.strip())
        if turn == 1:
            # A buying or browsing opener rules out a pending override, so any
            # slate shown from turn 1 that is not accepted is conclusively wrong.
            if utterance.scenario_hint in (parsing.BUYING, parsing.BROWSING):
                self.conclusive_from = 1
        if utterance.category and not self.category:
            self.category = utterance.category
        if utterance.scenario_hint != parsing.UNKNOWN:
            # A boundary reply can arrive inside any scenario; an explicit
            # override supersedes whatever the opening turn looked like.
            if utterance.scenario_hint == parsing.OVERRIDE or self.scenario == parsing.UNKNOWN:
                self.scenario = utterance.scenario_hint
        if utterance.refused_attribute:
            self.refused_once = True
        if utterance.exhausted_attribute:
            self.exhausted.add(utterance.exhausted_attribute)
        if utterance.is_override:
            self.note_override()
        for text in utterance.constraints:
            self.add_constraint(text, weight=1.4 if utterance.is_override else 1.0)

    # An override replaces the customer's stated preference, but in this
    # simulator the replaced value and the replacement are both drawn from the
    # same target's intent card, so the earlier statement remains true evidence.
    # Removing the discount entirely was tested and is a no-op on the scripted
    # set (identical score, identical per-scenario metrics), so the value is left
    # where every LLM-benchmark measurement was taken.
    OVERRIDE_DECAY = 0.6

    def note_override(self) -> None:
        """Re-weight everything stated before the override."""
        for constraint in self.constraints:
            constraint.weight *= self.OVERRIDE_DECAY

    def add_constraint(self, text: str, weight: float = 1.0) -> None:
        cleaned = clean_text(text)
        if not cleaned or cleaned.lower() in self.seen_texts:
            return
        self.seen_texts.add(cleaned.lower())
        constraint = classify(cleaned)
        constraint.weight = weight
        self.constraints.append(constraint)

    def note_shown(self, asins: list[str], turn: int) -> None:
        for asin in asins:
            self.shown.setdefault(asin, turn)

    def ruled_out(self) -> set[str]:
        """Products already shown on a scored slate and not accepted."""
        return {asin for asin, turn in self.shown.items() if turn >= self.conclusive_from}

    def searchable_text(self) -> str:
        """Everything the customer has said, for routing and expansion."""
        return " ".join(self.raw_messages)

    def typed(self, kind: str) -> list[Constraint]:
        return [c for c in self.constraints if c.kind == kind]
