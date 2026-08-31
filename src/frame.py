"""The structured query frame.

The agent's previous failure mode was architectural: it represented a
conversation as a bag of words that only ever grew, and searched over the whole
bag. Drift followed inevitably — common vocabulary accumulated until the largest
category in the catalog outvoted the one the customer actually named — and a bag
has nowhere to put "not pajamas".

A frame separates the three things a shopping request is made of:

* **type** — the thing itself ("flat sandals"). One slot. It is *sticky*: new
  evidence does not outvote it, only an explicit contradiction replaces it.
* **attributes** — modifiers of that thing (material, colour, budget, features).
  These accumulate.
* **negatives** — what the customer has ruled out. These subtract.

Because type is a slot rather than a tally, drift is not something to be tuned
against. It cannot happen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .catalog import Catalog
from .lexicon import normalize, terms as lexicon_terms
from .state import Constraint

# A challenger must beat the sitting type by this margin to replace it without an
# explicit contradiction. Stickiness is the whole point; make it hard to move.
CHALLENGE_MARGIN = 1.6

# How much more the head noun of a request counts than its modifiers when
# resolving a category name.
HEAD_WEIGHT = 4.0

# Above this, a lexical name match is trusted outright; below it, meaning decides.
LEXICAL_TRUST = 0.85
# Encoder cosines sit in a narrow band; below this nothing useful was found.
SEMANTIC_FLOOR = 0.60


@dataclass
class Frame:
    type_phrase: str | None = None        # canonical wording, e.g. "flat sandals"
    type_category: str | None = None      # resolved catalog category, lowercased
    type_confidence: float = 0.0
    attributes: list[Constraint] = field(default_factory=list)
    negatives: set[str] = field(default_factory=set)
    negative_categories: set[str] = field(default_factory=set)

    def propose_type(
        self, phrase: str, category: str | None, confidence: float, *, contradicted: bool = False
    ) -> bool:
        """Offer a new product type. Returns whether it was accepted.

        Accepted when there is no sitting type, when the customer has explicitly
        contradicted the sitting one, or when the challenger is decisively
        stronger. Otherwise the frame holds its ground — which is what stops a
        stray adjective from turning a sandal into a t-shirt.
        """
        if not phrase:
            return False
        if self.type_category and category == self.type_category and not contradicted:
            self.type_confidence = max(self.type_confidence, confidence)
            return False
        if (
            self.type_phrase is None
            or contradicted
            or confidence >= self.type_confidence * CHALLENGE_MARGIN
        ):
            self.type_phrase = phrase
            self.type_category = category
            self.type_confidence = confidence
            self.negative_categories.discard(category or "")
            return True
        return False

    def reject_type(self, catalog: Catalog) -> None:
        """The customer has ruled out the current type. Vacate the slot."""
        if self.type_category:
            self.negative_categories.add(self.type_category)
        self.type_phrase = None
        self.type_category = None
        self.type_confidence = 0.0

    def add_negative(self, text: str) -> None:
        for term in lexicon_terms(text):
            self.negatives.add(term)

    def describe(self) -> str:
        """The frame as a search string, type first."""
        parts = [self.type_phrase] if self.type_phrase else []
        parts.extend(c.text for c in self.attributes)
        return " ".join(p for p in parts if p)


def resolve_category(catalog: Catalog, phrase: str) -> tuple[str | None, float]:
    """Map a short product-type phrase onto a catalog category.

    Matching a canonical phrase like "flat sandals" against 1,115 category names
    is a far easier problem than routing a rambling conversation, which is why
    the extractor's job is to produce that phrase in the first place.

    Overlap is weighted by how informative each word is. Counting words equally
    resolved "men's short-sleeve shirt" to *men shorts*: stemming collapses
    "shorts" and "short" together, and the two common words "men" and "short"
    then outvoted the one word that actually identified the garment. Weighting by
    inverse document frequency makes "shirt" decisive and "men" nearly free.
    """
    ordered = [normalize(t) for t in lexicon_terms(phrase)]
    wanted = set(ordered)
    if not wanted:
        return None, 0.0

    # In an English noun phrase the head noun comes last, and it is the word that
    # says what the thing *is* — "men's short-sleeve shirt" is a shirt, not a
    # short. Weighting the head above its modifiers stops incidental words from
    # carrying a match, which is what sent that phrase to *men shorts*.
    head = ordered[-1]

    def mass(terms) -> float:
        return sum(catalog.lexicon.idf(t) * (HEAD_WEIGHT if t == head else 1.0) for t in terms) or 1e-9

    wanted_mass = mass(wanted)
    best_name, best_score = None, 0.0
    for name in catalog.by_category:
        name_terms = {normalize(t) for t in lexicon_terms(name)}
        if not name_terms:
            continue
        overlap = wanted & name_terms
        if not overlap:
            continue
        overlap_mass = mass(overlap)
        precision = overlap_mass / mass(name_terms)
        recall = overlap_mass / wanted_mass
        score = 2 * precision * recall / (precision + recall)
        if score > best_score:
            best_name, best_score = name, score

    # A lexical match on a category name is decisive when it is near-total, and
    # unreliable otherwise — the words that identify a category are often not the
    # words in its name. Below that bar, ask the encoder what the phrase means.
    import os

    if (
        best_score < LEXICAL_TRUST
        and catalog.dense is not None
        and os.environ.get("AGENT_SEMANTIC_CATEGORY", "1") == "1"
    ):
        semantic = catalog.dense.resolve_category(phrase, limit=3)
        if semantic:
            name, similarity = semantic[0]
            if similarity >= SEMANTIC_FLOOR and similarity > best_score:
                return name, min(similarity, 0.95)

    if best_name is None:
        # Nothing matched by name; fall back to where these words occur.
        routed = catalog.lexicon.classify(phrase, limit=1)
        if routed:
            return routed[0][0], 0.35
    return best_name, best_score
