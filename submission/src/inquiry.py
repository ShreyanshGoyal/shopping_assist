"""Choosing the question that splits the candidate set best.

A clarification turn is only worth spending if the answer changes what the agent
would show. Asking about colour when every remaining candidate is black learns
nothing; asking when the field is evenly split between black and blue halves it.

So each askable attribute is scored by the entropy of its value distribution
across the products still in contention, discounted by how many of those products
have a known value for it at all. The agent asks the attribute with the highest
expected information gain, and names the leading values in the question so the
customer can answer in one word.

Attribute vocabularies start from a small seed list and are widened using the
catalog-mined lexicon, so the words the agent recognises are the words this
catalog actually uses.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from .catalog import Catalog, Product
from .lexicon import normalize

# Seeds only. The lexicon widens each of these from real product titles.
ATTRIBUTE_SEEDS: dict[str, tuple[str, ...]] = {
    "color": ("black", "white", "blue", "red", "pink", "green", "brown", "gray", "purple", "navy"),
    "material": ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "denim", "mesh"),
    "size": ("petite", "plus", "wide", "narrow", "tall", "slim", "regular", "oversized"),
    "style": ("casual", "formal", "skinny", "relaxed", "vintage", "classic", "bohemian", "sporty", "elegant"),
    "use_case": ("running", "hiking", "gym", "work", "wedding", "beach", "travel", "yoga", "office", "winter"),
    "feature": ("pockets", "waterproof", "breathable", "stretch", "lightweight", "adjustable", "zipper", "elastic"),
}

# Attributes read straight off structured fields rather than title vocabulary.
STRUCTURED = ("brand", "budget", "category")

MIN_GAIN = 0.25          # below this, asking is not worth a turn
SAMPLE_CEILING = 400     # candidates inspected when estimating a split


@dataclass(frozen=True)
class Question:
    attribute: str
    values: tuple[str, ...]
    gain: float


def _price_band(product: Product) -> str | None:
    if product.price_value is None:
        return None
    price = product.price_value
    if price < 15:
        return "under $15"
    if price < 30:
        return "$15-30"
    if price < 60:
        return "$30-60"
    return "over $60"


class InquiryPlanner:
    """Ranks candidate questions by how much they would narrow the field."""

    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog
        self.vocabulary: dict[str, dict[str, str]] = {}
        for attribute, seeds in ATTRIBUTE_SEEDS.items():
            mapping: dict[str, str] = {}
            for seed in seeds:
                token = normalize(seed)
                mapping[token] = seed
                # Widen with words this catalog actually uses alongside the seed,
                # so 'stretch' also catches whatever this catalog calls it.
                for neighbour, similarity in catalog.lexicon.neighbours(token, k=4):
                    if similarity > 0.5:
                        mapping.setdefault(neighbour, seed)
            self.vocabulary[attribute] = mapping

    def _values(self, attribute: str, products: list[Product]) -> list[str]:
        if attribute == "brand":
            return [p.store for p in products if p.store]
        if attribute == "budget":
            return [band for band in (_price_band(p) for p in products) if band]
        if attribute == "category":
            return [p.coarse_category for p in products if p.coarse_category]
        mapping = self.vocabulary.get(attribute, {})
        found: list[str] = []
        for product in products:
            for term in product.title_terms:
                canonical = mapping.get(term)
                if canonical:
                    found.append(canonical)
                    break
        return found

    @staticmethod
    def _entropy(counts: Counter[str]) -> float:
        total = sum(counts.values())
        if total <= 1:
            return 0.0
        return -sum((n / total) * math.log2(n / total) for n in counts.values() if n)

    def rank_questions(self, products: list[Product], skip: set[str]) -> list[Question]:
        sample = products[:SAMPLE_CEILING]
        if len(sample) < 2:
            return []
        results: list[Question] = []
        for attribute in (*ATTRIBUTE_SEEDS, *STRUCTURED):
            if attribute in skip:
                continue
            values = self._values(attribute, sample)
            if len(values) < 2:
                continue
            counts = Counter(values)
            if len(counts) < 2:
                # Every candidate agrees: the answer cannot change the ranking.
                continue
            coverage = len(values) / len(sample)
            gain = self._entropy(counts) * coverage
            leading = tuple(value for value, _ in counts.most_common(2))
            results.append(Question(attribute, leading, gain))
        results.sort(key=lambda q: q.gain, reverse=True)
        return results

    def best(self, products: list[Product], skip: set[str]) -> Question | None:
        for question in self.rank_questions(products, skip):
            if question.gain >= MIN_GAIN:
                return question
        return None


def phrase(question: Question) -> str:
    """A natural question that names the leading options."""
    options = [value for value in question.values if value]
    choice = " or ".join(options[:2]) if len(options) >= 2 else (options[0] if options else "")
    templates = {
        "color": f"Any colour preference — {choice}?",
        "material": f"What material are you after — {choice}?",
        "size": f"What sizing do you need — {choice}?",
        "style": f"What style are you going for — {choice}?",
        "use_case": f"What will you be using it for — {choice}?",
        "feature": f"Anything specific it needs — {choice}?",
        "brand": f"Any brand preference — {choice}?",
        "budget": f"What's your budget — {choice}?",
        "category": f"Which are you after — {choice}?",
    }
    return templates.get(question.attribute, f"Do you have a preference for {question.attribute}?")
