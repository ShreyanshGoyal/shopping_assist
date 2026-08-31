"""A shopping vocabulary mined from the catalog itself.

Customers do not speak in category paths. They say "pants" when the catalog says
"Sport Specific Clothing Basketball", and "jeggings" when it says "Pull-on Denim
Legging". Rather than hand-authoring a thesaurus — fashion vocabulary is huge and
long-tail, and any list we wrote would have holes exactly where the catalog is
dense — the mapping is learned from 50,000 real product titles.

Two structures come out of one pass:

* **term to category**, so an everyday word routes to the parts of the catalog
  where it actually occurs;
* **category profiles per term**, so two words used in the same places can be
  recognised as near-synonyms without any external model.

Both are counts over the frozen catalog. No training, no embeddings, no network.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[a-z][a-z0-9]+")

# Words that carry no shopping intent; kept small on purpose, since the inverse
# document frequency weighting already discounts anything common.
NOISE = {
    "the", "and", "for", "with", "you", "your", "this", "that", "are", "not",
    "have", "has", "was", "but", "can", "will", "just", "some", "any", "get",
    "looking", "look", "want", "need", "like", "would", "really", "something",
    "please", "help", "find", "show", "still", "one", "more", "much", "very",
    "them", "those", "these", "什么",
}


def normalize(token: str) -> str:
    """Crude morphological folding so 'sandals' and 'sandal' agree."""
    if len(token) > 4:
        if token.endswith("ies"):
            return token[:-3] + "y"
        if token.endswith("sses") or token.endswith("shes") or token.endswith("ches"):
            return token[:-2]
        if token.endswith("s") and not token.endswith("ss"):
            return token[:-1]
    return token


def terms(text: str) -> list[str]:
    return [
        normalize(token)
        for token in TOKEN_RE.findall(text.lower())
        if len(token) > 2 and token not in NOISE
    ]


class Lexicon:
    """Term statistics over the catalog, used for routing and expansion."""

    def __init__(self) -> None:
        self._term_cat: dict[str, Counter[str]] = defaultdict(Counter)
        self._term_df: Counter[str] = Counter()
        self._cat_terms: dict[str, Counter[str]] = defaultdict(Counter)
        self._cat_size: Counter[str] = Counter()
        self._documents = 0
        self._idf: dict[str, float] = {}
        self._neighbours: dict[str, list[tuple[str, float]]] = {}

    # -- construction ------------------------------------------------------
    def add(self, text: str, category: str) -> None:
        self._documents += 1
        self._cat_size[category] += 1
        seen = set(terms(text))
        for term in seen:
            self._term_cat[term][category] += 1
            self._term_df[term] += 1
            self._cat_terms[category][term] += 1

    def finalize(self) -> None:
        total = max(self._documents, 1)
        self._idf = {
            term: math.log(1.0 + total / (1.0 + count))
            for term, count in self._term_df.items()
        }

    # -- lookups -----------------------------------------------------------
    def idf(self, term: str) -> float:
        return self._idf.get(term, math.log(1.0 + self._documents))

    def known(self, term: str) -> bool:
        return term in self._term_df

    def route(self, text: str, limit: int = 8) -> list[tuple[str, float]]:
        """Categories the customer's own words point at, best first.

        Each word votes for the categories it occurs in, in proportion to how
        concentrated it is there, weighted by how informative the word is. A word
        appearing everywhere contributes almost nothing; a word appearing in three
        categories contributes a lot.
        """
        scores: Counter[str] = Counter()
        for term in set(terms(text)):
            buckets = self._term_cat.get(term)
            if not buckets:
                continue
            occurrences = sum(buckets.values())
            if occurrences < 2:
                continue
            weight = self.idf(term)
            for category, count in buckets.most_common(20):
                scores[category] += weight * (count / occurrences)
        return scores.most_common(limit)

    def classify(self, text: str, limit: int = 8, top_terms: int = 6) -> list[tuple[str, float]]:
        """Which categories does this wording point at?

        Scored by pointwise mutual information rather than raw counts. A plain
        vote-sum lets the biggest bucket win on volume — "Shirts T-Shirts" holds
        1,354 products, so it contains a little of every common word and wins by
        size alone. PMI asks the sharper question: is this word *more* common in
        this category than in the catalog at large? A large bucket has to earn its
        score by concentration, not accumulate it by breadth.

        Only the most informative words are considered. A shopper names the thing
        they want in a handful of nouns; the rest of the sentence is filler that
        should not get a vote.
        """
        candidates = sorted(
            {term for term in terms(text) if self._term_df.get(term, 0) >= 2},
            key=self.idf,
            reverse=True,
        )[:top_terms]
        if not candidates:
            return []

        scores: Counter[str] = Counter()
        total = max(self._documents, 1)
        for term in candidates:
            buckets = self._term_cat.get(term)
            if not buckets:
                continue
            term_probability = self._term_df[term] / total
            weight = self.idf(term)
            for category, count in buckets.most_common(40):
                size = self._cat_size.get(category, 0)
                if size < 2:
                    continue
                lift = (count / size) / term_probability
                if lift > 1.0:
                    scores[category] += weight * math.log(lift)
        return scores.most_common(limit)

    def neighbours(self, term: str, k: int = 6) -> list[tuple[str, float]]:
        """Words used in the same parts of the catalog as this one.

        Similarity is cosine between the two words' category distributions, which
        makes 'jeggings' close to 'leggings' and 'denim' without anyone writing
        that down. Computed lazily and cached, since only query words need it.
        """
        if term in self._neighbours:
            return self._neighbours[term]
        buckets = self._term_cat.get(term)
        if not buckets or sum(buckets.values()) < 3:
            self._neighbours[term] = []
            return []

        norm = math.sqrt(sum(v * v for v in buckets.values()))
        candidates: Counter[str] = Counter()
        for category, _ in buckets.most_common(4):
            for other, count in self._cat_terms[category].most_common(120):
                if other != term:
                    candidates[other] += count

        scored: list[tuple[str, float]] = []
        for other in candidates:
            other_buckets = self._term_cat.get(other)
            if not other_buckets or self._term_df[other] < 3:
                continue
            shared = set(buckets) & set(other_buckets)
            if not shared:
                continue
            dot = sum(buckets[c] * other_buckets[c] for c in shared)
            other_norm = math.sqrt(sum(v * v for v in other_buckets.values()))
            if not other_norm:
                continue
            similarity = dot / (norm * other_norm)
            if similarity > 0.35:
                scored.append((other, similarity))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        self._neighbours[term] = scored[:k]
        return self._neighbours[term]

    def expand(self, text: str, per_term: int = 3) -> dict[str, float]:
        """Query terms plus near-synonyms, each with a confidence weight.

        Original words keep full weight; mined synonyms are discounted by their
        similarity, so expansion can add recall without swamping the evidence
        that the customer actually supplied.
        """
        weights: dict[str, float] = {}
        for term in set(terms(text)):
            weights[term] = max(weights.get(term, 0.0), 1.0)
            for other, similarity in self.neighbours(term, per_term):
                weights[other] = max(weights.get(other, 0.0), similarity * 0.6)
        return weights
