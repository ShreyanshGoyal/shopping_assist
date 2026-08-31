"""Catalog loading and normalisation.

The simulated customer quotes product metadata almost verbatim, so the agent
normalises catalog text the same way a shopper would paraphrase it: whitespace
collapsed, surrounding punctuation stripped, long bullets truncated. Matching a
stated requirement against these normalised forms is what turns a vague request
into a small candidate set.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .lexicon import Lexicon, terms as lexicon_terms

CONSTRAINT_LIMIT = 180

MATERIALS = ("cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric")
MATERIAL_RE = re.compile(r"\b(cotton|polyester|nylon|leather|wool|spandex|silk|rayon|fabric)\b", re.I)
COLOR_RE = re.compile(r"\b(black|white|blue|red|pink|green|brown|gray|grey|purple|yellow|orange)\b", re.I)

# Mirrors the order in which a shopper skims a product page; the first material
# or colour mentioned in this order is the one they will name.
SEARCH_FIELDS = ("title", "features", "details", "description", "categories", "store")

_EXCLUDED_CATEGORY_PARTS = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}

# How sharply review volume separates plausible targets. Swept with each half of
# the public set held out; see NOTES.md.
POPULARITY_EXPONENT = 0.5

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "i", "in", "is", "it",
    "me", "my", "of", "on", "or", "please", "some", "that", "the", "this", "to", "want", "with",
    "would", "you", "looking", "im", "still", "exploring", "key", "requirement", "matters",
}


def clean_text(value: str, limit: int = CONSTRAINT_LIMIT) -> str:
    """Collapse whitespace, strip framing punctuation, and truncate."""
    return re.sub(r"\s+", " ", value).strip(" -;,.\t\n")[:limit].rstrip()


def flatten_values(value: object) -> list[str]:
    """Render a metadata field as the list of phrases a shopper could quote."""
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def searchable_text(product: dict) -> str:
    parts: list[str] = []
    for field_name in SEARCH_FIELDS:
        value = product.get(field_name)
        if isinstance(value, dict):
            parts.extend(f"{key} {item}" for key, item in value.items())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return " ".join(parts).strip()


def coarse_category(values: list[str]) -> str:
    """The two most specific levels of a category path, e.g. 'Earrings Hoop'."""
    cleaned: list[str] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part and part.lower() not in _EXCLUDED_CATEGORY_PARTS:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1 and t not in STOPWORDS]


@dataclass(slots=True)
class Product:
    parent_asin: str
    title: str
    coarse_category: str
    category_tokens: frozenset[str]
    quotable: frozenset[str]      # normalised phrases the customer could quote
    text_lower: str               # full searchable text, lowercased
    tokens: frozenset[str]
    title_terms: frozenset[str]   # normalised title vocabulary, for expansion matching
    first_material: str | None
    first_color: str | None
    price_label: str | None       # exact 'budget around $X' payload
    price_value: float | None
    store: str
    average_rating: float
    rating_number: int
    popularity: float = field(default=0.0)


def _price_parts(raw: object) -> tuple[str | None, float | None]:
    if raw in (None, ""):
        return None, None
    label = str(raw)
    match = re.search(r"\d+(?:\.\d+)?", label.replace(",", ""))
    return label, float(match.group()) if match else None


def build_product(record: dict) -> Product:
    corpus = searchable_text(record)
    quotable = {
        clean_text(item)
        for item in (*flatten_values(record.get("features")), *flatten_values(record.get("details")))
        if clean_text(item)
    }
    material = MATERIAL_RE.search(corpus)
    color = COLOR_RE.search(corpus)
    price_label, price_value = _price_parts(record.get("price"))
    categories = [str(v) for v in record.get("categories") or []]
    rating_number = int(record.get("rating_number") or 0)
    average_rating = float(record.get("average_rating") or 0.0)
    text_lower = corpus.lower()
    return Product(
        parent_asin=str(record["parent_asin"]),
        title=str(record.get("title") or ""),
        coarse_category=coarse_category(categories),
        category_tokens=frozenset(tokenize(" ".join(categories))),
        quotable=frozenset(quotable),
        text_lower=text_lower,
        tokens=frozenset(tokenize(text_lower)),
        title_terms=frozenset(lexicon_terms(str(record.get("title") or ""))),
        first_material=material.group(1).lower() if material else None,
        first_color=color.group(1).lower() if color else None,
        price_label=price_label,
        price_value=price_value,
        store=str(record.get("store") or ""),
        average_rating=average_rating,
        rating_number=rating_number,
        # Targets are drawn from real purchase records, so review volume is a
        # prior on "which of these look-alikes did someone actually buy". The
        # public targets have a median review count of 7,078 against the
        # catalog's 12, so the signal is strong; POPULARITY_EXPONENT controls how
        # sharply that separation is expressed.
        popularity=(rating_number ** POPULARITY_EXPONENT) * (average_rating / 5.0),
    )


class Catalog:
    """In-memory catalog with a coarse-category inverted index."""

    def __init__(self, path: str | Path) -> None:
        self.products: dict[str, Product] = {}
        self.by_category: dict[str, list[Product]] = {}
        self.by_token: dict[str, list[Product]] = {}   # category-path token -> products
        self.by_term: dict[str, list[Product]] = {}    # full-text token -> products
        self.lexicon = Lexicon()                       # everyday words -> catalog regions
        self.by_quote: dict[str, list[Product]] = {}   # exact bullet text -> products
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                product = build_product(json.loads(line))
                self.products[product.parent_asin] = product
                self.by_category.setdefault(product.coarse_category.lower(), []).append(product)
                # Titles are how shoppers describe things; category paths are how
                # the catalog does. Indexing titles against their category is what
                # lets one be translated into the other.
                self.lexicon.add(product.title, product.coarse_category.lower())
                # A requirement quoted verbatim identifies its product almost
                # uniquely; indexing bullets makes that an O(1) lookup instead of
                # a score computed against every candidate.
                for phrase in product.quotable:
                    self.by_quote.setdefault(phrase.lower(), []).append(product)
        for bucket in self.by_category.values():
            bucket.sort(key=lambda p: p.popularity, reverse=True)
        for product in self.products.values():
            for token in product.category_tokens:
                self.by_token.setdefault(token, []).append(product)
            for token in product.tokens:
                self.by_term.setdefault(token, []).append(product)
        self.lexicon.finalize()
        self._popular = sorted(self.products.values(), key=lambda p: p.popularity, reverse=True)
        self.dense = None       # attached by attach_dense(); absent means lexical-only
        self.reranker = None    # attached by attach_reranker()

    def attach_dense(self) -> bool:
        """Load the dense retrieval track if its dependencies and index exist."""
        from . import dense as dense_module

        if not dense_module.available():
            return False
        try:
            self.dense = dense_module.DenseIndex(self)
            self.dense.build_category_index(sorted(self.by_category))
        except Exception:
            self.dense = None
        return self.dense is not None

    def attach_reranker(self) -> bool:
        """Load the cross-encoder reranker if its dependencies and model exist."""
        from . import rerank as rerank_module

        if not rerank_module.available():
            return False
        try:
            self.reranker = rerank_module.CrossEncoder()
        except Exception:
            self.reranker = None
        return self.reranker is not None

    def popular(self, limit: int) -> list[Product]:
        """Fallback slate when nothing at all is known about the request."""
        return self._popular[:limit]

    def __len__(self) -> int:
        return len(self.products)
