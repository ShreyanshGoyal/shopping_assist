"""Candidate generation and scoring.

Three routes feed one reranker:

* **Vocabulary routing** — the customer's own words are matched against a lexicon
  mined from catalog titles, which points at the regions of the catalog where
  those words actually occur. This is what lets "some new pants" reach the pants
  categories when the catalog calls them "Sport Specific Clothing Basketball".
* **Exact category** — when the opening turn names a category path outright.
* **Lexical** — rare vocabulary shared with the stated requirements, for when
  category is unknown or too broad to narrow anything.

The reranker then scores each candidate on how much of the stated requirement set
it satisfies. An exact quote of a product's own bullet counts for far more than
incidental word overlap, and a mined synonym counts for less than a word the
customer actually said.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .catalog import Catalog, Product, tokenize
from .lexicon import terms as lexicon_terms
from .state import SessionState

# Scoring weights, tuned on the 200 public development sessions.
W_QUOTE_EXACT = 12.0     # requirement is verbatim one of the product's bullets
W_QUOTE_SUBSTR = 6.0     # requirement appears inside the product's text
W_TOKEN_OVERLAP = 3.0    # partial lexical agreement
W_MATERIAL_PRIMARY = 3.0
W_MATERIAL_PRESENT = 1.0
W_COLOR_PRIMARY = 3.0
W_COLOR_PRESENT = 1.0
W_BUDGET_EXACT = 4.0
W_BUDGET_NEAR = 1.5
W_CATEGORY_EXACT = 15.0
W_CATEGORY_TOKENS = 2.5
# Catalogs of this size carry near-duplicate category nodes — "Bracelets Strand"
# beside "Bracelets Stretch", "Tops & Tees Tanks & Camis" beside "Tees & Blouses
# Tanks & Camis". Paying the full category bonus only on an exact node match
# buries a target that sits one node over from wherever the type resolved, which
# error analysis found happening repeatedly. A sibling earns a share of the bonus
# in proportion to how much of the category name it shares.
SIBLING_CREDIT = 0.55
# How many candidate category nodes share the type credit.
SOFT_CATEGORIES = 12
# Ceiling on soft credit, kept under the exact-match bonus so a category the
# customer states outright still outranks one the agent merely inferred.
W_CATEGORY_SOFT = 9.0
W_ROUTE = 6.0            # customer's vocabulary points at this product's category
W_EXPANSION = 4.0        # product title shares the customer's vocabulary
W_PROFILE_TAG = 0.15
W_POPULARITY = 0.02
W_NEGATIVE = 5.0         # customer explicitly ruled this wording out
W_DENSE = 5.0            # semantic agreement between the request and the product
W_RERANK = 4.0           # cross-encoder judgement of the request and product together

# Reranking is for settling near-ties. When retrieval already has a clear leader,
# reordering beneath it can only do harm, so the pass is skipped entirely.
RERANK_MARGIN = 0.15

# Reranking reads pairs jointly, which is expensive per item and only worth
# spending on products already plausible enough to reach the shortlist.
RERANK_DEPTH = 30

CANDIDATE_LIMIT = 1500
BUCKET_LIMIT = 600       # per routed category, most popular first
ROUTED_CATEGORIES = 6
DENSE_CANDIDATES = 400   # products pulled in by meaning alone


@dataclass
class QueryContext:
    """Everything derived once per turn and reused across every candidate."""
    routed: dict[str, float] = field(default_factory=dict)
    expansion: dict[str, float] = field(default_factory=dict)
    expansion_mass: float = 0.0
    profile_tags: frozenset[str] = frozenset()
    dense_scores: object = None      # cosine against every product, or None
    category_credit: dict = field(default_factory=dict)   # category -> soft type credit
    dense_floor: float = 0.0         # normalisation window, set from the candidate pool
    dense_range: float = 1.0
    dense_weight: float = 1.0        # reduced when exact quotes already decide it
    exact_hits: frozenset[str] = frozenset()


def build_context(catalog: Catalog, state: SessionState) -> QueryContext:
    # The frame, not the raw transcript. Routing over everything ever said is
    # what let common vocabulary accumulate until the largest bucket won.
    text = state.frame.describe() or state.searchable_text()
    routed_pairs = catalog.lexicon.route(text, limit=ROUTED_CATEGORIES) if text else []
    best = routed_pairs[0][1] if routed_pairs else 0.0
    routed = {name: (value / best if best else 0.0) for name, value in routed_pairs}

    expansion = catalog.lexicon.expand(text) if text else {}
    mass = sum(weight * catalog.lexicon.idf(term) for term, weight in expansion.items())
    dense_scores = catalog.dense.scores_for(text) if (catalog.dense is not None and text) else None

    # A requirement repeated verbatim from a product listing is near-conclusive.
    # Where that happens, semantic similarity can only add noise, so it stands
    # down rather than competing with evidence that is already decisive.
    exact: set[str] = set()
    for constraint in state.constraints:
        if constraint.kind != "phrase":
            continue
        for product in catalog.by_quote.get(constraint.text.lower(), ())[:200]:
            exact.add(product.parent_asin)

    # Likewise when the customer has named a category that exists verbatim in the
    # taxonomy: the structure is already certain and inference adds nothing.
    named_exactly = (
        state.frame.type_category is not None
        and state.frame.type_confidence >= 2.0
        and state.frame.type_category in catalog.by_category
    )

    return QueryContext(
        dense_scores=dense_scores,
        dense_weight=0.3 if (exact or named_exactly) else 1.0,
        exact_hits=frozenset(exact),
        routed=routed,
        expansion=expansion,
        expansion_mass=mass or 1.0,
        profile_tags=frozenset(
            str(tag).lower() for tag in (state.user_profile.get("preference_tags") or []) if tag
        ),
    )


# A named category is far stronger evidence than vocabulary routing, which only
# infers where a word tends to occur. When the customer names their category, its
# bucket is the search space — and a small bucket is the most precise signal in
# the catalog, not the least. Routing fills the gap when no category is named, and
# tops up the slate when a named bucket is too small to fill one.
TOPUP_THRESHOLD = 40


def committed_category(state: SessionState) -> str | None:
    """The type slot if the frame has one, else whatever the opening turn named."""
    if state.frame.type_category:
        return state.frame.type_category
    return state.category.lower() if state.category else None


def _category_candidates(catalog: Catalog, state: SessionState, ctx: QueryContext) -> list[Product]:
    committed = committed_category(state)
    exact = catalog.by_category.get(committed) if committed else None
    found: list[Product] = list(exact[:BUCKET_LIMIT]) if exact else []
    if len(found) >= TOPUP_THRESHOLD:
        return found
    for name in ctx.routed:
        if name in state.frame.negative_categories:
            continue
        bucket = catalog.by_category.get(name)
        if bucket:
            found.extend(bucket[:BUCKET_LIMIT])
    # The plausible-category set feeds candidates too, not just scoring: a target
    # one node over from the committed type has to be in the pool to be ranked.
    for name in ctx.category_credit:
        bucket = catalog.by_category.get(name)
        if bucket:
            found.extend(bucket[:BUCKET_LIMIT])
    return found


def _lexical_candidates(catalog: Catalog, state: SessionState, ctx: QueryContext) -> list[Product]:
    """Products sharing rare vocabulary with what the customer has said."""
    if not ctx.expansion:
        return []
    scores: dict[str, float] = {}
    pool: dict[str, Product] = {}
    total = max(len(catalog), 1)
    for term, weight in ctx.expansion.items():
        matches = catalog.by_term.get(term)
        if not matches or len(matches) > total * 0.2:
            continue
        contribution = weight / (1.0 + len(matches)) ** 0.5
        for product in matches:
            scores[product.parent_asin] = scores.get(product.parent_asin, 0.0) + contribution
            pool[product.parent_asin] = product
    ordered = sorted(pool.values(), key=lambda p: (scores[p.parent_asin], p.popularity), reverse=True)
    return ordered[:CANDIDATE_LIMIT]


def _dense_candidates(catalog: Catalog, ctx: QueryContext) -> list[Product]:
    """Products the request resembles in meaning, whatever words it used."""
    if ctx.dense_scores is None:
        return []
    return [
        catalog.products[asin]
        for asin, _ in catalog.dense.top(ctx.dense_scores, DENSE_CANDIDATES)
        if asin in catalog.products
    ]


def candidates(catalog: Catalog, state: SessionState, ctx: QueryContext) -> list[Product]:
    pool: dict[str, Product] = {}
    for product in _category_candidates(catalog, state, ctx):
        pool[product.parent_asin] = product
    for asin in ctx.exact_hits:
        product = catalog.products.get(asin)
        if product is not None:
            pool.setdefault(asin, product)
    for product in _dense_candidates(catalog, ctx):
        pool.setdefault(product.parent_asin, product)
    if len(pool) < 400:
        for product in _lexical_candidates(catalog, state, ctx):
            pool.setdefault(product.parent_asin, product)
    if not pool:
        pool = {p.parent_asin: p for p in catalog.popular(CANDIDATE_LIMIT)}
    return list(pool.values())


def score(product: Product, state: SessionState, ctx: QueryContext, catalog: Catalog) -> float:
    total = 0.0

    soft = ctx.category_credit.get(product.coarse_category.lower())
    if soft:
        total += W_CATEGORY_SOFT * soft

    committed = committed_category(state)
    if committed:
        wanted = committed
        if product.coarse_category.lower() == wanted:
            total += W_CATEGORY_EXACT
        else:
            wanted_tokens = set(tokenize(wanted))
            if wanted_tokens:
                shared = wanted_tokens & product.category_tokens
                if shared:
                    weight = sum(catalog.lexicon.idf(t) for t in shared)
                    whole = sum(catalog.lexicon.idf(t) for t in wanted_tokens) or 1e-9
                    similarity = min(weight / whole, 1.0)
                    total += W_CATEGORY_EXACT * SIBLING_CREDIT * similarity
                else:
                    total += 0.0

    total += W_ROUTE * ctx.routed.get(product.coarse_category.lower(), 0.0)

    if ctx.dense_scores is not None:
        raw = catalog.dense.score_of_index(product.parent_asin, ctx.dense_scores)
        # Cosines from this encoder sit in a narrow band, so an absolute value
        # says little. Normalising against the pool turns it into a usable rank
        # signal without letting it overwhelm a stated category.
        total += W_DENSE * ctx.dense_weight * min(max((raw - ctx.dense_floor) / ctx.dense_range, 0.0), 1.0)

    frame = state.frame
    if product.coarse_category.lower() in frame.negative_categories:
        return -1e6
    if frame.negatives:
        total -= W_NEGATIVE * sum(1 for term in product.title_terms if term in frame.negatives)

    if ctx.expansion:
        shared = 0.0
        for term in product.title_terms:
            weight = ctx.expansion.get(term)
            if weight:
                shared += weight * catalog.lexicon.idf(term)
        total += W_EXPANSION * min(shared / ctx.expansion_mass, 1.0)

    for constraint in state.constraints:
        weight = constraint.weight
        if constraint.kind == "phrase":
            lowered = constraint.text.lower()
            if constraint.text in product.quotable:
                total += W_QUOTE_EXACT * weight
            elif lowered and lowered in product.text_lower:
                total += W_QUOTE_SUBSTR * weight
            else:
                wanted = set(tokenize(constraint.text))
                if wanted:
                    overlap = len(wanted & product.tokens) / len(wanted)
                    total += W_TOKEN_OVERLAP * weight * overlap
        elif constraint.kind == "material":
            if product.first_material == constraint.value:
                total += W_MATERIAL_PRIMARY * weight
            elif constraint.value and constraint.value in product.text_lower:
                total += W_MATERIAL_PRESENT * weight
        elif constraint.kind == "color":
            if product.first_color == constraint.value:
                total += W_COLOR_PRIMARY * weight
            elif constraint.value and constraint.value in product.text_lower:
                total += W_COLOR_PRESENT * weight
        elif constraint.kind == "budget" and constraint.value:
            if product.price_label and product.price_label.replace(",", "") == constraint.value:
                total += W_BUDGET_EXACT * weight
            elif product.price_value is not None:
                try:
                    wanted_price = float(constraint.value)
                except ValueError:
                    continue
                spread = abs(product.price_value - wanted_price) / max(wanted_price, 1.0)
                if spread < 0.25:
                    total += W_BUDGET_NEAR * weight * (1.0 - spread * 4.0)

    if ctx.profile_tags:
        total += W_PROFILE_TAG * sum(1 for tag in ctx.profile_tags if tag in product.text_lower)

    # Targets are real purchase records: among equally plausible matches, the
    # frequently-bought one is the better guess.
    total += W_POPULARITY * product.popularity
    return total


def rank_scored(catalog: Catalog, state: SessionState, limit: int) -> list[tuple[float, str]]:
    """Best-to-worst candidates with their scores."""
    ctx = build_context(catalog, state)
    if ctx.dense_scores is not None:
        import numpy as np

        window = catalog.dense.top(ctx.dense_scores, DENSE_CANDIDATES)
        if window:
            values = np.array([v for _, v in window])
            ctx.dense_floor = float(values.min())
            ctx.dense_range = max(float(values.max() - values.min()), 1e-6)
    excluded = state.ruled_out()
    scored: list[tuple[float, float, str]] = []
    for product in candidates(catalog, state, ctx):
        if product.parent_asin in excluded:
            continue
        scored.append((score(product, state, ctx, catalog), product.popularity, product.parent_asin))
    scored.sort(reverse=True)
    scored = _rerank(catalog, state, ctx, scored)
    return [(value, asin) for value, _, asin in scored[:limit]]


def _rerank(catalog: Catalog, state: SessionState, ctx: QueryContext,
            scored: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    """Reorder the shortlist by joint request-product relevance.

    Blended into the retrieval score rather than replacing it: the exact-quote
    evidence is close to decisive where it fires, and the reranker's job is to
    settle the near-ties beneath it, not to overrule it. The same adaptive weight
    the dense track uses applies here — when a quote has already identified the
    product, the reranker is turned down rather than off.
    """
    if catalog.reranker is None or len(scored) < 2:
        return scored
    query = state.frame.describe()
    if not query.strip():
        return scored

    best, runner_up = scored[0][0], scored[1][0]
    if best > 0 and (best - runner_up) / best >= RERANK_MARGIN:
        return scored

    head = scored[:RERANK_DEPTH]
    products = [catalog.products[asin] for _, _, asin in head if asin in catalog.products]
    if len(products) < 2:
        return scored
    from .rerank import passage

    try:
        judgements = catalog.reranker.score(query, [passage(p) for p in products])
    except Exception:
        return scored

    # Squash the raw logits rather than stretching them across the shortlist.
    # Min-max normalisation guarantees a full-weight spread even when every
    # candidate looks alike to the model, which turns indifference into a strong
    # and arbitrary reordering signal.
    import math

    weight = W_RERANK * ctx.dense_weight
    adjusted = [
        (value + weight / (1.0 + math.exp(-judgement)), popularity, asin)
        for (value, popularity, asin), judgement in zip(head, judgements)
    ]
    adjusted.sort(reverse=True)
    return adjusted + scored[RERANK_DEPTH:]


def rank(catalog: Catalog, state: SessionState, top_k: int) -> list[str]:
    return [asin for _, asin in rank_scored(catalog, state, top_k)]
