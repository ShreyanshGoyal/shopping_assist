"""Sweep the shape of the target-popularity prior, holding out half the set.

Public targets have a median review count of 7,078 against the catalog's 12, so
review volume is a strong prior on which of several equally-matching products was
the one actually bought. This sweeps how sharply that is expressed, fitting
nothing: each configuration is scored on both halves independently and accepted
only if it improves both.
"""
from __future__ import annotations

from evaluator import local_evaluator as ev
from src import catalog as catalog_module
from src import ranking

CONFIGS = [
    ("exponent 0.5 (current)", 0.5, 0.02),
    ("exponent 0.3", 0.3, 0.02),
    ("exponent 0.7", 0.7, 0.02),
    ("exponent 0.7, weight 0.01", 0.7, 0.01),
    ("exponent 1.0, weight 0.004", 1.0, 0.004),
    ("exponent 0.6, weight 0.015", 0.6, 0.015),
]


def main() -> None:
    samples = ev.load_jsonl("data/public_set.jsonl")
    split_a = [s for i, s in enumerate(samples) if i % 2 == 0]
    split_b = [s for i, s in enumerate(samples) if i % 2 == 1]
    catalog_ids, categories, products = ev.catalog_index("data/catalog.jsonl")

    print(f"{'config':28s} {'split A':>9s} {'split B':>9s} {'both':>9s}")
    for label, exponent, weight in CONFIGS:
        catalog_module.POPULARITY_EXPONENT = exponent
        ranking.W_POPULARITY = weight
        from starter.agent import Agent   # rebuild so the exponent takes effect

        agent = Agent("data/catalog.jsonl")
        scores = []
        for split in (split_a, split_b, samples):
            scores.append(ev.evaluate(agent, split, catalog_ids, categories, products)
                          ["recommended_technical_score"])
        print(f"{label:28s} {scores[0]:9.4f} {scores[1]:9.4f} {scores[2]:9.4f}", flush=True)


if __name__ == "__main__":
    main()
