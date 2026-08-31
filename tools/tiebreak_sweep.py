"""Sweep the tie-break weights, validated on a held-out half of the public set.

The remaining scripted headroom sits in genuine near-ties: several products match
every disclosed requirement equally, and what separates them is the popularity
prior and the profile-tag bonus. Both are tuned here on split A and reported on
split B, so a gain has to survive on sessions it was not chosen against.
"""
from __future__ import annotations

from evaluator import local_evaluator as ev
from src import ranking
from starter.agent import Agent

CONFIGS = [
    ("popularity 0.02 (current)", 0.02, 0.15),
    ("popularity 0.00", 0.0, 0.15),
    ("popularity 0.01", 0.01, 0.15),
    ("popularity 0.04", 0.04, 0.15),
    ("popularity 0.08", 0.08, 0.15),
    ("popularity 0.02, profile 0.5", 0.02, 0.5),
    ("popularity 0.04, profile 0.5", 0.04, 0.5),
]


def main() -> None:
    samples = ev.load_jsonl("data/public_set.jsonl")
    split_a = [s for i, s in enumerate(samples) if i % 2 == 0]
    split_b = [s for i, s in enumerate(samples) if i % 2 == 1]
    catalog_ids, categories, products = ev.catalog_index("data/catalog.jsonl")
    agent = Agent("data/catalog.jsonl")

    print(f"{'config':30s} {'split A':>9s} {'split B':>9s} {'both':>9s}")
    for label, popularity, profile in CONFIGS:
        ranking.W_POPULARITY, ranking.W_PROFILE_TAG = popularity, profile
        scores = []
        for split in (split_a, split_b, samples):
            result = ev.evaluate(agent, split, catalog_ids, categories, products)
            scores.append(result["recommended_technical_score"])
        print(f"{label:30s} {scores[0]:9.4f} {scores[1]:9.4f} {scores[2]:9.4f}", flush=True)


if __name__ == "__main__":
    main()
