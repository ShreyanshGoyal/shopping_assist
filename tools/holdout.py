"""Split the public set and check the agent is not tuned to noise in it.

Scoring weights were set by hand while watching the 200 public sessions, which
is exactly the setup where a number can look good because it memorised its
sample. Splitting the set and scoring each half separately tests whether the
agent's performance is a property of the method or of those 200 sessions.

    python3 -m tools.holdout
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from starter.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", default="holdout_results.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    random.Random(args.seed).shuffle(samples)
    half = len(samples) // 2
    splits = {"A": samples[:half], "B": samples[half:]}

    catalog_ids, categories, products = catalog_index(args.catalog)
    agent = Agent(args.catalog)

    report = {}
    for name, rows in splits.items():
        result = evaluate(agent, rows, catalog_ids, categories, products)
        report[name] = {k: result[k] for k in
                        ("sample_count", "hit_rate_at_10", "mrr", "mttc",
                         "efficiency", "recommended_technical_score")}
        r = report[name]
        print(f"  split {name}: n={r['sample_count']:3d} score={r['recommended_technical_score']:.4f} "
              f"HR={r['hit_rate_at_10']:.3f} MRR={r['mrr']:.3f} MTTC={r['mttc']:.2f}")

    gap = abs(report["A"]["recommended_technical_score"] - report["B"]["recommended_technical_score"])
    print(f"\n  split-to-split gap: {gap:.4f}")
    print("  a small gap means the score is a property of the method, not of these 200 sessions")
    report["gap"] = round(gap, 6)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
