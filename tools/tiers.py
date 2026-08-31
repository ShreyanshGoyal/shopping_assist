"""Print the two-benchmark comparison in the terminal.

Both columns come from saved runs — the official evaluator on the left, the
language-model customer on the right — so the table can be shown on screen
without building slides. Nothing is computed here beyond the composite score.

    python3 -m tools.tiers
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

SCRIPTED = Path("results/scripted")
CUSTOMER = Path("results/llm_customer")

TIERS = [
    ("lexical  (submitted)", "results_frozen_tier1.json", "sim_n120_tier1.json"),
    ("+ dense retrieval",    "results_frozen.json",       "sim_n120_dense.json"),
    ("+ listwise reranking", "results_listwise.json",     "sim_n120_hedge.json"),
]


def scripted_score(name: str) -> float | None:
    path = SCRIPTED / name
    return json.load(open(path))["recommended_technical_score"] if path.exists() else None


def customer_score(name: str) -> float | None:
    path = CUSTOMER / name
    if not path.exists():
        return None
    rows = [s for s in json.load(open(path))["sessions"] if not s.get("aborted")]
    if not rows:
        return None
    hit = sum(s["hit"] for s in rows) / len(rows)
    mrr = statistics.fmean(s["reciprocal_rank"] for s in rows)
    mttc = statistics.fmean(s["first_hit_turn"] if s["first_hit_turn"] else 11 for s in rows)
    return 0.5 * hit + 0.3 * mrr + 0.2 * max(0.0, min(1.0, (11 - mttc) / 10))


def main() -> None:
    print()
    print("  Same agent. Two customers.")
    print()
    print(f"  {'':22s}{'quotes the catalog':>22s}{'paraphrases':>20s}")
    print(f"  {'':22s}{'(official evaluator)':>22s}{'(built for this)':>20s}")
    print("  " + "─" * 64)
    for label, left_file, right_file in TIERS:
        left, right = scripted_score(left_file), customer_score(right_file)
        left_text = f"{left:.4f}" if left is not None else "—"
        right_text = f"{right:.4f}" if right is not None else "—"
        print(f"  {label:22s}{left_text:>22s}{right_text:>20s}")
    print("  " + "─" * 64)
    print()
    print("  The columns move in opposite directions.")
    print("  What helps a real customer costs points against the evaluator.")
    print()


if __name__ == "__main__":
    main()
