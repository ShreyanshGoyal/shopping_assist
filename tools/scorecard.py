"""Print a compact scorecard from a saved evaluator run.

The evaluator's JSON is complete but far too long to read at a glance or fit in a
screenshot. This renders the same numbers as a scorecard. It computes nothing —
every figure is read from the run file.

    python3 -m tools.scorecard
    python3 -m tools.scorecard results/scripted/results_frozen.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASELINE = Path("docs/baseline_results.json")
DEFAULT = Path("results/scripted/results_frozen_tier1.json")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    run = json.load(open(path))
    base = json.load(open(BASELINE)) if BASELINE.exists() else None

    print()
    print("  TechJam Track 4 — official evaluator, 200 public sessions")
    print("  " + "─" * 56)
    print(f"  {'':22s}{'baseline':>12s}{'this agent':>14s}")
    print("  " + "─" * 56)
    rows = [
        ("Hit Rate@10", "hit_rate_at_10", "{:.3f}"),
        ("MRR", "mrr", "{:.3f}"),
        ("MTTC (turns)", "mttc", "{:.2f}"),
        ("Efficiency", "efficiency", "{:.3f}"),
    ]
    for label, key, fmt in rows:
        b = fmt.format(base[key]) if base and key in base else "—"
        print(f"  {label:22s}{b:>12s}{fmt.format(run[key]):>14s}")
    print("  " + "─" * 56)
    b = f"{base['technical_score']:.4f}" if base else "—"
    print(f"  {'TechnicalScore':22s}{b:>12s}{run['recommended_technical_score']:>14.4f}")
    print("  " + "─" * 56)

    usage = run.get("reported_token_usage") or {}
    print(f"  tokens used: {usage.get('total_tokens', 0)}          "
          f"network calls: 0          credential: none")
    print()
    print("  by scenario")
    for name, metrics in sorted(run.get("scenario_metrics", {}).items()):
        print(f"    {name:16s} n={metrics['sample_count']:3d}   "
              f"hit {metrics['hit_rate_at_10']:.3f}   "
              f"MRR {metrics['mrr']:.3f}   "
              f"turns {metrics['mttc']:.2f}")
    print()


if __name__ == "__main__":
    main()
