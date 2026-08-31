"""Replay a recorded language-model-customer session for viewing.

Reads a stored benchmark run — no API call, no credential, no cost — and prints
the conversation as it happened, so a session with a real paraphrasing customer
can be shown on screen alongside the scripted ones.

    python3 -m tools.show_llm_session                      # pick a good example
    python3 -m tools.show_llm_session --scenario use_case_led
    python3 -m tools.show_llm_session --sample public_0042
"""
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

DEFAULT = "results/llm_customer/sim_n120_hedge.json"


def wrap(label: str, text: str, width: int = 96) -> str:
    body = textwrap.fill(text, width=width, subsequent_indent=" " * (len(label) + 1))
    return f"{label} {body}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=DEFAULT)
    parser.add_argument("--sample", default="")
    parser.add_argument("--scenario", default="use_case_led")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    args = parser.parse_args()

    sessions = [s for s in json.load(open(args.run))["sessions"] if not s.get("aborted")]
    if args.sample:
        picked = next((s for s in sessions if s["sample_id"] == args.sample), None)
    else:
        # A converting session in the requested scenario, preferring one that took
        # a few turns — a one-turn win shows nothing about the conversation.
        candidates = [s for s in sessions
                      if s["scenario"] == args.scenario and s["hit"] and (s["first_hit_turn"] or 0) >= 3]
        picked = candidates[0] if candidates else next((s for s in sessions if s["hit"]), None)
    if picked is None:
        raise SystemExit("no matching session in that run")

    titles = {}
    with Path(args.catalog).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            titles[str(record["parent_asin"])] = str(record.get("title") or "")

    print("=" * 100)
    print(f"LANGUAGE-MODEL CUSTOMER  ·  {picked['sample_id']}  ·  "
          f"{picked['scenario']} / {picked['style']}")
    print("=" * 100)
    print(wrap("TARGET (hidden from the agent):", titles.get(picked["target"], picked["target"])[:150]))
    print("-" * 100)

    for entry in picked["transcript"]:
        print(wrap(f"\nTURN {entry['turn']}  CUSTOMER:", entry["customer"]))
        print(wrap("         AGENT:   ", str(entry.get("agent") or "")))
        shown = entry.get("shown") or []
        if shown:
            print(f"         SHOWS {len(shown)}:")
            for position, asin in enumerate(shown[:5], 1):
                mark = "   <<< TARGET" if asin == picked["target"] else ""
                print(f"            {position:2d}. {titles.get(asin, asin)[:74]}{mark}")
            if len(shown) > 5:
                print(f"            ... {len(shown) - 5} more")
        if entry.get("hit_rank"):
            print(f"\n{'=' * 100}")
            print(f"HIT at turn {entry['turn']}, rank {entry['hit_rank']}.")
            print("=" * 100)


if __name__ == "__main__":
    main()
