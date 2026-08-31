"""Where does retrieval put the target in the sessions we still miss?

The listwise reranker only ever sees retrieval's top 10. Extending it to select
ten from thirty can only rescue targets that retrieval ranks 11-30, so that band
is the size of the proposed lever; anything deeper is out of reach for this
mechanism regardless of how good the model is.

    python3 -m tools.miss_bands sim_n120_listwise.json
"""
from __future__ import annotations

import collections
import json
import sys

from src import ranking
from src.agent_impl import ShoppingAgent

BANDS = ["1-10", "11-30", "31-100", "101-300", ">300 / absent"]


def band_of(position: int | None) -> str:
    if position is None:
        return ">300 / absent"
    if position <= 10:
        return "1-10"
    if position <= 30:
        return "11-30"
    if position <= 100:
        return "31-100"
    return "101-300"


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "sim_n120_listwise.json"
    sessions = [s for s in json.load(open(path))["sessions"] if not s.get("aborted")]
    agent = ShoppingAgent("data/catalog.jsonl")

    bands: collections.Counter[str] = collections.Counter()
    miss_bands: collections.Counter[str] = collections.Counter()

    for record in sessions:
        target = record["target"]
        sid = f"d_{record['sample_id']}"
        agent.reset(sid, {})
        state = agent._sessions[sid]
        best = None
        for entry in record["transcript"]:
            agent.respond(sid, entry["customer"], entry["turn"], 10)
            shown, state.shown = state.shown, {}
            deep = ranking.rank_scored(agent.catalog, state, 300)
            state.shown = shown
            position = next((i + 1 for i, (_, a) in enumerate(deep) if a == target), None)
            if position and (best is None or position < best):
                best = position
        band = band_of(best)
        bands[band] += 1
        if not record["hit"]:
            miss_bands[band] += 1

    total = len(sessions)
    misses = sum(miss_bands.values())
    print(f"n={total} sessions, {misses} misses\n", flush=True)
    print("retrieval's BEST position for the target, all sessions:")
    for name in BANDS:
        if bands[name]:
            print(f"  {name:14s} {bands[name]:3d}  ({bands[name]/total:5.1%})")
    print(f"\nsame, restricted to the {misses} MISSED sessions:")
    for name in BANDS:
        if miss_bands[name]:
            print(f"  {name:14s} {miss_bands[name]:3d}  ({miss_bands[name]/misses:5.1%} of misses)")
    reachable = miss_bands["11-30"]
    print(f"\n  reachable by select-from-30: {reachable} of {misses} misses"
          f"  -> at best +{0.5*reachable/total:.3f} on the hit-rate term")
    print(f"  already top-10 yet still missed: {miss_bands['1-10']}"
          f"  (slate/turn budget, not retrieval)")


if __name__ == "__main__":
    main()
