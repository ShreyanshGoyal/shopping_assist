"""Where does the target actually sit, turn by turn?

Replays recorded benchmark transcripts against the agent and reports the target's
position in the *full* scored candidate list, not just the shortlist. That
separates the two failure modes the score cannot distinguish:

* the target never enters the candidate pool  -> a recall problem
* the target is in the pool but ranked low    -> a ranking problem

Costs nothing: the customer turns are replayed from the recording, and the
extractor's model calls hit the response cache from the original run.

    python3 -m tools.diagnose sim_n120_dense.json
"""
from __future__ import annotations

import collections
import json
import statistics
import sys

from src import ranking
from src.agent_impl import ShoppingAgent
from src.state import SessionState

DEEP = 5000   # how far down the ranking to look for the target


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "sim_n120_dense.json"
    sessions = [s for s in json.load(open(path))["sessions"] if not s.get("aborted")]
    agent = ShoppingAgent("data/catalog.jsonl")

    per_turn: dict[int, list[int | None]] = collections.defaultdict(list)
    final_rank: list[int | None] = []
    outcomes = collections.Counter()

    for record in sessions:
        target = record["target"]
        sid = f"diag_{record['sample_id']}"
        agent.reset(sid, {})
        state: SessionState = agent._sessions[sid]
        ranks: list[int | None] = []
        for entry in record["transcript"]:
            agent.respond(sid, entry["customer"], entry["turn"], 10)
            # The agent rules out what it has already shown, which would hide the
            # target from this measurement on the very turn it was found. Ask
            # "where would this rank on the evidence so far" instead.
            shown, state.shown = state.shown, {}
            deep = ranking.rank_scored(agent.catalog, state, DEEP)
            state.shown = shown
            position = next((i + 1 for i, (_, a) in enumerate(deep) if a == target), None)
            ranks.append(position)
            per_turn[entry["turn"]].append(position)
        best = min([r for r in ranks if r is not None], default=None)
        final_rank.append(best)
        if best is None:
            outcomes["never in pool"] += 1
        elif best == 1:
            outcomes["reached rank 1"] += 1
        elif best <= 10:
            outcomes["reached top 10"] += 1
        elif best <= 100:
            outcomes["rank 11-100"] += 1
        else:
            outcomes["rank >100"] += 1

    print(f"replayed {len(sessions)} sessions from {path}\n")
    print("best position the target ever reached:")
    for name, count in outcomes.most_common():
        print(f"  {name:18s} {count:4d}  ({count/len(sessions):5.1%})")

    found = [r for r in final_rank if r is not None]
    if found:
        print(f"\n  median best position {statistics.median(found):.0f}, "
              f"mean {statistics.fmean(found):.1f}")

    print("\ntarget's position by turn (median over sessions where it is rankable):")
    for turn in sorted(per_turn):
        values = [r for r in per_turn[turn] if r is not None]
        missing = len(per_turn[turn]) - len(values)
        if values:
            print(f"  turn {turn:2d}  n={len(per_turn[turn]):3d}  median rank {statistics.median(values):6.0f}"
                  f"   in top10 {sum(1 for v in values if v <= 10):3d}   unrankable {missing:3d}")


if __name__ == "__main__":
    main()
