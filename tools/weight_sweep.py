"""Offline sweep of retrieval weights against recorded transcripts.

Free: customer turns are replayed from a recording and extractor calls hit the
response cache. Reports where the target lands rather than a session score, since
the question is whether the ranking improves, not whether a particular slate
policy converted.
"""
import json
import statistics
import sys

from src import ranking
from src.agent_impl import ShoppingAgent

CONFIGS = [
    ("current             ", 15.0, 9.0, 5.0),
    ("category halved     ", 7.5, 4.5, 5.0),
    ("category quartered  ", 3.75, 2.25, 5.0),
    ("dense doubled       ", 15.0, 9.0, 10.0),
    ("cat half + dense dbl", 7.5, 4.5, 10.0),
    ("category near-off   ", 1.5, 1.0, 12.0),
]


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "results/llm_customer/sim_n120_dense.json"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 119
    sessions = [s for s in json.load(open(path))["sessions"] if not s.get("aborted")][:limit]
    agent = ShoppingAgent("data/catalog.jsonl")
    print(f"{len(sessions)} sessions\n")
    print(f"{'config':22s} {'rank1':>6s} {'top10':>6s} {'median':>7s}")
    for label, cat, soft, dense in CONFIGS:
        ranking.W_CATEGORY_EXACT, ranking.W_CATEGORY_SOFT, ranking.W_DENSE = cat, soft, dense
        best_ranks = []
        for record in sessions:
            target = record["target"]
            sid = f"{label}_{record['sample_id']}"
            agent.reset(sid, {})
            state = agent._sessions[sid]
            best = None
            for entry in record["transcript"]:
                agent.respond(sid, entry["customer"], entry["turn"], 10)
                shown, state.shown = state.shown, {}
                deep = ranking.rank_scored(agent.catalog, state, 100)
                state.shown = shown
                position = next((i + 1 for i, (_, a) in enumerate(deep) if a == target), None)
                if position and (best is None or position < best):
                    best = position
            best_ranks.append(best)
        found = [r for r in best_ranks if r]
        r1 = sum(1 for r in found if r == 1)
        t10 = sum(1 for r in found if r <= 10)
        median = statistics.median(found) if found else 0
        print(f"{label} {r1:5d}  {t10:5d}   {median:6.0f}", flush=True)


if __name__ == "__main__":
    main()
