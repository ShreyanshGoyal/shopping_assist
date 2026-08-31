"""Is there a turn-1 ranking lever on the scripted benchmark?

Sessions that convert on turn 2 could only have converted on turn 1 if the target
had been ranked first with turn-1 information alone. This replays the official
simulator and records where the target actually sat at turn 1, so the size of that
lever is measured rather than assumed.

    python3 -m tools.turn1_headroom
"""
from __future__ import annotations

import collections
import statistics

from evaluator import local_evaluator as ev
from src import ranking
from starter.agent import Agent


def main() -> None:
    samples = ev.load_jsonl("data/public_set.jsonl")
    catalog_ids, categories, products = ev.catalog_index("data/catalog.jsonl")
    agent = Agent("data/catalog.jsonl")
    inner = agent._agent

    turn1_rank: dict[str, int | None] = {}
    hit_turn: dict[str, int | None] = {}
    scenario: dict[str, str] = {}

    for sample in samples:
        sid = f"t1_{sample['sample_id']}"
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = ev.materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        agent.reset(sid, sample["user_profile"])
        disclosed: set[str] = set()
        boundary_used = False
        applied = sample["scenario_type"] != "intent_override"
        message = ev.initial_message(effective, ev.coarse_category(categories.get(target, [])), disclosed)

        found: int | None = None
        for turn in range(1, ev.MAX_TURNS + 1):
            response = agent.respond(sid, message, turn, ev.TOP_K)
            ranked = ev.normalize_recommendations(response.get("recommendations"), catalog_ids)
            if turn == 1:
                state = inner._sessions[sid]
                shown, state.shown = state.shown, {}
                deep = ranking.rank_scored(inner.catalog, state, 500)
                state.shown = shown
                turn1_rank[sample["sample_id"]] = next(
                    (i + 1 for i, (_, a) in enumerate(deep) if a == target), None
                )
            if applied and target in ranked:
                found = turn
                break
            if turn == ev.MAX_TURNS:
                break
            override = effective.get("behavior", {}).get("override") or {}
            if not applied and turn + 1 == int(override.get("turn", 3)):
                applied = True
                if override.get("new_value"):
                    disclosed.add(str(override["new_value"]))
                message = str(override.get("message", ""))
            else:
                message, boundary_used = ev.customer_reply(
                    effective, response.get("ask_attribute"), disclosed, boundary_used
                )
        hit_turn[sample["sample_id"]] = found
        scenario[sample["sample_id"]] = sample["scenario_type"]

    late = [s for s, t in hit_turn.items() if t and t >= 2]
    print(f"sessions converting at turn 2 or later: {len(late)} of {len(samples)}\n")
    ranks = [turn1_rank[s] for s in late]
    known = [r for r in ranks if r is not None]
    dist = collections.Counter(
        "rank 1" if r == 1 else
        "rank 2-4" if r <= 4 else
        "rank 5-10" if r <= 10 else
        "rank 11-50" if r <= 50 else "rank >50"
        for r in known
    )
    print("where the target already sat at TURN 1 in those sessions:")
    for name in ["rank 1", "rank 2-4", "rank 5-10", "rank 11-50", "rank >50"]:
        if dist[name]:
            print(f"  {name:12s} {dist[name]:3d}  ({dist[name]/len(late):5.1%})")
    unrankable = len(ranks) - len(known)
    if unrankable:
        print(f"  {'unrankable':12s} {unrankable:3d}  ({unrankable/len(late):5.1%})")
    if known:
        print(f"\n  median turn-1 rank {statistics.median(known):.0f}")
    reachable = sum(1 for r in known if r <= 4)
    print(f"\n  sessions where the target was already top-4 at turn 1: {reachable}"
          f"  ({reachable/len(late):.1%} of late converters)")

    # Intent-override sessions cannot convert before the override lands, so a
    # good turn-1 rank there is unusable. Only the rest are addressable.
    addressable = [
        s for s in late
        if scenario[s] != "intent_override" and turn1_rank[s] and turn1_rank[s] <= 4
    ]
    locked = reachable - len(addressable)
    print(f"  of those, intent-override (structurally locked): {locked}")
    print(f"  ADDRESSABLE turn-1 lever: {len(addressable)} sessions")

    gain = len(addressable) / len(samples) * 0.2 / 10 * 1.0
    print(f"\n  if all of them converted at turn 1 instead of turn 2:")
    print(f"    MTTC {statistics.fmean([t if t else 11 for t in hit_turn.values()]):.3f}"
          f" -> {statistics.fmean([1 if s in addressable else (hit_turn[s] or 11) for s in hit_turn]):.3f}")
    print(f"    approx composite gain +{gain:.4f} from efficiency alone")


if __name__ == "__main__":
    main()
