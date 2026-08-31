"""Audit every clarification the agent asks against the official simulator.

The scripted customer acts only on `ask_attribute`; the prose never reaches it.
So the only question that matters is whether each ask *bought* anything. This
replays the 200 public sessions and classifies every turn:

  productive   the reply disclosed at least one new requirement
  wasted       "I don't have an additional preference for X" - the ask cost a
               turn and returned nothing
  stonewall    the boundary scenario's one-time deflection, structurally
               unavoidable
  interrupted  the override turn, where the customer ignores the question and
               states the new intent instead - also unavoidable
  silent       the agent asked nothing while requirements were still outstanding

Also lists every session that converted below rank 1, which is where the
remaining points are.

    python3 -m tools.ask_audit
"""
from __future__ import annotations

import collections
import json
import re

from evaluator import local_evaluator as ev
from starter.agent import Agent

NO_MORE = re.compile(r"i don'?t have an additional preference for", re.I)
STONEWALL = re.compile(r"i don'?t have a preference for .*use your judg", re.I)
NUDGE = re.compile(r"ask me about one specific attribute", re.I)


def main() -> None:
    samples = ev.load_jsonl("data/public_set.jsonl")
    catalog_ids, categories, products = ev.catalog_index("data/catalog.jsonl")
    agent = Agent("data/catalog.jsonl")

    kinds: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    asked_attrs: collections.Counter[str] = collections.Counter()
    imperfect: list[dict] = []
    silent_with_work = 0

    for sample in samples:
        sid = f"audit_{sample['sample_id']}"
        scenario = sample["scenario_type"]
        target = str(sample["ground_truth"]["parent_asin"])
        card, behavior = ev.materialize_hidden_fields(sample, products)
        effective = {**sample, "intent_card": card, "behavior": behavior}
        agent.reset(sid, sample["user_profile"])
        disclosed: set[str] = set()
        boundary_used = False
        applied = scenario != "intent_override"
        message = ev.initial_message(effective, ev.coarse_category(categories.get(target, [])), disclosed)

        transcript: list[dict] = []
        hit_turn = best_rank = None
        for turn in range(1, ev.MAX_TURNS + 1):
            response = agent.respond(sid, message, turn, ev.TOP_K)
            attribute = response.get("ask_attribute")
            ranked = ev.normalize_recommendations(response.get("recommendations"), catalog_ids)
            asked_attrs[str(attribute)] += 1
            entry = {"turn": turn, "customer": message, "ask": attribute, "shown": len(ranked)}

            if applied and target in ranked:
                best_rank = ranked.index(target) + 1
                hit_turn = turn
                entry["kind"] = "hit"
                transcript.append(entry)
                break
            if turn == ev.MAX_TURNS:
                entry["kind"] = "last"
                transcript.append(entry)
                break

            override = effective.get("behavior", {}).get("override") or {}
            if not applied and turn + 1 == int(override.get("turn", 3)):
                applied = True
                if override.get("new_value"):
                    disclosed.add(str(override["new_value"]))
                message = str(override.get("message", ""))
                entry["kind"] = "interrupted"
            else:
                before = len(disclosed)
                message, boundary_used = ev.customer_reply(
                    effective, attribute, disclosed, boundary_used
                )
                if STONEWALL.search(message):
                    entry["kind"] = "stonewall"
                elif NUDGE.search(message):
                    entry["kind"] = "silent"
                    silent_with_work += 1
                elif NO_MORE.search(message):
                    entry["kind"] = "wasted"
                elif len(disclosed) > before:
                    entry["kind"] = "productive"
                else:
                    entry["kind"] = "wasted"
            kinds[scenario][entry["kind"]] += 1
            transcript.append(entry)

        if hit_turn is None or best_rank != 1:
            imperfect.append({
                "sample_id": sample["sample_id"], "scenario": scenario,
                "turn": hit_turn, "rank": best_rank, "transcript": transcript,
            })

    print("ASK OUTCOMES BY SCENARIO\n")
    order = ["productive", "wasted", "stonewall", "interrupted", "silent"]
    header = "  " + f"{'scenario':16s}" + "".join(f"{k:>13s}" for k in order)
    print(header)
    for scenario in sorted(kinds):
        row = "".join(f"{kinds[scenario][k]:>13d}" for k in order)
        print(f"  {scenario:16s}{row}")
    totals = collections.Counter()
    for counter in kinds.values():
        totals.update(counter)
    print("  " + "-" * (len(header) - 2))
    print(f"  {'total':16s}" + "".join(f"{totals[k]:>13d}" for k in order))

    print(f"\n  attributes asked: {dict(asked_attrs)}")
    print(f"  turns where the agent asked nothing while work remained: {silent_with_work}")

    print(f"\nSESSIONS NOT CONVERTING AT RANK 1: {len(imperfect)}")
    for record in imperfect:
        print(f"  {record['sample_id']}  {record['scenario']:16s} turn={record['turn']} rank={record['rank']}")
    with open("ask_audit.json", "w", encoding="utf-8") as handle:
        json.dump(imperfect, handle, indent=2)
    print("\n  full transcripts written to ask_audit.json")


if __name__ == "__main__":
    main()
