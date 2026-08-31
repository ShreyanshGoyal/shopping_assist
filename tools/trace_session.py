"""Replay one public session and print the conversation turn by turn.

Development tool. It reuses the organiser's simulator so what you see here is
exactly what the scorer sees — nothing is re-implemented or approximated.

Usage:  python3 -m tools.trace_session --sample public_0001
"""
from __future__ import annotations

import argparse
import textwrap

from evaluator import local_evaluator as ev
from starter.agent import Agent


def wrap(label: str, text: str, width: int = 92) -> str:
    body = textwrap.fill(text, width=width, subsequent_indent=" " * (len(label) + 1))
    return f"{label} {body}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="public_0001")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    args = parser.parse_args()

    samples = {s["sample_id"]: s for s in ev.load_jsonl(args.dataset)}
    sample = samples[args.sample]
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    agent = Agent(args.catalog)
    inner = agent._agent

    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = ev.materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}

    print("=" * 100)
    print(f"SESSION {sample['sample_id']}   scenario: {sample['scenario_type']}   difficulty: {sample['difficulty_bucket']}")
    print("=" * 100)
    print(wrap("TARGET (hidden from the agent):", products[target]["title"][:150]))
    print(f"   parent_asin {target}   category path: {' > '.join(categories[target])}")
    print("\nHIDDEN INTENT CARD — what the simulated customer is willing to say:")
    for key in ("hard_constraints", "soft_preferences"):
        for value in card[key]:
            print(wrap(f"   [{key[:4]}]", str(value)[:150]))
    print(wrap("\nAGENT SEES ONLY THIS PROFILE:", str(sample["user_profile"]["summary"])))
    print("-" * 100)

    session_id = "trace"
    agent.reset(session_id, sample["user_profile"])
    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = ev.initial_message(effective, ev.coarse_category(categories.get(target, [])), disclosed)

    for turn in range(1, ev.MAX_TURNS + 1):
        print(wrap(f"\nTURN {turn}  CUSTOMER:", message))
        response = agent.respond(session_id, message, turn, ev.TOP_K)
        funnel = inner._sessions[session_id].funnel[-1] if inner._sessions.get(session_id) and inner._sessions[session_id].funnel else None
        ranked = ev.normalize_recommendations(response.get("recommendations"), catalog_ids)
        if funnel:
            # The narrowing, shown as a funnel: this is the decision-tree
            # intuition the agent actually performs — the catalog collapsing as
            # requirements arrive — made visible rather than asserted.
            bucket = f" -> bucket {funnel['bucket']}" if funnel.get("bucket") else ""
            print(f"         FUNNEL:  {funnel['catalog']:,} catalog"
                  f"{bucket}"
                  f" -> {funnel['pool']:,} considered"
                  f" -> {len(ranked)} shown"
                  f"   ({funnel['constraints']} requirement(s) known"
                  f"{', type=' + repr(funnel['type']) if funnel.get('type') else ''})")
        print(wrap(f"         AGENT:   ", response["message"] + f"   [ask_attribute={response['ask_attribute']!r}]"))
        if ranked:
            print(f"         SHOWS {len(ranked)} product(s):")
            for position, asin in enumerate(ranked, 1):
                mark = "  <<< TARGET" if asin == target else ""
                print(f"            {position:2d}. {products[asin]['title'][:78]}{mark}")
        else:
            print("         SHOWS nothing this turn.")

        if not override_applied:
            print("         (scoring locked: this session is still waiting for its intent override)")
        if override_applied and target in ranked:
            rank = ranked.index(target) + 1
            print(f"\n{'=' * 100}\nHIT at turn {turn}, rank {rank}.  reciprocal rank = {1 / rank:.3f}\n{'=' * 100}")
            return
        if turn == ev.MAX_TURNS:
            break
        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            if override.get("new_value"):
                disclosed.add(str(override["new_value"]))
            message = str(override.get("message", ""))
        else:
            message, boundary_used = ev.customer_reply(effective, response.get("ask_attribute"), disclosed, boundary_used)
    print("\nNO HIT within 10 turns.")


if __name__ == "__main__":
    main()
