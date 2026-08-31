"""Run the agent against a language-model customer and score it.

Metrics are computed with the organiser's formulas so a run here is directly
comparable to a run of the official evaluator. What differs is only who plays
the customer — and therefore how much of the agent's score comes from genuinely
understanding a request rather than matching its wording.

    python3 -m sim.harness --sessions 40 --model <model-id>
    python3 -m sim.harness --sessions 8 --stub      # no credential needed

Targets are drawn from the official public sessions, so the same products are
used in both benchmarks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

from starter.agent import Agent
from .client import BaseClient, MissingCredential, make_client
from .customer import LLMCustomer
from .personas import SCENARIOS, STYLES

MAX_TURNS = 10
TOP_K = 10


def load_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_catalog(path: str | Path) -> tuple[dict[str, dict], set[str]]:
    products: dict[str, dict] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            products[str(record["parent_asin"])] = record
    return products, set(products)


def assign(sample_id: str, index: int | None = None) -> tuple[str, str]:
    """Give each session a scenario and a writing style.

    Hashing the id spreads assignments randomly, which leaves cells of one or two
    sessions and makes per-cell numbers unreadable. Cycling through the grid
    instead fills all 30 scenario-by-style combinations evenly, so a run of 120
    gives four sessions per cell and the breakdown means something.
    """
    scenarios, styles = sorted(SCENARIOS), sorted(STYLES)
    if index is None:
        digest = hashlib.sha256(sample_id.encode()).digest()
        return scenarios[digest[0] % len(scenarios)], styles[digest[1] % len(styles)]
    return scenarios[index % len(scenarios)], styles[(index // len(scenarios)) % len(styles)]


def normalize(payload: object, catalog_ids: set[str]) -> list[str]:
    """Same normalisation the official evaluator applies to a recommendation list."""
    if not isinstance(payload, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for item in payload:
        value = item.get("parent_asin", "") if isinstance(item, dict) else item
        asin = str(value).strip()
        if not asin or asin in seen or asin not in catalog_ids:
            continue
        seen.add(asin)
        result.append(asin)
        if len(result) >= TOP_K:
            break
    return result


class StubCustomer:
    """A scripted stand-in so the harness can be exercised without a credential."""

    def __init__(self, product: dict, scenario, style: str, profile: dict) -> None:
        self.product = product
        self.scenario = scenario
        self.turn = 0
        categories = [str(c) for c in (product.get("categories") or [])][1:]
        self.category = " ".join(categories[-2:]) if categories else "something"
        self.details = [str(f) for f in (product.get("features") or [])][:4]

    def opening(self) -> str:
        if self.scenario.maps_to == "buying" and self.details:
            return f"I need {self.category.lower()}. Something along the lines of {self.details[0][:60].lower()}."
        return f"I'm after {self.category.lower()}, still deciding though."

    def reply(self, assistant_message: str, shown_titles: list[str], turn: int) -> str:
        self.turn += 1
        if self.turn <= len(self.details):
            return f"What matters to me is {self.details[self.turn - 1][:70].lower()}."
        return "Nothing else really comes to mind."


def run_session(agent: Agent, sample: dict, products: dict, catalog_ids: set[str],
                client: BaseClient | None, verbose: bool, index: int | None = None) -> dict:
    target = str(sample["ground_truth"]["parent_asin"])
    scenario_name, style = assign(sample["sample_id"], index)
    scenario = SCENARIOS[scenario_name]
    profile = sample.get("user_profile") or {}

    if client is None:
        customer = StubCustomer(products[target], scenario, style, profile)
    else:
        customer = LLMCustomer(client, products[target], scenario, style, profile)

    session_id = f"sim_{sample['sample_id']}"
    agent.reset(session_id, profile)
    transcript: list[dict] = []
    try:
        message = customer.opening()
    except Exception as error:
        # The customer is a network service; losing it must abandon one session,
        # never the whole benchmark.
        return {
            "sample_id": sample["sample_id"], "scenario": scenario_name,
            "maps_to": scenario.maps_to, "style": style, "target": target,
            "hit": False, "first_hit_turn": None, "best_rank": None,
            "reciprocal_rank": 0.0, "aborted": repr(error), "transcript": [],
        }
    hit_turn: int | None = None
    best_rank: int | None = None
    aborted: str | None = None

    for turn in range(1, MAX_TURNS + 1):
        try:
            response = agent.respond(session_id, message, turn, TOP_K)
        except Exception as error:  # a crash is a miss, never a harness failure
            transcript.append({"turn": turn, "customer": message, "error": repr(error)})
            break
        ranked = normalize(response.get("recommendations"), catalog_ids)
        titles = [str(products[a].get("title") or "")[:110] for a in ranked]
        transcript.append({
            "turn": turn,
            "customer": message,
            "agent": response.get("message"),
            "ask_attribute": response.get("ask_attribute"),
            "shown": ranked,
            "shown_titles": titles,
        })
        if verbose:
            print(f"  [{turn}] customer: {message}")
            print(f"      agent   : {response.get('message')}  (asks {response.get('ask_attribute')!r}, shows {len(ranked)})")
        if target in ranked:
            best_rank = ranked.index(target) + 1
            hit_turn = turn
            transcript[-1]["hit_rank"] = best_rank
            if verbose:
                print(f"      -> HIT rank {best_rank}")
            break
        if turn == MAX_TURNS:
            break
        try:
            message = customer.reply(str(response.get("message") or ""), titles, turn + 1)
        except Exception as error:
            transcript.append({"turn": turn, "customer_error": repr(error)})
            aborted = repr(error)
            break

    return {
        "sample_id": sample["sample_id"],
        "scenario": scenario_name,
        "maps_to": scenario.maps_to,
        "style": style,
        "target": target,
        "hit": hit_turn is not None,
        "first_hit_turn": hit_turn,
        "best_rank": best_rank,
        "reciprocal_rank": 0.0 if best_rank is None else 1.0 / best_rank,
        "aborted": aborted,
        "transcript": transcript,
    }


def summarize(sessions: list[dict]) -> dict:
    if not sessions:
        return {"sample_count": 0}
    hit_rate = sum(int(s["hit"]) for s in sessions) / len(sessions)
    mrr = statistics.fmean(s["reciprocal_rank"] for s in sessions)
    mttc = statistics.fmean(
        s["first_hit_turn"] if s["first_hit_turn"] is not None else MAX_TURNS + 1 for s in sessions
    )
    efficiency = max(0.0, min(1.0, (11.0 - mttc) / 10.0))
    return {
        "sample_count": len(sessions),
        "hit_rate_at_10": round(hit_rate, 6),
        "mrr": round(mrr, 6),
        "mttc": round(mttc, 6),
        "efficiency": round(efficiency, 6),
        "technical_score": round(0.5 * hit_rate + 0.3 * mrr + 0.2 * efficiency, 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-customer benchmark")
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    parser.add_argument("--dataset", default="data/public_set.jsonl")
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--model", default="")
    parser.add_argument("--backend", default="auto", choices=["auto", "vertex", "aistudio"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stub", action="store_true", help="run without a model credential")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--hashed", action="store_true",
                        help="assign scenarios by hash (reproduces pre-stratification runs)")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--fresh", dest="resume", action="store_false")
    parser.add_argument("--output", default="sim_results.json")
    args = parser.parse_args()

    samples = load_jsonl(args.dataset)
    random.Random(args.seed).shuffle(samples)
    samples = samples[: args.sessions]
    products, catalog_ids = load_catalog(args.catalog)

    client = None
    if not args.stub:
        if not args.model:
            parser.error("--model is required unless --stub is set; run python3 -m tools.probe_models to see what works")
        try:
            client = make_client(args.model, args.backend)
        except MissingCredential as error:
            parser.error(str(error))

    agent = Agent(args.catalog)
    started = time.perf_counter()

    # Resume: completed sessions are reused rather than re-run, so an interrupted
    # benchmark costs nothing to finish.
    partial = Path(args.output).with_suffix(".partial.json")
    done_by_id: dict[str, dict] = {}
    if args.resume and partial.exists():
        try:
            done_by_id = {r["sample_id"]: r for r in json.loads(partial.read_text())}
            print(f"  resuming: {len(done_by_id)} sessions already complete")
        except (json.JSONDecodeError, KeyError, TypeError):
            done_by_id = {}

    results: list[dict] = []
    for index, sample in enumerate(samples, 1):
        if args.verbose:
            print(f"\n=== {index}/{len(samples)}  {sample['sample_id']} ===")
        cached = done_by_id.get(sample["sample_id"])
        if cached is not None and not cached.get("aborted"):
            results.append(cached)
        else:
            results.append(run_session(
                agent, sample, products, catalog_ids, client, args.verbose,
                index=None if args.hashed else index - 1,
            ))
            partial.write_text(json.dumps(results), encoding="utf-8")
        if not args.verbose:
            hits = sum(int(r["hit"]) for r in results)
            aborted = sum(1 for r in results if r.get("aborted"))
            note = f" | aborted {aborted}" if aborted else ""
            print(f"\r  {index}/{len(samples)} sessions | hits {hits}{note}", end="", flush=True)
    elapsed = time.perf_counter() - started
    print()

    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in results:
        grouped[record["scenario"]].append(record)
    by_style: dict[str, list[dict]] = defaultdict(list)
    for record in results:
        by_style[record["style"]].append(record)

    report = {
        "customer": "stub" if client is None else f"{client.backend}:{args.model}",
        "overall": summarize(results),
        "by_scenario": {name: summarize(rows) for name, rows in sorted(grouped.items())},
        "by_style": {name: summarize(rows) for name, rows in sorted(by_style.items())},
        "wall_clock_seconds": round(elapsed, 1),
        "aborted_sessions": sum(1 for r in results if r.get("aborted")),
        "token_usage": (
            {} if client is None else {
                "prompt_tokens": client.usage.prompt_tokens,
                "completion_tokens": client.usage.completion_tokens,
                "calls": client.usage.calls,
                "served_from_cache": client.usage.cached_calls,
            }
        ),
        "sessions": results,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    partial.unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in report.items() if k != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
