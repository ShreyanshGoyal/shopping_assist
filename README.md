# Shopping Copilot

**TechJam 2026 — Track 4: AI Conversational Search and Recommendations.** Solo entry.

A multi-turn agent that finds one hidden product in a frozen 50,000-item Amazon
catalog within ten turns — and a second benchmark, built specifically to find out
whether that agent understands anything at all.

> **Post-deadline note.** The submission closed on 1 September 2026. **Only this
> README has been changed since**, because it was still the organiser's
> participant-kit boilerplate and told a reader nothing about this entry. No code,
> data, results, or other documents have been touched. Every number below was
> measured before the deadline and is reproducible from a file in
> [`results/`](results/README.md).

---

## The short version

The organiser publishes the evaluator, so it was read before any code was written.
The "customer" in it is not a language model. It is a deterministic program that
derives everything it will ever say from the target product's own metadata, then
quotes those strings back at you.

That reframes the task. A high score against a shopper who quotes catalog text
proves string matching, not understanding. So two things were built: an agent that
maxes the official benchmark, and a second benchmark to find out whether that score
meant anything.

It largely did not. **The same agent scores 0.9746 against the official evaluator
and 0.348 against a language-model customer instructed never to quote the listing.**
Everything after that point was driven by the second number.

---

## Results

Against the organiser's evaluator, 200 public sessions:

| | TechnicalScore | HitRate@10 | MRR | MTTC |
|---|---|---|---|---|
| organiser weak-BM25 reference | 0.1067 | 0.125 | 0.068 | 9.81 |
| **submitted agent** | **0.9746** | **1.000** | **0.988** | **2.095** |

The submitted configuration runs on the Python standard library alone: no
third-party package, no model file, no network call, no credential, zero tokens,
about half a millisecond per turn. Verified on a fresh clone with a stock
interpreter, credentials removed, and network unavailable.

Both benchmarks, all three tiers:

| configuration | official evaluator | LLM customer |
|---|---|---|
| **tier 1 — lexical, submitted** | **0.9746** | 0.3479 |
| tier 2 — + dense retrieval | 0.9724 | 0.7054 |
| tier 3 — + listwise rerank + hedge | 0.8740 | **0.7903** |

The tiers move in opposite directions. The layer worth +0.085 against a shopper who
paraphrases costs 0.098 against one who quotes. Tier 1 ships because the rules
require reproduction under the published evaluator; the other two exist because the
second benchmark says that ranking is an artifact of the first.

Holdout validation on both halves of the public set: 0.9748 and 0.9700.

---

## The design

A **structured frame**, not a transcript. A sticky product-type slot that must be
*contradicted* rather than outvoted, accumulating attributes, and subtracting
negatives. An earlier bag-of-words version drifted badly — 17 of 24 sessions
collapsed into the catalog's largest category while the customer typed *"I want
shoes, not shirts!"*. Making type a slot makes that failure structurally impossible.

Four retrieval routes — stated category, catalog-mined vocabulary, rare-term
lexical, and dense embeddings — fused into one reranker, with an optional listwise
tier above it. Full diagram and component detail in
[`submission/README.md`](submission/README.md).

**On asking questions.** The simulator acts only on `ask_attribute` and never reads
the question text, and the open probe `other` matches *any* undisclosed requirement
while a named attribute matches only its own type. The optimal questioning policy
against this evaluator is therefore a constant. Four attempts at something more
adaptive — an entropy planner, contrastive questions, rescue gating, adaptive
widening — all scored worse. That is reported here rather than shipping a more
impressive-looking policy that measurably loses.

---

## What was falsified

Nine ideas were killed against a pre-registered acceptance bar, each with a
diagnosed cause and each still reachable behind an environment flag:

| idea | cost | why it lost |
|---|---|---|
| contrastive questions every turn | −0.055 | extracts less per turn than the open probe |
| rescue-gated questions | −0.061 | fires after the pool has already drifted |
| adaptive slate widening | −0.034 | widening spends rank to buy coverage already held |
| select-from-30 | −0.020 | more candidates, worse separation |
| cross-encoder rerank | −0.047 | rescores rather than promotes |
| entropy question planner | −0.011 | optimised splits over a pool that was drifting |

Full record with every number in [`NOTES.md`](NOTES.md); every measurement is
backed by a file in [`results/`](results/README.md) holding per-session transcripts,
so any claim here can be re-derived rather than taken on trust.

---

## Reproduce

Python 3.10+. The submitted configuration needs the standard library and nothing else.

```bash
python3 -m evaluator.local_evaluator
```

Writes `results.json` and prints TechnicalScore 0.9746. The adversarial benchmark
needs a model credential and is documented in [`docs/report.md`](docs/report.md) §4.

---

## Repository map

```text
submission/          the graded bundle — agent.py, src/, empty requirements.txt
docs/report.md       the full technical report, 11 sections
NOTES.md             the complete experimental record, in order, including failures
results/             every number in every document, with per-session transcripts
tools/               diagnostics — scorecards, sweeps, audits, funnel instrumentation
sim/                 the adversarial LLM-customer benchmark (development only)
src/                 agent implementation
```

Start with [`docs/report.md`](docs/report.md) for the argument, `NOTES.md` for the
evidence, [`submission/README.md`](submission/README.md) for the build.

---

## Provenance

Built on the organiser's participant kit. `evaluator/`, `starter/`, `data/` and the
`docs/` specification files are theirs and are unmodified — the evaluator in
particular was never edited, since the reported score depends on it being the
published one. The catalog and sessions derive from Amazon Reviews 2023 by McAuley
Lab, UCSD; see [`DATA_ATTRIBUTION.md`](DATA_ATTRIBUTION.md).

The agent never reads `ground_truth`. Private-label and target-pool reconstruction
were available and deliberately not attempted; see [`docs/report.md`](docs/report.md) §10.
