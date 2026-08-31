# Measurement record

Every number in `docs/report.md` and `NOTES.md` comes from a file here. Each one
holds full per-session transcripts, so any claim can be re-derived rather than
taken on trust.

## `scripted/` — the organiser's evaluator, 200 public sessions

| file | score | configuration |
|---|---|---|
| `results_frozen_tier1.json` | **0.9746** | submitted: lexical only |
| `results_frozen.json` | 0.9724 | + dense retrieval |
| `results_extractor_only.json` | 0.9316 | + model extraction |
| `results_listwise.json` | 0.8740 | + listwise reranking |

## `llm_customer/` — our adversarial benchmark

Stratified sessions with a language-model customer. `n120` files are the
reportable measurements at four sessions per cell; `n60` files are paired
experiments against a pre-registered acceptance bar.

| file | score | what it measures |
|---|---|---|
| `sim_n120_hedge.json` | **0.7903** | final: dense + listwise + width-1 hedge |
| `sim_n120_listwise.json` | 0.7481 | listwise without the hedge |
| `sim_n120_dense.json` | 0.7054 | retrieval only |
| `sim_n120_model.json` | 0.5453 | before the dense track |
| `sim_n52_tier1.json` | 0.2994 | submitted config, n=54 (stopped for budget) |

Rejected experiments, each paired against `sim_n60_hedge.json` (0.8002):

| file | score | outcome |
|---|---|---|
| `sim_n60_wide.json` | 0.7797 | select-from-30, −0.020 |
| `sim_n60_question.json` | 0.7453 | contrastive questions every turn, −0.055 |
| `sim_n60_rescue.json` | 0.7384 | rescue-gated questions, −0.061 |
| `sim_n60_rerank2.json` | 0.6834 | cross-encoder, corrected blend |
| `sim_n60_stall.json` | 0.6635 | adaptive slate widening, −0.034 |
| `sim_n60_rerank.json` | 0.6505 | cross-encoder, first blend, −0.047 |
| `sim_n60_infogain.json` | 0.6987 | entropy question planner, −0.011 |

Earlier builds (`sim_v3`, `sim_v4`, `sim_v6_*`, `sim_trial`, `sim_stub`) are kept
for the score progression in `NOTES.md`; they are not comparable to the current
agent.

## `ask_audit.json`

Every clarification the agent asks across the 200 public sessions, classified as
productive, wasted, stonewalled or interrupted. Backs the open-probe dominance
argument.
