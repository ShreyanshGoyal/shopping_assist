# Shopping Copilot - conversational search agent

A multi-turn agent that finds one hidden product in a frozen 50,000-item Amazon
catalog within ten turns, by accumulating what the shopper says into a structured
request and searching against that rather than against the raw transcript.

## Architecture

```
customer turn
     │
     ▼
  parsing ──────────► frame ◄────────── extraction (optional model tier)
  templates +        ├ type      one slot, sticky, replaced only on contradiction
  generic fallback   ├ attributes  material · colour · budget · quoted phrases
                     └ negatives   what the shopper has ruled out
                          │
                          ▼
              ┌──── candidate generation ────┐
              │  category route (stated)     │
              │  vocabulary route (mined)    │
              │  lexical route (rare terms)  │
              │  dense route (tier 2)        │
              └──────────────┬───────────────┘
                             ▼
                   scoring · quote match, category, negatives, popularity prior
                             ▼
              listwise rerank (tier 3, gated on near-ties)
                             ▼
              slate sizing ──► up to 10 parent_asin + one clarification
```

The frame is the design. An earlier version represented the conversation as a
growing bag of words and searched over it; drift was the inevitable result, with
17 of 24 sessions collapsing into the catalog's largest category regardless of
what the shopper asked for. Because the product type is a slot that must be
*contradicted* rather than a tally that can be outvoted, that failure mode is
structurally impossible here.

## Results

Two benchmarks. The **scripted evaluator** is the organiser's, and it is what
determines the official score. The **language-model customer** is ours: the same
sessions and the same metric formulas, but the customer is played by a model that
paraphrases, hedges and complains instead of quoting catalog text. It exists
because the first version of this agent scored 0.975 against the scripted
customer while relying almost entirely on matching its exact wording.

| tier | scripted (200) | LLM customer (120) | requires |
|---|---|---|---|
| organiser BM25 baseline | 0.1067 | - | - |
| **1 - lexical, graded config** | **0.9746** | **0.299** (n=54) | nothing |
| 2 - + dense retrieval | 0.9724 | 0.7054 | numpy, onnxruntime, tokenizers |
| 3 - + model extraction and listwise rerank | 0.8740 | **0.7903** | network + credential |

The tiers move in opposite directions, and that is the central finding. The
submitted configuration scores **0.9746 against a quoting customer and 0.299
against a paraphrasing one** — same agent, same targets, same metric. Against the
quoting customer, exact-match machinery is near-optimal and every model-based
layer can only overrule an answer that was already right. Against the
paraphrasing one, those layers are worth the gap between 0.299 and 0.790.

The tier-1 language-model figure is measured over 54 sessions; that run was
stopped part-way to preserve API budget and is reported at its true sample size. **Tier 1 is submitted**,
because the organiser confirmed that final results are generated exactly as the
published local evaluator.

## Quickstart (tier 1 - the graded configuration)

Python 3.10 or newer. No third-party package, no model file, no network, no
credential.

```bash
python3 -m evaluator.local_evaluator
```

Expected: `recommended_technical_score` of **0.9746**, hit rate 1.000, MRR 0.988,
MTTC 2.095, and zero reported tokens. Verified on a fresh clone with stock Python
3.12, credentials stripped and a minimal PATH.

## Optional tiers

**Tier 2 - dense retrieval.** Adds a local sentence-embedding channel over the
catalog. Worth roughly +0.16 on the language-model customer and −0.002 on the
scripted one.

```bash
pip install -r requirements-optional.txt
```

Then place `bge-small-en-v1.5` (`model.onnx` and `tokenizer.json`) under
`models/bge-small-en-v1.5/` and build the index once - about 25 minutes,
producing a 73 MB array cached to `.cache/embeddings/`:

```bash
python3 -m tools.build_embeddings
```

**Tier 3 - model extraction and listwise reranking.** Requires a Vertex AI
credential. Development-only; not part of the graded path.

```bash
AGENT_EXTRACTOR=model AGENT_LISTWISE=1 python3 -m evaluator.local_evaluator
```

Every tier is detected at construction and skipped when unavailable. Setting the
tier-3 variables with no credential present emits a warning and runs the offline
path; a full 200-session run under those conditions was verified to complete at
the offline score.

## Reproducing every number

Run from the repository root. Commands marked *(dev)* need the research tooling
and, where noted, a credential.

| claim | command |
|---|---|
| tier 1 score, 0.9746 | `python3 -m evaluator.local_evaluator` |
| tier 2 score, 0.9724 | install optional deps, build index, then the same command |
| every ask classified; open-probe dominance | `python3 -m tools.ask_audit` |
| where the target ranks, recall vs ranking | `python3 -m tools.diagnose` *(dev)* |
| which misses a deeper rerank window could reach | `python3 -m tools.miss_bands` *(dev)* |
| tie-break weights are optimal, both halves held out | `python3 -m tools.tiebreak_sweep` |
| popularity prior shape, both halves held out | `python3 -m tools.popularity_sweep` |
| category vs dense weighting is at a local optimum | `python3 -m tools.weight_sweep` *(dev)* |
| turn-1 ranking headroom, +0.0042 ceiling | `python3 -m tools.turn1_headroom` |
| language-model customer benchmark | `python3 -m sim.harness --sessions 120 --backend vertex --model <id>` *(dev, credential)* |
| a single session, turn by turn | `python3 -m tools.trace_session --sample public_0001` |

`NOTES.md` carries the full experimental record, including every negative result
with its diagnosis.

## Repository map

| path | what it is |
|---|---|
| `submission/` | the graded bundle: `agent.py`, `src/`, requirements, this README |
| `src/` | the agent — frame, retrieval routes, ranking, policy, optional tiers |
| `sim/` | the language-model customer benchmark: harness, personas, model client |
| `tools/` | diagnostics and sweeps; every table in the report maps to one |
| `results/scripted/` | organiser-evaluator runs, full per-session detail |
| `results/llm_customer/` | benchmark runs, including every rejected experiment |
| `results/README.md` | index mapping each result file to the claim it supports |
| `docs/report.md` | the written report |
| `docs/webinar_notes.md` | organiser's technical session, transcribed |
| `NOTES.md` | full experimental record, including every negative result |

If you only read three things: this README's results table, `docs/report.md` §6
for what was falsified, and `results/README.md` to check any number against its
source file.

## Limitations

**The remaining scripted points are irreducible.** Five of 200 sessions convert
below rank 1, and reading them individually shows genuine ties rather than
ranking errors: in `public_0002` the target scores 53.93 against 54.35 for
another leather belt with a buckle closure, and in `public_0144` the top three
polyester down jackets are separated by 0.16 points. The intent card discloses
only requirements such as `leather`, `100% Leather`, `Buckle closure` and
`Imported`, which dozens of catalog products satisfy identically. The practical
ceiling is about 0.992, the remainder being the intent-override lockout.

**Category assignment in the catalog is partly arbitrary, and that caps the
language-model result.** Supplying the agent the target's true category would
take rank-1 placement from 34.5% to 61.3% - 58 of 119 sessions improve. But the
correct node is inferable from the request only about half the time, because a
crew-neck t-shirt is filed under *Underwear Undershirts* and mules under
*Loafers* when a *Mules & Clogs* node exists. Most of that 58-session prize is
therefore unreachable by any method.

**Clarification questioning is a solved constant, not a strategy.** The scripted
simulator acts only on `ask_attribute` and never reads the question text, and the
open probe `other` matches any undisclosed requirement while a named attribute
matches only its own type. Four attempts at something smarter - an entropy
planner, contrastive question generation, a rescue-gated variant, and committing
early once the probe is spent - all measured worse. They remain in the code
behind environment flags with their results recorded.

**The language-model benchmark is not a leaderboard predictor.** It measures
robustness to a customer the official evaluation does not use. It is reported
because it is what exposed this agent's dependence on exact wording, and what
falsified eight subsequent ideas that looked sound.

## Contributions

Solo entry. 

## Disclosure

**Graded path:** zero tokens, zero network calls, no credential. Latency about
0.5 ms per turn at the median and 0.7 ms at the 95th percentile; roughly 6.5
seconds of one-off catalog indexing at startup; peak memory about 840 MB.
Verified offline on a fresh clone.

**Development only:** Gemini 3.5 Flash Lite via Vertex AI, used for the
language-model customer simulator and for the optional tier-3 layers. A
120-session benchmark run costs roughly 1.06M input and 35k output tokens.
Responses are cached by request, so repeat runs pay only for turns whose
conversation diverged. No credential is stored in the repository; the client
mints an OAuth token through the gcloud CLI on demand.

**Data:** the organiser's frozen catalog and public sessions only. No external
training data, no catalog modification, no attempt to reconstruct the private
target pool.

---

## Appendix: a demonstrated multi-turn session

Scripted intent-override session `public_0003`, produced by
`python3 -m tools.trace_session --sample public_0003`. The hidden target and
intent card are shown for the reader; the agent never sees either.

```
TARGET (hidden):  Casio Men's Wrist Watch AQ-800E-7A   [B09YMTWDXJ]
                  Clothing, Shoes & Jewelry > Men > Watches > Wrist Watches
INTENT CARD:      hard: Water Resistant · 3 Year Battery
                  soft: Day / Date Indicator · Stainless Steel Band
AGENT SEES:       "Prior purchases emphasize material, fit; ratings are usually positive."

TURN 1  CUSTOMER  I'm looking for Watches Wrist Watches. Stainless Steel Band
        AGENT     Anything else about this that matters to you?   [ask_attribute='other']
        SHOWS     1. Casio Men's Wrist Watch AQ-800E-7A          <<< TARGET, rank 1
        (scoring locked: an override session cannot convert before the new intent arrives)

TURN 2  CUSTOMER  For that, what matters is: Water Resistant; 3 Year Battery.
        AGENT     Anything else about this that matters to you?   [ask_attribute='other']
        SHOWS     1. Casio Men's Wrist Watch AQ-800E-7A          <<< TARGET, rank 1
                  2. St. Moritz Momentum M1 Cyclone Chrono Dive Watch
                  3. CURREN Quartz Analog Calendar Wrist Watch
                  ... 10 shown
        (still locked)

TURN 3  CUSTOMER  Actually, ignore my earlier preference. What I need is: Water Resistant.
        AGENT     Anything else about this that matters to you?   [ask_attribute='other']
        SHOWS     1. Casio Men's Wrist Watch AQ-800E-7A          <<< TARGET
        HIT at turn 3, rank 1.  reciprocal rank = 1.000
```

Three things this shows.

**The agent holds a single product at turn 1.** It is confident, and padding the
slate would only risk converting at a worse rank later. Withholding is a
deliberate policy: rank is worth roughly six times a turn under this metric.

**The override does not derail it.** The customer discards "Stainless Steel Band"
and states a different requirement, and the target stays at rank 1. The frame
demotes superseded statements rather than erasing them, because in this simulator
the discarded preference and its replacement are both drawn from the same
product's description - so the earlier statement remains true evidence.

**Turns 1 and 2 cannot score.** The evaluator refuses a hit until the new intent
has been delivered, which puts a floor of turn 3 on every override session and
caps the achievable MTTC at about 1.39 across the set. The agent is at rank 1
before the lock lifts and converts on the first turn it is allowed to.
