# Working notes — TechJam Conversational Search

## Score progression (200 public dev sessions)

| Version | TechnicalScore | HitRate@10 | MRR | MTTC |
|---|---|---|---|---|
| Organiser BM25 baseline | 0.1067 | 0.125 | 0.068 | 9.81 |
| v1 constraint tracking + category routing | 0.9037 | 1.000 | 0.715 | 1.54 |
| v2 + confidence-gated slate sizing | **0.9750** | 1.000 | 0.985 | 2.03 |

Theoretical ceiling is ~0.9922: intent-override sessions cannot convert before
turn 3-4, which pins MTTC at ~1.39 even with perfect ranking.

## What the task actually is

The simulated customer quotes product metadata close to verbatim — material,
colour, price and normalised bullets drawn from the target's own `features` and
`details`. So a stated requirement is a high-precision fingerprint, and the job
is to (a) drain the requirement set quickly and (b) match it against normalised
catalog text rather than bag-of-words BM25.

Two structural facts drive the design:

1. **Category routing is extremely strong.** The opening turn names the two most
   specific levels of the target's category path. There are 1,115 such buckets
   over 50,000 products, median size 8. Turn one alone narrows the catalog by
   three orders of magnitude for half the sessions.
2. **Rank is worth ~6x more than turns.** Moving one session from rank 5 to rank
   1 is worth 0.0012 of final score; the two extra turns it costs lose 0.0002.
   The baseline's instinct — pad the shortlist and convert early — is backwards.

## Design decisions worth defending

- **Open-ended probing.** A named attribute can only surface a requirement of
  that type; an open question reaches the whole requirement set. Its exhaustion
  is also informative: nothing more to add to an open question means nothing
  more to add at all, which is the agent's signal to stop asking and commit.
- **Override narrows, it does not contradict.** Both the original and the
  replacement preference derive from the same target, so earlier statements are
  demoted rather than erased. Hard slot erasure would lose information here.
- **Scenario-aware exclusion.** A slate shown and not accepted is provably
  wrong — except before an override lands, where hits are not scored. Sessions
  that open in buying or browsing form cannot be awaiting an override, so they
  exclude from turn 1; everything else waits until turn 4.
- **Offline by construction.** Submission rules warn that final scoring may run
  with network access disabled. The retrieval core uses no model and no network.

## Feasibility numbers (to disclose in the report)

- Index build: 6.5 s once at startup; peak RSS ~840 MB
- Latency: p50 0.5 ms, p95 0.7 ms per turn
- Tokens: 0 — no model call, no credential, no external service
- Full 200-session evaluation: ~9 s on an Apple M2

## Two benchmarks

The organiser's evaluator plays the customer from a fixed script that quotes
product metadata almost verbatim. `sim/` plays the customer with a language model
instead, so the same agent is measured against someone who paraphrases, hedges
and complains. Scores are computed with the organiser's own formulas, so the two
are directly comparable.

| Agent | Scripted (200) | LLM customer (same 6) |
|---|---|---|
| v2 verbatim matching | 0.9750 | 0.589 |
| v3 + catalog-mined lexicon routing | 0.9756 | 0.595 |
| v4 + information-gain questioning | 0.9558 | 0.106 |

The gap between the columns is the honest measure of how much of the scripted
score came from matching wording rather than understanding a request.

## Finding: query drift into the largest bucket

Information-gain questioning regressed the LLM benchmark by a factor of five. The
transcripts show why, and the cause is not the questioning.

Across 24 LLM sessions, only **5 final slates sat in the target's own category**,
and **13 collapsed into "Shirts T-Shirts"** — the catalog's largest bucket at
1,354 products. Routing sums term evidence over everything the customer has said,
and by the end of a session that is a median of ~900 characters. Common
vocabulary accumulates, the largest bucket appears in nearly every term's
candidate list, and it wins on volume.

Two things make it worse:

* The agent's own question vocabulary comes back in the answers. Ask "casual or
  classic", hear "casual", then route on the word the agent introduced.
* Information gain optimises splitting the *current* pool. Once the pool is
  wrong, it asks confident, well-formed questions about the wrong products, and
  the answers poison the query further. In one session the agent asked "Which are
  you after — Shirts T-Shirts or Tees & Blouses T-Shirts?" of a customer who had
  said "flat sandal" five times.

Information-gain questioning is therefore parked behind `AGENT_STRATEGY=infogain`
until retrieval is trustworthy. The idea is sound; it just cannot run on a pool
that drifts.

### Fix direction

1. Weight the opening turn far above later ones — the product type is stated first.
2. Suppress echo: discount vocabulary the agent introduced into the conversation.
3. Normalise routing so a large bucket cannot win on size alone.
4. Make the product type sticky once established, and require explicit
   contradiction to change it.

## v6: the structured frame

The bag-of-words design was replaced with a frame (`src/frame.py`): a sticky
product-type slot, accumulating attributes, and subtracting negatives. Extraction
became a swappable component (`src/extract.py`) so the same agent runs offline or
with model assistance.

| Agent (24 LLM sessions) | Score | HitRate | MRR | MTTC |
|---|---|---|---|---|
| v4 bag-of-words + planner | 0.211 | 0.292 | 0.127 | 9.62 |
| v6 frame + lexical extractor (offline) | 0.428 | 0.583 | 0.236 | 7.71 |
| v6 frame + model extractor | 0.578 | 0.696 | 0.466 | 6.48 |

Scripted evaluator is unchanged at 0.9752 with the offline extractor, so none of
this cost anything on the benchmark that is actually scored.

**Drift is gone.** v4 ended 17 of 24 sessions inside "Shirts T-Shirts"; v6's most
common final bucket appears twice. That is the architecture doing its job rather
than a weighting that happened to tune well.

**Honest caveat on the headline number.** v4 was the deliberately parked version
with the broken planner, so v4 -> v6 overstates the gain. The fair baseline is v3,
and on the six sessions where the two can be compared directly, v3 scored 0.595
and v6 scores 0.585 — a wash. The frame's contribution to *score* is therefore
not yet demonstrated; what is demonstrated is that drift stopped, that negatives
are representable, and that the offline/online gap is now measured at
0.150. Reproducing v3 across all 24 sessions would settle it.

**Where v6 still fails**, by writing style: terse 0.310 (hit rate 0.400), informal
0.470, plain 0.742. Short, clipped messages give the extractor almost nothing to
work with. By scenario, `mind_changer` is 0 for 1 — override handling is untested
and probably broken.

## v7: the dense retrieval track

### Why

Every failure the LLM-customer benchmark surfaced is a vocabulary failure. The
customer says "soft bottom like yoga mat"; the product says "cushioned EVA
footbed". Terse writers score worst of all styles (0.310) precisely because they
give almost no words to match on. No amount of lexical weighting fixes a request
that shares no tokens with its target.

The problem statement also asks for this directly. Pillar I specifies "a diverse
dense retrieval track for open-ended Browsing" and a pipeline "combining keyword,
category, and vector similarity". Before this build the system had no dense track
at all.

### What

`bge-small-en-v1.5` (33M parameters, 384 dimensions) run through ONNX Runtime on
CPU. The catalog is embedded once into a 50,000 x 384 float32 matrix cached to
disk; retrieval at query time is one matrix-vector product, a few milliseconds.
No network, no credential, no vector database — which is what the rules mean by
"entirely in-memory for light execution".

The track is optional by construction: `Catalog.attach_dense()` returns False when
numpy, onnxruntime or the prebuilt index are missing, and the agent runs
lexical-only. The offline lexical paths remain the baseline, not a fallback.

Fusion is additive into the existing scorer rather than a separate pipeline.
Cosines from this encoder occupy a narrow band, so an absolute value carries
little information; the score is normalised against the candidate pool before
being weighted, which keeps it from overwhelming a category the customer stated
outright.

### Result: +0.160 on the LLM-customer benchmark

Paired comparison, identical 120 stratified sessions, model extractor both sides.

| | Score | HitRate | MRR | MTTC |
|---|---|---|---|---|
| lexical only | 0.545 | 0.658 | 0.451 | 6.96 |
| **+ dense track** | **0.705** | **0.866** | **0.553** | **5.66** |

Every scenario and every writing style improved. The largest gains land exactly
where the lexical system was weakest — `use_case_led` +0.210 (the worst cell),
`vague` +0.211, `decisive` +0.205.

| by scenario | before | after |
|---|---|---|
| use_case_led | 0.331 | 0.541 |
| mind_changer | 0.422 | 0.592 |
| vague | 0.512 | 0.723 |
| decisive | 0.565 | 0.770 |
| indifferent | 0.669 | 0.694 |
| brand_led | 0.772 | 0.905 |

Session-level: 46 improved, 31 regressed, 42 unchanged. It is not uniformly
better — a third of the sessions it touched got worse — but the net is strongly
positive and the regressions are worth a separate look.

The cost is **-0.008 on the scripted evaluator** (0.9752 -> 0.9673), where the
customer quotes verbatim and dense mostly adds noise. Trading 0.008 on the
benchmark we have nearly maxed for 0.160 on the one that measures generalisation
is the trade this project exists to make.

### Encoder sanity check

Run against the three documented failures from the v4 transcripts:

| Query | Best match | Cosine |
|---|---|---|
| "soft bottom like yoga mat" | Santiro Flat Sandals **Yoga Mat Sole** | 0.678 |
| "need some new pants" | Curvify **Jeggings** (t-shirt scored 0.630) | 0.728 |
| "stretchy pull-on denim" | Curvify **Pull on Skinny Jeans** | 0.841 |

All three are cases the lexical system got wrong. End-to-end benchmark numbers
are pending.

## Benchmark methodology

Scenario and style were originally assigned by hashing the sample id, which left
cells of one or two sessions — `mind_changer` n=1, `chatty` n=1 — and made the
per-cell breakdown unreadable. Priorities were being argued from noise.

Assignment now cycles the 6 x 5 scenario-by-style grid, so a run of 120 gives
exactly 4 sessions per cell in all 30 cells. `--hashed` reproduces the older runs.

The harness also survives failure now: a customer-side error aborts one session
rather than the run, results are written incrementally, and `--resume` reuses
completed sessions. An earlier 29-minute run was lost to a single transient API
error before this existed.

### Cost

Measured over the 24-session run: 5,990 prompt tokens per session on the customer
side, plus roughly one 450-token extraction call per turn on the agent side. A
120-session run is therefore about **1.06M input and 35k output tokens** on
Gemini 3.5 Flash Lite via Vertex. Responses are cached by request, so repeat runs
after an agent change only pay for turns whose conversation actually diverged.

## Baseline at n=120 (stratified, 4 per cell)

Lexical retrieval, model extractor, LLM customer. Zero aborted sessions.

| | Score | HitRate | MRR | MTTC |
|---|---|---|---|---|
| overall | 0.545 | 0.658 | 0.451 | 6.96 |

### The small-sample breakdown was wrong, and it had been driving decisions

The overall figure barely moved (n=24 estimated 0.578, n=120 gives 0.545). The
per-cell breakdown reversed.

| by style | n=24 | n=120 |
|---|---|---|
| terse | **0.310 (worst)** | **0.608 (best)** |
| plain | 0.742 (best) | 0.595 |
| informal | 0.470 | 0.510 |
| chatty | 0.620 | 0.477 (worst) |

| by scenario | n=24 | n=120 |
|---|---|---|
| use_case_led | 0.640 | **0.331 (worst)** |
| mind_changer | 0.000 (n=1) | 0.422 |
| brand_led | 0.430 | **0.772 (best)** |

"Terse writers are our biggest hole" was stated in these notes and acted on. It
was five sessions of noise; terse is now the strongest style. Conversely
`use_case_led` looked healthy at n=3 and is in fact the weakest cell — customers
who describe the occasion ("something for a beach wedding") rather than the
product, which is the hardest type inference there is.

`mind_changer` at 0.422 over 20 sessions is genuinely weak, so the instinct to
fix intent override was directionally right even though the evidence offered for
it (one session, conflated with the already-solved scripted override) was not.

The lesson is methodological and belongs in the report: every per-cell number
before this run was unusable, and two separate roadmaps were prioritised off it.

Scripted evaluator remains 0.9752 lexical-only and 0.9673 with the dense track
enabled — dense costs a little where the customer quotes verbatim, which is the
expected trade and is only worth paying if the LLM-customer gain is larger.

## Exact-quote index, and letting dense stand down

Enabling the dense track cost 0.008 on the scripted evaluator, because a customer
quoting product bullets verbatim is already decided and semantic similarity can
only add noise on top.

`Catalog.by_quote` indexes every product bullet by its exact text, so a quoted
requirement is an O(1) lookup rather than a score computed against every
candidate. When such a quote fires — or when the customer names a category that
exists verbatim in the taxonomy — the dense contribution is scaled to 0.3.

Scripted: 0.9673 -> **0.9729**, against 0.9752 lexical-only. The dense track now
costs 0.002 there while adding 0.160 on the LLM customer.

## Clarification policy: order beats cleverness

Information-gain questioning regressed the scripted evaluator by 0.020 (0.9729 ->
0.9531), all of it MTTC. The scripted customer only answers a named attribute if
it holds a constraint of that type, so a sharp question frequently returns
nothing and costs a turn.

The fix was ordering rather than tuning. An open-ended probe is productive
against *any* customer because it reaches the whole requirement set instead of
one attribute of it, so it can never be a wasted turn. It now runs until it stops
paying; only then does information-gain questioning take over, by which point the
candidate pool it estimates entropy over is worth trusting.

With that ordering, infogain costs **nothing** on the scripted benchmark: 0.9729
either way. Whether it helps the LLM customer is measured separately.

## Negative result: information-gain questioning does not pay

Tried twice, measured both times, and it does not earn its turn.

| | Scripted | LLM customer (59 paired) |
|---|---|---|
| open-ended probing only | 0.9729 | 0.710 |
| + information-gain questioning | 0.9729 | 0.699 |

Neutral on the scripted benchmark once the ordering was fixed, and **-0.011** on
the LLM customer: 13 sessions improved, 18 regressed, 28 unchanged. Per-scenario
deltas look interesting (`mind_changer` +0.179) but each scenario is 10 sessions
here, which is the same small-sample trap documented above; they are not
reportable.

The idea is intuitive and it is explicitly one of the problem statement's
"Proactive Guidance" directions, which is why it was worth two attempts. It stays
in the repo behind `AGENT_STRATEGY=infogain`, off by default, with this result
recorded rather than the feature quietly shipped.

**Why it fails is more useful than the fact that it does.** Decomposing the
n=120 dense run:

```
103 hits / 16 misses.  Among hits: mean rank 2.91, mean turn 4.83
rank distribution: 51 hits at rank 1, 52 spread across ranks 2-10
headroom on MRR (all hits -> rank 1):  +0.134   <- largest single lever
headroom on HitRate (all misses fixed): +0.067
```

The binding constraint is *ranking*, not question efficiency. Asking a sharper
question cannot help an agent that already has the product in its candidate set
but ranks it fourth.

## Finding: slate withholding was tuned for the wrong customer

In the same run, **23 of 103 hits land exactly on turn 6** — `COMMIT_TURN`, where
the agent stops showing a single product and widens to ten. Those sessions had
the target in the top ten already and were being withheld.

Withholding is a bet that one more turn promotes the right product to rank 1. It
won decisively against the scripted customer, whose requirement set drains in two
turns. Against a language-model customer the conversation never runs dry, so
`information_left` stays true, the agent shows one product for five turns, and
half the eventual hits are not at rank 1 anyway — the bet does not pay.

So the obvious fix was to stop betting once the evidence stops arriving: widen
the shortlist when the leading candidate's score plateaus rather than at a fixed
turn. **It was measured and it is worse.**

| same 60 sessions | Score | HitRate | MRR | MTTC |
|---|---|---|---|---|
| fixed COMMIT_TURN=6 | 0.698 | 0.833 | 0.572 | 5.52 |
| widen on plateau | 0.664 | 0.817 | 0.493 | 5.63 |

The mechanism did what it was designed to do — turn-6 conversions fell from 22%
of hits to 16% — and the outcome got worse by 0.034, almost all of it MRR.
Converting a turn earlier locks in whatever rank the product currently holds, and
rank is worth roughly six times a turn.

The reasoning error is worth recording: "half the hits are not at rank 1, so
withholding is not paying" ignores the counterfactual. Withholding *was* paying;
those hits would have been worse without it. Reverted, kept behind `AGENT_STALL=1`.

Two ideas in a row — information-gain questioning and adaptive widening — were
intuitive, well-motivated, and measurably wrong. The only reason that is known is
the second benchmark. That is the argument for having built it.

## Negative result: cross-encoder reranking does not pay

The score decomposition said ranking was the binding constraint: 52 of 103 hits
landed at ranks 2-10, worth +0.134 of MRR, far more than the +0.067 available
from converting every miss. A cross-encoder reads request and product jointly
rather than comparing two independently-built representations, so it is the
textbook answer. `ms-marco-MiniLM-L-6-v2`, 87 MB, local, 121 ms for a 30-item
shortlist.

It was measured twice and it loses both times.

| same 59 sessions | Score | HitRate | MRR | MTTC |
|---|---|---|---|---|
| dense only | **0.698** | 0.831 | 0.579 | 5.53 |
| + reranker, min-max blend | 0.650 | 0.797 | 0.501 | 5.92 |
| + reranker, sigmoid blend and near-tie gate | 0.683 | 0.831 | 0.538 | 5.66 |

Scripted: 0.9729 without, 0.9702 with the first blend, 0.9714 with the second.

The first blend was straightforwardly wrong — min-max normalisation stretches
whatever the model returns across the full weight range, so thirty near-identical
candidates produce a confident-looking reordering built from noise. Replacing it
with a sigmoid of the raw logit, halving the weight, and skipping the pass
entirely when retrieval already has a clear leader recovered most of the loss.
Not all of it.

**Why it still loses.** The model is trained on MS MARCO: natural-language web
queries against prose passages. It is handed neither. The query is a frame
description — "flat sandals; stretchy fabric straps; yoga mat footbed" — and the
passage is title plus category plus bullet fragments. In a smoke test with a
natural-language query it ranked the sandal correctly and the t-shirt last, so
the model works; the mismatch is in how it is being asked.

More fundamentally, the retrieval score already carries domain evidence the
reranker has no access to: verbatim quote matches, the sticky type slot, explicit
negatives. A general-purpose relevance model breaking those ties on web-search
relevance is discarding better information for worse.

Off by default behind `AGENT_RERANK=1`. Worth one more attempt some day with the
raw customer utterance as the query rather than the frame description.

### Three negative results, and why they are the point

Information-gain questioning (-0.011), adaptive slate widening (-0.034) and
cross-encoder reranking (-0.015) were all well-motivated, all standard practice,
and all measurably wrong here. Two were argued for from the problem statement's
own suggested directions. None of them would have been caught by the scripted
evaluator, which moves less than 0.003 across all three.

That is the case for the second benchmark, and it is the honest version of the
innovation claim: not "we built a clever agent" but "we built the instrument that
told us which of our clever ideas were wrong".

## Error analysis: where the score actually leaks

Five experiments in a row moved nothing. The cause was diagnosing by hypothesis
instead of by measurement, so the next step was an instrument rather than an idea:
`tools/diagnose.py` replays recorded transcripts and reports where the target sits
in the **full** candidate ranking, separating two failures the score cannot
distinguish — never entering the pool (recall) versus entering it and ranking low.

It cost nothing to run: customer turns come from the recording and the extractor's
calls hit the response cache.

```
best position the target ever reached, n=119
  reached rank 1    36 (30.3%)
  reached top 10    50 (42.0%)
  rank 11-100       28 (23.5%)
  rank >100          3 ( 2.5%)
  never in pool      2 ( 1.7%)   <- recall is not the problem
```

Two bugs surfaced immediately from reading actual cases:

* `"men's short-sleeve shirt"` resolved to the category **men shorts**. Stemming
  collapses "shorts" and "short", and unweighted token overlap then let two
  incidental words outvote the one word naming the garment. Fixed by weighting
  overlap by inverse document frequency and by treating the phrase's head noun —
  the last word — as worth several times its modifiers.
* Targets sat in *sibling* categories: `Bracelets Strand` when the frame resolved
  to `Bracelets Stretch`, `Tops & Tees Tanks & Camis` versus `Tees & Blouses Tanks
  & Camis`. An exact node match paid +15 and a sibling at most +2.5, so a target
  one node away was buried. Siblings now earn a share of the bonus.

Together: scripted unchanged, LLM customer +0.002. Real bugs, no aggregate gain.

## The oracle test, and the ceiling it found

Hand the agent the target's true category and re-run the replay. Diagnostic only —
it reads ground truth, which the agent never does.

| | rank 1 | top 10 | median rank |
|---|---|---|---|
| as-is | 41/119 (34.5%) | 90 (75.6%) | 2 |
| oracle category | **73/119 (61.3%)** | **109 (91.6%)** | **1** |

**58 of 119 sessions improve purely from knowing the right aisle**, and only 10
stay outside the top ten once the category is right. Every experiment so far had
attacked within-category ranking — worth about 10 sessions — while the real
bottleneck was worth 58.

So the next question was whether that 58 is reachable. It mostly is not:

```
is the TRUE category reachable from the extracted type phrase?
  rank 1         28.6%      top 5   27.7%   (cumulative 56%)
  top 12          7.6%      top 50  16.8%
  not in top 50   8.4%      no type phrase  10.9%
```

And the unreachable cases explain why:

| extracted phrase | catalog's "true" category |
|---|---|
| crew neck t-shirt | Underwear Undershirts |
| mules | Shoes Loafers & Slip-Ons (a Mules & Clogs node exists) |
| side-clip band | Accessories Belts |
| knit jeggings | Women Jeans |

**The catalog's category assignment is partly arbitrary.** These are not retrieval
failures; the filing is not predictable from the product. A ceiling of roughly
this shape is therefore structural, and any approach that leans hard on resolving
one exact node inherits it.

That reframes the design question from "resolve the category better" to "depend on
the category less", which is what the weight sweep in `tools/weight_sweep.py`
tests: a +15 bonus for a node that is wrong most of the time may be doing harm.

### What was kept

* IDF-weighted, head-noun-weighted category name matching.
* Sibling credit instead of an all-or-nothing category bonus.
* Semantic category resolution via the encoder, which moved wrong-or-unset
  resolutions from 47 of 119 down to 33 (`AGENT_SEMANTIC_CATEGORY=0` disables).
* Soft credit spread over the plausible category nodes rather than one
  commitment — neutral offline, kept because it removes a single point of failure.

### Weight sweep: the current configuration is at a local optimum

`tools/weight_sweep.py`, 119 replayed sessions, free to run.

| config (category / soft / dense) | rank 1 | top 10 | median |
|---|---|---|---|
| **current (15 / 9 / 5)** | 40 | **87** | 2 |
| category halved (7.5 / 4.5 / 5) | **46** | 82 | 2 |
| category quartered (3.75 / 2.25 / 5) | 44 | 74 | 2 |
| dense doubled (15 / 9 / 10) | 29 | 60 | 5 |
| category halved + dense doubled | 39 | 68 | 3 |
| category near-off (1.5 / 1 / 12) | 40 | 67 | 3 |

The hypothesis that a frequently-wrong category bonus must be hurting is half
right: cutting it buys six rank-1 placements and loses five from the top ten. Hit
rate carries 0.5 of the composite against MRR's 0.3, so that trade loses. Doubling
the dense weight is plainly worse, so the semantic channel is not under-weighted
either.

No configuration beats the current one. The weights are not the remaining problem.

## Plateau, and what it consists of

Seven consecutive experiments produced no aggregate improvement on the LLM
customer: information-gain questioning, adaptive slate widening, cross-encoder
reranking twice, category-resolution fixes, soft category commitment, and a
weight sweep. Scripted moved less than 0.005 across all of them.

The error analysis explains why, and the explanation is structural rather than a
missing technique:

* **Recall is solved.** Only 2 of 119 targets never enter the candidate pool.
* **Category is the dominant lever and is largely unreachable.** Knowing the true
  node would take rank-1 from 34.5% to 61.3%, but the node is inferable from the
  request only about half the time, because the catalog files a crew-neck t-shirt
  under Underwear Undershirts and mules under Loafers.
* **What remains is genuine ambiguity.** With the correct category supplied, ten
  sessions still miss the top ten — targets like "Emmalise Women's Basic Casual
  Long Camisole" competing against "Active Basic Plain Basic Camisole". No
  ranking function separates those from a shopper's description.

This is worth stating plainly in the report rather than presenting the plateau as
unfinished work. The measured ceiling is a result, and the instrument that found
it is the contribution.

## What finally worked: LLM listwise reranking

Eight experiments in, the first clear win — and it succeeded precisely where the
cross-encoder failed, using the two fixes that post-mortem had identified but
never tested.

| paired, same 60 sessions | Score | HitRate | MRR | MTTC |
|---|---|---|---|---|
| retrieval only | 0.698 | 0.833 | 0.572 | 5.52 |
| **+ listwise rerank** | **0.762** | 0.817 | **0.773** | 4.90 |

**+0.064**, against a pre-registered acceptance bar of +0.020. 23 sessions
improved, 8 regressed, 29 unchanged. Rank-1 share of hits went from 28/50 to
**45/49** — the mechanism did exactly what it was built to do, and MRR gained
0.201, more than the +0.134 headroom the decomposition predicted, because
promoting the right product also pulled MTTC down by 0.6 turns.

### Why this one worked and the cross-encoder did not

Same shortlist, same task, same model family. Two differences, both diagnosed
from the cross-encoder's failure before this was built:

* **It reads the shopper's own words.** The cross-encoder was handed
  `frame.describe()` — "flat sandals; stretchy fabric straps; yoga mat footbed" —
  which is neither a natural-language query nor anything its training resembles.
  This gets the raw utterances, verbatim, in order.
* **Bounded authority.** The cross-encoder blended a score into every candidate,
  overruling domain evidence it could not see. This one is consulted only when
  retrieval has no clear leader (`LISTWISE_MARGIN`), and it promotes rather than
  rescores. When a verbatim quote has already identified the product, the call is
  skipped entirely — no tokens, no latency, no risk.

A wiring flaw was caught during the build and is worth recording: the reranker
originally ran on the truncated slate, so it never fired on the turns where the
agent shows a single product — exactly the turns where picking the right one
matters most. Reordering the top ten *before* truncation took it from 1 call to 9
across three smoke-test sessions.

### Caveats

* **Hit rate dipped 0.017** (one session). The reranker occasionally promotes a
  wrong candidate to rank 1, which costs a turn when the slate is width 1.
* Scenario splits at n=10 are not reportable, but `use_case_led` (-0.072) and
  `brand_led` (-0.070) moved against the trend while `mind_changer` (+0.222),
  `decisive` (+0.171) and `vague` (+0.156) moved with it. `use_case_led` was
  already the weakest cell at reliable sample size, so it is worth watching.
* This is the network-dependent configuration. The offline submission path does
  not include it, and the scripted score reported for submission is the offline
  one.

### The width-1 disagreement hedge

The listwise reranker's one regression was hit rate: promoting a wrong candidate
into a single-slot slate hides retrieval's leader entirely, and a wrong promotion
costs a whole turn. The fix is to show both whenever the reranker overrules the
leader and the slate would otherwise be width 1.

| paired, same 60 sessions | Score | HitRate | MRR | MTTC |
|---|---|---|---|---|
| listwise, no hedge | 0.762 | 0.817 | 0.773 | 4.90 |
| **+ hedge** | **0.800** | **0.867** | 0.797 | 4.62 |

**+0.038.** The hit-rate regression is not merely recovered but reversed — 52 hits
against 49 — and MRR and MTTC both improved as well. Same rank-versus-turn
arithmetic that justified withholding in the first place, applied to disagreement
rather than to uncertainty: costing one product rank 2 is far cheaper than losing
a turn.

Cumulative on this session set: retrieval only 0.698 -> **0.800**.

### Confirmed at n=120

| paired, 120 stratified sessions | Score | HitRate | MRR | MTTC |
|---|---|---|---|---|
| retrieval only | 0.705 | 0.866 | 0.553 | 5.66 |
| + listwise | 0.748 | 0.807 | 0.772 | 5.36 |
| **+ listwise + hedge** | **0.790** | 0.858 | 0.796 | 4.88 |

**+0.085 over retrieval alone.** The n=60 reading of 0.800 was mildly optimistic;
0.790 is the reportable figure. The hedge held at +0.042 and recovers almost all
of the hit-rate cost listwise introduced (0.807 back to 0.858, against retrieval's
0.866).

**90 of 103 hits land at rank 1**, up from 28 of 50 before listwise.

No scenario regressed, and the weakest cells gained most:

| scenario (n=20) | retrieval | final | delta |
|---|---|---|---|
| use_case_led | 0.541 | 0.700 | +0.159 |
| vague | 0.723 | 0.824 | +0.101 |
| mind_changer | 0.592 | 0.680 | +0.088 |
| indifferent | 0.694 | 0.762 | +0.068 |
| brand_led | 0.905 | 0.964 | +0.059 |
| decisive | 0.770 | 0.812 | +0.042 |

By writing style, the spread has narrowed to 0.756-0.839, with `terse` now the
strongest at 0.839 — the cell an early five-session sample had labelled the
biggest weakness at 0.310.

## Sizing the next lever before building it

With ranking largely solved, the binding constraint moved to misses. Measured
before building anything (`tools/miss_bands.py`), over the 22 misses in the n=120
listwise run, by where retrieval's best position for the target was:

```
  1-10             3  (13.6%)   already visible: slate/turn budget, not retrieval
  11-30            6  (27.3%)   <- reachable by widening the reranker's window
  31-100           6  (27.3%)
  101-300          3  (13.6%)
  >300 / absent    4  (18.2%)
```

Widening the reranker to select from thirty candidates can reach **6 of 22
misses**, a ceiling of +0.026 on the hit-rate term assuming perfect selection.
Thirteen of the 22 sit at rank 31 or worse and four never enter the top 300 at
all; no reranker window reaches those, because retrieval never gets close.

That reframes "the misses are worth +0.15" — most of that headroom is not
addressable by reranking at any depth. It also argued for building the wider
variant as an *extension* of the working prompt rather than a replacement: the
model names its best pick and runner-up exactly as before, and additionally
nominates a shortlist, so a malformed or missing shortlist degrades to the
mechanism that already earned +0.064 instead of trading it away.

### Rejected: select-from-30

Built as an extension of the working prompt — the model still names `best` and
`runner_up` exactly as before, and additionally nominates a shortlist — so that a
malformed or missing shortlist degrades to the mechanism that already earned
+0.064 rather than replacing it.

| paired, same 60 sessions | Score | HitRate | MRR | MTTC |
|---|---|---|---|---|
| listwise + hedge | **0.800** | 0.867 | 0.797 | 4.62 |
| + select-from-30 | 0.780 | 0.850 | 0.788 | 5.08 |

**-0.020**, against a pre-registered +0.020 bar. Seven sessions improved, seven
regressed, 46 unchanged: the deeper window found about as many new answers as it
disturbed. MTTC rose by 0.46 turns, which says the larger candidate block made the
model slower to commit.

The six reachable misses the band analysis predicted did not convert into wins.
Rank-1 accuracy held (45/51 against 44/52), so the mechanism itself is intact —
the wider window simply is not worth its dilution. Off by default;
`AGENT_LISTWISE_WIDE=1` reproduces it.

This is the value of having sized the lever first: the ceiling was +0.026 assuming
perfect selection, which is thin enough that a modest accuracy cost erases it. The
band analysis predicted a marginal result and got one.

## Rejected: turn-1 ranking prior

Sessions converting on turn 2 could only convert on turn 1 if the target were
already ranked first with turn-1 information alone. Measured rather than assumed
(`tools/turn1_headroom.py`):

```
141 sessions convert at turn 2 or later
   66 had the target at rank 1-4 at turn 1
   24 of those are intent-override — structurally locked out of early conversion
   -> 42 addressable

perfect turn-1 promotion: MTTC 2.145 -> 1.930, composite gain +0.0042
```

+0.0042 is the ceiling assuming a flawless prior, and the turn-1 ranks are
scattered — only a third sit in the 2-4 band where promotion is plausible. Not
worth a day against the listwise lever's +0.064. Dropped before implementation,
which is the diagnostic paying for itself.

## The open probe dominates every named question

The simulator's `customer_reply` matches a named attribute only against
requirements whose `classify_constraint` maps to that attribute, but matches
`other` against **any** requirement not yet disclosed. Two consequences follow,
and both were confirmed empirically with `tools/ask_audit.py` over the 200 public
sessions.

1. **`other` extracts at least as much as any named question, on every turn.**
   Measured: 415 open probes, every one either productive or interrupted by an
   override. Zero wasted.
2. **Once `other` returns nothing, nothing remains.** An empty answer to `other`
   proves there is no undisclosed requirement of any type, so every named
   question after it is provably worthless. Measured: 14 named asks, 14 wasted.

The audit also confirms two guards: the agent never emits `ask_attribute: null`
while requirements are outstanding (0 occurrences), and boundary sessions re-ask
after the one-time stonewall (6 stonewalls followed by 7 productive asks).

### Acting on point 2 makes the score worse

The obvious inference — commit to a full shortlist the moment `other` is spent,
since no further question can teach anything — was implemented and measured at
**0.9709 against 0.9724**. MTTC improved as predicted (2.145 to 2.115) and MRR
fell further (0.984 to 0.977).

Those turns are worthless for *information* but productive for *elimination*.
Showing a single product rules it out under the already-shown rule, so each extra
probing turn clears one wrong candidate from the shortlist the agent eventually
commits to. `public_0083` shows it directly: three post-exhaustion turns, then a
hit at rank 4 — without them the same slate would have been dirtier.

The named ladder is therefore kept, for elimination rather than for information.

## Negative result: candidate-grounded contrastive questions

The third and last attempt at the questioning lever, and the most informative.

Rather than choosing an attribute from a fixed taxonomy, the turn-planner call
that already reads the raw utterances and the shortlist was extended to also
write the question: one short question about the concrete property separating the
candidates still in contention, in plain shopper language, returning an empty
string when one candidate already matches everything stated. Marginal token cost,
since it rides an existing call. The model gates itself.

The questions it produces are genuinely good:

```
"Are you looking for a men's or women's style?"                      [style]
"Would you prefer the band to be a stretchy metal expansion band
 like your uncle's, or something else?"                             [feature]
"Would you prefer a gold-tone finish or a silver-tone metal band?"   [color]
```

| paired, same 60 sessions | Score | HitRate | MRR | MTTC |
|---|---|---|---|---|
| fixed open probe | **0.800** | 0.867 | 0.797 | 4.62 |
| + generated question | 0.745 | 0.817 | 0.734 | 5.17 |

**-0.055.** But the mechanism check tells a different story from the score:

```
previously-missed sessions that now convert:  6 of 8
previously-hit sessions now lost:             9
```

The mechanism *works* — targeted questions converted six of the eight sessions
the fixed probe missed, which is precisely what they were built for. They simply
cost more elsewhere than they earn there.

**Why.** The open probe extracts up to two requirements per turn regardless of
topic, because it matches anything undisclosed. A generated question is
necessarily narrow, so when its answer does not discriminate, that turn bought
less than the open probe would have. MTTC confirms it: 4.62 to 5.17, half a turn
slower. The questions are better per unit of information and worse per turn, and
turns are the scarce resource.

The scenario split says the same thing: `brand_led` +0.162 and `decisive` +0.044,
where customers have specific answers ready; `use_case_led` -0.281 and `vague`
-0.086, where a customer thinking in occasions cannot usefully answer "gold or
silver", and a wasted narrow question is worse than a wasted broad one.

### Negative result: the rescue gate

If a narrow question loses because it extracts less per turn than the open probe,
the natural repair is to stop asking it every turn and reserve it for sessions
already failing. That was built and measured.

The gate fires only when the session looks like a miss in progress: turn 4 or
later, retrieval has no clear leader, and the agent's confidence has stopped
improving. The question is **appended** to the open probe rather than replacing
it — "Anything else that matters — for instance, would you prefer a stretchable
metal band?" — so the turn keeps the probe's breadth and gains a discriminator.
`ask_attribute` stays `other` throughout, which makes the scripted risk zero by
construction, since the simulator reads only that field.

Selectivity worked as designed: the question fired on 3 of 61 turns, 5%, all
late and stalled.

| paired, same 59 sessions | Score | HitRate | MRR | MTTC |
|---|---|---|---|---|
| open probe only | **0.799** | 0.864 | 0.794 | 4.56 |
| question every turn | 0.758 | 0.831 | 0.747 | 5.07 |
| rescue-gated question | 0.738 | 0.797 | 0.747 | 5.20 |

**-0.061 — worse than asking every turn.** The mechanism check says why, and it
inverts the hypothesis:

```
rescue candidates (missed by open probe, converted by asking every turn):  6
  still converted under the gate:                                          2
casualties (hit by open probe, broken by asking every turn):               8
  protected by the gate:                                                   3
hits: open probe 51    every turn 49    gated 47
```

The gate kept **2 of 6** rescues while blocking only **3 of 8** casualties. It
filtered out more benefit than harm.

**The gating signal is anti-correlated with where questions help.** Confidence
stalling means the agent has no clear leader — which is also the state where a
narrow question is *least* likely to land, because the agent does not know enough
to know what to ask about. Questions pay off where the agent has a nearly-correct
picture and needs one discriminator, and those sessions do not look stalled.

There is also a measurement caveat worth stating: the intended gate was "the open
probe has already run dry", which cannot be observed against a language-model
customer. Constraint counts grow on every turn regardless of whether the turn
taught anything, because the customer always says something and the extractor
always converts it into a requirement. `dry_turns` was 0 across all 61 turns.
Confidence stalling was substituted as the nearest observable proxy, so this
result rejects the gate that could be built, not precisely the gate that was
designed.

### The two things called "questioning" can be decoupled

`customer_reply(sample, ask_attribute, disclosed, boundary_used)` does not take
the message text. The scripted customer never reads the question — only the
attribute string reaches it. So the scored decision and the visible one are
independent:

* **Scoreable questioning** is the choice of `ask_attribute`, and it is solved by
  a constant: `other` dominates every named attribute (derivation above).
* **Visible questioning** is the sentence a person reads. It cannot affect the
  score at all, and it is the only part of the system a demo viewer experiences.

Every experiment that lost — the entropy planner and this one — changed both
fields together, paying a scoring cost for a presentational gain. Emitting a
contrastive question in `message` while holding `ask_attribute` at `other` has
not been tested, and by construction cannot cost anything on the scripted set.

### Four attempts, four causes

| attempt | result | diagnosed cause |
|---|---|---|
| entropy planner over attribute types | -0.011 | optimised splits over a pool that was drifting |
| contrastive question, every turn | -0.055 | narrow question extracts less per turn than the open probe |
| contrastive question, rescue-gated | -0.061 | gating signal anti-correlated with where questions help |
| commit once the probe is exhausted | -0.0016 | those turns do elimination work, not information work |

The open probe survives all four, and for a derived reason rather than an
empirical one: it dominates every named attribute by construction, and on the
scored benchmark the question text is never read at all. This is the component
the organiser's own material calls central — "a better question can be more
valuable than another retrieval call" — and the honest finding is that against
this evaluator the optimal policy is a constant.

**Repository state:** the rescue gate is currently left enabled rather than
reverted, pending a decision. The frozen scripted configuration is unaffected
either way, since the gate requires a model client that the submission path does
not attach.

## The last scripted points are irreducible ties

Five of 200 sessions convert below rank 1: four intent-override, one buying.
Reading them individually:

```
public_0002  target 53.93 vs rank-1 54.35  (gap 0.42)   two leather belts, buckle closure
public_0144  top three 65.70 / 65.63 / 65.54 (spread 0.16)  three polyester down jackets, zipper
```

The intent card discloses only `leather`, `100% Leather`, `Buckle closure`,
`Imported` — requirements dozens of catalog products satisfy identically. These
are genuine ties, not ranking errors, and what separates them is the popularity
prior.

An initial hypothesis — that demoting pre-override constraints by 0.6 was
penalising the target's own earlier evidence — was tested by removing the
discount entirely. It is a **no-op**: identical overall score, identical
per-scenario metrics. The demotion was never the cause.

### The tie-break weights are already optimal

Swept with the halves of the public set held out from each other
(`tools/tiebreak_sweep.py`):

| config | split A | split B | both |
|---|---|---|---|
| **popularity 0.02 (current)** | **0.9748** | **0.9700** | **0.9724** |
| popularity 0.00 | 0.9576 | 0.9566 | 0.9571 |
| popularity 0.01 | 0.9693 | 0.9659 | 0.9676 |
| popularity 0.04 | 0.9744 | 0.9687 | 0.9716 |
| popularity 0.08 | 0.9703 | 0.9608 | 0.9656 |
| popularity 0.02, profile 0.5 | 0.9691 | 0.9692 | 0.9692 |

The current setting wins on both splits independently, so it is not a
split-specific artifact. Removing the popularity prior costs **0.015** — what is
written as a 0.02 tie-break term is one of the larger single contributors in the
system, which follows from targets being real purchase records.

## Holdout validation

Scoring weights were set by hand while watching all 200 public sessions — exactly
the setup where a number can look good because it memorised its sample. Splitting
the set and scoring each half separately (`tools/holdout.py`):

| split | n | score | HitRate | MRR | MTTC |
|---|---|---|---|---|---|
| A | 100 | 0.9711 | 1.000 | 0.983 | 2.18 |
| B | 100 | 0.9746 | 1.000 | 0.986 | 2.06 |

Gap of 0.0035, hit rate 1.000 on both halves. This is a stability check rather
than a full tune-and-validate — a stricter version would re-fit the weights on A
alone — but with hit rate saturated and MRR within 0.003 on both halves there is
little room for sample-specific tuning to be hiding.

## The three tiers, and what each costs

Audited what a reviewer actually gets from a bare clone, because the agent's
optional components degrade silently and it would be easy to report a number
nobody else can reproduce.

| scripted, 200 sessions | Score | Requires |
|---|---|---|
| **tier 1 — lexical only** | **0.9746** | nothing: Python standard library |
| tier 2 — + dense retrieval | 0.9724 | numpy, onnxruntime, tokenizers; 73 MB index, ~25 min to build |
| tier 3 — + model extractor and listwise | 0.8740 | network and a Vertex credential |

And on the LLM customer, the ordering inverts:

| LLM customer | Score |
|---|---|
| tier 1 — lexical only | ~0.60 |
| tier 2 — + dense | 0.714 |
| tier 3 — + listwise and hedge | **0.800** (n=60) |

Two things follow.

**Dense is documented but not installed by default — revised after the webinar.**
The original rationale was insurance: 0.002 on the scripted customer against
roughly +0.16 on a paraphrasing one, a 0.2% premium for a 16-point hedge. The
organiser's statement that *final results are generated exactly as the published
local evaluator* removes the risk being hedged, so the premium no longer buys
anything on the graded run.

The graded configuration is therefore **tier 1**, at 0.9746. Mechanically this is
just `requirements.txt`: it lists no third-party package, so a grader following
the setup instructions runs the lexical path, which the agent selects on its own
when numpy and onnxruntime are absent. The dense track stays fully implemented
and documented behind `requirements-optional.txt` — it is what carries the
LLM-customer result, and removing it would delete the evidence for the robustness
argument.

**The tiers are genuinely additive, not required.** A reviewer who clones the
repository and runs the evaluator immediately gets a working agent at 0.9746,
because `attach_dense()` and the extractor both degrade silently when their
dependencies are absent. Nothing in the default path needs a network, a
credential, or a model file. The README must state all three numbers rather than
only the best one, since which tier a grader exercises depends on what they
install.

## Environment

- Installing the gcloud CLI pulled in Homebrew Python 3.14, which shadows the
  python.org 3.12 the project started on. Everything runs on 3.14.7.
- Dense dependencies (numpy, onnxruntime, tokenizers) live in `.venv/`, so the
  system interpreter is untouched. Runs needing the dense track use
  `.venv/bin/python`; everything else runs on either.
- The simulator authenticates to Vertex with an OAuth token minted on demand by
  gcloud. No API key is stored in the repository.

## Remaining work

Scoring is frozen. What is left is packaging and evidence.

- Submission bundle per `docs/submission_rules.md`: entry file exporting `Agent`,
  `requirements.txt`, README with setup, reproduction steps and limitations.
- Written report: architecture, the two-benchmark method, the negative-result
  table, feasibility numbers, and the tier disclosure.
- Demo video: one scripted session and one language-model-customer session side
  by side, since the contrast is the argument.
- Devpost description with tools, APIs, libraries and datasets.

Closed and not to be reopened: the turn-1 ranking prior (+0.0042 ceiling,
measured), the paraphrased-scripted benchmark (the webinar confirmed the private
sessions use the published evaluator), and all four questioning experiments.
