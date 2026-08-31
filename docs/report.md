# Shopping Copilot: a conversational search agent, and the benchmark built to falsify it

TechJam 2026 — Track 4, AI Conversational Search and Recommendations

---

## Summary

A multi-turn agent that locates one hidden product in a frozen 50,000-item Amazon
catalog within ten turns.

| | TechnicalScore | HitRate@10 | MRR | MTTC |
|---|---|---|---|---|
| organiser weak-BM25 reference | 0.1067 | 0.125 | 0.068 | 9.81 |
| **submitted agent** | **0.9746** | **1.000** | **0.988** | **2.095** |

The submitted configuration runs on the Python standard library alone: no
third-party package, no model file, no network call, no credential, zero tokens,
about half a millisecond per turn. It was verified on a fresh clone with a stock
interpreter, credentials removed and network unavailable.

The more interesting result is not that number. It is what happened when a second
benchmark was built to test whether that number meant anything: it largely did
not, and the remaining effort went into what that benchmark revealed.

---

## 1. What the task actually is

The organiser publishes the evaluator, so it was read before any code was written.

The simulated customer is not a language model. `intent_card()` derives
everything it will ever say from the target product's own metadata: the first
material word appearing in its text, the first colour, verbatim bullets from its
`features` and `details`, and its price. The customer then quotes those strings.
`customer_reply()` takes the agent's `ask_attribute` and returns up to two
requirements not yet disclosed — and it never receives the agent's prose at all.

Three properties follow, each measured rather than assumed.

**Category routing is unusually powerful.** The opening turn names the two most
specific levels of the target's category path. There are 1,115 such nodes across
50,000 products, with a **median bucket size of 8**. For half of all sessions,
turn one alone narrows the catalog by three orders of magnitude.

**Rank is worth roughly six times a turn.** Under `0.50·HitRate + 0.30·MRR +
0.20·Efficiency`, moving one session from rank 5 to rank 1 gains 0.0012 of final
score; the two extra turns that costs lose 0.0002. The reference baseline's
instinct — pad the shortlist and convert early — is therefore backwards, and the
agent withholds instead.

**The open clarification probe dominates every named attribute.** `other` matches
any undisclosed requirement; a named attribute matches only requirements its
classifier maps to that type. So `other` extracts at least as much on every turn,
and an empty answer to it proves nothing remains. Audited across 200 sessions:
415 open probes, all productive or structurally interrupted; 14 named asks, all
14 returning nothing.

This analysis is disclosed openly. Identifying a simulated environment's dynamics
is ordinary practice in any benchmark of this kind, and it is discoverable from
the code participants are given; describing it vaguely would be worse on every
axis.

## 2. Architecture

```
customer turn
     │
     ▼
  parsing ──────────► frame ◄────────── extraction (optional model tier)
  templates +        ├ type      one slot, sticky, replaced only on contradiction
  generic fallback   ├ attributes  material · colour · budget · quoted phrases
                     └ negatives   what the shopper has ruled out
                          │
              ┌──── candidate generation ────┐
              │  category route (stated)     │
              │  vocabulary route (mined)    │
              │  lexical route (rare terms)  │
              │  dense route (optional)      │
              └──────────────┬───────────────┘
                             ▼
        scoring · quote match · category · negatives · popularity prior
                             ▼
             listwise rerank (optional, gated on near-ties)
                             ▼
        slate sizing ──► up to 10 parent_asin + one clarification
```

**The frame is the design decision that matters.** An earlier version represented
the conversation as a monotonically growing bag of words and searched over the
accumulation. Every failure mode traced to that: drift, because common vocabulary
piles up until the largest category wins on volume; echo, because the agent's own
question words re-entered the query through the customer's answers; and no way to
express negation at all. Because product type is now a slot that must be
*contradicted* rather than a tally that can be outvoted, drift is structurally
impossible rather than something to be tuned against.

**Vocabulary is mined from the catalog, not authored.** Shoppers say "pants"; the
catalog says "Sport Specific Clothing Basketball". Rather than hand-writing a
thesaurus — fashion vocabulary is long-tailed and any list would have holes
exactly where the catalog is dense — a term-to-category index and co-occurrence
clusters are built from 50,000 real product titles at startup.

**The popularity prior is load-bearing, not a tie-break.** Targets are real
purchase records: the public targets have a median review count of **7,078**
against the catalog's **12**. Removing the prior costs **0.015** of final score,
making it one of the larger single contributors in the system despite its small
weight.

**Slate sizing is a bet, deliberately made.** While the customer still has
something to disclose, the agent shows only what it can defend — often a single
product — and asks. It commits to a full shortlist once the requirement set is
drained. This converts later but at a better rank, which the six-to-one
arithmetic favours.

## 3. Mapping to the four pillars

The problem statement specifies four pillars. Each maps to a component, and each
claim below has a measurement behind it rather than an assertion.

| pillar | component | evidence |
|---|---|---|
| **I. Intent routing and hybrid pipeline** | Four retrieval routes — stated category, catalog-mined vocabulary, rare-term lexical, and dense embeddings — fused into one reranker, with an LLM semantic ranking tier above it | dense worth +0.16 on the LLM customer; listwise +0.085; route weights swept with both holdout halves |
| **II. Dialog strategy** | The frame: a sticky product-type slot, accumulating attributes, subtracting negatives. Intent override demotes superseded statements rather than erasing them | drift eliminated — bucket collapse went from 17 of 24 sessions to zero; override scenario scores HR 1.000, MRR 0.950 |
| **III. Self-evolution** | The frame *is* the distilled context; the raw transcript is never searched after the drift finding. Slate width adapts to confidence | withholding validated by the six-to-one rank-versus-turn arithmetic; adaptive widening tested and rejected at −0.034 |
| **IV. Evaluation** | Two benchmarks reported side by side, holdout validation on both halves, nine pre-registered experiments | 0.9746 scripted / 0.348 LLM customer for the same configuration; splits 0.9748 and 0.9700 |

Pillar II's "proactive guidance" deserves a direct answer, because it is the one
place this submission does something counter-intuitive. The scripted simulator
acts only on `ask_attribute` and never reads the question text, and the open
probe `other` matches any undisclosed requirement while a named attribute matches
only its own type. The optimal questioning policy against this evaluator is
therefore a constant, and four attempts at something more adaptive all scored
worse (§6). That is reported here rather than shipping a more impressive-looking
policy that measurably loses.

## 4. Method: two benchmarks

The organiser's evaluator determines the official score. But a customer that
quotes catalog text can be satisfied by matching wording, so a high score there
does not demonstrate understanding.

Hence an **adversarial second benchmark**: the same target products,
the same metric formulas, the same ten-turn limit, but the customer is played by
Gemini 3.5 Flash Lite and instructed never to quote the listing. Six behavioural
scenarios — including two the scripted simulator cannot express, a shopper who
describes the occasion rather than the product and one who leads with a brand —
crossed with five writing styles, assigned by cycling the 6×5 grid so that 120
sessions give exactly four per cell.

Responses are cached by request, which makes the benchmark reproducible and makes
repeated runs pay only for turns whose conversation actually diverged.

The first run answered the question it was built for:

| customer | TechnicalScore | HitRate | MRR |
|---|---|---|---|
| scripted (organiser) | 0.975 | 1.000 | 0.985 |
| language-model | 0.589 | 0.833 | 0.229 |

Hit rate largely held — category routing is genuine product behaviour — but
**MRR collapsed from 0.985 to 0.229**. Most of the score was exact wording.

It also found a defect the official evaluator could not see. Across 24 sessions,
**17 ended inside "Shirts T-Shirts"**, the catalog's largest bucket, while the
customer typed *"I do not want shirts at all, I want shoes!"*. That drove the
frame rebuild.

## 5. Results

| tier | scripted (200) | LLM customer (120) | requires |
|---|---|---|---|
| **1 — lexical, submitted** | **0.9746** | **0.348** | nothing |
| 2 — + dense retrieval | 0.9724 | 0.7054 | numpy, onnxruntime, tokenizers |
| 3 — + model extraction and listwise rerank | 0.8740 | **0.7903** | network, credential |

**The tiers move in opposite directions, and that is the central finding.** The
submitted configuration scores **0.9746 against a customer who quotes product
text and 0.348 against one who paraphrases** — the same agent, the same targets,
the same metric. Against the quoting customer, exact-match machinery is
near-optimal and every model-based layer can only overrule an answer that was
already right. Against the paraphrasing one, those layers are the difference
between 0.348 and 0.790.

All three tiers are measured over the same stratified sessions — 120, 119 and
120 after aborted sessions are excluded — so the column is directly comparable
row to row.

Two components carry the generalisation result:

**Dense retrieval** — `bge-small-en-v1.5`, 33M parameters, through ONNX Runtime on
CPU. The catalog is embedded once into a 50,000 × 384 matrix (73 MB) and
retrieval is a single matrix-vector product. Entirely in memory, no vector
database. Worth **+0.16** on the language-model customer for **−0.002** on the
scripted one. It closes exactly the gap it was built for: "soft bottom like yoga
mat" reaching a product whose description says "yoga mat sole".

**Listwise reranking** — one model call per turn that reads the shopper's raw
words alongside the shortlist and promotes its pick, but only when retrieval has
no clear leader, paired with a hedge that shows both candidates when the model
overrules a confident retrieval. Worth **+0.085**, taking rank-1 placement from
28 of 50 hits to **90 of 103**.

**Holdout validation.** With each half of the public set held out from the other,
the submitted configuration scores 0.9748 and 0.9700 — the score is not riding
noise in a particular sample.

## 6. What was falsified

The second benchmark's real value was killing plausible ideas. Each of
the following was measured against a pre-registered acceptance bar and rejected
with a diagnosed cause. All remain in the repository behind environment flags.

**Clarification questioning — four attempts:**

| attempt | result | diagnosed cause |
|---|---|---|
| entropy planner over attribute types | −0.011 | optimised splits over a drifting candidate pool |
| contrastive model-written questions, every turn | −0.055 | a narrow question extracts less per turn than an open one |
| the same, rescue-gated to failing sessions | −0.061 | the gating signal was anti-correlated with where questions help |
| commit once the open probe is spent | −0.0016 | those turns do elimination work, not information work |

The last is the most instructive. The dominance argument says a named question
after an exhausted open probe cannot teach anything — which is true, and acting
on it still scored worse, because showing one product per turn *eliminates* it
under the already-shown rule, so the shortlist the agent eventually commits to
has had its wrong candidates cleared.

**Ranking — three attempts:** a cross-encoder reranker at −0.047, then −0.015
after correcting a blend that stretched the model's output across the full weight
range regardless of its confidence; and adaptive slate widening at −0.034. The
cross-encoder post-mortem is what made listwise reranking succeed: it had been
handed a frame description instead of natural language, and given free rein
instead of bounded authority. Same shortlist, same model family, opposite result.

**Retrieval — two attempts:** widening the reranker to select ten from thirty
(−0.020), and re-weighting category against dense evidence (no configuration beat
the current one on both holdout halves).

## 7. Measured ceilings

Work stopped when the ceiling could be demonstrated, not when ideas ran out.

**Category resolution is the dominant lever and is largely unreachable.** An
oracle test — handing the agent the target's true category — takes rank-1
placement from **34.5% to 61.3%**, improving 58 of 119 sessions. But the correct
node is inferable from the customer's request only about half the time, because
the catalog files a crew-neck t-shirt under *Underwear Undershirts* and mules
under *Loafers* when a *Mules & Clogs* node exists. That is arbitrary filing, not
a retrieval failure, and no method recovers most of that 58-session prize.

**The remaining scripted points are genuine ties.** Five of 200 sessions convert
below rank 1. In `public_0002` the target scores 53.93 against 54.35 for another
leather belt with a buckle closure; in `public_0144` three polyester down jackets
sit within 0.16 points of each other. The disclosed requirements — `leather`,
`100% Leather`, `Buckle closure`, `Imported` — are satisfied identically by
dozens of catalog products. The practical ceiling is about **0.992**, the
remainder being the intent-override lockout, which prevents any session in that
scenario from converting before turn three.

## 8. Feasibility, cost and latency

**Submitted configuration:**

| | |
|---|---|
| tokens | 0 |
| network calls | 0 |
| credential | none |
| latency per turn | ~0.5 ms median, 0.7 ms p95 |
| startup | ~6.5 s one-off catalog indexing |
| peak memory | ~840 MB |
| full 200-session evaluation | ~20 s |

**Optional tiers:** dense retrieval adds ~11 ms per turn and a 73 MB cached
index built once in about 25 minutes. The model tier adds roughly one second per
turn, dominated by Vertex round-trip latency (930 ms median observed).

**Development cost:** Gemini 3.5 Flash Lite via Vertex AI, used only for the
customer simulator and the optional tiers. A 120-session benchmark run costs
roughly **1.06M input and 35k output tokens**; caching means repeat runs pay only
for diverged turns. No credential is stored in the repository — the client mints
an OAuth token through the gcloud CLI on demand.

## 9. Reproducibility

```bash
python3 -m evaluator.local_evaluator
```

Python 3.10 or newer, nothing installed. Verified from a fresh clone on stock
Python 3.12 with credentials stripped and a minimal PATH, producing 0.9746 and
zero tokens.

Optional tiers degrade rather than fail: setting the model-tier environment
variables with no credential present emits a warning and runs the offline path,
and a full 200-session run under those conditions completes at the offline score.
Every number in this report maps to a command, listed in the submission README.

## 10. Limitations, and what more time would buy

**The language-model benchmark is not a leaderboard predictor.** The organiser
confirmed that final results are generated exactly as the published local
evaluator, so it measures robustness to a customer the official evaluation does
not use. It is reported because it is what exposed this agent's dependence on
exact wording, and what falsified nine subsequent ideas that looked sound.

**Question wording cannot help under this evaluator**, because the simulator
never reads it. That is a property of the benchmark rather than of conversational
search, and it is the largest gap between what this task rewards and what a real
shopping assistant needs.

**The language-model benchmark uses one customer model.** Every session is played
by Gemini 3.5 Flash Lite. A different model would paraphrase differently, and the
absolute numbers would move; the ordering between tiers is what the design
supports, not the exact values.

**With more time**, in order of expected value: run the benchmark with a second
customer model to separate the agent's behaviour from one simulator's idiom; port
listwise reranking to a local quantised model so the +0.085 survives without a
network, completing the three-tier story with no external dependency; and index
the demand side — precompute plausible customer phrasings per product, a
generalisable form of the observation that made the exact-quote matcher work,
which should help the terse and occasion-led customers who share little
vocabulary with catalog text.

## 11. Team contributions

Solo entry. All design, implementation, experimentation and analysis by the
submitting participant.

---

*Full experimental record, including every negative result with its diagnosis, is
in `NOTES.md`.*
