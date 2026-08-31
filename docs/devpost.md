# Shopping Copilot — Devpost description

## What this is

A multi-turn shopping agent that finds one hidden product in a frozen 50,000-item
Amazon catalog within ten turns — and a second, adversarial benchmark built to
find out whether it actually understands anything.

The agent scores **0.9746** on the official evaluator against a weak-BM25
reference of **0.1067**, running on the Python standard library alone: no
third-party package, no model file, no network call, no credential, zero tokens.

---

## 1. The discovery: the customer is a program, and it quotes

The organiser ships the evaluator, so it came first, before any code. The
simulated customer is not a language model. `intent_card()` derives everything it
will ever say from the target product's own metadata — the first material word in
its text, the first colour, verbatim bullets from its `features` and `details`,
its price. The customer then quotes those strings back.

Two structural consequences followed, both measured:

- **The opening turn names the target's category**, and there are 1,115 category
  nodes across 50,000 products with a median bucket size of **8**. Turn one alone
  narrows the catalog by three orders of magnitude for half the sessions.
- **Rank is worth about six times a turn** under `0.50·HitRate + 0.30·MRR +
  0.20·Efficiency`. Moving one session from rank 5 to rank 1 gains 0.0012; the
  two extra turns it costs lose 0.0002. The baseline's instinct — pad the
  shortlist, convert early — is backwards.

Hence an interpretable offline scorer: category routing, typed constraint
slots, exact-quote matching against normalised catalog text, a popularity prior
(targets are real purchase records, and removing that prior costs 0.015), and a
confidence-gated slate that withholds rather than padding. That reached 0.9746
against a practical ceiling of ~0.992, the remainder being the intent-override
lockout and genuine ties.

**This is disclosed openly.** Identifying the environment's dynamics is standard
practice in any simulated benchmark, and the alternative — describing it vaguely
and hoping nobody reads the code — would be worse on every axis.

## 2. The honest question: does 0.975 measure understanding?

It measures wording. Hence the thing that would settle it: **an LLM-customer
benchmark**. Same target products, same metric formulas, same ten-turn limit —
but the customer is played by Gemini 3.5 Flash Lite, instructed never to quote
the listing and to describe the product in its own words, across six behavioural
scenarios and five writing styles, 120 stratified sessions with four per cell.

The first run answered the question:

| customer | TechnicalScore | MRR |
|---|---|---|
| scripted (organiser) | 0.975 | 0.985 |
| language-model | 0.589 | 0.229 |

Hit rate held — category routing is real product behaviour — but **MRR collapsed
from 0.985 to 0.229**. Most of the score was exact wording, exactly as suspected.

The benchmark then found a bug the official one could not see: across 24 sessions,
**17 collapsed into "Shirts T-Shirts"**, the catalog's largest bucket, while the
customer typed "I do not want shirts at all, I want shoes!". Representing a
conversation as a growing bag of words let common vocabulary accumulate until the
biggest category won on volume.

That drove the rebuild: a **structured frame** with a sticky product-type slot, a
set of accumulating attributes, and a set of negatives. Because type is a slot
that must be *contradicted* rather than a tally that can be outvoted, drift became
structurally impossible. Bucket collapse went from 17 of 24 to zero.

## 3. The four pillars, and the generalization tiers

**I — Intent routing and hybrid pipeline.** Four retrieval routes fused into one
reranker: the stated category, a vocabulary route from a lexicon mined out of
50,000 product titles (so "pants" reaches the pants categories the catalog calls
something else), a rare-term lexical route, and a **dense route** — `bge-small-en-v1.5`
through ONNX Runtime, the catalog embedded once into a 50,000 × 384 matrix,
retrieval by a single matrix-vector product. Entirely in-memory, no vector
database. **Worth +0.16 on the language-model customer.**

**II — Dialog strategy.** A frame that accumulates requirements across turns and
subtracts what the customer rules out. Intent override demotes superseded
statements rather than erasing them, because in this simulator the discarded
preference and its replacement both derive from the same product.

**III — Self-evolution.** The frame is the distilled context: the raw transcript
is never searched again after the drift finding. Slate width adapts to
confidence — the agent shows one product when it is sure and widens when the
requirement set is drained.

**IV — Evaluation.** Two benchmarks, reported side by side, plus holdout
validation on both halves of the public set (0.9711 / 0.9746).

On top of that, a **listwise reranking tier**: one model call per turn that reads
the shopper's raw words and the shortlist and promotes its pick — but only when
retrieval has no clear leader, and paired with a hedge that shows both candidates
when the model overrules a confident retrieval. **Worth +0.085 on the
language-model customer**, taking rank-1 placement from 28/50 hits to 90/103.

The tiers move in opposite directions, and that is the finding. **The submitted
configuration scores 0.9746 against a customer who quotes product text and 0.348
against one who paraphrases** — the same agent, the same targets, the same
metric. Dense retrieval alone more than doubles the second number; listwise
reranking takes it to 0.790. The offline configuration is the one submitted, because
the organiser confirmed final results are generated exactly as the published
evaluator — but both columns are reported, because only one of them is about
understanding.

## 4. What was falsified

The benchmark's real value was killing good ideas. Every one below was plausible,
several came from the problem statement's own suggested directions, and all were
measured and rejected with a diagnosed cause.

**Four attempts at smarter clarification questions:**

| attempt | result | why |
|---|---|---|
| entropy planner over attribute types | −0.011 | optimised splits over a drifting candidate pool |
| contrastive LLM questions, every turn | −0.055 | a narrow question extracts less per turn than an open one |
| contrastive questions, rescue-gated | −0.061 | the gating signal was anti-correlated with where questions help |
| commit once the open probe is spent | −0.0016 | those turns do elimination work, not information work |

The open probe survives on a derived argument: it matches *any* undisclosed
requirement while a named attribute matches only its own type, so it dominates by
construction — and the simulator never reads the question text at all. Against
this evaluator, the optimal questioning policy is a constant. That is reported
here rather than shipping something that looks clever and scores worse.

**Three attempts at better ranking:** a cross-encoder reranker twice (−0.047,
then −0.015 after fixing the blend) and adaptive slate widening (−0.034). The
cross-encoder post-mortem is what made listwise reranking work: it was being fed
a frame description instead of natural language, and given free rein instead of
bounded authority. Same shortlist, same model family, opposite result.

## 5. Measured ceilings

Work stopped at the point the ceiling could be shown, not at the point ideas ran out.

**Category resolution is the dominant lever and is largely unreachable.** An
oracle test — handing the agent the target's true category — takes rank-1
placement from **34.5% to 61.3%**, improving 58 of 119 sessions. But the correct
node is inferable from the request only about half the time, because the catalog
files a crew-neck t-shirt under *Underwear Undershirts* and mules under *Loafers*
when a *Mules & Clogs* node exists. That is arbitrary filing, not a retrieval
failure.

**The last scripted points are genuine ties.** Five of 200 sessions convert below
rank 1. In `public_0002` the target scores 53.93 against 54.35 for another leather
belt with a buckle closure; in `public_0144` three polyester down jackets sit
within 0.16 points. The disclosed requirements — `leather`, `100% Leather`,
`Buckle closure`, `Imported` — are satisfied identically by dozens of products.

**Weight sweeps confirm a local optimum.** Category-versus-dense weighting, the
tie-break weights, and the shape of the popularity prior were each swept with both
halves of the public set held out. No configuration beat the current one on both.

---

## Built with

**Development tools:** Claude Code, VS Code, git, macOS terminal.

**APIs:** Google Vertex AI — Gemini 3.5 Flash Lite, used *only* for the
language-model customer simulator and the optional tier-3 layers. Authenticated
with an OAuth token minted on demand by the gcloud CLI; no key is stored in the
repository. The graded submission makes zero API calls.

**Libraries and frameworks:** Python 3.10+ standard library (`json`, `re`,
`sqlite3`, `dataclasses`, `math`, `urllib`) for the graded path. Optional tier:
`numpy`, `onnxruntime`, `tokenizers`. No training framework — nothing is trained.

**Models:** `BAAI/bge-small-en-v1.5` (33M parameters, 384 dimensions, ONNX, CPU)
for dense retrieval; `cross-encoder/ms-marco-MiniLM-L-6-v2` for the rejected
reranking experiment, retained behind a flag with its negative result recorded;
Gemini 3.5 Flash Lite for the customer simulator and optional tiers.

**Datasets and assets:** the organiser's frozen 50,000-product catalog and 200
public sessions, derived from Amazon Reviews 2023 (McAuley Lab, UCSD). No
external training data, no catalog modification, no attempt to reconstruct the
private target pool.

## Costs and efficiency

**Graded path:** 0 tokens, 0 network calls, no credential. ~0.5 ms per turn
median, 0.7 ms at p95, ~6.5 s one-off startup indexing, ~840 MB peak memory.
Verified on a fresh clone with stock Python 3.12, credentials stripped and
network unavailable.

**Development:** roughly 1.06M input and 35k output tokens per 120-session
benchmark run, cached by request so repeat runs pay only for diverged turns.
