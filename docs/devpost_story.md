## Inspiration

The challenge ships its own evaluator, so I read it before writing any code.

The "customer" turned out not to be a language model at all. It is a small
deterministic program, and `intent_card()` derives everything it will ever say
from the target product's own metadata — the first material word in its text, the
first colour, verbatim bullets from its `features` and `details`, its price. The
customer then quotes those strings back at you.

That reframed the whole task. A high score against a customer who quotes catalog
text does not demonstrate understanding; it demonstrates string matching. So the
project became two things instead of one: an agent that maxes the official
benchmark, and a second benchmark built to find out whether that score meant
anything.

## What it does

Finds one hidden product in a frozen 50,000-item Amazon catalog within ten turns
of conversation, scoring **0.9746** on the official evaluator against a
weak-BM25 reference of **0.1067**.

The submitted configuration runs on the Python standard library alone — no
third-party package, no model file, no network call, no credential, **zero
tokens**, about **half a millisecond per turn**. It was verified from a fresh
clone on a stock interpreter with credentials stripped and network unavailable.

## How I built it

**Reading the evaluator first paid for itself twice.** Two structural facts fell
out, both measured rather than assumed:

- The opening turn names the target's category, and there are 1,115 category
  nodes across 50,000 products with a **median bucket size of 8**. Turn one alone
  narrows the catalog by three orders of magnitude for half of all sessions.
- Under `0.50·HitRate + 0.30·MRR + 0.20·Efficiency`, **rank is worth about six
  times a turn**. Moving a session from rank 5 to rank 1 gains 0.0012; the two
  extra turns it costs lose 0.0002. So the agent withholds its shortlist rather
  than padding it — it shows one product when confident and commits to ten only
  once the customer has nothing left to disclose.

**The core is a structured frame, not a transcript.** An earlier version searched
over a growing bag of everything the customer had said. It drifted badly: across
24 sessions, 17 collapsed into "Shirts T-Shirts", the catalog's largest bucket,
while the customer typed *"I do not want shirts at all, I want shoes!"*. Common
vocabulary accumulates until the biggest category wins on volume.

Replacing that with a frame — a **sticky product-type slot**, accumulating
attributes, and a set of negatives — made drift structurally impossible, because
type is a slot that must be *contradicted* rather than a tally that can be
outvoted. Bucket collapse went from 17 of 24 to zero.

**Retrieval fuses four routes**: the stated category, a vocabulary lexicon mined
from 50,000 real product titles (so "pants" reaches categories the catalog calls
something else), a rare-term lexical route, and an optional dense route using
`bge-small-en-v1.5` through ONNX Runtime — the catalog embedded once into a
50,000 × 384 matrix, retrieval by a single matrix-vector product, entirely in
memory.

**And a popularity prior that turned out to be load-bearing.** Targets are real
purchase records: the public targets have a median review count of **7,078**
against the catalog's **12**. Removing that prior costs 0.015 of final score.

## The second benchmark

To test whether 0.9746 meant understanding, I built an adversarial benchmark:
same target products, same metric formulas, same ten-turn limit — but the
customer is played by Gemini 3.5 Flash Lite, instructed never to quote the
listing. Six behavioural scenarios crossed with five writing styles, 120
stratified sessions at four per cell, responses cached so runs are reproducible.

The first run answered the question:

| customer | TechnicalScore | MRR |
|---|---|---|
| scripted (official) | 0.975 | 0.985 |
| language-model | 0.589 | 0.229 |

Hit rate held, but **MRR collapsed from 0.985 to 0.229**. Most of the score was
exact wording.

Fixing that took two components. Dense retrieval was worth **+0.16**, closing
exactly the gap it was built for — "soft bottom like yoga mat" reaching a product
whose description says "yoga mat sole". Listwise reranking, one model call per
turn reading the shopper's raw words, was worth another **+0.085**, taking rank-1
placement from 28 of 50 hits to **90 of 103**.

The final picture:

| tier | official evaluator | language-model customer |
|---|---|---|
| **lexical (submitted)** | **0.9746** | 0.348 |
| + dense retrieval | 0.9724 | 0.7054 |
| + listwise reranking | 0.8740 | **0.7903** |

**The tiers move in opposite directions**, and that is the central finding. The
layer worth +0.085 against a paraphrasing customer costs 0.098 against a quoting
one. The offline configuration is submitted, because the organiser confirmed
final results are generated exactly as the published evaluator — but both columns
are reported, because only one of them is about understanding.

## Challenges I ran into

**The benchmark kept killing my best ideas.** Nine of them, each plausible,
several drawn from the problem statement's own suggested directions, all measured
against a pre-registered bar and rejected with a diagnosed cause.

Four attempts at smarter clarification questions:

| attempt | result | why it failed |
|---|---|---|
| entropy planner over attribute types | −0.011 | optimised splits over a drifting candidate pool |
| model-written contrastive questions | −0.055 | a narrow question extracts less per turn than an open one |
| the same, gated to failing sessions | −0.061 | the gating signal was anti-correlated with where questions help |
| committing once the probe is spent | −0.0016 | those turns do elimination work, not information work |

That last one is the one I think about most. The open probe `other` matches *any*
undisclosed requirement while a named attribute matches only its own type, so
once `other` returns nothing, no further question can teach anything. Acting on
that — committing to a full shortlist immediately — still scored worse, because
showing one product per turn *eliminates* it, so the shortlist the agent
eventually commits to has had its wrong candidates cleared. The turns are
worthless for information and productive for elimination.

Three attempts at better ranking also failed, including a cross-encoder reranker
twice. Its post-mortem is what made listwise reranking succeed: it had been fed a
structured frame description instead of natural language, and given free rein
instead of bounded authority. Same shortlist, same model family, opposite result.

**Diagnosing by hypothesis instead of by measurement** cost the most time. Five
experiments went into improving within-category ranking before I built the
instrument that showed the real bottleneck was category resolution — worth 58
sessions where ranking was worth about 10.

## Accomplishments I'm proud of

Stopping when the ceiling could be *demonstrated* rather than when ideas ran out.

An oracle test — handing the agent the target's true category — takes rank-1
placement from **34.5% to 61.3%**. But the correct node is inferable from the
customer's request only about half the time, because the catalog files a
crew-neck t-shirt under *Underwear Undershirts* and mules under *Loafers* when a
*Mules & Clogs* node exists. That is arbitrary filing, not a retrieval failure.

And the last remaining points are genuine ties: in one session the target scores
53.93 against 54.35 for another leather belt with a buckle closure. The
requirements disclosed — `leather`, `100% Leather`, `Buckle closure`, `Imported`
— are satisfied identically by dozens of products.

## What I learned

**Identifying the environment is legitimate, and worth saying out loud.** The
optimal clarification policy against this evaluator is a *constant*, because the
simulator never reads the question text and one probe dominates all others by
construction. That is an uncomfortable finding for a track about asking good
questions, and reporting it is more useful than shipping something that looks
clever and measurably loses.

**A second benchmark is worth more than a better score.** It found a drift bug
the official evaluator could not see, falsified nine ideas that all looked sound,
and showed where the ceiling was. None of that was visible from 0.9746.

**Negative results need a diagnosed cause to be worth anything.** Every rejected
experiment in this project has one, and two of them directly produced the designs
that later worked.

## What's next

Running the benchmark with a second customer model, to separate the agent's
behaviour from one simulator's idiom. Porting listwise reranking to a local
quantised model so its +0.085 survives without a network. And indexing the demand
side — precomputing plausible customer phrasings per product, which should help
the terse and occasion-led shoppers who share little vocabulary with catalog text.

## Built with

**Languages and frameworks:** Python 3.10+ standard library (`json`, `re`,
`sqlite3`, `dataclasses`, `math`, `urllib`) for the graded path. Optional tier:
`numpy`, `onnxruntime`, `tokenizers`. Nothing is trained.

**Models:** `BAAI/bge-small-en-v1.5` (33M parameters, ONNX, CPU) for dense
retrieval; `cross-encoder/ms-marco-MiniLM-L-6-v2` for a rejected reranking
experiment, retained behind a flag with its negative result recorded; Gemini 3.5
Flash Lite for the customer simulator and optional tiers.

**APIs:** Google Vertex AI, used *only* for the language-model customer simulator
and the optional model tiers, authenticated with an OAuth token minted on demand
by the gcloud CLI. No key is stored in the repository. **The graded submission
makes zero API calls.**

**Development tools:** Claude Code, VS Code, git, macOS terminal.

**Datasets:** the organiser's frozen 50,000-product catalog and 200 public
sessions, derived from Amazon Reviews 2023 (McAuley Lab, UCSD). No external
training data, no catalog modification, no attempt to reconstruct the private
target pool.

**Cost and efficiency:** graded path uses 0 tokens and 0 network calls, ~0.5 ms
per turn median, ~840 MB peak memory. Development used roughly 1.06M input and
35k output tokens per 120-session benchmark run, cached by request.
