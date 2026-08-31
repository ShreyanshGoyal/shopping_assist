## Inspiration

The challenge ships its own evaluator, so I read it before writing code. The
"customer" isn't a language model — it's a deterministic program that derives
everything it will ever say from the target product's own metadata, then quotes
those strings back at you.

That reframed the task. A high score against a customer who quotes catalog text
proves string matching, not understanding. So I built two things: an agent that
maxes the official benchmark, and a second benchmark to find out whether that
score meant anything.

## What it does

Finds one hidden product in a frozen 50,000-item Amazon catalog within ten turns.

**0.9746** on the official evaluator against a weak-BM25 reference of **0.1067**
— on the Python standard library alone. No dependency, no model file, no network,
no credential, **zero tokens**, **0.5 ms per turn**. Verified from a fresh clone
with credentials stripped and network unavailable.

## How I built it

**A structured frame, not a transcript.** A sticky product-type slot,
accumulating attributes, and negatives. An earlier bag-of-words version drifted
badly — 17 of 24 sessions collapsed into the catalog's largest category while the
customer typed *"I want shoes, not shirts!"*. Making type a slot that must be
*contradicted*, rather than a tally that can be outvoted, made that impossible.

**Four fused retrieval routes:** the stated category, a vocabulary lexicon mined
from 50,000 product titles, a rare-term lexical route, and optional dense
embeddings (`bge-small-en-v1.5` via ONNX, 50,000 × 384, in memory).

**Withholding, not padding.** Rank is worth about six times a turn under this
metric, so the agent shows one product when confident and commits to ten only
once the customer has nothing left to disclose.

## The honest test

Same targets, same metrics — but the customer is Gemini 3.5 Flash Lite, told
never to quote the listing. 120 stratified sessions.

| tier | official evaluator | paraphrasing customer |
|---|---|---|
| **lexical (submitted)** | **0.9746** | 0.348 |
| + dense retrieval | 0.9724 | 0.7054 |
| + listwise reranking | 0.8740 | **0.7903** |

**The tiers move in opposite directions.** The layer worth +0.085 against a
paraphrasing customer costs 0.098 against a quoting one. The offline
configuration is submitted — the organiser confirmed final scoring uses the
published evaluator — but both columns are reported, because only one is about
understanding.

## Challenges I ran into

The second benchmark killed nine ideas, each measured against a pre-registered
bar:

| attempt | result | why |
|---|---|---|
| entropy-based question planner | −0.011 | optimised splits over a drifting pool |
| model-written contrastive questions | −0.055 | narrow questions extract less per turn than open ones |
| cross-encoder reranking (twice) | −0.047, −0.015 | fed a frame description, not natural language |
| adaptive slate widening | −0.034 | converting earlier locks in a worse rank |

The one I think about most: once the open probe returns nothing, no further
question can teach anything — yet committing immediately *still* scored worse,
because showing one product per turn eliminates it, cleaning the shortlist the
agent eventually commits to.

## What I learned

**The optimal questioning policy here is a constant.** The simulator never reads
the question text, and one probe dominates all others by construction. That's an
uncomfortable finding for a track about asking good questions, and reporting it
beats shipping something that looks clever and measurably loses.

**The ceiling is in the data.** An oracle test — handing the agent the true
category — lifts rank-1 placement from 34.5% to 61.3%. But that category is
inferable only about half the time, because the catalog files a crew-neck t-shirt
under *Underwear Undershirts*. Arbitrary filing, not a retrieval failure.

## Built with

**Python 3.10+ standard library** for the graded path; optional `numpy`,
`onnxruntime`, `tokenizers`. **Models:** `bge-small-en-v1.5` (ONNX, CPU),
`ms-marco-MiniLM-L-6-v2` (rejected, kept with its negative result), Gemini 3.5
Flash Lite. **APIs:** Google Vertex AI — used *only* for the customer simulator
and optional tiers, OAuth via gcloud, no key in the repository. **The submission
makes zero API calls.** **Data:** the organiser's frozen catalog and 200 public
sessions, from Amazon Reviews 2023 (McAuley Lab, UCSD) — no external training
data. **Tools:** Claude Code, VS Code, git.

**Cost:** graded path 0 tokens, ~840 MB, ~20 s for all 200 sessions. Development
used roughly 1.06M input tokens per benchmark run, cached by request.
