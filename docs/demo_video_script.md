# Demo video — shot list and narration

Target 3:30. Screen capture with voice-over. Upload to YouTube, **set to public**,
link from the Devpost description.

Record at 1920×1080. Terminal at a large font — 16pt minimum, judges may watch on
a laptop. Have every command pre-typed in a scratch file so nothing is typed live.

---

## 0:00–0:20 — The problem

**Show:** the problem statement's session example, or a single slide with the
scoring formula.

> "A customer wants one specific product out of fifty thousand. They will not tell
> you which. You get ten turns of conversation, and you are scored on whether you
> find it, how high you rank it, and how few turns you take."

Cut to the weak baseline number on screen: **0.1067**.

---

## 0:20–1:20 — The scripted evaluator, and one transcript

**Show:** a terminal, repository root.

```bash
python3 -m evaluator.local_evaluator
```

Let it run — it takes about twenty seconds, which is worth showing rather than
cutting, because the speed is part of the claim. Land on the output.

> "That is the organiser's evaluator, unmodified, on all two hundred public
> sessions. **0.9746**, hit rate 1.0. No API key, no network, no model file —
> standard library only, zero tokens."

**Then show one transcript** so the score is not just a number:

```bash
python3 -m tools.trace_session --sample public_0003
```

Point at three things on screen:
- Turn 1: **the agent shows a single product** — the target, at rank 1. "It is
  confident, so it does not pad. Rank is worth about six times a turn under this
  metric."
- Turn 3: the customer says *"Actually, ignore my earlier preference."* The
  target **stays at rank 1**.
- The lock note: "override sessions cannot score before turn three. It was
  already correct before it was allowed to be."

---

## 1:20–2:50 — The honest question

> "But that customer is a program, and it quotes the product's own description
> back at us. So we asked whether 0.975 measures understanding — and built a
> second benchmark to find out."

**Show:** `sim/personas.py` briefly — the six scenarios and five writing styles.

> "Same targets, same metrics, same ten turns. But the customer is a language
> model told never to quote the listing."

**Show the gap table on screen:**

| customer | score | MRR |
|---|---|---|
| scripted | 0.975 | 0.985 |
| language-model | 0.589 | 0.229 |

> "Hit rate held. MRR collapsed. Most of our score was wording."

**Now the payoff — play one LLM-customer session**, a `use_case_led` one, which
is the hardest scenario. Read two or three turns aloud in the customer's own
words. On screen, surface:
- the **frame state** — product type, accumulated attributes, negatives — so the
  structure is visible rather than asserted;
- a **listwise promotion**: retrieval had no clear leader, the model read the
  shopper's raw words and promoted its pick to rank 1.

> "The frame is why this does not drift. Product type is a slot that has to be
> contradicted, not a tally that can be outvoted. Before this, seventeen of
> twenty-four sessions collapsed into the catalog's biggest category while the
> customer typed 'I want shoes, not shirts'."

---

## 2:50–3:35 — What we measured, and where the ceiling is

**Show the two-benchmark tier table:**

| tier | scripted | LLM customer |
|---|---|---|
| 1 — lexical, submitted | **0.9746** | — |
| 2 — + dense retrieval | 0.9724 | 0.7054 |
| 3 — + listwise rerank | 0.8740 | **0.7903** |

> "The tiers move in opposite directions. The layer worth +0.085 against a
> paraphrasing customer costs 0.098 against a quoting one."

**Show the falsification table** — four questioning attempts, each with its cause.

> "The benchmark's real value was killing our own ideas. Four attempts at smarter
> clarification questions, all measured worse. The open probe wins by a derived
> argument: it matches any undisclosed requirement, and the simulator never reads
> the question text at all."

**Show the oracle ceiling:**

> "We stopped because we could show where the ceiling is. Handing the agent the
> target's true category takes rank-1 from 34.5% to 61.3%. But that category is
> inferable only about half the time — the catalog files a crew-neck t-shirt
> under Underwear Undershirts. That is arbitrary filing, not a retrieval failure."

---

## 3:35–3:50 — Close

**Show:** the clean-machine run — fresh clone, stock Python, credentials stripped.

> "Runs offline on a laptop. Zero tokens on the graded path, half a millisecond
> per turn, verified from a fresh clone on a stock interpreter with no network."

Hold on **0.9746**. End.

---

## Recording checklist

- [ ] Terminal font ≥16pt, light-on-dark, window at 1920×1080
- [ ] Commands pre-typed in a scratch file, pasted not typed
- [ ] `.env` **closed** and not visible in any editor tab or shell history
- [ ] No API key, project id, or billing page visible at any point
- [ ] Tables prepared as slides or a rendered markdown preview, not raw text
- [ ] One rehearsal pass for timing before the take
- [ ] Export 1080p, upload to YouTube, **visibility: Public**
- [ ] Paste the link into the Devpost description

## Capturing the LLM-customer session

The 1:20–2:50 segment needs a recorded session with visible frame state. Either
replay a stored transcript from `sim_n120_hedge.json` with an added state print,
or run one live — a live run costs a few hundred tokens and shows real latency.
Pick a `use_case_led` session that converts, since that scenario went from 0.541
to 0.700 and is the clearest illustration of what the tiers buy.
