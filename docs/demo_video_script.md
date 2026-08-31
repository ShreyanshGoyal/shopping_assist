# Demo video — shooting script

**Target 3:30 · screen capture with voice-over · YouTube, public**

Read the **SAY** lines aloud. They are written to be spoken, not read silently.
Everything in `code` is a command to paste — have them ready in a scratch file so
nothing is typed on camera.

---

## Before you press record

- [ ] Terminal font 16pt or larger, window roughly 1920×1080
- [ ] `.env` closed, Cloud console tab closed, no key or project id anywhere on screen
- [ ] Commands pre-typed in a scratch file
- [ ] `cd` into the repository already
- [ ] `Cmd + Shift + 5` → Options → Microphone → **your mic** → Record

---

# 1 · The problem — 0:20

**SHOW** — a terminal, nothing running yet.

**SAY**

> A customer wants one specific product out of fifty thousand.
> They won't tell you which one.
> You get ten turns of conversation to find it.
>
> The starter kit scores about ten percent.

---

# 2 · The score — 0:50

**RUN**

```
python3 -m tools.scorecard
```

**SHOW** — let the scorecard sit on screen. Point at the TechnicalScore row.

**SAY**

> This is the organiser's evaluator. Two hundred sessions.
>
> Zero point nine seven, against a baseline of point one one.
> Hit rate of one — it finds the product every time.
>
> And look at the bottom line. Zero tokens. No network. No API key.
> This is the Python standard library. It runs on a laptop in twenty seconds.

---

# 3 · One session — 0:40

**RUN**

```
python3 -m tools.trace_session --sample public_0001
```

**SHOW** — point at the FUNNEL line, then at the single product shown.

**SAY**

> Here's one session. The customer names a category.
>
> Fifty thousand products, down to a bucket of three hundred and twenty-nine,
> down to one recommendation.
>
> Notice it shows a single product, not ten.
> Under this metric, rank is worth about six times a turn — so padding the list
> to look thorough actually loses points. It waits until it's confident.
>
> Turn one. Rank one.

---

# 4 · The honest question — 1:20  ← the important one

**SAY** *(before running anything)*

> But here's the problem with that number.
>
> The customer in that evaluator is a program. It quotes the product's own
> description back at you. So a high score might just mean I matched their
> wording — not that anything understood anything.
>
> So I built a second benchmark to find out.
> Same products, same scoring. But the customer is a language model,
> told never to quote the listing.

**RUN**

```
python3 -m tools.show_llm_session --scenario use_case_led
```

**SHOW** — read the customer's turns aloud from the screen. Take your time here.

**SAY**

> She says she's going to the beach with her cousins, and wants
> wrist things that won't get ruined swimming.
>
> Nothing in that sentence appears in the catalog.
>
> The agent shows bracelets. She says they look too fancy, the metal would ruin
> in the ocean. It narrows. She asks for woven string ones with a beachy vibe.
> Then — a pack of about twenty, with ocean waves.
>
> Turn four. Rank one. A twenty-one piece surfer wave bracelet, waterproof.

---

# 5 · What that cost — 0:30

**SHOW** — the tier table, as a slide or a rendered markdown preview.

| tier | official | paraphrasing customer |
|---|---|---|
| lexical (submitted) | **0.9746** | 0.348 |
| + dense retrieval | 0.9724 | 0.7054 |
| + listwise reranking | 0.8740 | **0.7903** |

**SAY**

> Here's what that second benchmark showed.
>
> The same agent scores point nine seven against the customer who quotes,
> and point three four against the one who doesn't.
> Most of my score was word matching.
>
> Dense retrieval and reranking close that gap. But look at the columns —
> they move in opposite directions. What helps against a real customer
> costs points against this evaluator.
>
> I submit the offline version, because that's what gets scored.
> But I report both numbers. Only one of them is about understanding.

---

# 6 · The ceiling — 0:20

**SAY**

> That benchmark also killed nine of my own ideas. Every one measured,
> every one with a diagnosed reason it failed.
>
> And it showed me where the ceiling is. If I hand the agent the correct
> category, rank-one placement nearly doubles. But that category is only
> guessable about half the time — because the catalog files a crew-neck t-shirt
> under Underwear Undershirts.
>
> That's not a retrieval problem. That's the data.

---

# 7 · Close — 0:10

**SHOW** — the scorecard again, or the clean-machine run.

**SAY**

> Runs offline, on a laptop, for nothing.
> Zero point nine seven four six.

**Hold two seconds. Stop recording.**

---

## After

- [ ] Watch it back once — check no key, path, or console is visible
- [ ] Export 1080p
- [ ] Upload to YouTube, **visibility: Public**
- [ ] Paste the link into Devpost → Project Media → Video demo link

## If you run long

Cut section 6 first. Sections 4 and 5 are the entry — everything else is setup.
