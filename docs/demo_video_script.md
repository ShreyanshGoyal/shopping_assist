# Demo video — shooting script

**~3:30 · screen capture with voice-over · YouTube, public · no slides needed**

Read the **SAY** lines out loud. They're written the way people actually talk, so
say them naturally — pause where the line breaks, don't rush the numbers.

Everything in `code` is pasted, never typed on camera.

---

## Before you press record

- [ ] Terminal font 16pt+, window about 1920×1080
- [ ] `.env` closed, Cloud console closed
- [ ] Do Not Disturb on
- [ ] Commands pre-typed in a scratch file
- [ ] `Cmd + Shift + 5` → Options → **Microphone: your mic** → Record

---

# 1 · Setup — 0:20

**SHOW** — empty terminal.

**SAY**

> Okay, so — someone's shopping, and they've got one specific product in mind.
> It's somewhere in a catalog of fifty thousand.
>
> They're not going to tell you which one. You just get ten turns of conversation
> to figure it out.
>
> The starter code they give you gets about ten percent of them.

---

# 2 · The score — 0:45

**RUN** `python3 -m tools.scorecard`

**SHOW** — let it sit. Point at the TechnicalScore row, then the tokens line.

**SAY**

> This is their evaluator, all two hundred sessions.
>
> Point nine seven. The baseline's point one one.
> And hit rate is one — it finds the product in every single session.
>
> The bit I'd point at though is down here. Zero tokens. No network. No API key.
> This is just the Python standard library. Whole thing runs in twenty seconds
> on a laptop.

---

# 3 · One conversation — 0:40

**RUN** `python3 -m tools.trace_session --sample public_0001`

**SHOW** — point at the FUNNEL line, then the single product.

**SAY**

> Here's what one session looks like.
>
> Customer says what kind of thing they want. Fifty thousand products,
> down to a bucket of three twenty-nine, down to one.
>
> And notice — it only shows one product. Not ten.
>
> That's deliberate. The way they score this, getting rank one matters about six
> times more than saving a turn. So padding your list out to look thorough
> actually costs you. It waits until it's sure.
>
> Turn one, rank one.

---

# 4 · The catch — 1:20  ← this is the one that matters

**SAY** *(nothing running yet)*

> Here's my problem with that score though.
>
> The customer in their evaluator isn't a person, and it isn't an AI.
> It's a little program — and it quotes the product's own description
> straight back at you.
>
> So point nine seven might just mean I got good at matching their wording.
> Which isn't the same as understanding anything.
>
> So I built a second one to check. Same products, same scoring —
> but this time the customer's a language model, and it's told
> never to quote the listing.

**RUN** `python3 -m tools.show_llm_session --scenario use_case_led`

**SHOW** — read her turns off the screen. Slow down here.

**SAY**

> So she says she's off to the beach with her cousins, and she wants —
> her words — wrist things that won't get ruined when they go swimming.
>
> None of that is in the catalog. Not one word.
>
> It shows her bracelets. She says no, too fancy, the metal would wreck
> in seawater. So it narrows. She asks for woven string ones,
> something beachy. Closer. Then she says — a pack, about twenty,
> ocean waves on them.
>
> Turn four. Rank one. Twenty-one piece surfer wave bracelet, waterproof.

---

# 5 · What it cost — 0:35

**RUN** `python3 -m tools.tiers`

**SHOW** — the table. Point at the top row, both numbers.

**SAY**

> And this is what that second benchmark told me.
>
> Same agent. Point nine seven against the customer that quotes,
> point three five against the one that doesn't.
> So yeah — most of my score really was word matching.
>
> Adding proper search and reranking fixes it. Gets it up to point seven nine.
>
> But look at the two columns. They go in opposite directions.
> The stuff that helps a real customer actively costs me points on the
> thing that's actually being graded.
>
> So I ship the offline one, because that's what gets scored.
> But I'm reporting both. Only one of them's about understanding.

---

# 6 · Where it stops — 0:20

**SAY**

> That benchmark also killed nine of my own ideas. All measured,
> all with a reason they failed.
>
> And it showed me where the ceiling is. If I just hand the agent
> the right category, rank one nearly doubles. But you can only guess
> that category about half the time — because this catalog files
> a crew-neck t-shirt under Underwear Undershirts.
>
> That's not a search problem. That's the data.

---

# 7 · Close — 0:10

**RUN** `python3 -m tools.scorecard`

**SAY**

> Runs offline, on a laptop, costs nothing.
> Point nine seven four six.

**Hold two seconds. Stop.**

---

## After

- [ ] Watch it back — no key, no path, no console visible
- [ ] Trim the ends in QuickTime (`Cmd + T`, then `Cmd + S`)
- [ ] YouTube, **visibility: Public**
- [ ] Link into Devpost → Project Media → Video demo link

## Running long?

Cut section 6. Sections 4 and 5 are the submission — everything else is setup.
