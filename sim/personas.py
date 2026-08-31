"""Who the simulated customer is, and how they behave.

The organiser's four scenarios are reproduced so results stay comparable, plus
two that the scripted simulator cannot express: a shopper who describes the
occasion rather than the product, and one who leads with a brand. Writing styles
vary independently of scenario, because an agent that only understands tidy
prose is not robust.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    instruction: str
    maps_to: str          # closest organiser scenario, for comparable reporting
    allow_brand: bool = False


SCENARIOS: dict[str, Scenario] = {
    "decisive": Scenario(
        "decisive",
        "You know what you want. Open by saying what kind of item you're after and the "
        "single requirement that matters most to you. Be direct.",
        maps_to="buying",
    ),
    "vague": Scenario(
        "vague",
        "You are still making up your mind. Open with only a rough sense of what you want "
        "and say you're still looking around. Reveal specifics only when asked.",
        maps_to="browsing",
    ),
    "mind_changer": Scenario(
        "mind_changer",
        "Open by mentioning a preference that is NOT actually important to you. Around your "
        "third or fourth message, correct yourself: say you've changed your mind and state "
        "what you actually need.",
        maps_to="intent_override",
    ),
    "indifferent": Scenario(
        "indifferent",
        "You have few strong opinions and are happy to defer. The first time you are asked "
        "about a specific attribute, say you don't really mind and let the assistant choose. "
        "Answer later questions normally.",
        maps_to="boundary",
    ),
    "use_case_led": Scenario(
        "use_case_led",
        "You think in terms of the occasion, not the product. Describe what you need it FOR "
        "and how you want to feel, rather than listing attributes. Only give concrete "
        "specifics if the assistant asks directly.",
        maps_to="browsing",
    ),
    "brand_led": Scenario(
        "brand_led",
        "You have shopped this brand before and trust it. You may name the brand, but you "
        "still will not describe the exact product or quote its listing.",
        maps_to="buying",
        allow_brand=True,
    ),
}

STYLES: dict[str, str] = {
    "terse": "You type in short, clipped fragments. Rarely a full sentence. No pleasantries.",
    "chatty": "You are friendly and a little rambly, mentioning small irrelevant details about your life.",
    "plain": "You write in ordinary, clear, complete sentences.",
    "informal": "You write casually with lowercase, contractions and mild slang. Occasional typos are fine.",
    "esl": "English is your second language. Your grammar is imperfect but your meaning is clear.",
}


def profile_summary(profile: dict) -> str:
    """The anonymised profile, rendered as something a shopper would recognise."""
    tags = ", ".join(str(t) for t in (profile.get("preference_tags") or [])) or "no strong pattern"
    return f"In the past you have cared about: {tags}."
