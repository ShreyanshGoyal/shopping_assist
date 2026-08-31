"""A shopper played by a language model.

The contract with the rest of the harness is deliberately narrow: given the
conversation so far, produce the customer's next line. Whether the agent has
succeeded is never decided here — that stays an exact product-id comparison in
the harness, matching the organiser's rule that hits are always code matches.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .client import GeminiClient
from .personas import STYLES, SCENARIOS, Scenario, profile_summary

SYSTEM = """You are role-playing a shopper on a clothing and accessories website, \
talking to an AI shopping assistant. Stay in character at all times.

You already have ONE specific product in mind. Your goal is for the assistant to \
show you that product. But you behave like a real person, which means:

RULES YOU MUST FOLLOW
1. Never state the product's full title or model number.{brand_rule}
2. Never copy wording from the product listing. Describe things in your own everyday \
words, the way a shopper would from memory.
3. Do not dump everything at once. Say a little; let the assistant ask.
4. Answer what the assistant actually asked. If you have no opinion on it, say so \
plainly instead of inventing one.
5. If you are shown products that are not what you want, react briefly and naturally \
to what is wrong with them.
6. Keep every reply to at most two short sentences.
7. Never mention that you are an AI, that this is a test, or that you have a listing \
in front of you.

YOUR STYLE
{style}

YOUR SITUATION
{scenario}

{profile}

THE PRODUCT YOU HAVE IN MIND
{product}
"""


def describe_product(product: dict, allow_brand: bool) -> str:
    """The listing the customer is remembering, trimmed to what a shopper would recall."""
    lines = [f"Type of item: {' > '.join(str(c) for c in (product.get('categories') or [])[1:])}"]
    if allow_brand and product.get("store"):
        lines.append(f"Brand: {product['store']}")
    lines.append(f"It is described as: {str(product.get('title') or '')[:200]}")
    features = [str(f) for f in (product.get("features") or [])][:6]
    if features:
        lines.append("Its listed details include:")
        lines.extend(f"  - {f[:180]}" for f in features)
    if product.get("price") not in (None, ""):
        lines.append(f"It costs about ${product['price']}")
    return "\n".join(lines)


@dataclass
class LLMCustomer:
    client: GeminiClient
    product: dict
    scenario: Scenario
    style: str
    profile: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    @property
    def system_prompt(self) -> str:
        brand_rule = "" if self.scenario.allow_brand else " Never name the brand."
        return SYSTEM.format(
            brand_rule=brand_rule,
            style=STYLES.get(self.style, STYLES["plain"]),
            scenario=self.scenario.instruction,
            profile=profile_summary(self.profile),
            product=describe_product(self.product, self.scenario.allow_brand),
        )

    def opening(self) -> str:
        self.history.append({
            "role": "user",
            "text": "Begin the conversation. Send your first message to the shopping assistant.",
        })
        reply = self.client.generate(self.system_prompt, self.history)
        self.history.append({"role": "model", "text": reply})
        return reply

    def reply(self, assistant_message: str, shown_titles: list[str], turn: int) -> str:
        shown = "\n".join(f"  - {title[:110]}" for title in shown_titles) or "  (nothing yet)"
        prompt = (
            f"The assistant said:\n\"{assistant_message}\"\n\n"
            f"It is currently showing you:\n{shown}\n\n"
            f"This is message {turn} of at most 10. None of those is the product you want. "
            f"Reply in character."
        )
        self.history.append({"role": "user", "text": prompt})
        reply = self.client.generate(self.system_prompt, self.history)
        self.history.append({"role": "model", "text": reply})
        return reply


def build_scenario(name: str) -> Scenario:
    return SCENARIOS[name]
