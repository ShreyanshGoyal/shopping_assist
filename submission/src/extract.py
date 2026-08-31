"""Filling the query frame from what the customer said.

Two interchangeable implementations behind one method. The frame is the
architecture; the extractor is a component, and which one is installed changes
how well the frame is filled, never how the rest of the agent works.

* `LexicalExtractor` runs offline against the catalog-mined lexicon. No network,
  no credential, no dependency.
* `ModelExtractor` asks a language model to rewrite the conversation into a
  canonical product type, a list of attributes, and a list of rejections.

The organiser reserves the right to score with network access disabled, so the
model extractor degrades to the lexical one on any failure rather than taking the
session down with it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .catalog import Catalog
from .frame import Frame, resolve_category
from .lexicon import terms as lexicon_terms
from .state import Constraint, classify

# "not jeans", "those are pajamas", "stop showing me shirts", "I don't want heels"
NEGATION_PATTERNS = (
    re.compile(r"\b(?:not|no|isn'?t|aren'?t)\s+(?:a|an|the|those|these)?\s*([a-z][a-z\- ]{2,30})", re.I),
    re.compile(r"\bdon'?t\s+(?:want|need|like)\s+(?:a|an|the|any)?\s*([a-z][a-z\- ]{2,30})", re.I),
    re.compile(r"\b(?:stop|quit)\s+(?:showing|suggesting)\s+(?:me\s+)?(?:the\s+)?([a-z][a-z\- ]{2,30})", re.I),
    re.compile(r"\bthose\s+are\s+([a-z][a-z\- ]{2,30})", re.I),
)

# "I want shoes, not shirts" — the thing before the negation is the real target.
CORRECTION = re.compile(r"\b(?:i\s+)?(?:want|need|looking for)\s+([a-z][a-z\- ]{2,30}?)[,\s]+not\b", re.I)

REJECTS_TYPE = re.compile(
    r"\b(?:not (?:shoes|clothes|shirts|pants)|wrong (?:category|thing|item)|"
    r"different (?:kind|type)|none of (?:those|these|them))\b", re.I
)


def find_negatives(message: str) -> list[str]:
    found: list[str] = []
    for pattern in NEGATION_PATTERNS:
        for match in pattern.finditer(message):
            phrase = match.group(1).strip(" .,!?")
            if phrase and len(phrase) < 40:
                found.append(phrase)
    return found


@dataclass
class LexicalExtractor:
    """Offline frame filling. Weak at naming the product type, but free."""

    catalog: Catalog
    name: str = "lexical"

    def update(self, frame: Frame, message: str, turn: int) -> None:
        if not message:
            return

        for phrase in find_negatives(message):
            frame.add_negative(phrase)

        correction = CORRECTION.search(message)
        contradicted = bool(REJECTS_TYPE.search(message)) or correction is not None

        candidate = correction.group(1).strip() if correction else None
        if candidate is None and (frame.type_phrase is None or contradicted):
            # No canonical phrase available offline, so use the whole message and
            # let category resolution do what it can with it.
            candidate = message
        if candidate:
            category, score = resolve_category(self.catalog, candidate)
            if category and category not in frame.negative_categories:
                phrase = candidate if len(candidate) < 40 else category
                frame.propose_type(phrase, category, score, contradicted=contradicted)


PROMPT = """You turn a shopper's messages into a structured search request.

Return ONLY a JSON object, no markdown fence, with exactly these keys:
  "product_type": the kind of item they want, as a short canonical noun phrase a \
catalog would use (e.g. "flat sandals", "women's jeans", "calf socks"). Use the \
most specific type you can justify. If it is genuinely unclear, use "".
  "attributes":   a list of short phrases describing what they want it to be like \
(material, colour, fit, features, occasion, budget). Exclude the product type itself.
  "rejected":     a list of short phrases naming things they have said they do NOT \
want, including product types they have dismissed.
  "type_changed": true only if this latest message contradicts the product type \
recorded below, false otherwise.

Record so far:
  product type: {current_type}
  attributes:   {current_attributes}

Rules:
- Judge the product type mainly from what the shopper says they are looking for, \
not from words they used to reject something.
- Do not invent requirements they did not express.
- Keep every phrase under 8 words.
"""


@dataclass
class ModelExtractor:
    """Frame filling by a language model, with the lexical extractor as fallback."""

    catalog: Catalog
    client: object
    fallback: LexicalExtractor
    name: str = "model"
    failures: int = 0
    max_failures: int = 3

    def update(self, frame: Frame, message: str, turn: int) -> None:
        if not message:
            return
        if self.failures >= self.max_failures:
            # Persistent trouble: stop paying for calls and run offline.
            self.fallback.update(frame, message, turn)
            return

        prompt = PROMPT.format(
            current_type=frame.type_phrase or "(none yet)",
            current_attributes="; ".join(c.text for c in frame.attributes[-6:]) or "(none yet)",
        )
        try:
            raw = self.client.generate(prompt, [{"role": "user", "text": message}])
            payload = _parse_json(raw)
        except Exception:
            self.failures += 1
            self.fallback.update(frame, message, turn)
            return
        if payload is None:
            self.failures += 1
            self.fallback.update(frame, message, turn)
            return

        for phrase in payload.get("rejected") or []:
            if isinstance(phrase, str):
                frame.add_negative(phrase)
                rejected_category, score = resolve_category(self.catalog, phrase)
                if rejected_category and score > 0.6:
                    frame.negative_categories.add(rejected_category)

        contradicted = bool(payload.get("type_changed"))
        if contradicted and frame.type_category:
            frame.reject_type(self.catalog)

        product_type = payload.get("product_type")
        if isinstance(product_type, str) and product_type.strip():
            category, score = resolve_category(self.catalog, product_type)
            if category and category not in frame.negative_categories:
                # A model-supplied phrase is canonical by construction, so it is
                # trusted more than anything the lexical path can produce.
                frame.propose_type(product_type.strip(), category, max(score, 0.8) + 0.2,
                                   contradicted=contradicted)

        for phrase in payload.get("attributes") or []:
            if isinstance(phrase, str) and phrase.strip():
                _add_attribute(frame, phrase.strip())


def _add_attribute(frame: Frame, text: str) -> None:
    seen = {c.text.lower() for c in frame.attributes}
    if text.lower() in seen:
        return
    frame.attributes.append(classify(text))


def _parse_json(raw: str) -> dict | None:
    """Tolerate fenced or chatty output around the object."""
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\s*|\s*```$", "", text, flags=re.S)
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return None
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None
