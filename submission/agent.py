"""Submission entry point for the TechJam conversational search challenge.

Exports `Agent`, matching the interface in docs/submission_rules.md. The
implementation lives in `src/`; this file only adapts it and makes the package
importable however the harness chooses to load this file.

The agent runs on the Python standard library alone. Optional retrieval tiers are
detected at construction and skipped when their dependencies are absent, so this
path never requires a network, a credential or a model file.
"""
from __future__ import annotations

import sys
from pathlib import Path

# The harness imports this file directly, so `src` is not necessarily on the
# path yet. Resolving it here keeps the bundle self-contained under a plain
# import, an importlib load from a path, or execution from another directory.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.agent_impl import ShoppingAgent  # noqa: E402


class Agent:
    """Multi-turn shopping agent: finds one hidden product within ten turns."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self._agent = ShoppingAgent(catalog_path)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return self._agent.respond(session_id, user_message, turn, top_k)
