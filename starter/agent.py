"""Submission entry point.

The organiser's harness imports `Agent` from this module. The implementation
lives in `src/`; this file only adapts it to the required interface.
"""
from __future__ import annotations

from pathlib import Path

from src.agent_impl import ShoppingAgent


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self._agent = ShoppingAgent(catalog_path)

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._agent.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return self._agent.respond(session_id, user_message, turn, top_k)
