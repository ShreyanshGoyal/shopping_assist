"""Model client for the simulator.

Re-exported from the agent's own client so both speak to the same backends with
the same caching, without the simulator and the submission drifting apart.
"""
from src.llm import (  # noqa: F401
    AI_STUDIO_ROOT,
    SSL_CONTEXT,
    BaseClient,
    GeminiClient,
    MissingCredential,
    Usage,
    VertexClient,
    load_dotenv,
    make_client,
)
