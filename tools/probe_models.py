"""Find which models this machine can actually call, cheaply.

One two-token request per model. Reports quota, billing and enablement failures
distinctly from missing-model failures, so it is obvious whether the problem is
the account, the project, or the model id.

    python3 -m tools.probe_models              # auto-detect backend
    python3 -m tools.probe_models vertex       # force Vertex AI
"""
import json
import re
import sys

from sim.client import MissingCredential, make_client


def api_message(text: str) -> str:
    """Report what the API actually said, rather than guessing at a category."""
    match = re.search(r"\{.*", text, re.S)
    if match:
        try:
            payload = json.loads(match.group() + "}" * 3)
        except json.JSONDecodeError:
            payload = {}
        message = (payload.get("error") or {}).get("message")
        if message:
            return re.sub(r"\s+", " ", message)[:220]
    return re.sub(r"\s+", " ", text)[:220]

CANDIDATES = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-flash-lite-latest",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash",
]

backend = sys.argv[1] if len(sys.argv) > 1 else "auto"

for model in CANDIDATES:
    try:
        client = make_client(model, backend, max_output_tokens=8)
        reply = client.generate("Reply with exactly: ok", [{"role": "user", "text": "ping"}])
        print(f"  {client.backend:9s} {model:28s} WORKS  (replied {reply[:20]!r})")
    except MissingCredential as error:
        raise SystemExit(f"setup incomplete: {error}")
    except RuntimeError as error:
        print(f"  {model:38s} FAILS  {api_message(str(error))}")
