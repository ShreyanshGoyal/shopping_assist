"""Model clients.

Used by the agent's optional extraction layer and by the customer simulator.
Two backends behind one interface:

* **Vertex AI** — Google Cloud, billed to a project, authenticated with an OAuth
  token minted by the gcloud CLI. No API key exists or is stored.
* **AI Studio** — the developer API, authenticated with a key from `.env`.

Both speak the same `generateContent` request shape, so only the endpoint and the
authorisation header differ.

Responses are cached on disk keyed by the full request, including the backend and
model. Two runs of the same evaluation therefore bill once and produce identical
transcripts, which is what makes an LLM-driven benchmark usable as a regression
test rather than a noisy one-off.
"""
from __future__ import annotations

import hashlib
import json
import os
import ssl
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

CACHE_DIR = Path(".cache/sim")
AI_STUDIO_ROOT = "https://generativelanguage.googleapis.com/v1beta"


def _ssl_context() -> ssl.SSLContext:
    """Verified TLS context.

    Python installed from python.org ships without a usable CA bundle unless its
    'Install Certificates' step was run, which makes every HTTPS call fail. Point
    at certifi's bundle when present rather than weakening verification.
    """
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CONTEXT = _ssl_context()


def load_dotenv(path: str | Path = ".env") -> None:
    """Read KEY=VALUE lines into the environment. Values are never logged."""
    dotenv = Path(path)
    if not dotenv.exists():
        return
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class MissingCredential(RuntimeError):
    pass


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    cached_calls: int = 0

    def add(self, prompt: int, completion: int, cached: bool) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1
        if cached:
            self.cached_calls += 1


@dataclass
class BaseClient:
    model: str
    temperature: float = 0.0
    max_output_tokens: int = 256
    cache_dir: Path = field(default=CACHE_DIR)
    usage: Usage = field(default_factory=Usage)

    backend: str = "base"

    def __post_init__(self) -> None:
        load_dotenv()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._setup()

    def _setup(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _endpoint(self) -> str:  # pragma: no cover - overridden
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def generate(self, system: str, turns: list[dict], *, retries: int = 4) -> str:
        """One completion. `turns` is [{'role': 'user'|'model', 'text': ...}]."""
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": t["role"], "parts": [{"text": t["text"]}]} for t in turns],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        raw = json.dumps({"backend": self.backend, "model": self.model, **body}, sort_keys=True)
        fingerprint = hashlib.sha256(raw.encode()).hexdigest()[:32]
        cached = self.cache_dir / f"{fingerprint}.json"
        if cached.exists():
            record = json.loads(cached.read_text(encoding="utf-8"))
            self.usage.add(record.get("prompt_tokens", 0), record.get("completion_tokens", 0), cached=True)
            return record["text"]

        payload = self._post(body, retries)
        text = ""
        for candidate in payload.get("candidates", []):
            for part in candidate.get("content", {}).get("parts", []):
                text += part.get("text", "")
        meta = payload.get("usageMetadata", {})
        prompt_tokens = int(meta.get("promptTokenCount", 0))
        completion_tokens = int(meta.get("candidatesTokenCount", 0))
        cached.write_text(
            json.dumps({
                "text": text.strip(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }),
            encoding="utf-8",
        )
        self.usage.add(prompt_tokens, completion_tokens, cached=False)
        return text.strip()

    def _post(self, body: dict, retries: int) -> dict:
        last_error: Exception | None = None
        for attempt in range(retries):
            request = urllib.request.Request(
                self._endpoint(), data=json.dumps(body).encode(), headers=self._headers()
            )
            try:
                with urllib.request.urlopen(request, timeout=90, context=SSL_CONTEXT) as response:
                    return json.load(response)
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")
                # Rate limiting is transient; an exhausted quota, a billing
                # problem or a disabled API is not, and retrying only burns time.
                permanent = any(
                    marker in detail.lower()
                    for marker in ("depleted", "billing", "has not been used", "is disabled", "permission")
                )
                if permanent or error.code not in (429, 500, 502, 503, 504) or attempt == retries - 1:
                    raise RuntimeError(f"{self.backend} HTTP {error.code}: {detail[:500]}") from None
                last_error = error
                time.sleep(2 ** attempt)
            except urllib.error.URLError as error:
                if attempt == retries - 1:
                    raise RuntimeError(f"{self.backend} unreachable: {error.reason}") from None
                last_error = error
                time.sleep(2 ** attempt)
        raise RuntimeError(f"{self.backend} call failed: {last_error}")  # pragma: no cover


@dataclass
class VertexClient(BaseClient):
    """Vertex AI on Google Cloud, authenticated through the gcloud CLI.

    Uses an OAuth access token minted on demand, so no long-lived secret is ever
    written to the repository. Tokens last an hour; this refreshes well inside that.
    """

    project: str = ""
    location: str = ""
    backend: str = "vertex"
    _token: str = field(default="", repr=False)
    _token_expires: float = 0.0

    def _setup(self) -> None:
        self.project = self.project or os.environ.get("VERTEX_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or self._gcloud_project()
        self.location = self.location or os.environ.get("VERTEX_LOCATION") or "global"
        if not self.project:
            raise MissingCredential(
                "No Google Cloud project. Run 'gcloud config set project <PROJECT_ID>' "
                "or set VERTEX_PROJECT in .env"
            )
        self._access_token()

    @staticmethod
    def _run(args: list[str]) -> str:
        try:
            done = subprocess.run(args, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return done.stdout.strip() if done.returncode == 0 else ""

    def _gcloud_project(self) -> str:
        return self._run(["gcloud", "config", "get-value", "project"]).replace("(unset)", "")

    def _access_token(self) -> str:
        if self._token and time.time() < self._token_expires:
            return self._token
        token = self._run(["gcloud", "auth", "print-access-token"])
        if not token:
            raise MissingCredential(
                "Could not mint a Google Cloud access token. Install the gcloud CLI and run "
                "'gcloud auth application-default login', then 'gcloud auth login'."
            )
        self._token = token
        self._token_expires = time.time() + 2700  # refresh well inside the 1h lifetime
        return token

    def _host(self) -> str:
        return "aiplatform.googleapis.com" if self.location == "global" else f"{self.location}-aiplatform.googleapis.com"

    def _endpoint(self) -> str:
        return (
            f"https://{self._host()}/v1/projects/{self.project}/locations/{self.location}"
            f"/publishers/google/models/{self.model}:generateContent"
        )

    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "Authorization": f"Bearer {self._access_token()}"}


@dataclass
class GeminiClient(BaseClient):
    """AI Studio developer API, authenticated with a key from .env."""

    backend: str = "aistudio"
    _key: str = field(default="", repr=False)

    def _setup(self) -> None:
        self._key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        if not self._key:
            raise MissingCredential(
                "Set GEMINI_API_KEY in a .env file at the repository root. "
                "The key is never printed or committed."
            )

    def _endpoint(self) -> str:
        return f"{AI_STUDIO_ROOT}/models/{self.model}:generateContent?key={self._key}"

    @staticmethod
    def list_models() -> list[dict]:
        load_dotenv()
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
        if not key:
            raise MissingCredential("Set GEMINI_API_KEY in .env first.")
        request = urllib.request.Request(f"{AI_STUDIO_ROOT}/models?key={key}&pageSize=200")
        with urllib.request.urlopen(request, timeout=30, context=SSL_CONTEXT) as response:
            return json.load(response).get("models", [])


def make_client(model: str, backend: str = "auto", **kwargs) -> BaseClient:
    """Pick a backend. 'auto' prefers Vertex when a Cloud project is reachable."""
    load_dotenv()
    if backend == "vertex":
        return VertexClient(model=model, **kwargs)
    if backend == "aistudio":
        return GeminiClient(model=model, **kwargs)
    has_project = bool(
        os.environ.get("VERTEX_PROJECT")
        or os.environ.get("GOOGLE_CLOUD_PROJECT")
        or VertexClient._run(["gcloud", "config", "get-value", "project"]).replace("(unset)", "")
    )
    if has_project:
        return VertexClient(model=model, **kwargs)
    return GeminiClient(model=model, **kwargs)
