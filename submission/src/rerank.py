"""Cross-encoder reranking of the shortlist.

Retrieval scores a request against a product independently — the request becomes
a vector or a bag of terms, the product becomes another, and they meet only at
the end. A cross-encoder reads both together, so it can judge whether *this*
product answers *this* request rather than whether they occupy similar regions of
a vector space. That is where the remaining score lives: of 103 hits on the
n=120 benchmark, half landed at ranks 2-10, worth +0.134 of MRR.

Applied to a shortlist only. Scoring 30 pairs costs a few hundred milliseconds on
CPU; scoring 50,000 would be absurd, which is exactly why retrieval runs first.

Local, offline and optional, on the same terms as the dense track: missing
dependencies or model files mean the agent ranks without it rather than failing.
"""
from __future__ import annotations

from pathlib import Path

MODEL_DIR = Path("models/ms-marco-MiniLM-L-6-v2")
MAX_TOKENS = 256
BATCH = 16


def available() -> bool:
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        from tokenizers import Tokenizer  # noqa: F401
    except ImportError:
        return False
    return (MODEL_DIR / "model.onnx").exists() and (MODEL_DIR / "tokenizer.json").exists()


def passage(product) -> str:
    """The product as a short passage, in the order a shopper would read it."""
    parts = [product.title]
    if product.coarse_category:
        parts.append(product.coarse_category)
    parts.extend(sorted(product.quotable)[:3])
    return " ".join(p for p in parts if p)[:500]


class CrossEncoder:
    """Relevance of a (request, product) pair, read jointly."""

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding(length=None)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.inputs = {i.name for i in self.session.get_inputs()}

    def score(self, query: str, passages: list[str]) -> list[float]:
        import numpy as np

        if not query.strip() or not passages:
            return [0.0] * len(passages)
        out: list[float] = []
        for start in range(0, len(passages), BATCH):
            chunk = passages[start : start + BATCH]
            encoded = self.tokenizer.encode_batch([(query, p or " ") for p in chunk])
            ids = np.array([e.ids for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            types = np.array([e.type_ids for e in encoded], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
            logits = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
            out.extend(float(v) for v in logits.reshape(-1))
        return out
