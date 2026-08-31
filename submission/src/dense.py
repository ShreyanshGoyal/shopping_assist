"""Dense retrieval over the frozen catalog.

The lexical routes fail exactly where shoppers are most human: "soft bottom like
yoga mat" shares no vocabulary with "cushioned EVA footbed", and terse writers
("need some new pants") give almost no words to match on at all. A dense channel
closes that gap by comparing meaning rather than tokens.

Deliberately small and local. A 33M-parameter encoder runs on CPU through ONNX
Runtime, the catalog is embedded once into a 50,000 x 384 float32 matrix (~73 MB)
cached to disk, and retrieval is a single matrix multiply in numpy. That is the
"entirely in-memory, no vector database cluster" the rules require, and it needs
no network at query time.

Optional by construction: if numpy or onnxruntime is unavailable the agent runs
lexical-only rather than failing, so the offline baseline never depends on this.
"""
from __future__ import annotations

from pathlib import Path

MODEL_DIR = Path("models/bge-small-en-v1.5")
CACHE = Path(".cache/embeddings")
MAX_TOKENS = 192
BATCH = 64


def available() -> bool:
    try:
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        from tokenizers import Tokenizer  # noqa: F401
    except ImportError:
        return False
    return (MODEL_DIR / "model.onnx").exists() and (MODEL_DIR / "tokenizer.json").exists()


class Encoder:
    """Sentence embeddings via ONNX Runtime, CLS-pooled and L2-normalised."""

    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding(length=None)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 0          # let the runtime pick
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )
        self.inputs = {i.name for i in self.session.get_inputs()}

    def encode(self, texts: list[str]):
        import numpy as np

        out = []
        for start in range(0, len(texts), BATCH):
            chunk = [t or " " for t in texts[start : start + BATCH]]
            encoded = self.tokenizer.encode_batch(chunk)
            ids = np.array([e.ids for e in encoded], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
            # bge models are trained with CLS pooling.
            vectors = hidden[:, 0, :].astype(np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            out.append(vectors / np.maximum(norms, 1e-9))
        return np.vstack(out) if out else np.zeros((0, 384), dtype="float32")


def product_text(product) -> str:
    """What a shopper would recognise this product as."""
    parts = [product.title]
    if product.coarse_category:
        parts.append(product.coarse_category)
    parts.extend(sorted(product.quotable)[:4])
    return " ".join(parts)[:600]


class DenseIndex:
    """Embedded catalog plus query-time search."""

    def __init__(self, catalog, cache_dir: Path = CACHE) -> None:
        import numpy as np

        self.catalog = catalog
        self.encoder = Encoder()
        self.asins: list[str] = list(catalog.products)
        cache_dir.mkdir(parents=True, exist_ok=True)
        matrix_path = cache_dir / f"catalog-{len(self.asins)}.npy"
        order_path = cache_dir / f"catalog-{len(self.asins)}.ids"

        if matrix_path.exists() and order_path.exists():
            self.asins = order_path.read_text(encoding="utf-8").split("\n")
            self.matrix = np.load(matrix_path)
        else:
            texts = [product_text(catalog.products[a]) for a in self.asins]
            self.matrix = self.encoder.encode(texts)
            np.save(matrix_path, self.matrix)
            order_path.write_text("\n".join(self.asins), encoding="utf-8")
        self.position = {asin: i for i, asin in enumerate(self.asins)}

    def scores_for(self, query: str):
        """Cosine similarity of the query against every product.

        One 50,000 x 384 matrix-vector product: a few milliseconds, and it means
        any candidate from any route can be looked up without re-encoding.
        """
        if not query.strip():
            return None
        return self.matrix @ self.encoder.encode([query])[0]

    def top(self, scores, limit: int = 300) -> list[tuple[str, float]]:
        import numpy as np

        if scores is None:
            return []
        if limit >= len(scores):
            order = np.argsort(-scores)
        else:
            part = np.argpartition(-scores, limit)[:limit]
            order = part[np.argsort(-scores[part])]
        return [(self.asins[i], float(scores[i])) for i in order]

    def search(self, query: str, limit: int = 300) -> list[tuple[str, float]]:
        return self.top(self.scores_for(query), limit)

    def score_of_index(self, asin: str, scores) -> float:
        index = self.position.get(asin)
        return float(scores[index]) if index is not None else 0.0

    def encode_query(self, query: str):
        return self.encoder.encode([query])[0]

    def build_category_index(self, names: list[str]) -> None:
        """Embed the category names themselves.

        Resolving "waterproof bracelets" onto a node by token overlap fails on
        exactly the wording customers use — the words that identify a category
        are rarely the words in its name. Comparing meanings instead is the same
        trick that fixed retrieval, applied one level up.
        """
        import numpy as np

        self.category_names = list(names)
        readable = [n.replace("&", "and") for n in self.category_names]
        self.category_matrix = self.encoder.encode(readable)

    def resolve_category(self, phrase: str, limit: int = 5) -> list[tuple[str, float]]:
        import numpy as np

        if getattr(self, "category_matrix", None) is None or not phrase.strip():
            return []
        vector = self.encoder.encode([phrase])[0]
        scores = self.category_matrix @ vector
        order = np.argsort(-scores)[:limit]
        return [(self.category_names[i], float(scores[i])) for i in order]
