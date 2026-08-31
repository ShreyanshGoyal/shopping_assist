"""Embed the frozen catalog once. Run before using the dense retrieval track."""
import time

from src.catalog import Catalog
from src.dense import DenseIndex, available

if not available():
    raise SystemExit(
        "Dense track unavailable. Install numpy, onnxruntime and tokenizers, and "
        "place the encoder in models/bge-small-en-v1.5/."
    )
start = time.perf_counter()
catalog = Catalog("data/catalog.jsonl")
print(f"catalog loaded in {time.perf_counter()-start:.1f}s")
start = time.perf_counter()
index = DenseIndex(catalog)
print(f"index ready in {time.perf_counter()-start:.1f}s  shape={index.matrix.shape}  "
      f"{index.matrix.nbytes/1048576:.1f} MB")
