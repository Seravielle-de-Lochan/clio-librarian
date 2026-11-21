"""
Embeddings layer for Clio, the librarian.

Pluggable by design:
- Production: sentence-transformers, all-MiniLM-L6-v2 by default —
  small, fast, runs entirely locally, no cloud.
- Fallback (testing / no model available): a deterministic hashed bag-of-words
  embedder. NOT semantically smart — exists purely so the ingest/store/query
  pipeline can be tested end-to-end without downloading a model.

The rest of the system only ever calls `get_embedder()` and `.embed(texts)`.
Swapping the brain never touches the plumbing.
"""

from __future__ import annotations
import hashlib
import math
import re
from typing import List, Protocol


class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: List[str]) -> List[List[float]]: ...


class SentenceTransformerEmbedder:
    """Production embedder. Requires: pip install sentence-transformers"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        self.name = f"sentence-transformers/{model_name}"

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()


class HashedFallbackEmbedder:
    """
    Deterministic hashed bag-of-words with L2 normalisation.
    Overlapping vocabulary -> nearby vectors. Good enough to *test* the
    pipeline; NOT good enough to ship as the librarian's actual brain.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.name = f"hashed-fallback-{dim}d"
        self._token_re = re.compile(r"[a-zA-Z']{2,}")

    def _tokens(self, text: str) -> List[str]:
        return [t.lower() for t in self._token_re.findall(text)]

    def embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in self._tokens(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 128) % 2 == 0 else -1.0
                vec[idx] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out


def get_embedder(prefer_real: bool = True) -> Embedder:
    """Return the best available embedder. Never crashes for want of a model."""
    if prefer_real:
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            pass
    return HashedFallbackEmbedder()
