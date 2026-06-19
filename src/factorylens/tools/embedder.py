from __future__ import annotations

from functools import lru_cache
from typing import Protocol


EMBED_DIM = 384  # all-MiniLM-L6-v2


class Embedder(Protocol):
    """Hợp đồng tối thiểu mọi embedder phải có."""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Đổi list text -> list vector (mỗi vector dài self.dim)."""
        ...


class LocalEmbedder:
    """sentence-transformers chạy local. Import nặng -> lazy trong __init__."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy

        self._model = SentenceTransformer(model_name)
        self.dim = EMBED_DIM

    def embed(self, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings=True -> vector đã chuẩn hoá, cosine ổn định hơn
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]


@lru_cache
def get_default_embedder() -> Embedder:
    """Tải model 1 lần / process (đắt). Dùng cho ingest + retrieve thật."""
    return LocalEmbedder()