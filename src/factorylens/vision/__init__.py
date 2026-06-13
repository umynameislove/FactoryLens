"""Vision baseline helpers for FactoryLens."""

from factorylens.vision.anomaly import build_memory_bank, load_memory_bank, score_image
from factorylens.vision.embeddings import PatchEmbeddingExtractor

__all__ = [
    "PatchEmbeddingExtractor",
    "build_memory_bank",
    "load_memory_bank",
    "score_image",
]

