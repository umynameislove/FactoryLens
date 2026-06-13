"""Vision baseline helpers for FactoryLens."""

from factorylens.vision.anomaly import build_memory_bank, load_memory_bank, score_image
from factorylens.vision.embeddings import PatchEmbeddingExtractor
from factorylens.vision.heatmap import extract_regions, make_heatmap

__all__ = [
    "PatchEmbeddingExtractor",
    "build_memory_bank",
    "extract_regions",
    "load_memory_bank",
    "make_heatmap",
    "score_image",
]
