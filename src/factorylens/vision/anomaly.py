"""PatchCore-lite anomaly scoring over image patch embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from factorylens.vision.embeddings import PatchEmbeddingExtractor, feature_map_to_embeddings


DEFAULT_MEMORY_BANK_PATH = "data/memory_bank.npz"


def build_memory_bank(
    good_image_paths: list[str],
    out_path: str | None = DEFAULT_MEMORY_BANK_PATH,
    extractor: PatchEmbeddingExtractor | None = None,
    max_patches_per_image: int = 128,
    seed: int = 13,
) -> np.ndarray:
    """Build a memory bank from normal images and optionally save it.

    The returned array has shape ``(N, C)``. A small per-image patch subsample is
    used by default so CPU demos stay responsive.
    """

    if not good_image_paths:
        raise ValueError("good_image_paths must not be empty")

    extractor = extractor or PatchEmbeddingExtractor()
    rng = np.random.default_rng(seed)
    all_embeddings: list[np.ndarray] = []
    for image_path in good_image_paths:
        embeddings = extractor.extract_patch_embeddings(image_path)
        all_embeddings.append(_subsample_rows(embeddings, max_patches_per_image, rng))

    memory_bank = np.vstack(all_embeddings).astype(np.float32, copy=False)
    distance_scale = estimate_distance_scale(memory_bank)

    if out_path:
        save_memory_bank(out_path, memory_bank, distance_scale)

    return memory_bank


def score_image(
    image_path: str,
    memory_bank: np.ndarray | None = None,
    memory_bank_path: str = DEFAULT_MEMORY_BANK_PATH,
    extractor: PatchEmbeddingExtractor | None = None,
    distance_scale: float | None = None,
    anomaly_percentile: float = 95.0,
) -> tuple[float, np.ndarray]:
    """Score one image against a memory bank.

    Returns ``(anomaly_score, score_map)`` where both are normalized to ``[0, 1]``.
    ``score_map`` keeps the feature-map grid resolution; B6 upsamples it for a
    visual heatmap.
    """

    if memory_bank is None:
        memory_bank, loaded_scale = load_memory_bank(memory_bank_path)
        if distance_scale is None:
            distance_scale = loaded_scale
    if memory_bank.size == 0:
        raise ValueError("memory_bank must not be empty")

    extractor = extractor or PatchEmbeddingExtractor()
    feature_map = extractor.extract_feature_map(image_path)
    embeddings = feature_map_to_embeddings(feature_map)
    raw_scores = nearest_neighbor_distances(embeddings, memory_bank)

    _, height, width = feature_map.shape
    raw_score_map = raw_scores.reshape(height, width)
    scale = normalization_scale(distance_scale or estimate_distance_scale(memory_bank))
    score_map = np.clip(raw_score_map / max(scale, 1e-12), 0.0, 1.0).astype(np.float32)
    anomaly_score = float(np.clip(np.percentile(score_map, anomaly_percentile), 0.0, 1.0))
    return anomaly_score, score_map


def normalization_scale(distance_scale: float) -> float:
    """Return a stable scale for L2-normalized patch distances.

    Patch vectors are L2-normalized, so Euclidean distances naturally live in
    ``[0, 2]``. Memory-bank nearest-neighbor distances can be tiny for small
    smoke banks, so this lower bound uses the natural maximum distance to avoid
    immediate saturation.
    """

    return max(float(distance_scale), 2.0)


def save_memory_bank(path: str, memory_bank: np.ndarray, distance_scale: float) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        memory_bank=memory_bank.astype(np.float32, copy=False),
        distance_scale=np.array([distance_scale], dtype=np.float32),
    )


def load_memory_bank(path: str) -> tuple[np.ndarray, float]:
    data = np.load(path)
    memory_bank = data["memory_bank"].astype(np.float32, copy=False)
    distance_scale = float(data.get("distance_scale", np.array([1.0]))[0])
    return memory_bank, distance_scale


def nearest_neighbor_distances(
    embeddings: np.ndarray,
    memory_bank: np.ndarray,
    chunk_size: int = 1024,
) -> np.ndarray:
    """Return each embedding's Euclidean distance to the nearest memory patch."""

    embeddings = embeddings.astype(np.float32, copy=False)
    memory_bank = memory_bank.astype(np.float32, copy=False)
    distances: list[np.ndarray] = []
    bank_norm = np.sum(memory_bank * memory_bank, axis=1)[None, :]

    for start in range(0, len(embeddings), chunk_size):
        chunk = embeddings[start : start + chunk_size]
        chunk_norm = np.sum(chunk * chunk, axis=1)[:, None]
        squared = np.maximum(chunk_norm + bank_norm - 2.0 * chunk @ memory_bank.T, 0.0)
        distances.append(np.sqrt(np.min(squared, axis=1)))

    return np.concatenate(distances).astype(np.float32, copy=False)


def estimate_distance_scale(memory_bank: np.ndarray, sample_size: int = 2048) -> float:
    """Estimate a robust normal-distance scale from the memory bank itself."""

    if len(memory_bank) <= 1:
        return 1.0

    sample = memory_bank[:sample_size].astype(np.float32, copy=False)
    squared = np.maximum(
        np.sum(sample * sample, axis=1)[:, None]
        + np.sum(sample * sample, axis=1)[None, :]
        - 2.0 * sample @ sample.T,
        0.0,
    )
    np.fill_diagonal(squared, np.inf)
    nearest = np.sqrt(np.min(squared, axis=1))
    nearest = nearest[np.isfinite(nearest)]
    if nearest.size == 0:
        return 1.0

    return float(max(np.percentile(nearest, 95), 1e-6))


def _subsample_rows(
    embeddings: np.ndarray,
    max_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_rows <= 0 or len(embeddings) <= max_rows:
        return embeddings

    indexes = rng.choice(len(embeddings), size=max_rows, replace=False)
    return embeddings[np.sort(indexes)]
