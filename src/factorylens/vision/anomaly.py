"""PatchCore-lite anomaly scoring over image patch embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from factorylens.vision.embeddings import (
    PatchEmbeddingExtractor,
    feature_map_to_embeddings,
    normalize_embeddings,
)


DEFAULT_MEMORY_BANK_PATH = "data/memory_bank.npz"
CORESET_RATIO = 0.25
CORESET_MIN_ROWS = 256
CORESET_PROJECTION_DIM = 64


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
    all_embeddings: list[np.ndarray] = []
    for image_path in good_image_paths:
        embeddings = extractor.extract_patch_embeddings(image_path)
        all_embeddings.append(
            _uniform_candidate_rows(
                embeddings,
                max_patches_per_image,
            )
        )

    candidates = np.vstack(all_embeddings).astype(np.float32, copy=False)
    target_rows = _coreset_target_size(len(candidates))
    memory_bank = greedy_coreset(
        candidates,
        target_rows,
        seed=seed,
    )
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
    if embeddings.shape[1] != memory_bank.shape[1]:
        raise ValueError(
            "memory bank embedding dimension does not match the extractor; "
            "rebuild data/memory_bank.npz"
        )
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


def greedy_coreset(
    embeddings: np.ndarray,
    target_rows: int,
    *,
    seed: int = 13,
    projection_dim: int = CORESET_PROJECTION_DIM,
) -> np.ndarray:
    """Select a deterministic k-center coreset with a JL-style projection."""

    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("embeddings must have shape (N, C) with N > 0")
    if target_rows <= 0:
        raise ValueError("target_rows must be positive")
    if len(embeddings) <= target_rows:
        return embeddings.copy()

    projected = _project_for_coreset(
        embeddings,
        seed=seed,
        projection_dim=projection_dim,
    )
    centroid = projected.mean(axis=0, keepdims=True)
    first_index = int(
        np.argmax(np.sum((projected - centroid) ** 2, axis=1))
    )
    selected = np.empty(target_rows, dtype=np.int64)
    selected[0] = first_index
    minimum_distances = _squared_distances(
        projected,
        projected[first_index],
    )
    minimum_distances[first_index] = -1.0

    for index in range(1, target_rows):
        next_index = int(np.argmax(minimum_distances))
        selected[index] = next_index
        distances = _squared_distances(
            projected,
            projected[next_index],
        )
        minimum_distances = np.minimum(minimum_distances, distances)
        minimum_distances[selected[: index + 1]] = -1.0

    return embeddings[selected].astype(np.float32, copy=False)


def _uniform_candidate_rows(
    embeddings: np.ndarray,
    max_rows: int,
) -> np.ndarray:
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("patch embeddings must have shape (N, C) with N > 0")
    if max_rows <= 0 or len(embeddings) <= max_rows:
        return embeddings

    indexes = np.linspace(
        0,
        len(embeddings) - 1,
        num=max_rows,
        dtype=np.int64,
    )
    return embeddings[indexes]


def _coreset_target_size(candidate_count: int) -> int:
    if candidate_count <= CORESET_MIN_ROWS:
        return candidate_count
    return min(
        candidate_count,
        max(
            CORESET_MIN_ROWS,
            int(np.ceil(candidate_count * CORESET_RATIO)),
        ),
    )


def _project_for_coreset(
    embeddings: np.ndarray,
    *,
    seed: int,
    projection_dim: int,
) -> np.ndarray:
    if projection_dim <= 0:
        raise ValueError("projection_dim must be positive")
    if embeddings.shape[1] <= projection_dim:
        return normalize_embeddings(embeddings)

    rng = np.random.default_rng(seed)
    projection = rng.normal(
        0.0,
        1.0 / np.sqrt(projection_dim),
        size=(embeddings.shape[1], projection_dim),
    ).astype(np.float32)
    return normalize_embeddings(embeddings @ projection)


def _squared_distances(
    embeddings: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    difference = embeddings - reference[None, :]
    return np.sum(difference * difference, axis=1)
