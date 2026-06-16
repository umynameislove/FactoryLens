"""Simple nearest-prototype defect type classifier."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


DEFECT_TYPES = ("crack", "cut", "hole", "print")
MIN_DISTANCE_SCALE = 1e-6


class PatchEmbeddingExtractorLike(Protocol):
    def extract_patch_embeddings(self, image_path: str) -> np.ndarray:
        """Return patch embeddings for one image."""


@dataclass(frozen=True)
class DefectTypePrototype:
    label: str
    embedding: np.ndarray
    source_count: int
    intra_class_radius: float | None = None


@dataclass(frozen=True)
class DefectTypeResult:
    label: str
    confidence: float
    distances: dict[str, float]


def classify_defect_type(
    image_path: str,
    reference_paths_by_label: Mapping[str, Sequence[str]] | None = None,
    *,
    prototypes: Sequence[DefectTypePrototype] | None = None,
    extractor: PatchEmbeddingExtractorLike | None = None,
) -> DefectTypeResult:
    """Classify one defect image as crack/cut/hole/print.

    This is a lightweight baseline. It averages normalized patch embeddings into
    one image vector, compares that vector with one prototype per defect type,
    and reports the nearest label with a conservative evidence score. The
    confidence combines nearest-label separation, distance-probability lift over
    chance, in-class prototype fit, and the amount of reference support. It is
    still not a calibrated production probability.
    """

    resolved_extractor = extractor or _default_extractor()
    resolved_prototypes = list(prototypes or [])
    if not resolved_prototypes:
        if reference_paths_by_label is None:
            raise ValueError("reference_paths_by_label or prototypes is required")
        resolved_prototypes = build_defect_type_prototypes(
            reference_paths_by_label,
            extractor=resolved_extractor,
        )

    prototypes_by_label = _validate_prototypes(resolved_prototypes)
    query_embedding = extract_image_embedding(image_path, resolved_extractor)
    distances = {
        label: _euclidean_distance(query_embedding, prototype.embedding)
        for label, prototype in prototypes_by_label.items()
    }

    ranked = sorted(distances.items(), key=lambda item: item[1])
    best_label = ranked[0][0]
    confidence = _confidence_from_distances(
        distances,
        best_label=best_label,
        prototypes_by_label=prototypes_by_label,
    )
    return DefectTypeResult(
        label=best_label,
        confidence=confidence,
        distances=distances,
    )


def build_defect_type_prototypes(
    reference_paths_by_label: Mapping[str, Sequence[str]],
    *,
    extractor: PatchEmbeddingExtractorLike | None = None,
) -> list[DefectTypePrototype]:
    """Build one centroid prototype for each defect type."""

    _validate_reference_labels(reference_paths_by_label)
    resolved_extractor = extractor or _default_extractor()
    prototypes: list[DefectTypePrototype] = []
    for label in DEFECT_TYPES:
        paths = list(reference_paths_by_label[label])
        embeddings = [
            extract_image_embedding(image_path, resolved_extractor)
            for image_path in paths
        ]
        centroid = _normalize_vector(np.mean(np.vstack(embeddings), axis=0))
        intra_class_radius = _estimate_intra_class_radius(embeddings, centroid)
        prototypes.append(
            DefectTypePrototype(
                label=label,
                embedding=centroid,
                source_count=len(paths),
                intra_class_radius=intra_class_radius,
            )
        )

    return prototypes


def extract_image_embedding(
    image_path: str,
    extractor: PatchEmbeddingExtractorLike,
) -> np.ndarray:
    """Return one normalized image embedding from patch embeddings."""

    patch_embeddings = extractor.extract_patch_embeddings(image_path)
    if patch_embeddings.ndim != 2 or patch_embeddings.shape[0] == 0:
        raise ValueError("patch embeddings must have shape (N, C) with N > 0")

    return _normalize_vector(np.mean(patch_embeddings, axis=0))


def _validate_reference_labels(
    reference_paths_by_label: Mapping[str, Sequence[str]],
) -> None:
    labels = set(reference_paths_by_label)
    expected = set(DEFECT_TYPES)
    if labels != expected:
        missing = ", ".join(sorted(expected - labels)) or "none"
        extra = ", ".join(sorted(labels - expected)) or "none"
        raise ValueError(f"reference labels mismatch; missing={missing}; extra={extra}")

    empty_labels = [
        label for label, paths in reference_paths_by_label.items() if not paths
    ]
    if empty_labels:
        raise ValueError(f"reference labels must not be empty: {', '.join(empty_labels)}")


def _validate_prototypes(
    prototypes: Sequence[DefectTypePrototype],
) -> dict[str, DefectTypePrototype]:
    prototypes_by_label = {prototype.label: prototype for prototype in prototypes}
    labels = set(prototypes_by_label)
    expected = set(DEFECT_TYPES)
    if labels != expected or len(prototypes_by_label) != len(prototypes):
        expected_text = ", ".join(DEFECT_TYPES)
        raise ValueError(f"prototypes must cover exactly these labels: {expected_text}")

    return prototypes_by_label


def _default_extractor() -> PatchEmbeddingExtractorLike:
    from factorylens.vision.embeddings import PatchEmbeddingExtractor

    return PatchEmbeddingExtractor()


def _normalize_vector(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError("embedding vector must be one-dimensional")

    norm = float(np.linalg.norm(vector))
    return vector / max(norm, eps)


def _euclidean_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left.astype(np.float32) - right.astype(np.float32)))


def _estimate_intra_class_radius(
    embeddings: Sequence[np.ndarray],
    centroid: np.ndarray,
) -> float | None:
    if len(embeddings) <= 1:
        return None

    distances = np.array(
        [_euclidean_distance(embedding, centroid) for embedding in embeddings],
        dtype=np.float32,
    )
    positive_distances = distances[distances > MIN_DISTANCE_SCALE]
    if positive_distances.size == 0:
        return None

    return float(np.percentile(positive_distances, 75))


def _confidence_from_distances(
    distances: Mapping[str, float],
    *,
    best_label: str,
    prototypes_by_label: Mapping[str, DefectTypePrototype],
) -> float:
    ordered_distances = np.array(
        [distances[label] for label in DEFECT_TYPES],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(ordered_distances)) or np.any(ordered_distances < 0):
        return 0.0

    best_distance = distances[best_label]
    runner_up_distance = sorted(distances.values())[1]
    relative_margin = 0.0
    if runner_up_distance > 0:
        relative_margin = (runner_up_distance - best_distance) / runner_up_distance

    positive_distances = ordered_distances[ordered_distances > MIN_DISTANCE_SCALE]
    distance_scale = (
        float(np.median(positive_distances))
        if positive_distances.size
        else 1.0
    )
    logits = -ordered_distances / max(distance_scale, MIN_DISTANCE_SCALE)
    logits = logits - np.max(logits)
    probabilities = np.exp(logits)
    probabilities = probabilities / np.sum(probabilities)

    best_index = DEFECT_TYPES.index(best_label)
    chance_level = 1.0 / len(DEFECT_TYPES)
    probability_lift = (float(probabilities[best_index]) - chance_level) / (
        1.0 - chance_level
    )
    best_prototype = prototypes_by_label[best_label]
    source_support = _source_support(best_prototype.source_count)
    prototype_fit = _prototype_fit(
        best_distance,
        radius=best_prototype.intra_class_radius,
        fallback_scale=distance_scale,
    )
    raw_confidence = (
        0.55 * max(probability_lift, 0.0)
        + 0.45 * max(relative_margin, 0.0)
    )
    raw_confidence *= source_support * prototype_fit
    return float(np.clip(raw_confidence, 0.0, 0.99))


def _source_support(source_count: int) -> float:
    """Return a support penalty for tiny reference sets.

    One sample can identify a nearest neighbor, but it cannot describe normal
    variation for a defect type, so confidence stays intentionally low.
    """

    return float(np.clip(0.35 + 0.15 * max(source_count, 0), 0.0, 1.0))


def _prototype_fit(
    distance: float,
    *,
    radius: float | None,
    fallback_scale: float,
) -> float:
    if radius is None or radius <= MIN_DISTANCE_SCALE:
        return 0.75

    normalized_distance = distance / max(radius, fallback_scale * 0.05)
    return float(np.exp(-normalized_distance))
