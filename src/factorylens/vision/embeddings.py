"""Torchvision feature extraction for the PatchCore-lite baseline."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


class PatchEmbeddingExtractor:
    """Extract mid-level ResNet patch embeddings from a single image.

    By default this tries to use ImageNet-pretrained ResNet18 weights. If the
    weights are not available offline, the extractor can fall back to random
    weights so local smoke tests still run. Production/demo runs should prefer
    cached pretrained weights for more meaningful anomaly scores.
    """

    def __init__(
        self,
        image_size: int = 224,
        pretrained: bool = True,
        allow_untrained_fallback: bool = True,
        device: str | None = None,
    ) -> None:
        self.image_size = image_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = _build_resnet18_features(
            pretrained=pretrained,
            allow_untrained_fallback=allow_untrained_fallback,
        ).to(self.device)
        self.model.eval()
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )

    def extract_feature_map(self, image_path: str) -> np.ndarray:
        """Return a feature map with shape ``(C, H, W)``."""

        image = Image.open(Path(image_path)).convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            features = self.model(tensor).squeeze(0).detach().cpu().numpy()
        return features.astype(np.float32, copy=False)

    def extract_patch_embeddings(self, image_path: str) -> np.ndarray:
        """Return L2-normalized patch embeddings with shape ``(N, C)``."""

        feature_map = self.extract_feature_map(image_path)
        return feature_map_to_embeddings(feature_map)


def feature_map_to_embeddings(feature_map: np.ndarray) -> np.ndarray:
    """Convert ``(C, H, W)`` feature map to normalized ``(H*W, C)`` patches."""

    if feature_map.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")

    channels, height, width = feature_map.shape
    embeddings = feature_map.reshape(channels, height * width).T.astype(np.float32)
    return normalize_embeddings(embeddings)


def normalize_embeddings(embeddings: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize embeddings row-wise."""

    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (N, C)")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, eps)


def _build_resnet18_features(
    pretrained: bool,
    allow_untrained_fallback: bool,
) -> nn.Sequential:
    _configure_torch_cache()
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    try:
        model = models.resnet18(weights=weights)
    except Exception as exc:
        if not pretrained or not allow_untrained_fallback:
            raise
        warnings.warn(
            "Could not load pretrained ResNet18 weights; falling back to random "
            f"weights for local smoke execution. Original error: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        model = models.resnet18(weights=None)

    # Keep layers through layer2: detailed enough for localization, still small.
    return nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
    )


def _configure_torch_cache() -> None:
    cache_dir = Path(os.environ.get("FACTORYLENS_TORCH_CACHE", "data/torch"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache_dir))
