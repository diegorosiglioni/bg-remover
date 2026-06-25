"""Shared fixtures and synthetic-image helpers for the test suite.

No test loads the real BiRefNet model or hits the network. The segmenter seam is
always exercised through an injected fake (:class:`FakeSegmenter`), and the
matting/defringe solvers run on tiny images so they stay fast.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG so synthetic images are reproducible across runs."""
    return np.random.default_rng(1234)


def solid_rgb(height: int, width: int, color=(120, 130, 140)) -> np.ndarray:
    """An (H, W, 3) uint8 image filled with a single colour."""
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:] = np.array(color, dtype=np.uint8)
    return img


def disc_alpha(size: int, radius_frac: float = 0.4) -> np.ndarray:
    """A centred filled disc as float32 alpha in [0, 1], shape (size, size)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy = cx = (size - 1) / 2.0
    r = radius_frac * size
    inside = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    return inside.astype(np.float32)


class FakeSegmenter:
    """In-memory Segmenter implementing both the modular and one-shot seams.

    ``predict`` returns a stored soft alpha; ``remove_oneshot`` returns a stored
    RGBA. Both ignore the pixels of the incoming image, so tests stay
    deterministic without invoking any model.
    """

    def __init__(self, soft_alpha: np.ndarray | None = None,
                 oneshot_rgba: np.ndarray | None = None) -> None:
        self._soft_alpha = soft_alpha
        self._oneshot_rgba = oneshot_rgba
        self.predict_calls = 0
        self.oneshot_calls = 0

    def predict(self, image: np.ndarray) -> np.ndarray:
        self.predict_calls += 1
        if self._soft_alpha is not None:
            return self._soft_alpha
        return disc_alpha(image.shape[0])

    def remove_oneshot(self, image: np.ndarray, config) -> np.ndarray:
        self.oneshot_calls += 1
        if self._oneshot_rgba is not None:
            return self._oneshot_rgba
        h, w = image.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = image
        rgba[..., 3] = (disc_alpha(h) * 255).astype(np.uint8)
        return rgba


class PredictOnlySegmenter:
    """A Segmenter that only implements ``predict`` (no one-shot support)."""

    def __init__(self, soft_alpha: np.ndarray) -> None:
        self._soft_alpha = soft_alpha

    def predict(self, image: np.ndarray) -> np.ndarray:
        return self._soft_alpha
