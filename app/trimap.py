"""Trimap synthesis: soft alpha -> {background, unknown, foreground}.

A matting refiner needs to be told which pixels are certainly foreground,
certainly background, and which are uncertain (the edge band it should solve
for). This module turns a soft segmentation alpha into that trimap.

The unknown band is produced by eroding both the confident-foreground and
confident-background regions, which widens the uncertain zone symmetrically
around the subject boundary by a configurable amount. Eroding (rather than only
keeping the naturally-soft zone) guarantees a usable band even when the input
mask is nearly binary.

This module is engine-independent: identical on CPU (phase 1) and GPU (phase 2).
"""

from __future__ import annotations

import cv2
import numpy as np

from app.config import Config

# Trimap label values (float convention shared across the pipeline).
TRIMAP_BG = 0.0
TRIMAP_UNKNOWN = 0.5
TRIMAP_FG = 1.0


def synthesize_trimap(soft_alpha: np.ndarray, config: Config) -> np.ndarray:
    """Build a trimap from a soft alpha map.

    Pixels at/above ``config.trimap_foreground_threshold`` are candidate
    foreground; pixels at/below ``config.trimap_background_threshold`` are
    candidate background. Both candidate regions are then eroded by a kernel
    sized to ``config.trimap_band_pct`` of the longer image edge, opening an
    unknown band of roughly twice that width around the boundary.

    Args:
        soft_alpha: Soft foreground probability, shape ``(H, W)``, dtype float,
            range ``[0, 1]``.
        config: Pipeline configuration (thresholds and band width).

    Returns:
        Trimap, shape ``(H, W)``, dtype ``float32`` with values in
        ``{0.0, 0.5, 1.0}`` (background / unknown / foreground).

    Raises:
        ValueError: If ``soft_alpha`` is not a 2-D array.
    """
    if soft_alpha.ndim != 2:
        raise ValueError(
            f"soft_alpha must be 2-D (H, W); got shape {soft_alpha.shape!r}."
        )

    alpha = soft_alpha.astype(np.float32, copy=False)
    height, width = alpha.shape

    foreground = (alpha >= config.trimap_foreground_threshold).astype(np.uint8)
    background = (alpha <= config.trimap_background_threshold).astype(np.uint8)

    # Kernel radius scaled to image size; at least 1px so a band always exists.
    radius = max(1, round(config.trimap_band_pct * max(height, width)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))

    sure_foreground = cv2.erode(foreground, kernel)
    sure_background = cv2.erode(background, kernel)

    trimap = np.full((height, width), TRIMAP_UNKNOWN, dtype=np.float32)
    trimap[sure_background.astype(bool)] = TRIMAP_BG
    trimap[sure_foreground.astype(bool)] = TRIMAP_FG
    return trimap
