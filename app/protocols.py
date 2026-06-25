"""Structural interfaces (seams) the pipeline depends on.

The pipeline never imports a concrete engine. It depends only on these
:class:`typing.Protocol` interfaces, so swapping the phase-1 CPU engine for the
phase-2 GPU engine is an implementation change behind a stable contract rather
than a rewrite.

Seams:
    Segmenter:      coarse subject/background separation -> soft alpha.
        phase 1 -> ``RembgSegmenter`` (BiRefNet ONNX via rembg).
        phase 2 -> ``TorchBiRefNetSegmenter`` (BiRefNet on torch/CUDA).
    MattingRefiner: sub-pixel alpha refinement from image + trimap.
        phase 1 -> ``PyMattingRefiner`` (pymatting closed-form matting).
        phase 2 -> ``ViTMatteRefiner`` (ViTMatte on torch/CUDA).

Array conventions (shared by every implementation):
    image:  ``np.ndarray`` of shape ``(H, W, 3)``, dtype ``uint8``, RGB, sRGB.
    alpha:  ``np.ndarray`` of shape ``(H, W)``, dtype ``float32``, range ``[0, 1]``.
    trimap: ``np.ndarray`` of shape ``(H, W)``, dtype ``float32``, values in
            ``{0.0, 0.5, 1.0}`` (background / unknown / foreground).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Segmenter(Protocol):
    """Coarse foreground segmentation seam.

    Implementations load their model once (at construction) and must be safe to
    reuse across many ``predict`` calls without reloading weights.
    """

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Estimate a soft foreground alpha for an image.

        Args:
            image: RGB image, shape ``(H, W, 3)``, dtype ``uint8``.

        Returns:
            Soft alpha matte, shape ``(H, W)``, dtype ``float32`` in ``[0, 1]``,
            where 1.0 is confident foreground and 0.0 confident background. The
            map is intentionally left soft (not thresholded) so downstream
            trimap synthesis can choose its own bands.
        """
        ...


@runtime_checkable
class OneShotSegmenter(Protocol):
    """One-shot removal seam: image straight to finished RGBA.

    A :class:`Segmenter` may *additionally* implement this when its engine can
    produce a matted RGBA in a single call (the rembg path). The pipeline checks
    for this capability structurally — never against a concrete class — so any
    future or test-injected one-shot engine is accepted on equal footing.
    """

    def remove_oneshot(self, image: np.ndarray, config: object) -> np.ndarray:
        """Produce finished RGBA from an RGB image in a single call.

        Args:
            image: RGB image, shape ``(H, W, 3)``, dtype ``uint8``.
            config: Pipeline configuration (the engine reads its own fields).

        Returns:
            RGBA image, shape ``(H, W, 4)``, dtype ``uint8``.
        """
        ...


@runtime_checkable
class MattingRefiner(Protocol):
    """Sub-pixel alpha refinement seam.

    Takes the full-resolution image plus a trimap and returns a refined alpha
    with clean, sub-pixel edges suitable for halo-free compositing.
    """

    def refine(self, image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """Refine alpha from an image and a trimap.

        Args:
            image: RGB image, shape ``(H, W, 3)``, dtype ``uint8``.
            trimap: Trimap, shape ``(H, W)``, dtype ``float32`` with values in
                ``{0.0, 0.5, 1.0}`` (background / unknown / foreground).

        Returns:
            Refined alpha matte, shape ``(H, W)``, dtype ``float32`` in
            ``[0, 1]``.
        """
        ...
