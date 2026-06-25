"""Coarse segmentation (the Segmenter seam).

Phase 1 implements :class:`RembgSegmenter`, which wraps a single ``rembg``
session backed by a BiRefNet ONNX model on ONNX Runtime (CPU). The session is
created once at construction and reused for every image, so BiRefNet weights are
loaded once, not per request.

The same session serves both pipeline modes without ever running the model
twice for one image:

* ``predict`` returns a *soft* mask (``only_mask=True``) for the modular chain.
* ``remove_oneshot`` returns finished RGBA (``alpha_matting=True``) for the
  one-shot path.

:class:`TorchBiRefNetSegmenter` is a documented phase-2 stub. ``build_segmenter``
selects the implementation from ``config.engine``.
"""

from __future__ import annotations

import numpy as np

from rembg import remove, new_session

from app.config import Config
from app.protocols import Segmenter

# ONNX Runtime execution providers per device. Phase 1 is CPU-only; phase 2 on
# a CUDA box would install onnxruntime-gpu and use CUDAExecutionProvider, but
# the torch engine is the intended phase-2 path (see docs/ARCHITECTURE.md).
_PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
}


class SegmentationError(RuntimeError):
    """Raised when model loading or inference fails."""


class RembgSegmenter:
    """BiRefNet-via-rembg segmenter (phase 1, CPU ONNX).

    Implements :class:`~app.protocols.Segmenter`. Holds one ``rembg`` session
    for its lifetime; construct once and reuse.
    """

    def __init__(self, config: Config) -> None:
        """Create the rembg session for the configured model.

        Args:
            config: Pipeline configuration (``model_name`` and ``device``).

        Raises:
            SegmentationError: If the model cannot be loaded (e.g. unknown model
                name or a failed download).
        """
        # Imported lazily so the engine-independent modules and tests do not pay
        # rembg's heavy import (onnxruntime, scipy, ...) unless segmentation runs.

        self._config = config
        # Direct lookup, not a silent fallback: an unmapped device (e.g. a typo
        # like "gpu") must fail loudly instead of running on CPU unannounced.
        if config.device not in _PROVIDERS:
            raise SegmentationError(
                f"No ONNX execution-provider mapping for device {config.device!r}; "
                f"known devices: {sorted(_PROVIDERS)}."
            )
        providers = _PROVIDERS[config.device]
        try:
            self._session = new_session(config.model_name, providers=providers)
        except Exception as exc:
            raise SegmentationError(
                f"Failed to load rembg model {config.model_name!r}: {exc}"
            ) from exc

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Estimate a soft foreground alpha for an image.

        Uses ``only_mask=True`` so the raw model probability mask is returned
        (no post-processing, no thresholding), keeping it soft for downstream
        trimap synthesis.

        Args:
            image: RGB image, shape ``(H, W, 3)``, dtype ``uint8``.

        Returns:
            Soft alpha matte, shape ``(H, W)``, dtype ``float32`` in ``[0, 1]``.

        Raises:
            ValueError: If ``image`` is not an ``(H, W, 3)`` uint8 array.
            SegmentationError: If inference fails.
        """
        _validate_image(image)

        try:
            mask = remove(
                image,
                session=self._session,
                only_mask=True,
                post_process_mask=False,
            )
        except Exception as exc:
            raise SegmentationError(f"Segmentation inference failed: {exc}") from exc

        return _mask_to_alpha(mask, image.shape[:2])

    def remove_oneshot(self, image: np.ndarray, config: Config) -> np.ndarray:
        """Run rembg's built-in one-shot removal (alpha matting) to RGBA.

        Reuses the same session as :meth:`predict`, so BiRefNet runs once for
        this call. rembg internally performs closed-form matting plus pymatting
        foreground estimation, which is the halo-killer for this path.

        Args:
            image: RGB image, shape ``(H, W, 3)``, dtype ``uint8``.
            config: Pipeline configuration (alpha-matting thresholds, optional
                flat background colour).

        Returns:
            RGBA image, shape ``(H, W, 4)``, dtype ``uint8``.

        Raises:
            ValueError: If ``image`` is not an ``(H, W, 3)`` uint8 array.
            SegmentationError: If inference fails.
        """
        _validate_image(image)

        try:
            rgba = remove(
                image,
                session=self._session,
                alpha_matting=True,
                alpha_matting_foreground_threshold=config.alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=config.alpha_matting_background_threshold,
                alpha_matting_erode_size=config.alpha_matting_erode_size,
                bgcolor=config.background_color,
            )
        except Exception as exc:
            raise SegmentationError(f"One-shot removal failed: {exc}") from exc

        rgba = np.asarray(rgba, dtype=np.uint8)
        if rgba.ndim != 3 or rgba.shape[2] != 4:
            raise SegmentationError(
                f"Expected RGBA output, got shape {rgba.shape!r}."
            )
        return rgba


class TorchBiRefNetSegmenter:
    """BiRefNet on torch/CUDA (phase 2). Documented stub in phase 1.

    See ``docs/ARCHITECTURE.md`` section 8 for the implementation plan.
    """

    def __init__(self, config: Config) -> None:
        """Reject construction in phase 1.

        Raises:
            NotImplementedError: Always, in phase 1.
        """
        raise NotImplementedError(
            "TorchBiRefNetSegmenter is a phase-2 (GPU/torch) component. "
            "See docs/ARCHITECTURE.md section 8 for the migration plan."
        )

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Not implemented in phase 1.

        Raises:
            NotImplementedError: Always, in phase 1.
        """
        raise NotImplementedError("TorchBiRefNetSegmenter is a phase-2 component.")


def build_segmenter(config: Config) -> Segmenter:
    """Construct the segmenter implementation selected by configuration.

    Args:
        config: Pipeline configuration (``engine``).

    Returns:
        A :class:`~app.protocols.Segmenter` implementation.

    Raises:
        SegmentationError: If ``config.engine`` is ``"torch"`` (phase-2 only in
            this build).
        ValueError: If ``config.engine`` is unknown.
    """
    if config.engine == "rembg":
        return RembgSegmenter(config)
    if config.engine == "torch":
        raise SegmentationError(
            "engine='torch' is a phase-2 (GPU) component and is not available "
            "in this phase-1 build. Set engine='rembg' (the default), or "
            "implement the torch seam per docs/GUIDA_GPU.md."
        )
    raise ValueError(f"Unknown engine: {config.engine!r}")


def _validate_image(image: np.ndarray) -> None:
    """Validate that an array is an ``(H, W, 3)`` uint8 image.

    Args:
        image: Candidate image array.

    Raises:
        ValueError: If shape or dtype is invalid.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        shape = getattr(image, "shape", None)
        raise ValueError(f"image must be (H, W, 3) ndarray; got shape {shape!r}.")
    if image.dtype != np.uint8:
        raise ValueError(f"image must be dtype uint8; got {image.dtype}.")


def _mask_to_alpha(mask: np.ndarray, shape: tuple) -> np.ndarray:
    """Normalize a rembg mask to a float32 alpha in ``[0, 1]``.

    Args:
        mask: rembg mask output; ``(H, W)`` or ``(H, W, 1)``, dtype ``uint8``.
        shape: Expected ``(H, W)`` of the source image.

    Returns:
        Soft alpha, shape ``(H, W)``, dtype ``float32`` in ``[0, 1]``.

    Raises:
        SegmentationError: If the mask shape is unexpected.
    """
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if mask.shape != shape:
        raise SegmentationError(
            f"Mask shape {mask.shape!r} does not match image {shape!r}."
        )
    return (mask.astype(np.float32) / 255.0).clip(0.0, 1.0)
