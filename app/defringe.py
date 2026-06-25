"""Defringe: recover true foreground colour so edges carry no white halo.

This is the step that actually kills the white fringe (see ``docs/ARCHITECTURE.md``
section 1). Every edge pixel of a subject shot on white satisfies::

    observed = alpha * F + (1 - alpha) * white

Compositing the *observed* RGB over a new background leaves the
``(1 - alpha) * white`` term behind, which glows as a halo over dark
backgrounds. The fix is to solve for the true foreground colour ``F`` and
composite that instead.

Two steps, matching the design brief:
    a. Optional edge-aware smoothing of the alpha via a guided filter (the
       original image as guide). Off by default; enable only if halos persist.
    b. Foreground colour estimation via pymatting ``estimate_foreground_ml``.

Engine-independent: identical on CPU (phase 1) and GPU (phase 2).
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from pymatting import estimate_foreground_ml

from app.config import Config


class DefringeError(RuntimeError):
    """Raised when foreground colour estimation fails."""


def defringe(
    image: np.ndarray, alpha: np.ndarray, config: Config
) -> Tuple[np.ndarray, np.ndarray]:
    """Recover foreground colour and optionally refine alpha edges.

    Args:
        image: RGB image, shape ``(H, W, 3)``, dtype ``uint8``.
        alpha: Alpha matte, shape ``(H, W)``, dtype float, range ``[0, 1]``.
        config: Pipeline configuration (guided-filter flag and parameters,
            foreground-estimation regularization).

    Returns:
        Tuple ``(foreground, alpha_out)`` where ``foreground`` is the recovered
        true foreground colour as ``float32`` ``(H, W, 3)`` in ``[0, 1]`` and
        ``alpha_out`` is the (possibly guided-filter-refined) alpha as
        ``float32`` ``(H, W)`` in ``[0, 1]``.

    Raises:
        ValueError: If image/alpha shapes are inconsistent or malformed.
        DefringeError: If the foreground-estimation solver fails.
    """
    _validate(image, alpha)

    image_norm = image.astype(np.float64) / 255.0
    alpha_out = alpha.astype(np.float32, copy=True)

    # Step (a): optional edge-aware alpha smoothing guided by the image.
    if config.guided_filter_enabled:
        alpha_out = _guided_refine(image, alpha_out, config)

    # Step (b): solve for true foreground colour F (halo-killer).
    try:
        foreground = estimate_foreground_ml(
            image_norm,
            alpha_out.astype(np.float64),
            regularization=config.pymatting_regularization,
        )
    except Exception as exc:  # pymatting/scipy solver errors, MemoryError, ...
        raise DefringeError(f"Foreground estimation failed: {exc}") from exc
    foreground = np.clip(foreground, 0.0, 1.0).astype(np.float32)
    return foreground, alpha_out


def _guided_refine(
    image: np.ndarray, alpha: np.ndarray, config: Config
) -> np.ndarray:
    """Edge-aware smoothing of alpha using the image as guide.

    Args:
        image: RGB guide image, shape ``(H, W, 3)``, dtype ``uint8``.
        alpha: Alpha matte, shape ``(H, W)``, dtype ``float32`` in ``[0, 1]``.
        config: Pipeline configuration (guided-filter radius and eps).

    Returns:
        Refined alpha, shape ``(H, W)``, dtype ``float32`` in ``[0, 1]``.
    """
    guide = image.astype(np.float32) / 255.0
    refined = cv2.ximgproc.guidedFilter(
        guide=guide,
        src=alpha,
        radius=config.guided_filter_radius,
        eps=config.guided_filter_eps,
    )
    return np.clip(refined, 0.0, 1.0).astype(np.float32)


def _validate(image: np.ndarray, alpha: np.ndarray) -> None:
    """Validate image/alpha shape and dtype agreement.

    Args:
        image: Candidate RGB image array.
        alpha: Candidate alpha array.

    Raises:
        ValueError: If shapes or dtype are invalid.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3); got {image.shape!r}.")
    if image.dtype != np.uint8:
        raise ValueError(f"image must be dtype uint8; got {image.dtype}.")
    if alpha.ndim != 2:
        raise ValueError(f"alpha must be (H, W); got {alpha.shape!r}.")
    if alpha.shape != image.shape[:2]:
        raise ValueError(
            f"alpha shape {alpha.shape!r} does not match image "
            f"{image.shape[:2]!r}."
        )
