"""Sub-pixel alpha refinement (the MattingRefiner seam).

Phase 1 implements :class:`PyMattingRefiner` using pymatting's closed-form
matting (``estimate_alpha_cf``), which solves for a smooth, sub-pixel alpha in
the trimap's unknown band given the image as guidance. This is engine-agnostic
CPU code and is exactly the step phase 2 replaces with ViTMatte behind the same
:class:`~app.protocols.MattingRefiner` contract.

:class:`ViTMatteRefiner` is a documented phase-2 stub: it raises
``NotImplementedError`` pointing at ``docs/ARCHITECTURE.md`` so the seam is
visible and callers fail loudly rather than silently degrading.
"""

from __future__ import annotations

import numpy as np
from pymatting import estimate_alpha_cf

from app.config import Config


class MattingError(RuntimeError):
    """Raised when alpha refinement fails."""


class PyMattingRefiner:
    """Closed-form matting refiner (phase 1, CPU).

    Implements :class:`~app.protocols.MattingRefiner`.
    """

    def __init__(self, config: Config) -> None:
        """Initialize the refiner.

        Args:
            config: Pipeline configuration. Retained for parity with other
                refiners and future tuning; closed-form matting itself needs no
                runtime parameters here.
        """
        self._config = config

    def refine(self, image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """Refine alpha from an image and trimap via closed-form matting.

        Args:
            image: RGB image, shape ``(H, W, 3)``, dtype ``uint8``.
            trimap: Trimap, shape ``(H, W)``, dtype float with values in
                ``{0.0, 0.5, 1.0}``.

        Returns:
            Refined alpha matte, shape ``(H, W)``, dtype ``float32`` in
            ``[0, 1]``.

        Raises:
            ValueError: If image/trimap shapes are inconsistent or malformed.
            MattingError: If the underlying solver fails.
        """
        _validate(image, trimap)

        image_norm = image.astype(np.float64) / 255.0
        trimap_norm = trimap.astype(np.float64)

        try:
            alpha = estimate_alpha_cf(image_norm, trimap_norm)
        except Exception as exc:  # pymatting raises various solver errors
            raise MattingError(f"Closed-form matting failed: {exc}") from exc

        return np.clip(alpha, 0.0, 1.0).astype(np.float32)


class ViTMatteRefiner:
    """ViTMatte refiner (phase 2, GPU). Documented stub in phase 1.

    Implements :class:`~app.protocols.MattingRefiner` once wired. See
    ``docs/ARCHITECTURE.md`` section 8 for the implementation plan.
    """

    def __init__(self, config: Config) -> None:
        """Reject construction in phase 1.

        Raises:
            NotImplementedError: Always, in phase 1.
        """
        raise NotImplementedError(
            "ViTMatteRefiner is a phase-2 (GPU/torch) component. "
            "See docs/ARCHITECTURE.md section 8 for the migration plan."
        )

    def refine(self, image: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        """Not implemented in phase 1.

        Raises:
            NotImplementedError: Always, in phase 1.
        """
        raise NotImplementedError("ViTMatteRefiner is a phase-2 component.")


def _validate(image: np.ndarray, trimap: np.ndarray) -> None:
    """Validate image/trimap shape and dtype agreement.

    Args:
        image: Candidate RGB image array.
        trimap: Candidate trimap array.

    Raises:
        ValueError: If shapes or dtype are invalid.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be (H, W, 3); got {image.shape!r}.")
    if image.dtype != np.uint8:
        raise ValueError(f"image must be dtype uint8; got {image.dtype}.")
    if trimap.ndim != 2:
        raise ValueError(f"trimap must be (H, W); got {trimap.shape!r}.")
    if trimap.shape != image.shape[:2]:
        raise ValueError(
            f"trimap shape {trimap.shape!r} does not match image "
            f"{image.shape[:2]!r}."
        )
