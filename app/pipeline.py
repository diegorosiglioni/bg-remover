"""Pipeline orchestration: the single public entrypoint.

``remove_background(source, config)`` is the only function an embedding app
needs. It decodes the input, runs the configured pipeline mode, and returns an
RGBA array:

* ``rembg_oneshot`` (default): one rembg call with built-in alpha matting.
* ``modular``: segment -> trimap -> matting refine -> defringe -> composite,
  the engine-independent chain that phase 2 reuses unchanged.

Heavy model sessions are expensive to build, so segmenters are cached by their
identifying configuration (engine, device, model name). Repeated
``remove_background`` calls with the same model reuse one loaded session.
:class:`BackgroundRemover` exposes the same behaviour as a reusable object (used
by the CLI) and accepts an injected segmenter for testing.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import numpy as np

from app.composite import composite_rgba
from app.config import Config
from app.defringe import defringe
from app.holes import fill_alpha_holes
from app.ingest import ImageSource, load_image_with_box
from app.matting import PyMattingRefiner
from app.protocols import MattingRefiner, OneShotSegmenter, Segmenter
from app.segmentation import build_segmenter
from app.trimap import synthesize_trimap

# Cache of loaded segmenters keyed by the config fields that determine the
# loaded model. Keeps repeated remove_background calls from reloading weights.
_SegmenterKey = Tuple[str, str, str]
_SEGMENTER_CACHE: Dict[_SegmenterKey, Segmenter] = {}
# Guards the cache so concurrent remove_background calls do not each build a
# (costly) duplicate model session for the same key.
_SEGMENTER_CACHE_LOCK = threading.Lock()


class PipelineError(RuntimeError):
    """Raised when the pipeline cannot complete for a non-input reason."""


def remove_background(
    source: ImageSource, config: Optional[Config] = None
) -> np.ndarray:
    """Remove the background from an image and return RGBA.

    This is the stable public entrypoint. The underlying segmenter is cached by
    model identity, so calling this repeatedly with the same configuration does
    not reload the model.

    Args:
        source: Any source accepted by :func:`app.ingest.load_image` (path,
            bytes, PIL image, or ndarray).
        config: Pipeline configuration. Defaults to :class:`Config` defaults
            (rembg engine, CPU, ``birefnet-general``, one-shot mode).

    Returns:
        RGBA image, shape ``(H, W, 4)``, dtype ``uint8``.

    Raises:
        IngestError: If the input cannot be decoded.
        SegmentationError: If model loading or inference fails.
        PipelineError: If the configuration is internally inconsistent.
    """
    config = config or Config()
    segmenter = _get_cached_segmenter(config)
    return BackgroundRemover(config, segmenter=segmenter).process(source)


def clear_segmenter_cache() -> None:
    """Drop all cached segmenters (releasing their model sessions)."""
    with _SEGMENTER_CACHE_LOCK:
        _SEGMENTER_CACHE.clear()


def _get_cached_segmenter(config: Config) -> Segmenter:
    """Return a cached segmenter for the config's model, building it if needed.

    Args:
        config: Pipeline configuration.

    Returns:
        A shared :class:`~app.protocols.Segmenter` for the configured model.
    """
    key: _SegmenterKey = (config.engine, config.device, config.model_name)
    with _SEGMENTER_CACHE_LOCK:
        segmenter = _SEGMENTER_CACHE.get(key)
        if segmenter is None:
            segmenter = build_segmenter(config)
            _SEGMENTER_CACHE[key] = segmenter
        return segmenter


class BackgroundRemover:
    """Reusable background remover holding a segmenter and refiner.

    Construct once and call :meth:`process` for each image to amortize model
    loading across a batch (see ``app/cli.py``).
    """

    def __init__(
        self, config: Config, segmenter: Optional[Segmenter] = None
    ) -> None:
        """Initialize the remover.

        Args:
            config: Pipeline configuration.
            segmenter: Optional pre-built segmenter to reuse (e.g. a shared or
                test-injected instance). When ``None``, one is built from the
                config.
        """
        self._config = config
        self._segmenter: Segmenter = segmenter or build_segmenter(config)
        self._refiner: MattingRefiner = PyMattingRefiner(config)

    def process(self, source: ImageSource) -> np.ndarray:
        """Run the configured pipeline on one image.

        Args:
            source: Any source accepted by :func:`app.ingest.load_image`.

        Returns:
            RGBA image, shape ``(H, W, 4)``, dtype ``uint8``.

        Raises:
            IngestError: If the input cannot be decoded.
            SegmentationError: If inference fails.
            PipelineError: If the configuration is internally inconsistent.
        """
        image, pad_box = load_image_with_box(source, self._config)

        if self._config.pipeline_mode == "rembg_oneshot":
            rgba = self._run_oneshot(image)
        elif self._config.pipeline_mode == "modular":
            rgba = self._run_modular(image)
        else:
            raise PipelineError(
                f"Unknown pipeline_mode: {self._config.pipeline_mode!r}."
            )

        # Repair saliency drops before cropping so the fill runs on the full
        # canonical frame (and against the canonical source image).
        if self._config.fill_holes:
            rgba = fill_alpha_holes(image, rgba, self._config)

        if self._config.restore_original_size and pad_box is not None:
            left, top, right, bottom = pad_box
            rgba = rgba[top:bottom, left:right]
        return rgba

    def _run_oneshot(self, image: np.ndarray) -> np.ndarray:
        """Run the rembg one-shot path.

        Args:
            image: Canonical RGB image, ``(H, W, 3)`` uint8.

        Returns:
            RGBA image, ``(H, W, 4)`` uint8.

        Raises:
            PipelineError: If the active segmenter does not support one-shot
                removal (e.g. the phase-2 torch engine).
        """
        if not isinstance(self._segmenter, OneShotSegmenter):
            raise PipelineError(
                "pipeline_mode 'rembg_oneshot' requires a one-shot-capable "
                f"segmenter; {type(self._segmenter).__name__} has no "
                "remove_oneshot. Use 'modular' instead."
            )
        return self._segmenter.remove_oneshot(image, self._config)

    def _run_modular(self, image: np.ndarray) -> np.ndarray:
        """Run the engine-independent modular chain.

        Args:
            image: Canonical RGB image, ``(H, W, 3)`` uint8.

        Returns:
            RGBA image, ``(H, W, 4)`` uint8.
        """
        soft_alpha = self._segmenter.predict(image)
        trimap = synthesize_trimap(soft_alpha, self._config)
        alpha = self._refiner.refine(image, trimap)
        foreground, alpha = defringe(image, alpha, self._config)
        return composite_rgba(foreground, alpha, self._config)
