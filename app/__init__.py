"""High-precision white-background removal.

Phase 1 (current): CPU-only, rembg + ONNX Runtime, importable library + CLI.
Phase 2 (later): GPU torch BiRefNet + ViTMatte behind the same seams, served
over FastAPI. See ``docs/ARCHITECTURE.md``.

The stable public surface for embedding apps is :func:`remove_background`,
:class:`BackgroundRemover`, and :class:`Config`. Engine seams are exposed for
testing and phase-2 implementers.
"""

from __future__ import annotations

from app.config import Config
from app.pipeline import (
    BackgroundRemover,
    PipelineError,
    clear_segmenter_cache,
    remove_background,
)
from app.protocols import MattingRefiner, Segmenter

__all__ = [
    "Config",
    "Segmenter",
    "MattingRefiner",
    "remove_background",
    "BackgroundRemover",
    "PipelineError",
    "clear_segmenter_cache",
]

__version__ = "0.1.0"
