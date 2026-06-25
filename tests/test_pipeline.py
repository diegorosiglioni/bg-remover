"""Pipeline orchestration with an injected fake segmenter (no model load)."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import Config
from app.pipeline import BackgroundRemover, PipelineError
from tests.conftest import FakeSegmenter, PredictOnlySegmenter, disc_alpha, solid_rgb


def test_oneshot_path_uses_segmenter():
    seg = FakeSegmenter()
    cfg = Config(canonical_size=None, pipeline_mode="rembg_oneshot")
    remover = BackgroundRemover(cfg, segmenter=seg)
    out = remover.process(solid_rgb(32, 32))
    assert out.shape == (32, 32, 4)
    assert seg.oneshot_calls == 1


def test_modular_path_runs_full_chain():
    size = 24
    seg = FakeSegmenter(soft_alpha=disc_alpha(size, radius_frac=0.35))
    cfg = Config(canonical_size=None, pipeline_mode="modular")
    remover = BackgroundRemover(cfg, segmenter=seg)
    out = remover.process(solid_rgb(size, size, (180, 90, 40)))
    assert out.shape == (size, size, 4)
    assert seg.predict_calls == 1


def test_oneshot_requires_oneshot_capable_segmenter():
    # A segmenter without remove_oneshot must be rejected for the oneshot mode,
    # via capability, not a concrete-class check (finding H4).
    seg = PredictOnlySegmenter(disc_alpha(16))
    cfg = Config(canonical_size=None, pipeline_mode="rembg_oneshot")
    remover = BackgroundRemover(cfg, segmenter=seg)
    with pytest.raises(PipelineError):
        remover.process(solid_rgb(16, 16))


def test_unknown_pipeline_mode_raises():
    seg = FakeSegmenter()
    cfg = Config(canonical_size=None)
    remover = BackgroundRemover(cfg, segmenter=seg)
    # Force an invalid mode past pydantic validation to exercise the guard.
    bad = cfg.model_copy()
    object.__setattr__(bad, "pipeline_mode", "nonsense")
    object.__setattr__(remover, "_config", bad)
    with pytest.raises(PipelineError):
        remover.process(solid_rgb(16, 16))


def test_restore_original_size_crops_back():
    size = 16
    seg = FakeSegmenter()
    cfg = Config(canonical_size=32, restore_original_size=True,
                 pipeline_mode="rembg_oneshot")
    remover = BackgroundRemover(cfg, segmenter=seg)
    out = remover.process(solid_rgb(size, size))
    # Output cropped back to the original (pre-pad) dimensions.
    assert out.shape[0] <= 32 and out.shape[1] <= 32
