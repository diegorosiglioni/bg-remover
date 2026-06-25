"""Matting refiner and defringe on tiny images (solver runs, but fast)."""

from __future__ import annotations

import numpy as np
import pytest

from app.composite import _normalize_foreground  # noqa: F401  (import smoke)
from app.config import Config
from app.defringe import defringe
from app.matting import PyMattingRefiner
from app.trimap import synthesize_trimap
from tests.conftest import disc_alpha, solid_rgb


def test_refiner_validation():
    refiner = PyMattingRefiner(Config())
    with pytest.raises(ValueError):
        refiner.refine(np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.float32))
    with pytest.raises(ValueError):
        refiner.refine(
            np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((5, 5), dtype=np.float32)
        )


def test_refiner_returns_float32_alpha_in_range():
    size = 24
    image = solid_rgb(size, size, (180, 60, 60))
    alpha = disc_alpha(size, radius_frac=0.35)
    trimap = synthesize_trimap(alpha, Config())
    out = PyMattingRefiner(Config()).refine(image, trimap)
    assert out.dtype == np.float32
    assert out.shape == (size, size)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_defringe_validation():
    with pytest.raises(ValueError):
        defringe(
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((5, 5), dtype=np.float32),
            Config(),
        )


def test_defringe_returns_clean_foreground_and_alpha():
    size = 24
    image = solid_rgb(size, size, (200, 200, 200))
    alpha = disc_alpha(size, radius_frac=0.35)
    fg, alpha_out = defringe(image, alpha, Config())
    assert fg.dtype == np.float32 and fg.shape == (size, size, 3)
    assert alpha_out.dtype == np.float32 and alpha_out.shape == (size, size)
    assert fg.min() >= 0.0 and fg.max() <= 1.0
    np.testing.assert_array_equal(alpha_out, alpha)  # no guided filter by default
