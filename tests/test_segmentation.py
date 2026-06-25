"""Segmentation helpers and engine selection (no real model load)."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import Config
from app.segmentation import (
    SegmentationError,
    _mask_to_alpha,
    _validate_image,
    build_segmenter,
)


def test_mask_to_alpha_normalizes_and_clips():
    mask = np.array([[0, 128, 255]], dtype=np.uint8)
    alpha = _mask_to_alpha(mask, (1, 3))
    assert alpha.dtype == np.float32
    assert alpha.shape == (1, 3)
    np.testing.assert_allclose(alpha, [[0.0, 128 / 255, 1.0]], atol=1e-6)


def test_mask_to_alpha_squeezes_trailing_channel():
    mask = np.zeros((2, 2, 1), dtype=np.uint8)
    alpha = _mask_to_alpha(mask, (2, 2))
    assert alpha.shape == (2, 2)


def test_mask_to_alpha_shape_mismatch_raises():
    with pytest.raises(SegmentationError):
        _mask_to_alpha(np.zeros((2, 2), dtype=np.uint8), (3, 3))


def test_validate_image_rejects_bad_shape_and_dtype():
    with pytest.raises(ValueError):
        _validate_image(np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError):
        _validate_image(np.zeros((4, 4, 3), dtype=np.float32))


def test_build_segmenter_torch_raises():
    with pytest.raises(SegmentationError):
        build_segmenter(Config(engine="torch"))
