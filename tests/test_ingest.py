"""Ingest: decoding, canonicalization, and the pad box (characterization)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.config import Config
from app.ingest import IngestError, load_image, load_image_with_box
from tests.conftest import solid_rgb


def test_load_from_ndarray_no_canonical():
    arr = solid_rgb(10, 20)
    out = load_image(arr, Config(canonical_size=None))
    assert out.shape == (10, 20, 3)
    assert out.dtype == np.uint8


def test_grayscale_array_expands_to_rgb():
    gray = np.full((8, 8), 100, dtype=np.uint8)
    out = load_image(gray, Config(canonical_size=None))
    assert out.shape == (8, 8, 3)
    assert (out == 100).all()


def test_rgba_array_drops_alpha():
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    out = load_image(rgba, Config(canonical_size=None))
    assert out.shape == (8, 8, 3)
    np.testing.assert_array_equal(out[0, 0], [255, 0, 0])


def test_padding_centers_small_input():
    arr = solid_rgb(10, 20, (10, 20, 30))
    out, box = load_image_with_box(arr, Config(canonical_size=40, pad_color=(255, 255, 255)))
    assert out.shape == (40, 40, 3)
    left, top, right, bottom = box
    assert (right - left, bottom - top) == (20, 10)
    # Padded border is white, interior is the source colour.
    np.testing.assert_array_equal(out[0, 0], [255, 255, 255])
    np.testing.assert_array_equal(out[top + 1, left + 1], [10, 20, 30])


def test_restore_crop_roundtrips_padded_input():
    arr = solid_rgb(10, 20, (5, 6, 7))
    out, box = load_image_with_box(arr, Config(canonical_size=40))
    left, top, right, bottom = box
    cropped = out[top:bottom, left:right]
    assert cropped.shape == (10, 20, 3)


def test_oversized_input_is_downscaled():
    arr = solid_rgb(100, 50)
    out, box = load_image_with_box(arr, Config(canonical_size=40))
    assert out.shape == (40, 40, 3)
    # Longer edge mapped to canonical_size.
    left, top, right, bottom = box
    assert max(right - left, bottom - top) == 40


def test_empty_bytes_raises():
    with pytest.raises(IngestError):
        load_image(b"", Config())


def test_unsupported_type_raises():
    with pytest.raises(IngestError):
        load_image(12345, Config())  # type: ignore[arg-type]


def test_bad_array_dtype_raises():
    with pytest.raises(IngestError):
        load_image(np.zeros((4, 4, 3), dtype=np.float32), Config())
