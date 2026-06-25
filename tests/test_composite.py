"""Compositing, PNG encoding, and checkerboard preview (characterization)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.composite import (
    composite_over_checkerboard,
    composite_rgba,
    encode_png,
    make_checkerboard,
)
from app.config import Config
from tests.conftest import disc_alpha, solid_rgb


def test_composite_rgba_straight_alpha():
    fg = solid_rgb(8, 8, (200, 100, 50)).astype(np.float32) / 255.0
    alpha = disc_alpha(8)
    out = composite_rgba(fg, alpha, Config())
    assert out.shape == (8, 8, 4)
    assert out.dtype == np.uint8
    # Straight alpha: RGB preserved regardless of alpha.
    np.testing.assert_array_equal(out[..., 3], (alpha * 255).astype(np.uint8))


def test_composite_rgba_flat_background_is_opaque():
    fg = solid_rgb(8, 8, (255, 0, 0)).astype(np.float32) / 255.0
    alpha = np.zeros((8, 8), dtype=np.float32)  # fully transparent fg
    cfg = Config(background_color=(0, 0, 255, 255))
    out = composite_rgba(fg, alpha, cfg)
    assert (out[..., 3] == 255).all()
    # With zero alpha everywhere the result is pure background blue.
    np.testing.assert_array_equal(out[0, 0, :3], [0, 0, 255])


def test_composite_rgba_premultiplied():
    fg = solid_rgb(4, 4, (255, 255, 255)).astype(np.float32) / 255.0
    alpha = np.full((4, 4), 0.5, dtype=np.float32)
    cfg = Config(premultiplied_output=True)
    out = composite_rgba(fg, alpha, cfg)
    # Premultiplied: rgb = fg * alpha = 0.5 -> ~128.
    assert abs(int(out[0, 0, 0]) - 128) <= 1


def test_encode_png_roundtrips():
    rgba = np.zeros((6, 6, 4), dtype=np.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = 255
    data = encode_png(rgba)
    decoded = np.array(Image.open(io.BytesIO(data)).convert("RGBA"))
    np.testing.assert_array_equal(decoded, rgba)


def test_encode_png_rejects_bad_shape():
    with pytest.raises(ValueError):
        encode_png(np.zeros((4, 4, 3), dtype=np.uint8))


def test_make_checkerboard_pattern():
    board = make_checkerboard(32, 32, square=16)
    assert board.shape == (32, 32, 3)
    # Opposite corners of a 2x2 square grid differ.
    assert not np.array_equal(board[0, 0], board[0, 16])


def test_checkerboard_rejects_bad_rgba():
    with pytest.raises(ValueError):
        composite_over_checkerboard(np.zeros((4, 4, 3), dtype=np.uint8))
