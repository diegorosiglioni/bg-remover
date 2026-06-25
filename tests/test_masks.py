"""Grayscale mask storage: path mirroring, round-trip, and resize."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.masks import load_mask, mask_path, resize_mask, save_mask


def test_mask_path_mirrors_tree_with_suffix():
    src = Path("input/object_A/img1.jpg")
    root = Path("input")
    masks = Path("masks")
    assert mask_path(src, root, masks) == Path("masks/object_A/img1_mask.png")


def test_mask_path_flat_when_outside_root():
    src = Path("/somewhere/else/pic.png")
    assert mask_path(src, Path("input"), Path("masks")) == Path("masks/pic_mask.png")


def test_save_then_load_round_trip(tmp_path: Path):
    mask = np.zeros((4, 6), dtype=bool)
    mask[1:3, 2:5] = True
    dest = tmp_path / "sub" / "m_mask.png"

    save_mask(mask, dest)
    assert dest.exists()
    # White = keep: stored PNG is 255 inside, 0 outside.
    gray = np.asarray(Image.open(dest).convert("L"))
    assert gray[1, 2] == 255
    assert gray[0, 0] == 0
    # Round-trips back to the same boolean mask.
    assert np.array_equal(load_mask(dest), mask)


def test_resize_mask_nearest_preserves_shape():
    mask = np.ones((4, 4), dtype=bool)
    resized = resize_mask(mask, (8, 6))  # (width, height)
    assert resized.shape == (6, 8)
    assert resized.all()
