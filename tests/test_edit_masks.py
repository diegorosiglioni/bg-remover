"""Mask-editor logic that does not require a display.

The interactive loop (windowing/keys) is not exercised here; the drawing model,
mask I/O, and image decode are.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from edit_masks import (
    _EditState,
    _Viewport,
    _fit_scale,
    _imread_unicode,
    _load_initial_mask,
    _save,
    _snap_45,
    _wheel_delta,
)
from app.masks import save_mask
from tests.conftest import solid_rgb


def _state(h: int = 20, w: int = 20) -> _EditState:
    disp = solid_rgb(h, w, (30, 40, 50))[:, :, ::-1].copy()  # BGR
    return _EditState(disp, np.zeros((h, w), dtype=np.uint8))


def test_fill_encloses_drawn_boundary():
    state = _state()
    # Draw a rough rectangular boundary with a small brush.
    state.brush = 1
    for x in range(4, 16):
        state.paint(x, 4, erase=False)
        state.paint(x, 15, erase=False)
    for y in range(4, 16):
        state.paint(4, y, erase=False)
        state.paint(15, y, erase=False)

    state.fill()
    assert state.mask[10, 10] == 255  # interior filled
    assert state.mask[0, 0] == 0  # outside untouched


def test_erase_removes_from_mask():
    state = _state()
    state.brush = 2
    state.mask[8:13, 8:13] = 255
    state.paint(10, 10, erase=True)
    assert state.mask[10, 10] == 0


def test_draw_paints_directly_into_mask():
    state = _state()
    state.brush = 2
    state.paint(10, 10, erase=False)
    assert state.mask[10, 10] == 255  # brush adds to the mask immediately


def test_clear_and_reset():
    disp = solid_rgb(10, 10, (1, 2, 3))[:, :, ::-1].copy()
    initial = np.zeros((10, 10), dtype=np.uint8)
    initial[2:8, 2:8] = 255
    state = _EditState(disp, initial)

    state.paint(5, 5, erase=False)
    state.clear()
    assert not state.mask.any()

    state.reset()
    assert np.array_equal(state.mask, initial)


def test_load_initial_mask_absent_is_zero(tmp_path: Path):
    src = tmp_path / "in" / "obj" / "a.jpg"
    mask = _load_initial_mask(src, tmp_path / "in", tmp_path / "masks", (12, 8))
    assert mask.shape == (8, 12)
    assert not mask.any()


def test_load_initial_mask_resizes_existing(tmp_path: Path):
    src = tmp_path / "in" / "obj" / "a.jpg"
    save_mask(np.ones((4, 4), dtype=bool), tmp_path / "masks" / "obj" / "a_mask.png")
    mask = _load_initial_mask(src, tmp_path / "in", tmp_path / "masks", (10, 6))
    assert mask.shape == (6, 10)
    assert (mask == 255).all()


def test_save_upscales_to_native(tmp_path: Path):
    src = tmp_path / "in" / "obj" / "a.jpg"
    state = _state(5, 5)
    state.mask[:] = 255
    _save(state, src, tmp_path / "in", tmp_path / "masks", (10, 8))

    out = tmp_path / "masks" / "obj" / "a_mask.png"
    gray = np.asarray(Image.open(out).convert("L"))
    assert gray.shape == (8, 10)
    assert (gray == 255).all()


def test_imread_unicode_round_trip(tmp_path: Path):
    path = tmp_path / "pic.png"
    Image.fromarray(solid_rgb(6, 6, (200, 100, 50)), mode="RGB").save(path)
    bgr = _imread_unicode(path)
    assert bgr is not None
    assert bgr.shape == (6, 6, 3)


def test_fit_scale_fits_within_screen():
    assert _fit_scale((1000, 500), (1920, 1080)) <= 1.912
    assert _fit_scale((4000, 2000), (1920, 1080)) < 1.0


def test_snap_45_snaps_to_axes_and_diagonal():
    # Mostly-horizontal drag snaps to pure horizontal (same length).
    assert _snap_45((0, 0), (10, 2)) == (10, 0)
    # Mostly-vertical snaps to pure vertical.
    assert _snap_45((0, 0), (1, 10)) == (0, 10)
    # ~45° drag snaps to the diagonal.
    x, y = _snap_45((0, 0), (10, 9))
    assert x == y and x > 0


def test_wheel_delta_sign():
    # Wheel up: positive delta in the high word.
    assert _wheel_delta(120 << 16) > 0
    # Wheel down: negative (signed short) delta.
    assert _wheel_delta((-120 & 0xFFFF) << 16) < 0


def test_viewport_to_image_and_zoom_keeps_point_fixed():
    vp = _Viewport((100, 100), (200, 200))
    # The image point under a cursor stays put across a zoom.
    before = vp.to_image(120, 80)
    vp.zoom_at(120, 80, 2.0)
    after = vp.to_image(120, 80)
    assert abs(before[0] - after[0]) <= 1
    assert abs(before[1] - after[1]) <= 1


def test_viewport_pan_shifts_origin():
    vp = _Viewport((100, 100), (200, 200))
    x0, _ = vp.to_image(0, 0)
    vp.pan(10, 0)  # drag content right -> view origin moves left in image space
    x1, _ = vp.to_image(0, 0)
    assert x1 < x0


def test_undo_restores_previous_stroke():
    state = _state()
    snapshot = state.mask.copy()
    state.push_undo()
    state.mask[5:10, 5:10] = 255
    assert state.mask.any()
    state.undo()
    assert np.array_equal(state.mask, snapshot)


def test_draw_segment_paints_a_line():
    state = _state()
    state.brush = 1
    state.draw_segment((2, 2), (2, 17), erase=False)
    assert state.mask[10, 2] == 255  # midpoint of the vertical segment
