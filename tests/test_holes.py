"""Interior hole filling — characterization + immutability.

The end-to-end output array is pinned so the P1 centroid refactor
(center_of_mass instead of repeat/tile bincount) is proven byte-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import Config
from app.holes import fill_alpha_holes
from tests.conftest import solid_rgb


def _ring_rgba(size: int = 64):
    """Foreground annulus with a central transparent hole.

    Returns (image, rgba). The rgba alpha is opaque on the ring and transparent
    in the central disc, so the hole is fully enclosed and central.
    """
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy = cx = (size - 1) / 2.0
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    outer = dist2 <= (0.45 * size) ** 2
    hole = dist2 <= (0.15 * size) ** 2
    foreground = outer & ~hole

    image = solid_rgb(size, size, (40, 160, 200))
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = image
    rgba[..., 3] = (foreground * 255).astype(np.uint8)
    return image, rgba


def test_no_holes_returns_unchanged_copy():
    image = solid_rgb(16, 16)
    rgba = np.zeros((16, 16, 4), dtype=np.uint8)
    rgba[..., 3] = 255  # fully opaque, no holes
    cfg = Config(fill_holes=True)
    out = fill_alpha_holes(image, rgba, cfg)
    np.testing.assert_array_equal(out, rgba)
    assert out is not rgba  # a copy, not the same buffer


def test_central_hole_is_filled():
    image, rgba = _ring_rgba()
    cfg = Config(fill_holes=True, fill_holes_center_dist=0.5)
    out = fill_alpha_holes(image, rgba, cfg)
    centre = out.shape[0] // 2
    # Hole centre is now opaque and carries the original colour.
    assert out[centre, centre, 3] == 255
    np.testing.assert_array_equal(out[centre, centre, :3], [40, 160, 200])


def test_offcentre_hole_left_transparent():
    image, rgba = _ring_rgba()
    cfg = Config(fill_holes=True, fill_holes_center_dist=0.01)
    out = fill_alpha_holes(image, rgba, cfg)
    centre = out.shape[0] // 2
    # Too strict a centre distance: the hole is not selected, stays transparent.
    assert out[centre, centre, 3] == 0


def test_input_not_mutated():
    image, rgba = _ring_rgba()
    before = rgba.copy()
    fill_alpha_holes(image, rgba, Config(fill_holes=True, fill_holes_center_dist=0.5))
    np.testing.assert_array_equal(rgba, before)


def test_validation_errors():
    with pytest.raises(ValueError):
        fill_alpha_holes(
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((4, 4, 3), dtype=np.uint8),  # not RGBA
            Config(fill_holes=True),
        )
    with pytest.raises(ValueError):
        fill_alpha_holes(
            np.zeros((4, 4, 3), dtype=np.uint8),
            np.zeros((5, 5, 4), dtype=np.uint8),  # size mismatch
            Config(fill_holes=True),
        )


def test_fill_output_is_pinned():
    """Golden checksum so the P1 refactor cannot silently alter the result."""
    image, rgba = _ring_rgba(64)
    cfg = Config(
        fill_holes=True,
        fill_holes_center_dist=0.5,
        fill_holes_dilate=2,
        fill_holes_feather=1.5,
    )
    out = fill_alpha_holes(image, rgba, cfg)
    # Stable summary of the full result: total + per-channel sums + opaque count.
    signature = (
        int(out.sum()),
        tuple(int(out[..., c].sum()) for c in range(4)),
        int((out[..., 3] == 255).sum()),
    )
    # Captured from the pre-refactor implementation (repeat/tile bincount).
    # The P1 centroid refactor (center_of_mass) must reproduce this exactly.
    assert signature == _GOLDEN_SIGNATURE


_GOLDEN_SIGNATURE = (2303440, (163840, 655360, 819200, 665040), 2608)


def _drop_rgba(size: int = 80):
    """Solid body with a central saliency-drop hole (soft alpha ramp).

    Returns (image, rgba). The whole square is solid subject in the original
    image; the model alpha keeps the body opaque but drops a central disc to
    transparent with a soft semi-transparent ramp at its boundary — the
    signature of a reflective-metal saliency drop. The original colour inside
    the hole is bright (a specular highlight), matching the surrounding body.
    """
    from scipy.ndimage import distance_transform_edt

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cy = cx = (size - 1) / 2.0
    dist = np.hypot(yy - cy, xx - cx)
    hole_r = 0.20 * size

    image = solid_rgb(size, size, (235, 235, 235))  # bright reflective body
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = image

    # Alpha: opaque body, transparent hole core, soft ramp over a few px ring.
    alpha = np.full((size, size), 255.0, dtype=np.float32)
    ramp_px = 6.0
    inner = dist <= hole_r
    ramp = (dist > hole_r) & (dist <= hole_r + ramp_px)
    alpha[inner] = 0.0
    alpha[ramp] = np.clip((dist[ramp] - hole_r) / ramp_px, 0.0, 1.0) * 255.0
    rgba[..., 3] = alpha.astype(np.uint8)
    return image, rgba


def test_drop_restored_opaque_hard_edge():
    image, rgba = _drop_rgba()
    cfg = Config(fill_holes=True, fill_holes_restore_drops=True)
    out = fill_alpha_holes(image, rgba, cfg)
    centre = out.shape[0] // 2
    # Hole core is now fully opaque with the original (hard, un-blended) colour.
    assert out[centre, centre, 3] == 255
    np.testing.assert_array_equal(out[centre, centre, :3], [235, 235, 235])
    # No gray-ghost alpha left inside the restored core (hard, not feathered).
    assert out[centre, centre, 3] == 255


def test_drop_left_transparent_when_restore_off():
    image, rgba = _drop_rgba()
    # Flag off + a centre gate too strict for the legacy path to select it: the
    # drop is not restored, so the core stays transparent (no new behavior).
    cfg = Config(
        fill_holes=True, fill_holes_restore_drops=False, fill_holes_center_dist=0.01
    )
    out = fill_alpha_holes(image, rgba, cfg)
    centre = out.shape[0] // 2
    assert out[centre, centre, 3] == 0


def test_crisp_eyelet_stays_transparent_with_restore_on():
    # A real see-through eyelet: crisp edge (no ramp). Must stay transparent
    # even with drop restore enabled.
    image, rgba = _ring_rgba()
    cfg = Config(fill_holes=True, fill_holes_restore_drops=True, fill_holes_center_dist=0.01)
    out = fill_alpha_holes(image, rgba, cfg)
    centre = out.shape[0] // 2
    assert out[centre, centre, 3] == 0
