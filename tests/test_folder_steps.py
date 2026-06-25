"""Decoupled pipeline steps: generate_masks then apply_masks.

The AI engine is an injected fake (no model loads). Covers per-folder dispatch,
the masks/ mirror layout, the apply cut, missing-mask handling, and mask resize.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.folder_pipeline import apply_masks, generate_masks
from app.masks import save_mask
from tests.conftest import solid_rgb


class FakeRemover:
    """Remover whose alpha is a centred square, recording processed paths."""

    def __init__(self) -> None:
        self.processed: list[Path] = []

    def process(self, source) -> np.ndarray:
        self.processed.append(Path(source))
        rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
        h, w = rgb.shape[:2]
        alpha = np.zeros((h, w), dtype=np.uint8)
        alpha[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 255
        return np.dstack([rgb, alpha])


def _write_png(path: Path, color=(10, 20, 30), size=(8, 8)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(solid_rgb(size[1], size[0], color), mode="RGB").save(path)


def _build_tree(root: Path) -> None:
    _write_png(root / "object_A" / "1.jpg")
    _write_png(root / "object_A" / "2.jpg")
    _write_png(root / "object_B_pendant" / "1.jpg")
    _write_png(root / "object_B_pendant" / "2.jpg")


def test_generate_masks_dispatch_and_layout(tmp_path: Path):
    root = tmp_path / "in"
    masks = tmp_path / "masks"
    _build_tree(root)
    remover = FakeRemover()

    ok, fail, skip = generate_masks(root, masks, remover)
    assert (ok, fail, skip) == (4, 0, 0)
    # pendant: 2 inferences; masked: 1 inference (its 2nd image) -> 3 total.
    assert len(remover.processed) == 3

    # Mask files mirror the tree with a _mask suffix.
    for rel in ("object_A/1_mask.png", "object_A/2_mask.png",
                "object_B_pendant/1_mask.png", "object_B_pendant/2_mask.png"):
        mpath = masks / rel
        assert mpath.exists()
        gray = np.asarray(Image.open(mpath).convert("L"))
        # White centre (keep), black corner (remove).
        assert gray[4, 4] == 255
        assert gray[0, 0] == 0


def test_apply_masks_round_trip(tmp_path: Path):
    root = tmp_path / "in"
    masks = tmp_path / "masks"
    out = tmp_path / "out"
    _build_tree(root)
    generate_masks(root, masks, FakeRemover())

    ok, fail, skip = apply_masks(root, out, masks)
    assert (ok, fail, skip) == (4, 0, 0)
    png = out / "object_A" / "1.png"
    assert png.exists()
    rgba = np.asarray(Image.open(png).convert("RGBA"))
    assert rgba[4, 4, 3] == 255  # kept
    assert rgba[0, 0, 3] == 0  # removed
    assert tuple(rgba[4, 4, :3]) == (10, 20, 30)  # RGB untouched inside mask


def test_apply_masks_missing_mask_fails(tmp_path: Path):
    root = tmp_path / "in"
    masks = tmp_path / "masks"
    out = tmp_path / "out"
    _write_png(root / "object_A" / "1.jpg")
    masks.mkdir()  # exists but empty -> no mask for the image

    ok, fail, skip = apply_masks(root, out, masks)
    assert (ok, fail, skip) == (0, 1, 0)
    assert not list(out.rglob("*.png"))


def test_apply_masks_resizes_mask_to_source(tmp_path: Path):
    root = tmp_path / "in"
    masks = tmp_path / "masks"
    out = tmp_path / "out"
    _write_png(root / "object_A" / "1.jpg", size=(8, 8))
    # Hand-authored mask at a *different* size than the source (4x4 all-keep).
    save_mask(np.ones((4, 4), dtype=bool), masks / "object_A" / "1_mask.png")

    ok, fail, skip = apply_masks(root, out, masks)
    assert (ok, fail, skip) == (1, 0, 0)
    rgba = np.asarray(Image.open(out / "object_A" / "1.png").convert("RGBA"))
    assert rgba.shape == (8, 8, 4)
    assert (rgba[..., 3] == 255).all()  # resized 4x4 keep-all -> full 8x8 opaque


def test_generate_masks_skip_existing(tmp_path: Path):
    root = tmp_path / "in"
    masks = tmp_path / "masks"
    _write_png(root / "object_B_pendant" / "1.jpg")
    remover = FakeRemover()

    generate_masks(root, masks, remover)
    assert len(remover.processed) == 1
    # Second run with skip_existing must not re-infer.
    ok, fail, skip = generate_masks(root, masks, remover, skip_existing=True)
    assert (ok, fail, skip) == (0, 0, 1)
    assert len(remover.processed) == 1
