"""Folder-aware pipeline: classification, grouping, mask, and dispatch.

The AI engine is exercised through an injected fake remover so no model loads.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.folder_pipeline import (
    apply_mask,
    group_images_by_folder,
    is_pendant,
    mask_from_alpha,
    process_masked_folder,
    process_pendant_folder,
    run_pipeline,
    select_reference,
)
from tests.conftest import solid_rgb


class FakeRemover:
    """A remover whose alpha is a centred square, recording processed paths."""

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


# --- pure helpers -----------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("ring_pendant", True),
        ("Pendant_gold", True),
        ("PENDANT", True),
        ("rectangle", False),
        ("bracelet", False),
    ],
)
def test_is_pendant(name, expected):
    assert is_pendant(name) is expected


def test_mask_from_alpha_threshold():
    alpha = np.array([[0, 127, 128, 255]], dtype=np.uint8)
    mask = mask_from_alpha(alpha, threshold=128)
    assert mask.tolist() == [[False, False, True, True]]


def test_select_reference_second_alphabetical():
    imgs = [Path("a.jpg"), Path("b.jpg"), Path("c.jpg")]
    assert select_reference(imgs) == Path("b.jpg")


def test_select_reference_single_falls_back_to_first():
    assert select_reference([Path("only.jpg")]) == Path("only.jpg")


def test_select_reference_empty_raises():
    with pytest.raises(ValueError):
        select_reference([])


def test_apply_mask_keeps_interior_transparent_outside():
    rgb = solid_rgb(4, 4, (50, 60, 70))
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    rgba = apply_mask(rgb, mask)
    assert rgba.shape == (4, 4, 4)
    assert rgba[1, 1, 3] == 255
    assert rgba[0, 0, 3] == 0
    # RGB preserved untouched inside the mask.
    assert tuple(rgba[1, 1, :3]) == (50, 60, 70)


def test_apply_mask_resizes_mask_to_image():
    rgb = solid_rgb(8, 8, (1, 2, 3))
    mask = np.ones((4, 4), dtype=bool)  # different size -> must resize up
    rgba = apply_mask(rgb, mask)
    assert rgba.shape == (8, 8, 4)
    assert (rgba[..., 3] == 255).all()


# --- grouping ---------------------------------------------------------------


def test_group_images_by_folder_recursive_sorted(tmp_path: Path):
    _write_png(tmp_path / "obj_a" / "z.jpg")
    _write_png(tmp_path / "obj_a" / "a.jpg")
    _write_png(tmp_path / "obj_b_pendant" / "img.png")
    (tmp_path / "obj_a" / "note.txt").write_text("x")

    groups = group_images_by_folder(tmp_path)
    assert set(groups) == {tmp_path / "obj_a", tmp_path / "obj_b_pendant"}
    assert [p.name for p in groups[tmp_path / "obj_a"]] == ["a.jpg", "z.jpg"]


# --- branches ---------------------------------------------------------------


def test_process_pendant_folder_runs_engine_per_image(tmp_path: Path):
    folder = tmp_path / "in" / "ring_pendant"
    out = tmp_path / "out"
    _write_png(folder / "1.jpg")
    _write_png(folder / "2.jpg")
    remover = FakeRemover()

    ok, fail, skip = process_pendant_folder(
        remover, sorted(folder.glob("*.jpg")), tmp_path / "in", out
    )
    assert (ok, fail, skip) == (2, 0, 0)
    assert len(remover.processed) == 2  # one inference per image
    assert (out / "ring_pendant" / "1.png").exists()
    assert (out / "ring_pendant" / "2.png").exists()


def test_process_masked_folder_one_inference_shared_mask(tmp_path: Path):
    folder = tmp_path / "in" / "rectangle"
    out = tmp_path / "out"
    _write_png(folder / "a.jpg")
    _write_png(folder / "b.jpg")
    _write_png(folder / "c.jpg")
    remover = FakeRemover()

    ok, fail, skip = process_masked_folder(
        remover, sorted(folder.glob("*.jpg")), tmp_path / "in", out
    )
    assert (ok, fail, skip) == (3, 0, 0)
    # Exactly one inference, on the second image alphabetically (b.jpg).
    assert remover.processed == [folder / "b.jpg"]
    for name in ("a", "b", "c"):
        png = out / "rectangle" / f"{name}.png"
        assert png.exists()
        rgba = np.asarray(Image.open(png).convert("RGBA"))
        # Centre kept opaque, corner transparent (the shared square mask).
        assert rgba[4, 4, 3] == 255
        assert rgba[0, 0, 3] == 0


def test_masked_folder_reference_failure_fails_all(tmp_path: Path):
    folder = tmp_path / "in" / "rectangle"
    out = tmp_path / "out"
    _write_png(folder / "a.jpg")
    _write_png(folder / "b.jpg")

    class Boom:
        def process(self, source):
            raise RuntimeError("model exploded")

    ok, fail, skip = process_masked_folder(
        Boom(), sorted(folder.glob("*.jpg")), tmp_path / "in", out
    )
    assert (ok, fail, skip) == (0, 2, 0)
    assert not list(out.rglob("*.png"))


def test_run_pipeline_dispatches_by_name(tmp_path: Path):
    root = tmp_path / "in"
    out = tmp_path / "out"
    _write_png(root / "object_A" / "1.jpg")
    _write_png(root / "object_A" / "2.jpg")
    _write_png(root / "object_B_pendant" / "1.jpg")
    _write_png(root / "object_B_pendant" / "2.jpg")
    remover = FakeRemover()

    ok, fail, skip = run_pipeline(root, out, remover)
    assert (ok, fail, skip) == (4, 0, 0)
    # pendant: 2 inferences; masked: 1 inference -> 3 total.
    assert len(remover.processed) == 3
    assert (out / "object_A" / "1.png").exists()
    assert (out / "object_B_pendant" / "2.png").exists()


def test_skip_existing(tmp_path: Path):
    folder = tmp_path / "in" / "rectangle"
    out = tmp_path / "out"
    _write_png(folder / "a.jpg")
    _write_png(folder / "b.jpg")
    (out / "rectangle").mkdir(parents=True)
    (out / "rectangle" / "a.png").write_bytes(b"existing")
    remover = FakeRemover()

    ok, fail, skip = process_masked_folder(
        remover, sorted(folder.glob("*.jpg")), tmp_path / "in", out, skip_existing=True
    )
    assert (ok, fail, skip) == (1, 0, 1)
