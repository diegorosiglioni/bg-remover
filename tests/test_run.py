"""run.py CLI: step dispatch, arg validation, and a no-model apply round-trip.

The generate step's model is monkeypatched to a fake so no weights load.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

import run
from tests.conftest import solid_rgb
from tests.test_folder_steps import FakeRemover, _build_tree


def test_bad_threshold_returns_2():
    assert run.main(["--step", "generate_masks", "--mask-threshold", "300"]) == 2


def test_generate_missing_root_returns_2(tmp_path: Path):
    code = run.main(["--step", "generate_masks", "--root", str(tmp_path / "nope")])
    assert code == 2


def test_apply_missing_masks_dir_returns_2(tmp_path: Path):
    root = tmp_path / "in"
    _build_tree(root)
    code = run.main(
        ["--step", "apply_masks", "--root", str(root), "--masks", str(tmp_path / "no")]
    )
    assert code == 2


def test_generate_then_apply_end_to_end(tmp_path: Path, monkeypatch):
    root = tmp_path / "in"
    masks = tmp_path / "masks"
    out = tmp_path / "out"
    _build_tree(root)

    # Avoid loading the real model: run.BackgroundRemover(config) -> FakeRemover.
    monkeypatch.setattr(run, "BackgroundRemover", lambda config: FakeRemover())

    gen = run.main(
        ["--step", "generate_masks", "--root", str(root), "--masks", str(masks)]
    )
    assert gen == 0
    # Masked (non-pendant) folder stores one shared mask, not one per image.
    assert (masks / "object_A" / "_shared_mask.png").exists()

    app = run.main(
        [
            "--step", "apply_masks",
            "--root", str(root),
            "--masks", str(masks),
            "--out", str(out),
        ]
    )
    assert app == 0
    rgba = np.asarray(Image.open(out / "object_A" / "1.png").convert("RGBA"))
    assert rgba[4, 4, 3] == 255
    assert rgba[0, 0, 3] == 0
