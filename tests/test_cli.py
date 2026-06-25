"""CLI helpers: colour parsing, discovery, output-path containment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.cli import _parse_color, discover_images, process_folder
from app.config import Config
from app.pipeline import BackgroundRemover
from tests.conftest import FakeSegmenter, solid_rgb


def test_parse_color_rgb_defaults_alpha():
    assert _parse_color("10,20,30") == (10, 20, 30, 255)


def test_parse_color_rgba():
    assert _parse_color("1,2,3,4") == (1, 2, 3, 4)


@pytest.mark.parametrize("bad", ["1,2", "1,2,3,4,5", "a,b,c", "300,0,0"])
def test_parse_color_rejects_bad(bad):
    with pytest.raises(ValueError):
        _parse_color(bad)


def test_discover_images_filters_and_sorts(tmp_path: Path):
    (tmp_path / "b.png").write_bytes(b"x")
    (tmp_path / "a.jpg").write_bytes(b"x")
    (tmp_path / "note.txt").write_text("x")
    found = discover_images(tmp_path)
    assert [p.name for p in found] == ["a.jpg", "b.png"]


def _write_png(path: Path, color=(10, 20, 30)):
    Image.fromarray(solid_rgb(8, 8, color), mode="RGB").save(path)


def test_process_folder_writes_outputs(tmp_path: Path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    _write_png(in_dir / "one.png")
    _write_png(in_dir / "two.png")

    cfg = Config(canonical_size=None, pipeline_mode="rembg_oneshot")
    remover = BackgroundRemover(cfg, segmenter=FakeSegmenter())
    n_ok, n_fail, n_skip = process_folder(remover, sorted(in_dir.glob("*.png")), in_dir, out_dir)
    assert (n_ok, n_fail, n_skip) == (2, 0, 0)
    assert (out_dir / "one.png").exists()
    assert (out_dir / "two.png").exists()


def test_process_folder_skip_existing(tmp_path: Path):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    _write_png(in_dir / "one.png")
    (out_dir / "one.png").write_bytes(b"already")

    cfg = Config(canonical_size=None, pipeline_mode="rembg_oneshot")
    remover = BackgroundRemover(cfg, segmenter=FakeSegmenter())
    n_ok, n_fail, n_skip = process_folder(
        remover, [in_dir / "one.png"], in_dir, out_dir, skip_existing=True
    )
    assert (n_ok, n_fail, n_skip) == (0, 0, 1)
