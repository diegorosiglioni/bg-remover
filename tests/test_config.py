"""Config validation behaviour."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Config


def test_defaults_are_phase1():
    cfg = Config()
    assert cfg.engine == "rembg"
    assert cfg.device == "cpu"
    assert cfg.model_name == "birefnet-general"
    assert cfg.pipeline_mode == "rembg_oneshot"


def test_color_components_rejected_out_of_range():
    with pytest.raises(ValidationError):
        Config(pad_color=(300, 0, 0))
    with pytest.raises(ValidationError):
        Config(background_color=(0, 0, 0, 300))


def test_color_components_accepted_in_range():
    cfg = Config(pad_color=(0, 128, 255), background_color=(1, 2, 3, 4))
    assert cfg.pad_color == (0, 128, 255)
    assert cfg.background_color == (1, 2, 3, 4)


def test_valid_model_names_accepted():
    # Real rembg model identifiers must keep working.
    for name in ("birefnet-general", "birefnet-general-lite", "u2net", "isnet-general-use"):
        assert Config(model_name=name).model_name == name


@pytest.mark.parametrize(
    "bad",
    [
        "../evil",
        "..\\evil",
        "/abs/path",
        "C:\\abs\\path",
        "foo/bar",
        "foo\x00bar",
        "",
        "   ",
    ],
)
def test_model_name_rejects_path_like_values(bad):
    # Path separators / traversal / control chars must be rejected so the value
    # cannot steer pooch's download path. See security finding S1/S2.
    with pytest.raises(ValidationError):
        Config(model_name=bad)
