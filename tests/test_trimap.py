"""Trimap synthesis behaviour (characterization)."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import Config
from app.trimap import TRIMAP_BG, TRIMAP_FG, TRIMAP_UNKNOWN, synthesize_trimap
from tests.conftest import disc_alpha


def test_trimap_rejects_non_2d():
    with pytest.raises(ValueError):
        synthesize_trimap(np.zeros((4, 4, 1), dtype=np.float32), Config())


def test_trimap_values_and_bands():
    alpha = disc_alpha(64, radius_frac=0.4)
    trimap = synthesize_trimap(alpha, Config())
    assert trimap.dtype == np.float32
    assert set(np.unique(trimap)).issubset({TRIMAP_BG, TRIMAP_UNKNOWN, TRIMAP_FG})
    # Centre is solidly foreground; corners solidly background; an unknown band
    # exists around the disc edge.
    assert trimap[32, 32] == TRIMAP_FG
    assert trimap[0, 0] == TRIMAP_BG
    assert (trimap == TRIMAP_UNKNOWN).any()


def test_trimap_is_deterministic():
    alpha = disc_alpha(48)
    a = synthesize_trimap(alpha, Config())
    b = synthesize_trimap(alpha, Config())
    np.testing.assert_array_equal(a, b)
