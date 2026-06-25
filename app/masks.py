"""Grayscale mask storage shared by the generate/apply pipeline steps.

A *mask* is a per-image binary foreground map persisted as an 8-bit grayscale
PNG: **white (255) = keep**, **black (0) = remove**. Masks mirror the input tree
under a ``masks/`` root, one file per source image with a ``_mask`` suffix::

    input/object_A/img1.jpg   ->   masks/object_A/img1_mask.png

Persisting the mask between the two pipeline steps is what lets a user hand-edit
it (see ``edit_masks.py``) and re-run only the apply step. The on-disk grayscale
form is the seam: both the AI generator and the manual editor write the same
format, and the apply step neither knows nor cares which produced it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

# Suffix appended to the source stem for its mask file (kept as a constant so
# both the writer and the path resolver agree on the convention).
MASK_SUFFIX = "_mask"

# Filename of the single *shared* mask stored once per folder for the masked
# (non-pendant) branch, where every image in the folder uses the same silhouette.
# Storing one file instead of one per image avoids writing N byte-identical PNGs.
# The leading underscore keeps it out of the way of per-image ``<stem>_mask.png``
# files (which still override it for hand-edited single-image corrections).
SHARED_MASK_NAME = "_shared_mask.png"

# A grayscale pixel at/above this value is treated as foreground when a stored
# mask is read back into a boolean array.
_MASK_READ_THRESHOLD = 128


def mask_path(source: Path, root: Path, masks_dir: Path) -> Path:
    """Return the mask file location mirroring ``source`` under ``masks_dir``.

    Args:
        source: Source image path (under ``root``).
        root: Input root the relative subpath is taken from.
        masks_dir: Mask root that mirrors the input tree.

    Returns:
        ``masks_dir/<relative dir>/<stem>_mask.png``. When ``source`` is not
        under ``root`` (e.g. an ad-hoc path passed to the editor), the mask is
        placed flat in ``masks_dir`` as ``<stem>_mask.png``.
    """
    try:
        relative = source.relative_to(root)
        parent = relative.parent
    except ValueError:
        parent = Path(".")
    return masks_dir / parent / f"{source.stem}{MASK_SUFFIX}.png"


def shared_mask_path(folder: Path, root: Path, masks_dir: Path) -> Path:
    """Return the shared-mask file for ``folder`` mirrored under ``masks_dir``.

    One file per folder, used by the masked (non-pendant) branch so a single
    silhouette is stored once instead of duplicated per image.

    Args:
        folder: The source folder (the parent dir of its images).
        root: Input root the relative subpath is taken from.
        masks_dir: Mask root that mirrors the input tree.

    Returns:
        ``masks_dir/<relative dir>/_shared_mask.png``. When ``folder`` is not
        under ``root``, the shared mask is placed flat in ``masks_dir``.
    """
    try:
        relative = folder.relative_to(root)
    except ValueError:
        relative = Path(".")
    return masks_dir / relative / SHARED_MASK_NAME


def save_mask(mask: np.ndarray, destination: Path) -> None:
    """Write a boolean foreground mask as a grayscale PNG (white = keep).

    Args:
        mask: Boolean mask, shape ``(H, W)`` (True = foreground/keep).
        destination: Output PNG path; parent directories are created.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    gray = np.where(mask, np.uint8(255), np.uint8(0))
    Image.fromarray(gray, mode="L").save(destination)


def load_mask(path: Path) -> np.ndarray:
    """Read a grayscale mask PNG back into a boolean foreground mask.

    Args:
        path: Mask PNG path.

    Returns:
        Boolean mask, shape ``(H, W)`` (True where the stored value is
        ``>= 128``, i.e. white = keep).
    """
    gray = np.asarray(Image.open(path).convert("L"))
    return gray >= _MASK_READ_THRESHOLD


def resize_mask(mask: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize a boolean mask to ``(width, height)``.

    Args:
        mask: Boolean mask, shape ``(H, W)``.
        size_wh: Target ``(width, height)`` in pixels.

    Returns:
        Boolean mask at the requested size.
    """
    img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    resized = img.resize(size_wh, Image.NEAREST)
    return np.asarray(resized) >= _MASK_READ_THRESHOLD
