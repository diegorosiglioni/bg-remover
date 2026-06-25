"""Process a single image: AI mask -> white-hole carve -> save mask + cutout.

A targeted one-image counterpart to ``run.py`` (which always walks a whole
folder). Useful to (re)do just one pendant with the white-hole carving, e.g.::

    python carve_one.py "input/pendant/photo_01.jpg" \
        --masks masks --out output --carve-white 250 --carve-min-area 0.002

The mask is written under ``--masks`` mirroring ``--root`` (so ``edit_masks.py``
finds it), and the transparent PNG under ``--out`` with the same relative path.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from app.composite import encode_png
from app.config import Config
from app.folder_pipeline import (
    DEFAULT_CARVE_MIN_AREA,
    DEFAULT_CARVE_WHITE,
    DEFAULT_MASK_THRESHOLD,
    apply_mask,
    carve_white_holes,
    mask_from_alpha,
)
from app.ingest import load_image
from app.masks import mask_path, save_mask
from app.pipeline import BackgroundRemover

# Native-resolution ingest (EXIF + sRGB, no padding/scaling) for the cutout, so
# the written PNG keeps the source's own dimensions.
_NATIVE_INGEST = Config(canonical_size=None)


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Define and parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python carve_one.py",
        description="Run the model on ONE image, carve white holes, save mask+PNG.",
    )
    parser.add_argument("image", help="Source image path.")
    parser.add_argument(
        "--root",
        default=None,
        help="Input root for mirroring mask/output paths (default: image's parent).",
    )
    parser.add_argument("--masks", default="masks", help="Mask root.")
    parser.add_argument("--out", default="output", help="Output root.")
    parser.add_argument("--model", default=None, help="rembg/ONNX model name.")
    parser.add_argument(
        "--mask-threshold", type=int, default=DEFAULT_MASK_THRESHOLD, metavar="0-255"
    )
    parser.add_argument(
        "--carve-white", type=int, default=DEFAULT_CARVE_WHITE, metavar="0-255",
        help="Grayscale cut for carving white holes; 0 disables. Try 250.",
    )
    parser.add_argument(
        "--carve-min-area", type=float, default=DEFAULT_CARVE_MIN_AREA, metavar="FRAC",
        help="Minimum carved-hole area as a fraction of the image.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point. Returns 0 on success, 2 on bad input."""
    args = _parse_args(argv)
    source = Path(args.image)
    if not source.is_file():
        print(f"error: not a file: {source}", file=sys.stderr)
        return 2
    if not 0 <= args.carve_white <= 255:
        print("error: --carve-white must be in [0, 255]", file=sys.stderr)
        return 2
    if not 0.0 <= args.carve_min_area <= 1.0:
        print("error: --carve-min-area must be in [0.0, 1.0]", file=sys.stderr)
        return 2

    root = Path(args.root) if args.root else source.parent
    masks_dir = Path(args.masks)
    out_dir = Path(args.out)

    # Keep interior holes transparent: carving white holes is the whole point
    # here, so don't let fill_alpha_holes (config.fill_holes) re-cover them first.
    overrides: dict = {"fill_holes": False}
    if args.model is not None:
        overrides["model_name"] = args.model
    config = Config(**overrides)
    print(
        f"Loading model {config.model_name!r} "
        f"(engine={config.engine}, device={config.device})..."
    )
    remover = BackgroundRemover(config)

    rgba = remover.process(source)
    mask = mask_from_alpha(rgba[..., 3], args.mask_threshold)
    before = int(mask.sum())
    mask = carve_white_holes(mask, rgba[..., :3], args.carve_white, args.carve_min_area)
    carved = before - int(mask.sum())

    # Save the grayscale mask (so edit_masks.py can load/correct it).
    mask_dst = mask_path(source, root, masks_dir)
    save_mask(mask, mask_dst)

    # Cut the native-resolution source with the carved mask and save the PNG.
    rgb = load_image(source, _NATIVE_INGEST)
    out_rgba = apply_mask(rgb, mask)
    try:
        rel = source.relative_to(root).with_suffix(".png")
    except ValueError:
        rel = Path(source.stem + ".png")
    out_dst = out_dir / rel
    out_dst.parent.mkdir(parents=True, exist_ok=True)
    out_dst.write_bytes(encode_png(out_rgba))

    print(f"  carved {carved} px white holes")
    print(f"  mask   -> {mask_dst}")
    print(f"  cutout -> {out_dst}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
