"""Two-step folder pipeline: generate masks, then apply them.

This decouples the expensive model inference from the cheap cutout step so a
user can hand-edit masks (see ``edit_masks.py``) in between:

    python run.py --step generate_masks   # runs the model, writes masks/
    # ...optionally edit masks with edit_masks.py...
    python run.py --step apply_masks       # reads masks/, writes output/

``generate_masks`` dispatches per folder exactly like ``app.folder_pipeline``:
folders whose name contains "pendant" get a per-image AI mask; other folders
share one mask built from their second image. ``apply_masks`` loads no model: it
reads each stored grayscale mask and cuts the native-resolution source, resizing
the mask to the source dimensions when they differ.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Sequence

from app.config import Config
from app.folder_pipeline import (
    DEFAULT_CARVE_MIN_AREA,
    DEFAULT_CARVE_WHITE,
    DEFAULT_MASK_THRESHOLD,
    apply_masks,
    generate_masks,
    group_images_by_folder,
)
from app.pipeline import BackgroundRemover

logger = logging.getLogger(__name__)

# Default working folders (overridable on the command line).
DEFAULT_ROOT = "input"
DEFAULT_OUT = "output"
DEFAULT_MASKS = "masks"


def _run_generate(args: argparse.Namespace) -> int:
    """Execute the ``generate_masks`` step.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code (``0`` all ok, ``1`` any failed, ``2`` bad input).
    """
    root = Path(args.root)
    masks_dir = Path(args.masks)
    if not root.is_dir():
        print(f"error: input root not found: {root}", file=sys.stderr)
        return 2
    if not group_images_by_folder(root):
        print(f"error: no images found under {root}", file=sys.stderr)
        return 2

    # Any enclosed hole whose centroid is within --fill-radius (default 0.16) of
    # the image centre is re-covered with the opaque original image content;
    # holes farther out stay transparent. restore_drops is disabled so the rule
    # is purely the centre radius (size and material do not matter).
    overrides: dict = {
        "fill_holes": True,
        "fill_holes_center_dist": args.fill_radius,
        "fill_holes_restore_drops": False,
    }
    if args.model is not None:
        overrides["model_name"] = args.model
    config = Config(**overrides)

    masks_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Loading model {config.model_name!r} "
        f"(engine={config.engine}, device={config.device})..."
    )
    remover = BackgroundRemover(config)

    n_ok, n_fail, n_skip = generate_masks(
        root, masks_dir, remover, args.mask_threshold, args.skip_existing,
        args.carve_white, args.carve_min_area,
    )
    _print_summary(n_ok, n_fail, n_skip)
    return 1 if n_fail else 0


def _run_apply(args: argparse.Namespace) -> int:
    """Execute the ``apply_masks`` step.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code (``0`` all ok, ``1`` any failed, ``2`` bad input).
    """
    root = Path(args.root)
    out_dir = Path(args.out)
    masks_dir = Path(args.masks)
    if not root.is_dir():
        print(f"error: input root not found: {root}", file=sys.stderr)
        return 2
    if not masks_dir.is_dir():
        print(
            f"error: masks directory not found: {masks_dir}; "
            "run --step generate_masks first",
            file=sys.stderr,
        )
        return 2
    if not group_images_by_folder(root):
        print(f"error: no images found under {root}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    n_ok, n_fail, n_skip = apply_masks(
        root, out_dir, masks_dir, preview_dir, args.skip_existing
    )
    _print_summary(n_ok, n_fail, n_skip)
    return 1 if n_fail else 0


def _print_summary(n_ok: int, n_fail: int, n_skip: int) -> None:
    """Print the final per-run tally.

    Args:
        n_ok: Number of images that succeeded.
        n_fail: Number that failed.
        n_skip: Number that were skipped.
    """
    total = n_ok + n_fail + n_skip
    print(
        f"\nDone: {n_ok} succeeded, {n_fail} failed, {n_skip} skipped, "
        f"{total} total."
    )


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Define and parse command-line arguments.

    Args:
        argv: Optional argument vector.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="python run.py",
        description=(
            "Two-step folder pipeline: --step generate_masks runs the model and "
            "writes grayscale masks; --step apply_masks reads them and writes "
            "transparent PNGs."
        ),
    )
    parser.add_argument(
        "--step",
        required=True,
        choices=("generate_masks", "apply_masks"),
        help="Which pipeline step to run.",
    )
    parser.add_argument(
        "--root", default=DEFAULT_ROOT, help="Input root (subfolders = objects)."
    )
    parser.add_argument(
        "--masks", default=DEFAULT_MASKS, help="Mask root (mirrors the input tree)."
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT, help="Output root (apply_masks step)."
    )
    parser.add_argument("--model", default=None, help="rembg/ONNX model name.")
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=DEFAULT_MASK_THRESHOLD,
        metavar="0-255",
        help="Alpha cut for binarizing silhouettes (default 128).",
    )
    parser.add_argument(
        "--carve-white",
        type=int,
        default=DEFAULT_CARVE_WHITE,
        metavar="0-255",
        help=(
            "generate_masks: carve enclosed white holes (eyelets, gaps) from the "
            "mask. Grayscale cut for 'white'; 0 disables (default). Try 250 for "
            "jewellery on a white background."
        ),
    )
    parser.add_argument(
        "--carve-min-area",
        type=float,
        default=DEFAULT_CARVE_MIN_AREA,
        metavar="FRAC",
        help=(
            "Minimum carved-hole area as a fraction of the image (default "
            f"{DEFAULT_CARVE_MIN_AREA}); smaller white blobs (highlights, stones) "
            "are kept. Only used when --carve-white > 0."
        ),
    )
    parser.add_argument(
        "--fill-radius",
        type=float,
        default=0.16,
        metavar="0-1",
        help=(
            "generate_masks: max distance of a hole's centroid from the image "
            "centre for it to be re-covered with the opaque original content, "
            "normalized so 0 is the centre and 1.0 is a corner (half the "
            "diagonal). Default 0.16. Holes farther out stay transparent."
        ),
    )
    parser.add_argument(
        "--preview-dir",
        default=None,
        help="Also write a checkerboard preview per image (apply_masks step).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose mask (generate) or output (apply) already exists.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Emit DEBUG logs and tracebacks."
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``python run.py``.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: ``0`` all images succeeded, ``1`` any failed, ``2``
        bad arguments or an empty/missing input tree.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if not 0 <= args.mask_threshold <= 255:
        print("error: --mask-threshold must be in [0, 255]", file=sys.stderr)
        return 2
    if not 0 <= args.carve_white <= 255:
        print("error: --carve-white must be in [0, 255]", file=sys.stderr)
        return 2
    if not 0.0 <= args.carve_min_area <= 1.0:
        print("error: --carve-min-area must be in [0.0, 1.0]", file=sys.stderr)
        return 2
    if not 0.0 < args.fill_radius <= 1.0:
        print("error: --fill-radius must be in (0.0, 1.0]", file=sys.stderr)
        return 2

    if args.step == "generate_masks":
        return _run_generate(args)
    return _run_apply(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
