"""Command-line batch interface.

Processes every image in an input directory and writes a transparent (or
flat-background) PNG per file to an output directory. The model is loaded once
and reused across the whole batch via :class:`~app.pipeline.BackgroundRemover`.
Each file is handled independently: a bad image is logged and skipped rather
than aborting the run.

Usage::

    python -m app.cli IN_DIR OUT_DIR [options]

Configuration defaults come from :class:`~app.config.Config` (and ``BGR_*``
environment variables); command-line flags override them for the run.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from app.composite import composite_over_checkerboard, encode_png
from app.config import Config
from app.pipeline import BackgroundRemover
from PIL import Image

logger = logging.getLogger(__name__)

# Image extensions treated as inputs (lower-case, with leading dot).
IMAGE_EXTENSIONS = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``python -m app.cli``.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: ``0`` if every discovered image succeeded, ``1`` if
        any file failed, ``2`` if no images were found or arguments were
        invalid.
    """
    args = _parse_args(argv)

    # Errors always print a one-line summary to stderr; --verbose additionally
    # emits full tracebacks (logged at DEBUG) so a failing batch is debuggable.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not in_dir.is_dir():
        print(f"error: input directory not found: {in_dir}", file=sys.stderr)
        return 2

    try:
        config = _build_config(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    images = discover_images(in_dir, recursive=args.recursive)
    if not images:
        print(f"error: no images found in {in_dir}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading model {config.model_name!r} "
        f"(engine={config.engine}, device={config.device}, mode={config.pipeline_mode})..."
    )
    remover = BackgroundRemover(config)

    n_ok, n_fail, n_skip = process_folder(
        remover, images, in_dir, out_dir, preview_dir, skip_existing=args.skip_existing
    )

    print(
        f"\nDone: {n_ok} succeeded, {n_fail} failed, {n_skip} skipped, "
        f"{len(images)} total."
    )
    return 1 if n_fail else 0


def process_folder(
    remover: BackgroundRemover,
    images: List[Path],
    in_dir: Path,
    out_dir: Path,
    preview_dir: Optional[Path] = None,
    skip_existing: bool = False,
) -> Tuple[int, int, int]:
    """Process a list of images, isolating per-file failures.

    Args:
        remover: A constructed :class:`~app.pipeline.BackgroundRemover`.
        images: Image paths to process.
        in_dir: Input root (used to preserve relative subpaths in outputs).
        out_dir: Output root for transparent PNG results.
        preview_dir: Optional root for checkerboard preview PNGs. When given, an
            opaque preview (result over a transparency checkerboard) is written
            per image so cutouts are easy to inspect.
        skip_existing: When True, skip any image whose output PNG already exists
            (no re-inference), so an interrupted batch resumes instead of
            restarting from zero.

    Returns:
        Tuple ``(n_succeeded, n_failed, n_skipped)``.
    """
    n_ok = 0
    n_fail = 0
    n_skip = 0
    total = len(images)
    for index, path in enumerate(images, start=1):
        relative = path.relative_to(in_dir).with_suffix(".png")
        destination = out_dir / relative
        # Defense in depth: never write outside the output root even if a
        # relative subpath or a symlinked out_dir would resolve elsewhere.
        if not _is_within(destination, out_dir):
            n_fail += 1
            print(
                f"[{index}/{total}] {path.name} FAILED: output path escapes "
                f"{out_dir} -> {destination}",
                file=sys.stderr,
            )
            continue
        if skip_existing and destination.exists():
            n_skip += 1
            print(f"[{index}/{total}] {path.name} -> skip (exists)")
            continue
        try:
            rgba = remover.process(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(encode_png(rgba))
            n_ok += 1
            print(f"[{index}/{total}] {path.name} -> {destination}")
            # Preview is a non-essential extra: a failure here must not mark the
            # image as failed when its primary PNG was already written.
            if preview_dir is not None:
                try:
                    _write_preview(rgba, preview_dir / relative)
                except Exception as exc:
                    print(
                        f"[{index}/{total}] {path.name} preview failed: {exc}",
                        file=sys.stderr,
                    )
                    logger.debug("preview failure for %s", path, exc_info=True)
        except Exception as exc:  # isolate: one bad file must not stop the batch
            n_fail += 1
            print(
                f"[{index}/{total}] {path.name} FAILED: {exc}",
                file=sys.stderr,
            )
            # Full traceback (and the exception type) is preserved for --verbose
            # so a batch that fails on many files can be diagnosed.
            logger.debug("processing failure for %s", path, exc_info=True)
    return n_ok, n_fail, n_skip


def _is_within(path: Path, root: Path) -> bool:
    """Return True if ``path`` resolves inside ``root``.

    Args:
        path: Candidate output path (need not exist yet).
        root: Output root directory.

    Returns:
        True if ``path`` is ``root`` or a descendant of it after resolution.
    """
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _write_preview(rgba: np.ndarray, destination: Path) -> None:
    """Write an opaque checkerboard preview of an RGBA result.

    Args:
        rgba: RGBA result, shape ``(H, W, 4)``, dtype ``uint8``.
        destination: Output path for the preview PNG.
    """
    preview = composite_over_checkerboard(rgba)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(preview, mode="RGB").save(destination)


def discover_images(directory: Path, recursive: bool = False) -> List[Path]:
    """Find image files in a directory.

    Args:
        directory: Directory to scan.
        recursive: When True, descend into subdirectories.

    Returns:
        Sorted list of image file paths with recognized extensions.
    """
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(
        p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _build_config(args: argparse.Namespace) -> Config:
    """Build a Config from defaults/env plus any provided CLI overrides.

    Only flags the user actually passed are applied, so environment variables
    and defaults still take effect for the rest.

    Args:
        args: Parsed command-line arguments.

    Returns:
        A validated :class:`~app.config.Config`.

    Raises:
        ValueError: If an override value is invalid (e.g. malformed colour).
    """
    overrides: dict = {}
    if args.engine is not None:
        overrides["engine"] = args.engine
    if args.device is not None:
        overrides["device"] = args.device
    if args.model is not None:
        overrides["model_name"] = args.model
    if args.mode is not None:
        overrides["pipeline_mode"] = args.mode
    if args.band_pct is not None:
        overrides["trimap_band_pct"] = args.band_pct
    if args.guided_filter:
        overrides["guided_filter_enabled"] = True
    if args.premultiplied:
        overrides["premultiplied_output"] = True
    if args.restore_original_size:
        overrides["restore_original_size"] = True
    if args.fill_holes:
        overrides["fill_holes"] = True
    if args.fill_holes_dilate is not None:
        overrides["fill_holes_dilate"] = args.fill_holes_dilate
    if args.fill_holes_feather is not None:
        overrides["fill_holes_feather"] = args.fill_holes_feather
    if args.fill_holes_restore_drops:
        overrides["fill_holes_restore_drops"] = True
    if args.fill_holes_drop_edge_feather is not None:
        overrides["fill_holes_drop_edge_feather"] = args.fill_holes_drop_edge_feather
    if args.bg_color is not None:
        overrides["background_color"] = _parse_color(args.bg_color)
    return Config(**overrides)


def _parse_color(text: str) -> Tuple[int, int, int, int]:
    """Parse an ``R,G,B`` or ``R,G,B,A`` string into an RGBA tuple.

    Args:
        text: Comma-separated integer components in ``[0, 255]``.

    Returns:
        RGBA tuple; alpha defaults to 255 when only RGB is given.

    Raises:
        ValueError: If the string is malformed or out of range.
    """
    parts = [p.strip() for p in text.split(",")]
    if len(parts) not in (3, 4):
        raise ValueError(f"background color must be 'R,G,B' or 'R,G,B,A'; got {text!r}")
    try:
        values = [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"background color components must be integers: {text!r}") from exc
    if any(v < 0 or v > 255 for v in values):
        raise ValueError(f"background color components must be in [0, 255]: {text!r}")
    if len(values) == 3:
        values.append(255)
    return (values[0], values[1], values[2], values[3])


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Define and parse command-line arguments.

    Args:
        argv: Optional argument vector.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Batch white-background removal (phase 1, CPU).",
    )
    parser.add_argument("--in-dir", help="Directory of input images.")
    parser.add_argument("--out-dir", help="Directory for output PNGs.")
    parser.add_argument(
        "--recursive", action="store_true", help="Recurse into subdirectories."
    )
    parser.add_argument(
        "--preview-dir",
        default=None,
        help="Also write an opaque checkerboard preview per image to this folder.",
    )
    parser.add_argument(
        "--engine", choices=("rembg", "torch"), default=None, help="Segmentation engine."
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default=None, help="Compute device."
    )
    parser.add_argument("--model", default=None, help="rembg/ONNX model name.")
    parser.add_argument(
        "--mode",
        choices=("rembg_oneshot", "modular"),
        default=None,
        help="Pipeline mode.",
    )
    parser.add_argument(
        "--band-pct", type=float, default=None, help="Trimap unknown-band width fraction."
    )
    parser.add_argument(
        "--bg-color", default=None, help="Flat background as 'R,G,B' or 'R,G,B,A'."
    )
    parser.add_argument(
        "--guided-filter", action="store_true", help="Enable guided-filter alpha refine."
    )
    parser.add_argument(
        "--premultiplied", action="store_true", help="Emit premultiplied alpha."
    )
    parser.add_argument(
        "--restore-original-size",
        action="store_true",
        help="Crop canonicalization padding off outputs so each PNG matches "
        "its original input dimensions instead of the canonical size.",
    )
    parser.add_argument(
        "--fill-holes",
        action="store_true",
        help="Fill central enclosed holes with the original image content "
        "(keep them as in the original instead of transparent). Off-centre "
        "holes stay transparent.",
    )
    parser.add_argument(
        "--fill-holes-dilate",
        type=int,
        default=None,
        metavar="PX",
        help="Pixels to grow the hole fill past its boundary so it covers the "
        "matting-contaminated ring around the hole (cleaner edge). Default 2; "
        "use 0 to fill only the exact hole.",
    )
    parser.add_argument(
        "--fill-holes-feather",
        type=float,
        default=None,
        metavar="SIGMA",
        help="Gaussian sigma (px) for feathering the fill boundary into the "
        "surrounding result, avoiding a hard seam. Default 1.5; use 0 for a "
        "hard edge.",
    )
    parser.add_argument(
        "--fill-holes-restore-drops",
        action="store_true",
        help="Content-aware repair: restore saliency drops (the model dropping "
        "part of a solid reflective body) hard-edged from the original at full "
        "opacity, while leaving real ring eyelets transparent. Requires "
        "--fill-holes.",
    )
    parser.add_argument(
        "--fill-holes-drop-edge-feather",
        type=float,
        default=None,
        metavar="SIGMA",
        help="Gaussian sigma (px) softening only the outer alpha edge of a "
        "restored drop. Default 0 = hard edge matching the true subject border.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose output PNG already exists (resume a batch "
        "instead of restarting from zero).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit full tracebacks for per-file failures (logged at DEBUG).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
