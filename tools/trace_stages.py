"""Dump every pipeline stage for ONE image, to see how it is transformed.

Replicates ``BackgroundRemover.process`` step by step but writes each
intermediate as a PNG so the hole-fill defect can be inspected visually.

Usage:
    python -m tools.trace_stages IN_IMAGE OUT_DIR [--no-fill]

Stages written to OUT_DIR (prefixed 0..):
    00_canonical_rgb.png      what the model actually sees (post ingest)
    01_model_alpha.png        raw alpha matte from the segmenter (grayscale)
    02_model_over_checker.png model RGBA composited over checkerboard
    03_enclosed_holes.png      all enclosed transparent holes (white = hole)
    04_selected_fill_mask.png  central holes selected to fill (white = fill)
    05_fill_weight.png         dilate+feather blend weight (0..255)
    06_before_fill_checker.png result before fill, over checkerboard
    07_after_fill_checker.png  result after fill, over checkerboard (final)
    08_after_fill_alpha.png    final alpha matte (grayscale)
Plus zoomed crops around the fill region (``*_crop.png``) when a hole was filled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from app.composite import composite_over_checkerboard
from app.config import Config
from app.ingest import load_image_with_box
from app.pipeline import BackgroundRemover


def _save_gray(mask: np.ndarray, path: Path) -> None:
    """Save a boolean or 0..255 array as a grayscale PNG."""
    if mask.dtype == bool:
        arr = (mask.astype(np.uint8)) * 255
    else:
        arr = np.clip(mask, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def _save_rgb(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr, mode="RGB").save(path)


def _crop_box(mask: np.ndarray, pad: int = 80) -> tuple[int, int, int, int] | None:
    """Bounding box of True pixels, padded, or None if mask empty."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    h, w = mask.shape
    top = max(0, int(ys.min()) - pad)
    bottom = min(h, int(ys.max()) + pad)
    left = max(0, int(xs.min()) - pad)
    right = min(w, int(xs.max()) + pad)
    return left, top, right, bottom


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    in_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    flags = sys.argv[3:]
    fill = "--no-fill" not in flags
    restore_drops = "--restore-drops" in flags
    out_dir.mkdir(parents=True, exist_ok=True)

    config = Config(fill_holes=fill, fill_holes_restore_drops=restore_drops)
    remover = BackgroundRemover(config)

    # --- stage 0: ingest (what the model sees) ---
    image, pad_box = load_image_with_box(in_path, config)
    _save_rgb(image, out_dir / "00_canonical_rgb.png")
    print(f"canonical image: {image.shape}, pad_box={pad_box}")

    # --- stage 1/2: model output (before any hole fill) ---
    rgba = remover._run_oneshot(image)  # noqa: SLF001 - tracing internals on purpose
    _save_gray(rgba[..., 3], out_dir / "01_model_alpha.png")
    _save_rgb(composite_over_checkerboard(rgba), out_dir / "02_model_over_checker.png")
    _save_rgb(composite_over_checkerboard(rgba), out_dir / "06_before_fill_checker.png")

    # --- stage 3/4/5: hole detection + selection + blend weight ---
    from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter

    from app.holes import _select_holes  # noqa: SLF001

    h, w = rgba.shape[:2]
    alpha = rgba[..., 3]
    cutoff = int(round(np.clip(config.fill_holes_threshold, 0.0, 1.0) * 255.0))
    foreground = alpha >= cutoff
    enclosed = binary_fill_holes(foreground) & ~foreground
    _save_gray(enclosed, out_dir / "03_enclosed_holes.png")
    print(f"enclosed hole pixels: {int(enclosed.sum())}")

    fill_mask = _select_holes(enclosed, h, w, config) if enclosed.any() else np.zeros_like(enclosed)
    _save_gray(fill_mask, out_dir / "04_selected_fill_mask.png")
    print(f"selected fill pixels: {int(fill_mask.sum())}")

    if restore_drops and enclosed.any():
        from app.holes import _classify_drops  # noqa: SLF001

        drop_mask = _classify_drops(image, rgba, enclosed, config)
        _save_gray(drop_mask, out_dir / "04b_drop_mask.png")
        print(f"classified drop pixels: {int(drop_mask.sum())}")
        # Update the crop region to the drop so the close-up frames the fix.
        if drop_mask.any():
            fill_mask = drop_mask

    if fill_mask.any():
        dilate = int(config.fill_holes_dilate)
        region = binary_dilation(fill_mask, iterations=dilate) if dilate > 0 else fill_mask
        weight = region.astype(np.float32)
        if config.fill_holes_feather > 0:
            weight = np.clip(gaussian_filter(weight, sigma=float(config.fill_holes_feather)), 0, 1)
        weight = np.maximum(weight, fill_mask.astype(np.float32))
        _save_gray(weight * 255.0, out_dir / "05_fill_weight.png")

    # --- stage 7/8: final result after fill ---
    final = remover.process(in_path)
    # process() crops when restore_original_size; default keeps canonical size.
    _save_rgb(composite_over_checkerboard(final), out_dir / "07_after_fill_checker.png")
    _save_gray(final[..., 3], out_dir / "08_after_fill_alpha.png")

    # --- zoom crops around the filled region for close inspection ---
    box = _crop_box(fill_mask) if fill_mask.any() else None
    if box is not None:
        left, top, right, bottom = box
        print(f"crop box around fill: {box}")
        for name in ("00_canonical_rgb", "02_model_over_checker", "07_after_fill_checker"):
            src = np.asarray(Image.open(out_dir / f"{name}.png").convert("RGB"))
            _save_rgb(src[top:bottom, left:right], out_dir / f"{name}_crop.png")

    print(f"done -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
