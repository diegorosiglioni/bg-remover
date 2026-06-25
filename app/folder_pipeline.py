"""Folder-aware batch pipeline with two per-folder strategies.

Given an input root whose immediate (or nested) subfolders each hold the images
of one object, this processes every image to a background-removed PNG and
mirrors the input tree under an output root. The strategy is chosen per folder
by its name:

* **Folders whose name contains "pendant"** -> run every image through the full
  AI engine (:class:`~app.pipeline.BackgroundRemover`), the same per-image
  removal the standard CLI does.
* **Folders without "pendant"** -> *shared-mask* removal. The subject silhouette
  is taken from the **second image alphabetically**: that one image is run
  through the AI engine, its alpha is binarized into one mask, and that single
  mask is applied to **every** image in the folder (RGB kept inside the mask,
  transparent outside). One inference per folder instead of one per image, and a
  perfectly consistent cutout across a turntable/aligned set.

The mask is the binarized engine alpha used directly (no contour-fill step), so
interior openings in the silhouette stay transparent as the model saw them.

Run as a module::

    python -m app.folder_pipeline ROOT OUT_DIR [options]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from app.cli import IMAGE_EXTENSIONS, _is_within, _write_preview
from app.composite import encode_png
from app.config import Config
from app.ingest import load_image
from app.masks import load_mask, mask_path, resize_mask, save_mask
from app.pipeline import BackgroundRemover

logger = logging.getLogger(__name__)

# Substring (case-insensitive) in a folder name that selects the per-image AI
# branch instead of the shared-mask branch.
PENDANT_TOKEN = "pendant"

# Default alpha cut: a pixel is foreground when its 0..255 alpha is at/above it.
DEFAULT_MASK_THRESHOLD = 128

# Native-resolution ingest config: decode + EXIF + sRGB, but no canonical
# padding/scaling, so each target image is masked and written at its own
# original dimensions.
_NATIVE_INGEST = Config(canonical_size=None)


class Remover(Protocol):
    """The single capability this pipeline needs from the AI engine."""

    def process(self, source) -> np.ndarray:  # pragma: no cover - structural
        """Return an ``(H, W, 4)`` uint8 RGBA cutout for one image source."""
        ...


def is_pendant(folder_name: str) -> bool:
    """Return True if a folder name selects the per-image AI branch.

    Args:
        folder_name: The bare folder name (not a full path).

    Returns:
        True when ``"pendant"`` appears anywhere in the name (case-insensitive).
    """
    return PENDANT_TOKEN in folder_name.lower()


def group_images_by_folder(root: Path) -> Dict[Path, List[Path]]:
    """Group every image under ``root`` by its containing directory.

    Walks the tree to any depth; each directory that *directly* holds images
    becomes one group (one object). Images are returned sorted by name so the
    "second image alphabetically" selection is well defined.

    Args:
        root: Input root directory.

    Returns:
        Mapping of folder path -> sorted list of image paths in that folder.
        Folders with no images are omitted.
    """
    groups: Dict[Path, List[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            groups.setdefault(path.parent, []).append(path)
    for folder in groups:
        groups[folder].sort(key=lambda p: p.name.lower())
    return dict(sorted(groups.items(), key=lambda kv: str(kv[0]).lower()))


def select_reference(images: Sequence[Path]) -> Path:
    """Pick the silhouette reference: the second image alphabetically.

    Args:
        images: Folder images, already sorted by name.

    Returns:
        ``images[1]`` when at least two exist, otherwise ``images[0]``.

    Raises:
        ValueError: If ``images`` is empty.
    """
    if not images:
        raise ValueError("cannot select a reference from an empty image list")
    return images[1] if len(images) >= 2 else images[0]


def mask_from_alpha(alpha: np.ndarray, threshold: int = DEFAULT_MASK_THRESHOLD) -> np.ndarray:
    """Binarize an alpha channel into a boolean foreground mask.

    Args:
        alpha: Alpha channel, shape ``(H, W)``, dtype ``uint8`` (0..255).
        threshold: A pixel is foreground when ``alpha >= threshold``.

    Returns:
        Boolean mask, shape ``(H, W)``.
    """
    return alpha >= threshold


def apply_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Cut ``rgb`` to RGBA using a boolean mask, resizing the mask if needed.

    The mask defines the kept interior; pixels outside it become fully
    transparent. RGB pixels are left untouched (no defringe) so the result is
    exactly the original colour inside the silhouette.

    Args:
        rgb: Source image, shape ``(H, W, 3)``, dtype ``uint8``.
        mask: Boolean foreground mask. Resized (nearest) to the image size when
            its shape differs, so one folder mask fits images of any size.

    Returns:
        RGBA image, shape ``(H, W, 4)``, dtype ``uint8``.
    """
    height, width = rgb.shape[:2]
    if mask.shape != (height, width):
        mask = _resize_mask(mask, (width, height))
    alpha = np.where(mask, np.uint8(255), np.uint8(0))
    return np.dstack([rgb, alpha])


def _resize_mask(mask: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize a boolean mask to ``(width, height)``.

    Thin alias kept for backward compatibility; delegates to
    :func:`app.masks.resize_mask`.

    Args:
        mask: Boolean mask, shape ``(H, W)``.
        size_wh: Target ``(width, height)`` in pixels.

    Returns:
        Boolean mask at the requested size.
    """
    return resize_mask(mask, size_wh)


def _destination(path: Path, root: Path, out_dir: Path) -> Path:
    """Mirror an input path under ``out_dir`` with a ``.png`` suffix.

    Args:
        path: Source image path.
        root: Input root the relative subpath is taken from.
        out_dir: Output root.

    Returns:
        The output PNG path (subfolders preserved, extension forced to ``.png``
        because the cutout carries transparency).
    """
    return out_dir / path.relative_to(root).with_suffix(".png")


def _write_result(
    rgba: np.ndarray,
    destination: Path,
    preview_dir: Optional[Path],
    relative: Path,
) -> None:
    """Encode an RGBA result to ``destination`` and optionally a preview.

    Args:
        rgba: RGBA result, shape ``(H, W, 4)`` uint8.
        destination: Output PNG path.
        preview_dir: Optional root for a checkerboard preview; ``None`` to skip.
        relative: Image path relative to the input root (for the preview path).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(encode_png(rgba))
    if preview_dir is not None:
        try:
            _write_preview(rgba, preview_dir / relative)
        except Exception as exc:  # preview is non-essential; never fail on it
            logger.debug("preview failure for %s: %s", relative, exc, exc_info=True)


def _guard_destination(
    path: Path, root: Path, out_dir: Path, skip_existing: bool
) -> Tuple[Optional[Path], Optional[str]]:
    """Resolve and validate the output path for one image.

    Args:
        path: Source image path.
        root: Input root.
        out_dir: Output root.
        skip_existing: When True, an already-present output is a skip.

    Returns:
        Tuple ``(destination, status)``. ``status`` is ``None`` to proceed,
        ``"skip"`` when the output exists and ``skip_existing`` is set, or an
        error string when the destination escapes ``out_dir`` (``destination``
        is then ``None``).
    """
    destination = _destination(path, root, out_dir)
    if not _is_within(destination, out_dir):
        return None, f"output path escapes {out_dir} -> {destination}"
    if skip_existing and destination.exists():
        return destination, "skip"
    return destination, None


def process_pendant_folder(
    remover: Remover,
    images: Sequence[Path],
    root: Path,
    out_dir: Path,
    preview_dir: Optional[Path] = None,
    skip_existing: bool = False,
) -> Tuple[int, int, int]:
    """Per-image AI removal for a pendant folder.

    Args:
        remover: The AI engine (anything with ``process(path) -> RGBA``).
        images: Images in the folder.
        root: Input root (for mirrored output paths).
        out_dir: Output root.
        preview_dir: Optional checkerboard-preview root.
        skip_existing: Skip images whose output PNG already exists.

    Returns:
        Tuple ``(n_ok, n_fail, n_skip)``.
    """
    n_ok = n_fail = n_skip = 0
    for path in images:
        destination, status = _guard_destination(path, root, out_dir, skip_existing)
        if status == "skip":
            n_skip += 1
            print(f"  {path.name} -> skip (exists)")
            continue
        if destination is None:
            n_fail += 1
            print(f"  {path.name} FAILED: {status}", file=sys.stderr)
            continue
        try:
            rgba = remover.process(path)
            _write_result(rgba, destination, preview_dir, path.relative_to(root))
            n_ok += 1
            print(f"  {path.name} -> {destination}")
        except Exception as exc:  # isolate: one bad file must not stop the batch
            n_fail += 1
            print(f"  {path.name} FAILED: {exc}", file=sys.stderr)
            logger.debug("pendant processing failure for %s", path, exc_info=True)
    return n_ok, n_fail, n_skip


def process_masked_folder(
    remover: Remover,
    images: Sequence[Path],
    root: Path,
    out_dir: Path,
    mask_threshold: int = DEFAULT_MASK_THRESHOLD,
    preview_dir: Optional[Path] = None,
    skip_existing: bool = False,
) -> Tuple[int, int, int]:
    """Shared-mask removal for a non-pendant folder.

    The silhouette of the second image (alphabetically) is computed once by the
    AI engine, binarized, and applied to every image in the folder.

    Args:
        remover: The AI engine (``process(path) -> RGBA``).
        images: Images in the folder (sorted by name).
        root: Input root.
        out_dir: Output root.
        mask_threshold: Alpha cut for binarizing the reference silhouette.
        preview_dir: Optional checkerboard-preview root.
        skip_existing: Skip images whose output PNG already exists.

    Returns:
        Tuple ``(n_ok, n_fail, n_skip)``. If the reference silhouette cannot be
        built, every image is counted as failed.
    """
    reference = select_reference(images)
    try:
        ref_rgba = remover.process(reference)
        mask = mask_from_alpha(ref_rgba[..., 3], mask_threshold)
    except Exception as exc:
        print(
            f"  reference {reference.name} FAILED: {exc}; whole folder skipped",
            file=sys.stderr,
        )
        logger.debug("mask reference failure for %s", reference, exc_info=True)
        return 0, len(images), 0

    if not mask.any():
        print(
            f"  warning: empty mask from {reference.name} "
            "(all-transparent silhouette); outputs will be fully transparent",
            file=sys.stderr,
        )
    print(f"  mask from {reference.name} ({int(mask.sum())} px foreground)")

    n_ok = n_fail = n_skip = 0
    for path in images:
        destination, status = _guard_destination(path, root, out_dir, skip_existing)
        if status == "skip":
            n_skip += 1
            print(f"  {path.name} -> skip (exists)")
            continue
        if destination is None:
            n_fail += 1
            print(f"  {path.name} FAILED: {status}", file=sys.stderr)
            continue
        try:
            rgb = load_image(path, _NATIVE_INGEST)
            rgba = apply_mask(rgb, mask)
            _write_result(rgba, destination, preview_dir, path.relative_to(root))
            n_ok += 1
            print(f"  {path.name} -> {destination}")
        except Exception as exc:  # isolate per-file failures
            n_fail += 1
            print(f"  {path.name} FAILED: {exc}", file=sys.stderr)
            logger.debug("masked processing failure for %s", path, exc_info=True)
    return n_ok, n_fail, n_skip


def run_pipeline(
    root: Path,
    out_dir: Path,
    remover: Remover,
    mask_threshold: int = DEFAULT_MASK_THRESHOLD,
    preview_dir: Optional[Path] = None,
    skip_existing: bool = False,
) -> Tuple[int, int, int]:
    """Process every folder under ``root``, dispatching by folder name.

    Args:
        root: Input root.
        out_dir: Output root (mirrors the input tree).
        remover: The AI engine (``process(path) -> RGBA``).
        mask_threshold: Alpha cut for the shared-mask branch.
        preview_dir: Optional checkerboard-preview root.
        skip_existing: Skip images whose output PNG already exists.

    Returns:
        Tuple ``(n_ok, n_fail, n_skip)`` aggregated across all folders.
    """
    groups = group_images_by_folder(root)
    total_ok = total_fail = total_skip = 0
    for folder, images in groups.items():
        rel = folder.relative_to(root) if folder != root else Path(".")
        if is_pendant(folder.name):
            print(f"[pendant] {rel} ({len(images)} images) -> AI per image")
            ok, fail, skip = process_pendant_folder(
                remover, images, root, out_dir, preview_dir, skip_existing
            )
        else:
            print(f"[masked]  {rel} ({len(images)} images) -> shared mask")
            ok, fail, skip = process_masked_folder(
                remover, images, root, out_dir, mask_threshold, preview_dir, skip_existing
            )
        total_ok += ok
        total_fail += fail
        total_skip += skip
    return total_ok, total_fail, total_skip


# --------------------------------------------------------------------------- #
# Step 1: generate_masks  (run the model, persist grayscale masks to masks/)
# --------------------------------------------------------------------------- #


def _native_size_wh(path: Path) -> Tuple[int, int]:
    """Return the native ``(width, height)`` of a source image.

    Uses the no-canonicalization ingest so the size matches exactly what the
    apply step loads (EXIF orientation applied, no padding/scaling).

    Args:
        path: Source image path.

    Returns:
        ``(width, height)`` in pixels.
    """
    rgb = load_image(path, _NATIVE_INGEST)
    height, width = rgb.shape[:2]
    return width, height


def _save_one_mask(
    mask: np.ndarray, source: Path, root: Path, masks_dir: Path
) -> Path:
    """Persist one boolean mask for ``source`` under ``masks_dir``.

    Args:
        mask: Boolean foreground mask (True = keep).
        source: Source image the mask belongs to.
        root: Input root.
        masks_dir: Mask root mirroring the input tree.

    Returns:
        The written mask path.
    """
    destination = mask_path(source, root, masks_dir)
    save_mask(mask, destination)
    return destination


def generate_pendant_masks(
    remover: Remover,
    images: Sequence[Path],
    root: Path,
    masks_dir: Path,
    mask_threshold: int = DEFAULT_MASK_THRESHOLD,
    skip_existing: bool = False,
) -> Tuple[int, int, int]:
    """Per-image AI masks for a pendant folder.

    Each image is run through the engine; its alpha is binarized and saved as a
    grayscale mask. One inference per image.

    Args:
        remover: The AI engine (``process(path) -> RGBA``).
        images: Images in the folder.
        root: Input root.
        masks_dir: Mask root (mirrors the input tree).
        mask_threshold: Alpha cut for binarizing each silhouette.
        skip_existing: Skip images whose mask PNG already exists.

    Returns:
        Tuple ``(n_ok, n_fail, n_skip)``.
    """
    n_ok = n_fail = n_skip = 0
    for path in images:
        destination = mask_path(path, root, masks_dir)
        if skip_existing and destination.exists():
            n_skip += 1
            print(f"  {path.name} -> skip (mask exists)")
            continue
        try:
            rgba = remover.process(path)
            mask = mask_from_alpha(rgba[..., 3], mask_threshold)
            save_mask(mask, destination)
            n_ok += 1
            print(f"  {path.name} -> {destination}")
        except Exception as exc:  # isolate: one bad file must not stop the batch
            n_fail += 1
            print(f"  {path.name} FAILED: {exc}", file=sys.stderr)
            logger.debug("pendant mask failure for %s", path, exc_info=True)
    return n_ok, n_fail, n_skip


def generate_masked_masks(
    remover: Remover,
    images: Sequence[Path],
    root: Path,
    masks_dir: Path,
    mask_threshold: int = DEFAULT_MASK_THRESHOLD,
    skip_existing: bool = False,
) -> Tuple[int, int, int]:
    """Shared-mask generation for a non-pendant folder.

    One inference on the second image (alphabetically) yields a single
    silhouette; it is saved as a mask for *every* image in the folder, resized
    to each image's native dimensions so the stored mask overlays its source
    cleanly in the editor.

    Args:
        remover: The AI engine (``process(path) -> RGBA``).
        images: Images in the folder (sorted by name).
        root: Input root.
        masks_dir: Mask root.
        mask_threshold: Alpha cut for binarizing the reference silhouette.
        skip_existing: Skip images whose mask PNG already exists.

    Returns:
        Tuple ``(n_ok, n_fail, n_skip)``. If the reference silhouette cannot be
        built, every image is counted as failed.
    """
    reference = select_reference(images)
    try:
        ref_rgba = remover.process(reference)
        mask = mask_from_alpha(ref_rgba[..., 3], mask_threshold)
    except Exception as exc:
        print(
            f"  reference {reference.name} FAILED: {exc}; whole folder skipped",
            file=sys.stderr,
        )
        logger.debug("mask reference failure for %s", reference, exc_info=True)
        return 0, len(images), 0

    if not mask.any():
        print(
            f"  warning: empty mask from {reference.name} "
            "(all-transparent silhouette)",
            file=sys.stderr,
        )
    print(f"  mask from {reference.name} ({int(mask.sum())} px foreground)")

    n_ok = n_fail = n_skip = 0
    for path in images:
        destination = mask_path(path, root, masks_dir)
        if skip_existing and destination.exists():
            n_skip += 1
            print(f"  {path.name} -> skip (mask exists)")
            continue
        try:
            width, height = _native_size_wh(path)
            sized = mask if mask.shape == (height, width) else resize_mask(
                mask, (width, height)
            )
            save_mask(sized, destination)
            n_ok += 1
            print(f"  {path.name} -> {destination}")
        except Exception as exc:  # isolate per-file failures
            n_fail += 1
            print(f"  {path.name} FAILED: {exc}", file=sys.stderr)
            logger.debug("masked mask failure for %s", path, exc_info=True)
    return n_ok, n_fail, n_skip


def generate_masks(
    root: Path,
    masks_dir: Path,
    remover: Remover,
    mask_threshold: int = DEFAULT_MASK_THRESHOLD,
    skip_existing: bool = False,
) -> Tuple[int, int, int]:
    """Step 1: run the model and persist a grayscale mask per source image.

    Dispatches each folder by name exactly like :func:`run_pipeline`: pendant
    folders get a per-image AI mask, others share one mask from their 2nd image.

    Args:
        root: Input root.
        masks_dir: Mask root (mirrors the input tree).
        remover: The AI engine (``process(path) -> RGBA``).
        mask_threshold: Alpha cut for binarizing silhouettes.
        skip_existing: Skip images whose mask PNG already exists.

    Returns:
        Tuple ``(n_ok, n_fail, n_skip)`` aggregated across all folders.
    """
    groups = group_images_by_folder(root)
    total_ok = total_fail = total_skip = 0
    for folder, images in groups.items():
        rel = folder.relative_to(root) if folder != root else Path(".")
        if is_pendant(folder.name):
            print(f"[pendant] {rel} ({len(images)} images) -> AI mask per image")
            ok, fail, skip = generate_pendant_masks(
                remover, images, root, masks_dir, mask_threshold, skip_existing
            )
        else:
            print(f"[masked]  {rel} ({len(images)} images) -> shared mask")
            ok, fail, skip = generate_masked_masks(
                remover, images, root, masks_dir, mask_threshold, skip_existing
            )
        total_ok += ok
        total_fail += fail
        total_skip += skip
    return total_ok, total_fail, total_skip


# --------------------------------------------------------------------------- #
# Step 2: apply_masks  (read masks/, cut sources, write output/)
# --------------------------------------------------------------------------- #


def apply_masks(
    root: Path,
    out_dir: Path,
    masks_dir: Path,
    preview_dir: Optional[Path] = None,
    skip_existing: bool = False,
) -> Tuple[int, int, int]:
    """Step 2: cut every source image with its stored mask, writing ``out_dir``.

    No model is loaded. Each image's mask is read from ``masks_dir`` and applied
    to the native-resolution source; the mask is resized to the source dims when
    they differ (so a hand-edited mask of any size still fits).

    Args:
        root: Input root.
        out_dir: Output root (mirrors the input tree).
        masks_dir: Mask root the per-image masks are read from.
        preview_dir: Optional checkerboard-preview root.
        skip_existing: Skip images whose output PNG already exists.

    Returns:
        Tuple ``(n_ok, n_fail, n_skip)`` aggregated across all folders. An image
        whose mask file is missing is counted as failed.
    """
    groups = group_images_by_folder(root)
    n_ok = n_fail = n_skip = 0
    for folder, images in groups.items():
        rel = folder.relative_to(root) if folder != root else Path(".")
        print(f"[apply]   {rel} ({len(images)} images)")
        for path in images:
            destination, status = _guard_destination(
                path, root, out_dir, skip_existing
            )
            if status == "skip":
                n_skip += 1
                print(f"  {path.name} -> skip (exists)")
                continue
            if destination is None:
                n_fail += 1
                print(f"  {path.name} FAILED: {status}", file=sys.stderr)
                continue
            source_mask = mask_path(path, root, masks_dir)
            if not source_mask.exists():
                n_fail += 1
                print(
                    f"  {path.name} FAILED: no mask at {source_mask}; "
                    "run --step generate_masks first",
                    file=sys.stderr,
                )
                continue
            try:
                rgb = load_image(path, _NATIVE_INGEST)
                mask = load_mask(source_mask)
                rgba = apply_mask(rgb, mask)
                _write_result(rgba, destination, preview_dir, path.relative_to(root))
                n_ok += 1
                print(f"  {path.name} -> {destination}")
            except Exception as exc:  # isolate per-file failures
                n_fail += 1
                print(f"  {path.name} FAILED: {exc}", file=sys.stderr)
                logger.debug("apply failure for %s", path, exc_info=True)
    return n_ok, n_fail, n_skip


def _build_config(args: argparse.Namespace) -> Config:
    """Build the engine config from CLI overrides.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A :class:`~app.config.Config` for the AI engine.
    """
    overrides: dict = {}
    if args.model is not None:
        overrides["model_name"] = args.model
    return Config(**overrides)


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Define and parse command-line arguments.

    Args:
        argv: Optional argument vector.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="python -m app.folder_pipeline",
        description=(
            "Folder-aware background removal: 'pendant' folders use the AI model "
            "per image; other folders share one mask from their 2nd image."
        ),
    )
    parser.add_argument("root", help="Input root (subfolders = objects).")
    parser.add_argument("out_dir", help="Output root (mirrors the input tree).")
    parser.add_argument("--model", default=None, help="rembg/ONNX model name.")
    parser.add_argument(
        "--mask-threshold",
        type=int,
        default=DEFAULT_MASK_THRESHOLD,
        metavar="0-255",
        help="Alpha cut for the shared-mask branch (default 128).",
    )
    parser.add_argument(
        "--preview-dir",
        default=None,
        help="Also write a checkerboard preview per image to this folder.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose output PNG already exists.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Emit DEBUG logs and tracebacks."
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``python -m app.folder_pipeline``.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` if every image succeeded, ``1`` if any failed, ``2`` for bad
        arguments or an empty/missing input tree.
    """
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    if not root.is_dir():
        print(f"error: input root not found: {root}", file=sys.stderr)
        return 2
    if not 0 <= args.mask_threshold <= 255:
        print("error: --mask-threshold must be in [0, 255]", file=sys.stderr)
        return 2

    groups = group_images_by_folder(root)
    if not groups:
        print(f"error: no images found under {root}", file=sys.stderr)
        return 2

    try:
        config = _build_config(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Loading model {config.model_name!r} "
        f"(engine={config.engine}, device={config.device})..."
    )
    remover = BackgroundRemover(config)

    n_ok, n_fail, n_skip = run_pipeline(
        root, out_dir, remover, args.mask_threshold, preview_dir, args.skip_existing
    )
    total = n_ok + n_fail + n_skip
    print(f"\nDone: {n_ok} succeeded, {n_fail} failed, {n_skip} skipped, {total} total.")
    return 1 if n_fail else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
