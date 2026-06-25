"""Phase 1 image ingest: decode any reasonable input to a canonical array.

The rest of the pipeline assumes a single, predictable image representation
(see ``app/protocols.py``): an ``(H, W, 3)`` ``uint8`` RGB array in the sRGB
colour space with no embedded ICC profile and EXIF orientation already applied.
This module is the one place that normalizes arbitrary uploads into that form.

Accepted sources: filesystem path, raw encoded bytes, ``PIL.Image.Image``, or an
existing ``numpy`` array. Anything undecodable raises :class:`IngestError` with a
useful message, so a bad upload never crashes the caller.

Phase 2 will extend this module with a dual-resolution representation
(full-res edge truth + model-res resize) and ``pyvips`` tiling for >~30 MP
inputs. Phase 1 keeps a single full-resolution array.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError

from app.config import Config

logger = logging.getLogger(__name__)

# Source types this module knows how to turn into a canonical RGB array.
ImageSource = Union[str, Path, bytes, bytearray, Image.Image, np.ndarray]


class IngestError(ValueError):
    """Raised when an input cannot be decoded into a valid RGB image."""


def load_image(
    source: ImageSource, config: Optional[Config] = None
) -> np.ndarray:
    """Decode and normalize any supported source to a canonical RGB array.

    The returned array is the contract every downstream stage relies on:
    full-resolution, 3-channel RGB, ``uint8``, sRGB, EXIF orientation applied,
    ICC profile dropped.

    When ``config.canonical_size`` is set (default ``2362``), every image is
    normalized to a true ``canonical_size`` x ``canonical_size`` square: inputs
    that fit are centered on a ``config.pad_color`` canvas with no scaling
    (lossless), and oversized inputs are downscaled to fit (aspect preserved)
    before centering. The white border is later removed as background.

    Args:
        source: A filesystem path (``str`` / ``Path``), raw encoded image bytes,
            a ``PIL.Image.Image``, or a ``numpy`` array. Arrays are assumed to
            already be sRGB; ``(H, W, 3)`` uint8 is taken as RGB and ``(H, W)``
            or ``(H, W, 1)`` as grayscale (expanded to RGB). ``(H, W, 4)`` RGBA
            arrays are flattened by dropping the alpha channel.
        config: Optional pipeline configuration. Only its canonicalization
            fields (``canonical_size``, ``pad_color``) are read here. When
            ``None``, defaults are used.

    Returns:
        RGB image as ``np.ndarray`` of shape ``(H, W, 3)``, dtype ``uint8``.

    Raises:
        IngestError: If the source is empty, of an unsupported type, or cannot
            be decoded as an image.
    """
    array, _ = load_image_with_box(source, config)
    return array


# Crop box (left, top, right, bottom) locating the original image inside a
# canonicalized array, or ``None`` when no padding was applied.
PadBox = Optional[Tuple[int, int, int, int]]


def load_image_with_box(
    source: ImageSource, config: Optional[Config] = None
) -> Tuple[np.ndarray, PadBox]:
    """Like :func:`load_image`, but also return where the original sits.

    The second return value is the crop box ``(left, top, right, bottom)`` of
    the original image within the canonicalized array, or ``None`` when no
    padding was applied. Cropping a same-sized output (e.g. the RGBA result)
    with this box restores the original input dimensions losslessly.

    Args:
        source: See :func:`load_image`.
        config: See :func:`load_image`.

    Returns:
        Tuple ``(array, pad_box)`` where ``array`` is the canonical RGB image
        and ``pad_box`` is the original's crop box, or ``None`` if unpadded.

    Raises:
        IngestError: As in :func:`load_image`.
    """
    config = config or Config()
    image = _to_pil(source)

    # Apply EXIF orientation before any geometry- or colour-dependent step so
    # width/height and pixel positions are physically correct.
    image = ImageOps.exif_transpose(image)

    # Colour-manage to sRGB when an ICC profile is present, then drop it. With
    # no profile we assume the data is already sRGB (the common web case).
    image = _to_srgb(image)

    # Collapse any mode (palette, grayscale, RGBA, CMYK, ...) to plain RGB.
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Canonicalize size by white-padding small inputs to a uniform square.
    image, pad_box = _pad_to_canonical(image, config.canonical_size, config.pad_color)

    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise IngestError(
            f"Decoded image has unexpected shape {array.shape!r}; expected (H, W, 3)."
        )
    return array, pad_box


def _pad_to_canonical(
    image: Image.Image,
    canonical_size: Optional[int],
    pad_color: Tuple[int, int, int],
) -> Tuple[Image.Image, PadBox]:
    """Fit an image to a true ``canonical_size`` x ``canonical_size`` square.

    The result is always exactly ``canonical_size`` on both edges:

    * Inputs that fit within the square (both dimensions <= ``canonical_size``)
      are centered on a ``pad_color`` canvas with no scaling — lossless, the
      subject keeps its original pixels.
    * Inputs that exceed the square in either dimension are first downscaled
      (aspect-ratio preserved, Lanczos) to fit, then centered on the canvas.
      This is the only case that resamples pixels.

    Args:
        image: An RGB image (mode already normalized).
        canonical_size: Target square edge in pixels, or ``None`` to disable.
        pad_color: RGB fill for the added border.

    Returns:
        Tuple ``(image, pad_box)``. When the input is already exactly
        ``canonical_size`` square, the original image and ``None``. Otherwise a
        new ``canonical_size`` square RGB image with the source centered, plus
        the crop box ``(left, top, right, bottom)`` locating the (possibly
        downscaled) source within it.
    """
    if not canonical_size:
        return image, None

    width, height = image.size

    # Downscale oversized inputs so they fit the square (aspect preserved).
    if width > canonical_size or height > canonical_size:
        scale = canonical_size / max(width, height)
        width = max(1, round(width * scale))
        height = max(1, round(height * scale))
        image = image.resize((width, height), Image.LANCZOS)

    if (width, height) == (canonical_size, canonical_size):
        return image, None

    left = (canonical_size - width) // 2
    top = (canonical_size - height) // 2
    canvas = Image.new("RGB", (canonical_size, canonical_size), pad_color)
    canvas.paste(image, (left, top))
    return canvas, (left, top, left + width, top + height)


def _to_pil(source: ImageSource) -> Image.Image:
    """Turn a supported source into a loaded ``PIL.Image.Image``.

    Args:
        source: See :func:`load_image`.

    Returns:
        A decoded PIL image (mode not yet normalized).

    Raises:
        IngestError: If the source is empty, an unsupported type, or undecodable.
    """
    if isinstance(source, Image.Image):
        return source

    if isinstance(source, np.ndarray):
        return _pil_from_array(source)

    if isinstance(source, (bytes, bytearray)):
        if len(source) == 0:
            raise IngestError("Cannot decode image: input bytes are empty.")
        try:
            image = Image.open(io.BytesIO(bytes(source)))
            image.load()
            return image
        except (UnidentifiedImageError, OSError) as exc:
            raise IngestError(f"Cannot decode image from bytes: {exc}") from exc

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise IngestError(f"Image path does not exist or is not a file: {path}")
        try:
            image = Image.open(path)
            image.load()
            return image
        except (UnidentifiedImageError, OSError) as exc:
            raise IngestError(f"Cannot decode image at {path}: {exc}") from exc

    raise IngestError(f"Unsupported image source type: {type(source).__name__}")


def _pil_from_array(array: np.ndarray) -> Image.Image:
    """Build a PIL image from a numpy array, validating shape and dtype.

    Args:
        array: Image array. Supports ``(H, W)``, ``(H, W, 1)``, ``(H, W, 3)``,
            and ``(H, W, 4)``; dtype must be ``uint8``.

    Returns:
        A PIL image (grayscale arrays expanded to RGB by PIL on conversion).

    Raises:
        IngestError: If the array has an unsupported shape or dtype.
    """
    if array.dtype != np.uint8:
        raise IngestError(
            f"Array image must be dtype uint8, got {array.dtype}."
        )
    if array.ndim == 2:
        return Image.fromarray(array, mode="L")
    if array.ndim == 3:
        channels = array.shape[2]
        if channels == 1:
            return Image.fromarray(array[:, :, 0], mode="L")
        if channels == 3:
            return Image.fromarray(array, mode="RGB")
        if channels == 4:
            return Image.fromarray(array, mode="RGBA")
    raise IngestError(
        f"Array image has unsupported shape {array.shape!r}; "
        "expected (H, W), (H, W, 1), (H, W, 3), or (H, W, 4)."
    )


def _to_srgb(image: Image.Image) -> Image.Image:
    """Convert an image to sRGB using its embedded ICC profile, then drop it.

    If the image carries no ICC profile it is assumed to already be sRGB and is
    returned unchanged (aside from a defensive profile drop). A malformed or
    unconvertible profile is treated as "assume sRGB" rather than a hard error,
    since the goal is robustness against arbitrary uploads.

    Args:
        image: The (already EXIF-transposed) source image.

    Returns:
        An image in the sRGB space with no embedded ICC profile.
    """
    icc = image.info.get("icc_profile")
    if not icc:
        return image

    try:
        source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        srgb_profile = ImageCms.createProfile("sRGB")
        converted = ImageCms.profileToProfile(
            image,
            source_profile,
            srgb_profile,
            outputMode="RGB",
        )
        if converted is not None:
            converted.info.pop("icc_profile", None)
            return converted
    except (ImageCms.PyCMSError, OSError, ValueError) as exc:
        # Unreadable/unsupported profile: fall back to assuming sRGB, but make
        # the silent colour reinterpretation auditable. A wide-gamut or CMYK
        # input whose profile fails here is processed as-is, which can shift
        # colours and subtly degrade the mask (security/MLE finding H2).
        logger.warning(
            "ICC profile conversion failed; assuming sRGB without conversion: %s",
            exc,
        )

    image.info.pop("icc_profile", None)
    return image
