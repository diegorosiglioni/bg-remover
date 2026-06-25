"""Compositing: pack recovered foreground + alpha into output RGBA.

Given the defringed foreground colour ``F`` and the alpha matte, this produces
the final image. Two output shapes:

* Transparent: an RGBA image carrying ``F`` and alpha. Straight (unassociated)
  alpha by default — the PNG standard, halo-free because ``F`` is already clean.
  Premultiplied (associated) alpha is available via configuration for pipelines
  that require it.
* Flat background: ``F`` composited over a solid colour, yielding an opaque
  RGBA image where ``result = alpha * F + (1 - alpha) * background``.

Engine-independent: identical on CPU (phase 1) and GPU (phase 2).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.config import Config

# Default checkerboard appearance for transparency previews.
_CHECKER_SQUARE = 16
_CHECKER_LIGHT = (255, 255, 255)
_CHECKER_DARK = (204, 204, 204)


def composite_rgba(
    foreground: np.ndarray, alpha: np.ndarray, config: Config
) -> np.ndarray:
    """Compose the final RGBA image from foreground colour and alpha.

    Args:
        foreground: True foreground colour, shape ``(H, W, 3)``. Accepts
            ``float`` in ``[0, 1]`` or ``uint8`` in ``[0, 255]``.
        alpha: Alpha matte, shape ``(H, W)``, dtype float, range ``[0, 1]``.
        config: Pipeline configuration (background colour, premultiply flag).

    Returns:
        RGBA image, shape ``(H, W, 4)``, dtype ``uint8``. Opaque when
        ``config.background_color`` is set, otherwise carrying the alpha matte.

    Raises:
        ValueError: If foreground/alpha shapes are inconsistent or malformed.
    """
    fg = _normalize_foreground(foreground)
    _validate(fg, alpha)
    a = np.clip(alpha.astype(np.float32), 0.0, 1.0)

    if config.background_color is not None:
        return _composite_over_background(fg, a, config.background_color)

    rgb = fg * a[..., None] if config.premultiplied_output else fg
    rgba = np.dstack([rgb, a])
    return _to_uint8(rgba)


def encode_png(rgba: np.ndarray) -> bytes:
    """Encode an RGBA array as PNG bytes.

    Args:
        rgba: RGBA image, shape ``(H, W, 4)``, dtype ``uint8``.

    Returns:
        PNG-encoded bytes.

    Raises:
        ValueError: If the array is not ``(H, W, 4)`` uint8.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError(
            f"encode_png expects (H, W, 4) uint8; got shape {rgba.shape!r} "
            f"dtype {rgba.dtype}."
        )
    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def make_checkerboard(
    height: int,
    width: int,
    square: int = _CHECKER_SQUARE,
    light: tuple = _CHECKER_LIGHT,
    dark: tuple = _CHECKER_DARK,
) -> np.ndarray:
    """Build a two-tone checkerboard image.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        square: Side length of each checker square in pixels.
        light: RGB tuple for light squares (0..255).
        dark: RGB tuple for dark squares (0..255).

    Returns:
        RGB image, shape ``(height, width, 3)``, dtype ``uint8``.

    Raises:
        ValueError: If ``square`` is not positive.
    """
    if square <= 0:
        raise ValueError(f"square must be positive; got {square}.")
    yy, xx = np.mgrid[0:height, 0:width]
    is_dark = ((xx // square) + (yy // square)) % 2 == 1
    board = np.empty((height, width, 3), dtype=np.uint8)
    board[~is_dark] = np.array(light, dtype=np.uint8)
    board[is_dark] = np.array(dark, dtype=np.uint8)
    return board


def composite_over_checkerboard(
    rgba: np.ndarray, square: int = _CHECKER_SQUARE
) -> np.ndarray:
    """Composite a straight-alpha RGBA image over a transparency checkerboard.

    Produces an opaque RGB preview so a cutout is visible at a glance. Assumes
    straight (unassociated) alpha, which is the pipeline's default output.

    Args:
        rgba: RGBA image, shape ``(H, W, 4)``, dtype ``uint8``.
        square: Checker square side length in pixels.

    Returns:
        Opaque RGB image, shape ``(H, W, 3)``, dtype ``uint8``.

    Raises:
        ValueError: If ``rgba`` is not an ``(H, W, 4)`` uint8 array.
    """
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError(
            f"composite_over_checkerboard expects (H, W, 4) uint8; "
            f"got shape {rgba.shape!r} dtype {rgba.dtype}."
        )
    height, width = rgba.shape[:2]
    board = make_checkerboard(height, width, square=square).astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3].astype(np.float32)[..., None] / 255.0
    blended = alpha * rgb + (1.0 - alpha) * board
    return _to_uint8(blended)


def _composite_over_background(
    fg: np.ndarray, alpha: np.ndarray, background_color: tuple
) -> np.ndarray:
    """Composite foreground over a solid background, returning opaque RGBA.

    Args:
        fg: Foreground colour, shape ``(H, W, 3)``, float in ``[0, 1]``.
        alpha: Alpha matte, shape ``(H, W)``, float in ``[0, 1]``.
        background_color: RGBA tuple in ``[0, 255]``; alpha component ignored
            for the colour blend (output is forced opaque).

    Returns:
        Opaque RGBA image, shape ``(H, W, 4)``, dtype ``uint8``.
    """
    bg_rgb = np.array(background_color[:3], dtype=np.float32) / 255.0
    a = alpha[..., None]
    blended = a * fg + (1.0 - a) * bg_rgb
    opaque = np.ones(fg.shape[:2], dtype=np.float32)
    rgba = np.dstack([blended, opaque])
    return _to_uint8(rgba)


def _normalize_foreground(foreground: np.ndarray) -> np.ndarray:
    """Coerce foreground to float32 ``[0, 1]``.

    Args:
        foreground: Foreground array, float ``[0, 1]`` or uint8 ``[0, 255]``.

    Returns:
        Foreground as ``float32`` in ``[0, 1]``.
    """
    if foreground.dtype == np.uint8:
        return foreground.astype(np.float32) / 255.0
    return np.clip(foreground.astype(np.float32), 0.0, 1.0)


def _to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert a float image in ``[0, 1]`` to ``uint8`` with rounding.

    Args:
        array: Float array in ``[0, 1]``.

    Returns:
        ``uint8`` array in ``[0, 255]``.
    """
    return np.clip(np.round(array * 255.0), 0, 255).astype(np.uint8)


def _validate(foreground: np.ndarray, alpha: np.ndarray) -> None:
    """Validate foreground/alpha shape agreement.

    Args:
        foreground: Candidate foreground array (float, normalized).
        alpha: Candidate alpha array.

    Raises:
        ValueError: If shapes are invalid.
    """
    if foreground.ndim != 3 or foreground.shape[2] != 3:
        raise ValueError(f"foreground must be (H, W, 3); got {foreground.shape!r}.")
    if alpha.ndim != 2:
        raise ValueError(f"alpha must be (H, W); got {alpha.shape!r}.")
    if alpha.shape != foreground.shape[:2]:
        raise ValueError(
            f"alpha shape {alpha.shape!r} does not match foreground "
            f"{foreground.shape[:2]!r}."
        )
