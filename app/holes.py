"""Interior hole filling (keep central holes as in the original).

When the cutout would leave a transparent hole *fully enclosed* by the subject
near the centre of the frame, this restores the original image content there
(opaque) instead of leaving it see-through. It targets small, central holes —
a ring's eye, a small gap the user wants kept solid — while leaving larger or
off-centre holes transparent.

:func:`fill_alpha_holes` selects a hole when it is **central** — its centroid is
within ``fill_holes_center_dist`` of the image centre (distance normalized so 0
is the centre and 1.0 is a corner). Size does not matter: a large central
opening (a horseshoe gap) is filled too.

Selected holes are made opaque and the original image content is pasted back
in. The strict hole interior is copied 1:1 from the source; the fill is then
grown a few pixels (``fill_holes_dilate``) so it also covers the thin
matting-contaminated ring around the hole, and the outer boundary is feathered
(``fill_holes_feather``) so the patch blends seamlessly into the surrounding
result instead of leaving a hard seam. Holes that touch the image border are
never enclosed, so they are never considered.

Engine-independent: operates on the final RGBA plus the canonical source image,
so it behaves identically for both pipeline modes and for phase 2.
"""

from __future__ import annotations

import numpy as np

from app.config import Config


def fill_alpha_holes(
    image: np.ndarray, rgba: np.ndarray, config: Config
) -> np.ndarray:
    """Fill central enclosed holes with the original image content.

    Args:
        image: Canonical source RGB image, shape ``(H, W, 3)``, dtype ``uint8``
            (same spatial size as ``rgba``). Supplies the colour pasted into the
            holes; the hole core is copied 1:1, the dilated boundary is feathered
            into the surrounding result.
        rgba: Pipeline RGBA result, shape ``(H, W, 4)``, dtype ``uint8``.
        config: Pipeline configuration (``fill_holes_threshold``,
            ``fill_holes_center_dist``, ``fill_holes_dilate``,
            ``fill_holes_feather``).

    Returns:
        A new RGBA array with qualifying holes made opaque and pasted from the
        original image, with a dilated and feathered edge. The input is not
        mutated. When nothing qualifies, an unchanged copy is returned.

    Raises:
        ValueError: If shapes/dtypes are invalid or sizes disagree.
    """
    _validate(image, rgba)

    # Lazy import: scipy is a transitive dependency (via pymatting) but only the
    # hole-filling path needs it, so keep it out of the common import cost.
    from scipy.ndimage import binary_fill_holes

    height, width = rgba.shape[:2]
    alpha = rgba[..., 3]
    cutoff = int(round(np.clip(config.fill_holes_threshold, 0.0, 1.0) * 255.0))
    foreground = alpha >= cutoff

    # All transparent regions fully enclosed by foreground (border-touching
    # background is excluded by construction).
    enclosed = binary_fill_holes(foreground) & ~foreground
    if not enclosed.any():
        return rgba.copy()

    # Content-aware path (opt-in): split enclosed holes into saliency drops
    # (restore hard-edged from the original) and the rest (legacy eyelet fill).
    drop_mask = (
        _classify_drops(image, rgba, enclosed, config)
        if config.fill_holes_restore_drops
        else np.zeros_like(enclosed)
    )

    # Legacy geometry-selected central holes, excluding anything already claimed
    # as a drop so the two passes operate on disjoint pixels.
    fill_mask = _select_holes(enclosed, height, width, config) & ~drop_mask

    if not drop_mask.any() and not fill_mask.any():
        return rgba.copy()

    out = rgba
    if drop_mask.any():
        out = _restore_drop(image, out, drop_mask, rgba, config)
    if fill_mask.any():
        out = _paste_original(image, out, fill_mask, config)
    # _restore_drop / _paste_original always return a fresh array, so `out` is
    # never the input buffer here; the early-return above covers the no-op case.
    return out


def _paste_original(
    image: np.ndarray,
    rgba: np.ndarray,
    fill_mask: np.ndarray,
    config: Config,
) -> np.ndarray:
    """Paste original content into the holes with a dilated, feathered edge.

    The hole core (``fill_mask``) is copied 1:1 from ``image``. The fill is
    optionally grown by ``fill_holes_dilate`` pixels so it also overwrites the
    thin matting-contaminated ring around the hole, then the boundary is
    softened with a Gaussian of sigma ``fill_holes_feather`` so the patch blends
    into the surrounding result instead of leaving a hard seam.

    Args:
        image: Canonical source RGB image, shape ``(H, W, 3)``, dtype ``uint8``.
        rgba: Pipeline RGBA result, shape ``(H, W, 4)``, dtype ``uint8``.
        fill_mask: Boolean mask of the selected hole cores.
        config: Pipeline configuration (``fill_holes_dilate``,
            ``fill_holes_feather``).

    Returns:
        A new RGBA array with the holes filled. The input is not mutated.
    """
    from scipy.ndimage import binary_dilation, gaussian_filter

    dilate = int(config.fill_holes_dilate)
    region = (
        binary_dilation(fill_mask, iterations=dilate) if dilate > 0 else fill_mask
    )

    # Soft per-pixel blend weight: 1.0 inside the fill, fading to 0.0 outside.
    weight = region.astype(np.float32)
    feather = float(config.fill_holes_feather)
    if feather > 0.0:
        weight = np.clip(gaussian_filter(weight, sigma=feather), 0.0, 1.0)
    # The hole core must stay fully original even after blurring pulled it down.
    weight = np.maximum(weight, fill_mask.astype(np.float32))

    out = rgba.copy()
    w = weight[..., None]
    # Blend original over the existing result colour. Interior pixels are opaque
    # in the source, and the surrounding band is foreground (alpha ~ 1), so this
    # is correct for straight alpha and a close match for premultiplied alpha
    # (where colour == straight colour at alpha 1).
    src = image.astype(np.float32)
    dst = out[..., :3].astype(np.float32)
    blended = w * src + (1.0 - w) * dst
    out[..., :3] = np.clip(np.round(blended), 0, 255).astype(np.uint8)

    # Drive alpha opaque across the blended region (weighted so the feather band
    # does not punch a hard alpha edge of its own).
    alpha = out[..., 3].astype(np.float32)
    new_alpha = np.maximum(alpha, weight * 255.0)
    out[..., 3] = np.clip(np.round(new_alpha), 0, 255).astype(np.uint8)
    return out


# Alpha boundaries (0..255) for the soft-ramp test. A pixel between these is
# "uncertain" matting — the signature of a saliency drop on reflective material.
_RAMP_LOW = 26  # ~0.1
_RAMP_HIGH = 230  # ~0.9
# Ring width (px) sampled just outside a hole to characterize the bordering
# subject material and the alpha ramp.
_BORDER_RING_PX = 6
# How much darker (0..255 luminance) a hole interior may be than its bordering
# subject and still count as a (bright/specular) drop. Rejects genuine dark gaps.
_DROP_DARK_TOL = 40.0


def _classify_drops(
    image: np.ndarray,
    rgba: np.ndarray,
    enclosed: np.ndarray,
    config: Config,
) -> np.ndarray:
    """Mask of enclosed holes that are saliency drops (not ring eyelets).

    The discriminating signal is the **alpha ramp** at the hole boundary, not
    colour. On these white-lightbox catalogue shots the original pixels inside a
    real see-through eyelet (white background) and inside a dropped specular
    metal body (white reflection) are both near-white, so colour cannot separate
    them. What differs is model confidence: a correctly removed eyelet has a
    crisp alpha edge (little ramp), whereas a saliency drop on reflective metal
    leaves a wide soft semi-transparent ramp. A component is therefore a drop
    when it is bordered by solid subject, its boundary band is mostly a soft
    ramp (``fill_holes_drop_ramp_min``), and its interior is not markedly darker
    than the bordering subject (specular drops are bright, so this rejects
    genuine dark enclosed gaps).

    Args:
        image: Canonical source RGB, ``(H, W, 3)`` uint8.
        rgba: Pipeline RGBA result, ``(H, W, 4)`` uint8.
        enclosed: Boolean mask of all enclosed transparent holes.
        config: Pipeline configuration (drop thresholds).

    Returns:
        Boolean mask (subset of ``enclosed``) of components classified as drops.
    """
    from scipy.ndimage import binary_dilation, label

    labels, count = label(enclosed)
    if count == 0:
        return np.zeros_like(enclosed)

    alpha = rgba[..., 3]
    img = image.astype(np.float32)
    ramp_min = float(config.fill_holes_drop_ramp_min)

    drop = np.zeros_like(enclosed)
    for cid in range(1, count + 1):
        component = labels == cid
        # Boundary band = a ring just outside the hole.
        band = binary_dilation(component, iterations=_BORDER_RING_PX) & ~component
        subject_ring = band & (alpha >= _RAMP_HIGH)
        if not subject_ring.any():
            continue  # no solid subject bordering it -> cannot be an enclosed body drop

        interior_lum = float(img[component].mean())
        subject_lum = float(img[subject_ring].mean())

        ramp = band & (alpha > _RAMP_LOW) & (alpha < _RAMP_HIGH)
        ramp_frac = float(ramp.sum()) / float(band.sum()) if band.any() else 0.0

        # Soft, uncertain boundary (drop) + interior not darker than the
        # bordering subject (reflective/bright, not a genuine dark gap).
        if ramp_frac >= ramp_min and interior_lum >= subject_lum - _DROP_DARK_TOL:
            drop |= component

    return drop


def _restore_drop(
    image: np.ndarray,
    work_rgba: np.ndarray,
    drop_mask: np.ndarray,
    orig_rgba: np.ndarray,
    config: Config,
) -> np.ndarray:
    """Seamlessly restore a saliency drop from the original image.

    The repair has two zones:

    * **Core** — the hole plus the contaminated semi-transparent ramp ring
      around it (found by flooding the hole through the ramp band, stopping at
      solid foreground and at background). Here the pipeline result is broken, so
      the original is written 1:1 and alpha is forced to 255.
    * **Blend skirt** — a margin grown past the core into the surrounding SOLID
      subject (``fill_holes_drop_blend_margin``). There the original and the
      pipeline result already agree (opaque subject), so feathering the original
      over the result with a smoothstep falloff puts the only RGB transition on
      matching pixels — invisible even at high zoom, with continuous tone, grain
      and detail. No seam, no halo.

    Args:
        image: Canonical source RGB, ``(H, W, 3)`` uint8.
        work_rgba: Current RGBA being built up (may already carry other repairs).
        drop_mask: Boolean mask of hole cores classified as drops.
        orig_rgba: The unmodified pipeline RGBA, used to find the ramp ring from
            the model's original alpha (independent of prior repairs).
        config: Pipeline configuration (``fill_holes_drop_max_grow``,
            ``fill_holes_drop_blend_margin``, ``fill_holes_drop_edge_feather``).

    Returns:
        A new RGBA array with the drop region restored. Input is not mutated.
    """
    from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter

    orig_alpha = orig_rgba[..., 3]
    # Core: flood the hole through the contaminated ramp band only. The flood
    # stops at solid foreground (alpha >= _RAMP_HIGH) and at background, so it
    # ends exactly at the subject's true edge. Capped to bound runaway growth.
    ramp = (orig_alpha > _RAMP_LOW) & (orig_alpha < _RAMP_HIGH)
    flood_through = ramp | drop_mask
    max_grow = int(config.fill_holes_drop_max_grow)
    core = (
        binary_dilation(drop_mask, mask=flood_through, iterations=max_grow)
        if max_grow > 0
        else drop_mask.copy()
    )
    core |= drop_mask

    # Blend skirt: extend the core into the surrounding SOLID subject so the RGB
    # transition lands where original and result already match.
    margin = int(config.fill_holes_drop_blend_margin)
    solid_fg = orig_alpha >= _RAMP_HIGH
    if margin > 0:
        skirt = binary_dilation(core, iterations=margin) & solid_fg & ~core
    else:
        skirt = np.zeros_like(core)

    # Per-pixel weight: 1.0 across the core, smoothstep falloff to 0.0 across the
    # skirt by distance from the core boundary.
    weight = core.astype(np.float32)
    if margin > 0 and skirt.any():
        dist_out = distance_transform_edt(~core)
        t = np.clip(1.0 - dist_out / float(margin), 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)  # smoothstep, C1-continuous at both ends
        weight = np.where(skirt, smooth.astype(np.float32), weight)
        weight[core] = 1.0

    out = work_rgba.copy()
    w = weight[..., None]
    src = image.astype(np.float32)
    dst = out[..., :3].astype(np.float32)
    blended = w * src + (1.0 - w) * dst
    out[..., :3] = np.clip(np.round(blended), 0, 255).astype(np.uint8)

    # Alpha: opaque across the core AND the skirt. The skirt is solid subject,
    # so forcing it fully opaque removes the faint alpha arc that the model's
    # near-opaque (but <255) ramp would otherwise leave at the core boundary.
    out[..., 3][core | skirt] = 255
    feather = float(config.fill_holes_drop_edge_feather)
    if feather > 0.0:
        # Optional: soften only the outer alpha edge; RGB stays as blended above.
        alpha_w = np.clip(gaussian_filter(core.astype(np.float32), sigma=feather), 0.0, 1.0)
        alpha_w = np.maximum(alpha_w, core.astype(np.float32))
        new_alpha = np.maximum(out[..., 3].astype(np.float32), alpha_w * 255.0)
        out[..., 3] = np.clip(np.round(new_alpha), 0, 255).astype(np.uint8)
    return out


def _select_holes(
    enclosed: np.ndarray,
    height: int,
    width: int,
    config: Config,
) -> np.ndarray:
    """Pick the enclosed holes whose centroid is central.

    Args:
        enclosed: Boolean mask of all enclosed transparent holes.
        height: Frame height in pixels.
        width: Frame width in pixels.
        config: Pipeline configuration (centre-distance threshold).

    Returns:
        Boolean mask of the holes to fill (subset of ``enclosed``).
    """
    from scipy.ndimage import center_of_mass, label

    labels, count = label(enclosed)
    if count == 0:
        return np.zeros_like(enclosed)

    ids = np.arange(1, count + 1)
    # Per-component centroids without materializing full-frame coordinate grids
    # (the previous repeat/tile arrays were O(H*W) float64 each — hundreds of MB
    # at the canonical size). center_of_mass returns the same mean-of-coordinates
    # centroid as (y, x) per label.
    centroids = np.asarray(center_of_mass(enclosed, labels, ids), dtype=np.float64)
    cy = centroids[:, 0]
    cx = centroids[:, 1]

    # Radial distance of each centroid from the image centre, normalized so the
    # centre is 0 and a corner is 1.0 (half the image diagonal).
    center_y = height / 2.0
    center_x = width / 2.0
    half_diag = float(np.hypot(width, height)) / 2.0
    dist = np.hypot(cy - center_y, cx - center_x) / half_diag

    keep_ids = ids[dist <= float(np.clip(config.fill_holes_center_dist, 0.0, 1.0))]
    if keep_ids.size == 0:
        return np.zeros_like(enclosed)

    return np.isin(labels, keep_ids)


def _validate(image: np.ndarray, rgba: np.ndarray) -> None:
    """Validate image/rgba shape and dtype agreement.

    Args:
        image: Candidate source RGB image.
        rgba: Candidate RGBA result.

    Raises:
        ValueError: If shapes or dtypes are invalid, or sizes disagree.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(
            f"image must be (H, W, 3) uint8; got shape {image.shape!r} "
            f"dtype {image.dtype}."
        )
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError(
            f"rgba must be (H, W, 4) uint8; got shape {rgba.shape!r} "
            f"dtype {rgba.dtype}."
        )
    if rgba.shape[:2] != image.shape[:2]:
        raise ValueError(
            f"rgba size {rgba.shape[:2]!r} does not match image "
            f"{image.shape[:2]!r}."
        )
