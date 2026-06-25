"""Central configuration for the background-removal pipeline.

All tunable behaviour lives here so that no thresholds, model names, device
choices, or engine selections are hardcoded inside the processing modules. The
same :class:`Config` object is threaded through every stage of
``remove_background`` and is the single place that changes between the phase-1
(CPU / rembg / ONNX) and phase-2 (GPU / torch / BiRefNet + ViTMatte) deployments.

Settings may be overridden via environment variables prefixed with ``BGR_``
(for example ``BGR_DEVICE=cuda`` or ``BGR_PIPELINE_MODE=modular``).
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Engine and device identifiers selected at runtime. "torch" / "cuda" are
# accepted now but only wired up in phase 2; the phase-1 engine is "rembg".
EngineName = Literal["rembg", "torch"]
DeviceName = Literal["cpu", "cuda"]
PipelineMode = Literal["rembg_oneshot", "modular"]


class Config(BaseSettings):
    """Runtime configuration for a single background-removal invocation.

    Attributes are grouped by pipeline stage. Defaults target the phase-1
    CPU deployment; phase 2 flips ``engine`` and ``device`` without touching
    any other module (see ``docs/ARCHITECTURE.md``).

    Attributes:
        engine: Which segmentation engine backs the :class:`Segmenter` seam.
            ``"rembg"`` (phase 1, ONNX) or ``"torch"`` (phase 2, BiRefNet).
        device: Compute device. ``"cpu"`` for phase 1; ``"cuda"`` for phase 2.
        model_name: rembg/ONNX model id for the phase-1 segmenter. Default
            ``"birefnet-general"``; switch to ``"birefnet-general-lite"`` only
            if per-image latency is too high in the embedding app.
        pipeline_mode: ``"rembg_oneshot"`` runs a single rembg ``remove`` call
            with built-in alpha matting (default CPU path). ``"modular"`` runs
            the engine-independent chain (segment -> trimap -> matting refine ->
            defringe -> composite), which phase 2 reuses unchanged.
        trimap_foreground_threshold: Soft-alpha value at/above which a pixel is
            definite foreground (1.0) in the synthesized trimap.
        trimap_background_threshold: Soft-alpha value at/below which a pixel is
            definite background (0.0) in the synthesized trimap.
        trimap_band_pct: Width of the dilated "unknown" band as a fraction of
            the longer image edge (e.g. 0.005 == 0.5%).
        alpha_matting_foreground_threshold: rembg one-shot parameter; pixels
            brighter than this in the coarse mask seed the foreground.
        alpha_matting_background_threshold: rembg one-shot parameter; pixels
            darker than this seed the background.
        alpha_matting_erode_size: rembg one-shot erosion (px) applied to the
            mask before matting.
        pymatting_regularization: Regularization passed to pymatting foreground
            estimation in the modular defringe stage.
        guided_filter_enabled: When True, apply an edge-aware guided filter to
            the alpha before foreground estimation. Off by default; enable only
            if test images show residual halo after foreground estimation.
        guided_filter_radius: Guided-filter window radius (px).
        guided_filter_eps: Guided-filter regularization (squared-color units).
        fill_holes: When True, fill an enclosed transparent hole near the image
            centre with the original image content (opaque), instead of leaving
            it transparent. Any hole whose centroid is within
            ``fill_holes_center_dist`` of the centre is kept "as in the
            original" (whatever its size); holes farther out stay transparent.
            Off by default.
        fill_holes_threshold: Alpha value (``[0, 1]``) at/above which a pixel
            counts as foreground when detecting enclosed holes. Default 0.5.
        fill_holes_center_dist: Maximum distance of the hole's centroid from the
            image centre for the hole to be filled, normalized so ``0`` is the
            centre and ``1.0`` is a corner (half the image diagonal). Holes
            farther out stay transparent. Default 0.15.
        fill_holes_dilate: Number of pixels to grow the filled region outward
            past the strict hole boundary, so the original image also covers the
            thin matting-contaminated ring around the hole (which would otherwise
            leave a dirty seam). Default 2. Set 0 to fill only the exact hole.
        fill_holes_feather: Gaussian sigma (px) for softening the boundary
            between the pasted original content and the surrounding result. The
            hole core stays 100% original; only the outer band is blended so the
            edge is seamless instead of a hard line. Default 1.5. Set 0 for a
            hard edge.
        background_color: Optional flat RGBA fill for the output. ``None`` means
            a transparent background.
        canonical_size: Target edge (px) for ingest canonicalization. Every
            image becomes a true ``canonical_size`` x ``canonical_size`` square:
            inputs that fit are centered on a ``pad_color`` canvas with no
            scaling (lossless); oversized inputs are downscaled to fit (aspect
            preserved) before centering. ``None`` disables canonicalization.
            Default ``2362`` (default canonical size).
        pad_color: RGB fill used for the canonicalization border. Default white
            ``(255, 255, 255)`` so the padded margin is removed as background.
        restore_original_size: When True, crop the canonicalization padding back
            off the final RGBA so the output matches the source dimensions
            instead of ``canonical_size``. Lossless for inputs that only needed
            padding; for oversized inputs (which were downscaled to fit) the
            crop returns the downscaled size, not the original pixels. Default
            False (output stays at the canonical size). No effect on inputs that
            were not padded.
        premultiplied_output: When True, emit premultiplied (associated) alpha
            RGBA (RGB already multiplied by alpha). When False (default), emit
            straight (unassociated) alpha, which is the PNG standard and renders
            correctly in all viewers. Foreground colour is defringed either way,
            so straight alpha is halo-free; premultiplied is offered for
            compositing pipelines that require it. Ignored when
            ``background_color`` is set (the output is then flat and opaque).
    """

    model_config = SettingsConfigDict(
        env_prefix="BGR_",
        env_file=".env",
        extra="ignore",
        # "model_name" would otherwise collide with pydantic's protected
        # "model_" namespace; disable that protection explicitly.
        protected_namespaces=(),
    )

    # --- engine / device selection (the phase-1 <-> phase-2 switch) ---
    engine: EngineName = "rembg"
    device: DeviceName = "cpu"
    model_name: str = "birefnet-general"
    pipeline_mode: PipelineMode = "rembg_oneshot"

    # --- trimap synthesis (modular mode) ---
    trimap_foreground_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    trimap_background_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    trimap_band_pct: float = Field(default=0.005, gt=0.0, le=0.5)

    # --- rembg one-shot alpha matting ---
    alpha_matting_foreground_threshold: int = Field(default=240, ge=0, le=255)
    alpha_matting_background_threshold: int = Field(default=10, ge=0, le=255)
    alpha_matting_erode_size: int = Field(default=10, ge=0)

    # --- modular defringe (pymatting + optional guided filter) ---
    pymatting_regularization: float = Field(default=1e-5, gt=0.0)
    guided_filter_enabled: bool = False
    guided_filter_radius: int = Field(default=4, ge=1)
    guided_filter_eps: float = Field(default=1e-4, gt=0.0)

    # --- interior hole filling (saliency-drop fix) ---
    fill_holes: bool = True
    fill_holes_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    fill_holes_center_dist: float = Field(default=0.15, gt=0.0, le=1.0)
    fill_holes_dilate: int = Field(default=2, ge=0)
    fill_holes_feather: float = Field(default=1.5, ge=0.0)

    # --- saliency-drop repair (content-aware, opt-in) ---
    # When True, enclosed holes are classified per-component: a hole whose
    # interior original colour matches the surrounding subject (not the
    # background) AND whose alpha boundary is a soft semi-transparent ramp is a
    # *saliency drop* (the model under-segmented a solid reflective body). Drops
    # are restored hard-edged from the original at full opacity, covering the
    # contaminated ramp ring — instead of the cosmetic feather used for ring
    # eyelets, which left a gray halo/seam. Holes that look like a ring eye
    # (background-coloured interior, crisp edge) stay transparent. Off by default
    # so existing fill_holes behaviour is unchanged.
    fill_holes_restore_drops: bool = True
    # Minimum fraction of the hole's boundary band that is a soft alpha ramp
    # (neither solid foreground nor solid background) for it to count as a drop.
    # This model-uncertainty signal — not colour — separates a real see-through
    # eyelet (crisp edge) from a dropped reflective body (soft ramp), since both
    # are near-white in the original on a white-lightbox shot.
    fill_holes_drop_ramp_min: float = Field(default=0.20, ge=0.0, le=1.0)
    # Max pixels the restore grows from the hole through the contaminated ramp
    # band (stops at solid foreground and at background). Caps runaway growth.
    fill_holes_drop_max_grow: int = Field(default=40, ge=0)
    # Pixels the restore extends past the contaminated core into the surrounding
    # SOLID subject, where the original and the pipeline result already match.
    # The original is feathered across this skirt so the seam lands on matching
    # pixels and is invisible even at high zoom. Default 24. Set 0 for a hard
    # core-only restore.
    fill_holes_drop_blend_margin: int = Field(default=24, ge=0)
    # Gaussian sigma (px) feathering only the outer alpha edge of a restored
    # drop. Default 0 = hard edge (matches the true subject boundary).
    fill_holes_drop_edge_feather: float = Field(default=0.0, ge=0.0)

    # --- ingest canonicalization ---
    canonical_size: Optional[int] = Field(default=2362, gt=0)
    pad_color: Tuple[int, int, int] = (255, 255, 255)
    restore_original_size: bool = True

    # --- composite ---
    background_color: Optional[Tuple[int, int, int, int]] = None
    premultiplied_output: bool = False

    @field_validator("pad_color", "background_color")
    @classmethod
    def _components_in_byte_range(
        cls, value: Optional[Tuple[int, ...]]
    ) -> Optional[Tuple[int, ...]]:
        """Reject colour components outside [0, 255] at config time.

        Without this, an out-of-range component (e.g. ``(300, 0, 0)``) only
        fails later with a cryptic error deep inside PIL. Validating here gives
        a clear message at construction.
        """
        if value is not None and any(c < 0 or c > 255 for c in value):
            raise ValueError(f"colour components must be in [0, 255]; got {value!r}")
        return value

    @field_validator("model_name")
    @classmethod
    def _model_name_is_a_plain_identifier(cls, value: str) -> str:
        """Reject model names that could steer the model download/cache path.

        ``model_name`` is passed to ``rembg.new_session`` -> ``pooch`` where it
        becomes both a filename component and a URL path segment. A value
        containing path separators, ``..``, or control characters could write
        the downloaded ``.onnx`` outside the model cache or alter the download
        URL. rembg model ids are plain slugs (e.g. ``birefnet-general``), so we
        constrain to that shape. See security findings S1/S2.
        """
        if not value or not value.strip():
            raise ValueError("model_name must be a non-empty model identifier.")
        if value != value.strip():
            raise ValueError(f"model_name must not have surrounding whitespace; got {value!r}")
        forbidden = set('/\\') | {"..", "\x00"}
        if any(token in value for token in forbidden):
            raise ValueError(
                "model_name must be a plain model identifier without path "
                f"separators, '..', or NUL; got {value!r}"
            )
        if any(ord(c) < 0x20 for c in value):
            raise ValueError(f"model_name must not contain control characters; got {value!r}")
        return value
