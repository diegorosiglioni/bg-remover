# Architecture

High-precision removal of solid white backgrounds, with **no white halo** at the
edges. This document explains what the system does today (phase 1) and how it
grows into a GPU service later (phase 2), so that someone new to the project can
read it once and understand the whole thing.

---

## 1. The problem: white halos

A photo of a subject on a pure white background is, at the edges, a *blend* of
the subject and the background. Every edge pixel obeys the compositing equation:

```
observed = alpha * F + (1 - alpha) * B
```

- `observed` — the RGB you actually see in the file.
- `alpha`   — coverage of the subject in that pixel (1 = fully subject, 0 = fully background).
- `F`       — the subject's *true* colour at that pixel.
- `B`       — the background colour (here, white = (255, 255, 255)).

A naive background remover estimates `alpha`, then composites the **observed**
RGB over the new background:

```
result = alpha * observed + (1 - alpha) * new_background
```

This is wrong at every semi-transparent edge pixel. Because `observed` already
contains `(1 - alpha) * white`, the leftover white never leaves. Over a dark
background the edges glow — the classic **white halo / fringe**.

### The fix: foreground colour estimation

The halo is killed by recovering the true foreground colour `F` and compositing
**that**, not `observed`:

```
result = alpha * F + (1 - alpha) * new_background
```

Solving for `F` per pixel (given `observed` and `alpha`) is *foreground
estimation*. We use pymatting's `estimate_foreground_ml`, which propagates clean
colour from confident interior pixels outward into the semi-transparent edge.
With `F` recovered, edge pixels carry the subject's real colour and the white
contamination is gone.

> **This is the single most important idea in the project.** Good alpha alone is
> not enough; you must also recover `F`. Both phase 1 and phase 2 do this — only
> the engines that produce `alpha` differ.

In phase 1 this happens two ways depending on `pipeline_mode`:
- `rembg_oneshot`: rembg's built-in `alpha_matting=True` internally calls
  pymatting `estimate_alpha_cf` + `estimate_foreground_ml`. The halo-killer is
  on by default, for free.
- `modular`: we run `estimate_foreground_ml` ourselves in `defringe.py` against
  our own trimap, so the exact same step is reusable, untouched, in phase 2.

---

## 2. Two phases, one contract

The project is deliberately split so the expensive GPU work is deferred and the
CPU build embedded in the app today is not throwaway.

| | Phase 1 (now) | Phase 2 (later) |
|---|---|---|
| Where it runs | CPU-only PC, inside a local app | GPU server (NVIDIA / CUDA) |
| Packaging | Importable library + CLI | Same library wrapped in FastAPI + Docker |
| Segmentation | BiRefNet **ONNX** via `rembg` | BiRefNet on **torch** (HR weights) |
| Matting refine | pymatting closed-form | **ViTMatte** sub-pixel matting |
| Dependencies | rembg, onnxruntime, pymatting, opencv, numpy, pillow | + torch, transformers, fastapi, uvicorn |
| Defringe (foreground `F`) | pymatting `estimate_foreground_ml` | **identical** |
| Trimap / composite | engine-independent modules | **identical** |

The thing that makes phase 2 a *swap, not a rewrite* is that everything talks
through two narrow interfaces ("seams"), and the halo-critical math
(trimap, defringe, composite) is engine-independent and already correct in phase 1.

---

## 3. The seams (interfaces)

Defined in `app/protocols.py` as `typing.Protocol` interfaces. The pipeline
imports **these**, never a concrete engine.

### `Segmenter` — coarse separation
```python
def predict(self, image: np.ndarray) -> np.ndarray:  # (H,W,3) uint8 -> (H,W) float32 [0,1]
```
Returns a **soft** alpha (not thresholded), so downstream trimap synthesis can
choose its own bands.

- Phase 1: `RembgSegmenter` — holds one `rembg` session (`new_session("birefnet-general")`)
  and calls `remove(img, session=s, only_mask=True)` to get the soft mask.
- Phase 2: `TorchBiRefNetSegmenter` — BiRefNet HR on CUDA, FP16, loaded once.

### `MattingRefiner` — sub-pixel alpha
```python
def refine(self, image: np.ndarray, trimap: np.ndarray) -> np.ndarray:  # -> (H,W) float32 [0,1]
```
- Phase 1: `PyMattingRefiner` — pymatting `estimate_alpha_cf(image, trimap)`.
- Phase 2: `ViTMatteRefiner` — ViTMatte (`hustvl/vitmatte-*`) on CUDA.

Both phase-2 implementations are documented stubs in phase 1 (they raise
`NotImplementedError` with a pointer to this document). Swapping them in is a
config change (`engine`, `device`) plus installing the phase-2 dependencies — no
pipeline code changes.

### Array conventions (every implementation obeys these)
- `image`: `(H, W, 3)`, `uint8`, RGB, sRGB.
- `alpha`: `(H, W)`, `float32`, `[0, 1]`.
- `trimap`: `(H, W)`, `float32`, values in `{0.0, 0.5, 1.0}` (bg / unknown / fg).

---

## 4. The two pipeline modes

Selected by `Config.pipeline_mode`. Both share the same `rembg` session, so
**BiRefNet runs exactly once per image** in either mode.

### `rembg_oneshot` (phase-1 default)
```
remove(image, session=s, alpha_matting=True, ...) -> RGBA
```
One call. rembg segments, builds an internal trimap from the
`alpha_matting_*` thresholds, runs closed-form matting **and**
`estimate_foreground_ml`, and returns premultiplied RGBA. Fastest path, fewest
moving parts, and the foreground-estimation halo-killer is built in. Use this
unless you need finer control.

### `modular`
```
soft_alpha = segmenter.predict(image)            # rembg only_mask=True (or torch in phase 2)
trimap     = synth_trimap(soft_alpha, config)    # threshold + dilate unknown band
alpha      = refiner.refine(image, trimap)       # pymatting (or ViTMatte in phase 2)
F          = estimate_foreground(image, alpha)   # estimate_foreground_ml  -> halo-killer
rgba       = composite(F, alpha, config)         # premultiplied RGBA, optional flat bg
```
Every step is engine-independent except `segmenter` and `refiner`, which are the
two seams. **This is the exact chain phase 2 runs** — phase 2 only swaps the two
seam implementations. Use `modular` when you want explicit control over the
trimap band, the optional guided filter, or to validate the phase-2 path on CPU.

When to use which:
- `rembg_oneshot`: production CPU default; good edges, minimal latency.
- `modular`: tuning, debugging halos, or exercising the phase-2 code path early.

---

## 5. Configuration (every option)

All in `app/config.py` (`Config`, pydantic-settings). Override via env vars
prefixed `BGR_` (e.g. `BGR_PIPELINE_MODE=modular`) or a `.env` file. Nothing is
hardcoded in the processing modules.

| Option | Default | Meaning |
|---|---|---|
| `engine` | `rembg` | Segmentation backend seam. `rembg` (phase 1) or `torch` (phase 2). |
| `device` | `cpu` | Compute device. `cpu` now; `cuda` in phase 2. |
| `model_name` | `birefnet-general` | rembg/ONNX model id. Fall back to `birefnet-general-lite` only if per-image latency is too high. |
| `pipeline_mode` | `rembg_oneshot` | `rembg_oneshot` (one call) or `modular` (explicit chain). |
| `trimap_foreground_threshold` | `0.95` | Soft-alpha >= this becomes definite foreground in the trimap. |
| `trimap_background_threshold` | `0.05` | Soft-alpha <= this becomes definite background. |
| `trimap_band_pct` | `0.005` | Unknown-band dilation as a fraction of the longer edge (0.5%). |
| `alpha_matting_foreground_threshold` | `240` | rembg one-shot: mask brightness seeding foreground. |
| `alpha_matting_background_threshold` | `10` | rembg one-shot: mask brightness seeding background. |
| `alpha_matting_erode_size` | `10` | rembg one-shot: mask erosion (px) before matting. |
| `pymatting_regularization` | `1e-5` | Regularization for modular foreground estimation. |
| `guided_filter_enabled` | `False` | Edge-aware guided filter on alpha before foreground estimation. Enable only if halos persist. |
| `guided_filter_radius` | `4` | Guided-filter window radius (px). |
| `guided_filter_eps` | `1e-4` | Guided-filter regularization. |
| `canonical_size` | `2362` | Ingest target edge (px). Every image becomes a true `canonical_size`×`canonical_size` square: fitting inputs are centered on a `pad_color` canvas (lossless), oversized inputs are downscaled to fit first. `None` disables it. |
| `pad_color` | `(255, 255, 255)` | RGB fill for the canonicalization border (white, so the margin is removed as background). |
| `restore_original_size` | `True` | When `True`, crop the canonicalization padding back off the final RGBA so output matches the original input dimensions. |
| `fill_holes` | `True` | Keep a central enclosed hole as in the original image (restore source pixels, opaque) instead of transparent. Off-centre holes stay transparent; hole size does not matter. |
| `fill_holes_threshold` | `0.5` | Alpha at/above which a pixel counts as foreground when detecting enclosed holes. |
| `fill_holes_center_dist` | `0.15` | Max distance of the hole centroid from the image centre to be filled (0 = centre, 1.0 = corner / half-diagonal). |
| `fill_holes_dilate` | `2` | Pixels the restored hole grows into the surrounding subject before compositing, so the patch overlaps the alpha boundary. |
| `fill_holes_feather` | `1.5` | Gaussian sigma (px) blending the restored patch edge into the result (cosmetic feather for ring eyelets). `0` = hard edge. |
| `fill_holes_restore_drops` | `True` | Content-aware repair: hard-restore *saliency drops* (model dropping part of a solid reflective body) at full opacity, covering the contaminated ramp ring; leave real ring eyelets transparent. Classified by alpha-ramp softness, not colour. |
| `fill_holes_drop_ramp_min` | `0.20` | Min fraction of the hole boundary band that is a soft alpha ramp for it to count as a drop (vs. a crisp-edged eyelet). |
| `fill_holes_drop_max_grow` | `40` | Max px the restore grows from the hole through the contaminated ramp band (stops at solid foreground/background). Caps runaway growth. |
| `fill_holes_drop_blend_margin` | `24` | Px the restore extends past the contaminated core into surrounding solid subject, feathered so the seam lands on matching pixels. `0` = hard core-only restore. |
| `fill_holes_drop_edge_feather` | `0.0` | Gaussian sigma (px) softening only the outer alpha edge of a restored drop. `0` = hard edge matching the true subject border. |
| `background_color` | `None` | Flat RGBA fill; `None` = transparent. |
| `premultiplied_output` | `False` | `True` = premultiplied (associated) alpha RGBA; `False` = straight (unassociated) alpha, the PNG standard (viewer-correct, halo-free because `F` is defringed). Ignored when `background_color` is set. |

---

## 6. Module layout

```
app/
  __init__.py        Public surface: remove_background, BackgroundRemover, Config.
  config.py          Config (pydantic-settings). The only place behaviour is tuned.
  protocols.py       Segmenter, OneShotSegmenter & MattingRefiner Protocols. The seams.
  ingest.py          Decode bytes/path -> RGB uint8 ndarray. EXIF transpose,
                     convert to sRGB, drop ICC profile. (Phase 2 adds dual-res
                     + pyvips tiling for very large images.)
  segmentation.py    RembgSegmenter (phase 1). TorchBiRefNetSegmenter (phase-2 stub).
  trimap.py          soft alpha -> trimap; threshold + dilate unknown band.   [engine-independent]
  matting.py         PyMattingRefiner (phase 1). ViTMatteRefiner (phase-2 stub). [engine-independent core]
  defringe.py        estimate_foreground_ml + optional guided filter.          [engine-independent]
  holes.py           Fill interior saliency-drop holes (enclosed gaps).        [engine-independent]
  composite.py       Premultiplied RGBA PNG; flat/custom background option.     [engine-independent]
  pipeline.py        remove_background(image, config) -> RGBA. Wires both modes.
  cli.py             Batch a folder (in_dir -> out_dir) with per-file isolation.
docs/
  ARCHITECTURE.md    This file.
  GUIDA_GPU.md       Phase-2 (GPU/CUDA) migration guide.
```

Modules marked **engine-independent** are byte-for-byte identical between phase 1
and phase 2 — they are where the halo problem is actually solved, so they are
built and tested first.

---

## 7. CLI usage (phase 1)

The library entrypoint is `remove_background(image, config)`. The CLI batches a
folder:

```bash
# transparent PNGs for every image in ./in -> ./out
python -m app.cli ./in ./out

# tune via env (modular chain, lite model)
BGR_PIPELINE_MODE=modular BGR_MODEL_NAME=birefnet-general-lite python -m app.cli ./in ./out
```

Each file is processed independently; a bad image logs an error and is skipped
rather than aborting the batch.

---

## 8. The phase-2 migration — exactly what changes

Nothing in the halo-critical math changes. The migration is:

1. **Install phase-2 deps** (separate requirements file): `torch`, `torchvision`,
   `transformers`, `huggingface-hub`, `accelerate`, `fastapi`, `uvicorn`.
2. **Implement the two seam stubs**:
   - `TorchBiRefNetSegmenter.predict` — BiRefNet HR (`ZhengPeng7/BiRefNet_HR`)
     via `AutoModelForImageSegmentation.from_pretrained(..., trust_remote_code=True)`,
     ImageNet-normalized 2048² input, `model(x)[-1].sigmoid()`, upsample to full
     res, loaded once, FP16 autocast.
   - `ViTMatteRefiner.refine` — `VitMatteImageProcessor` +
     `VitMatteForImageMatting` (`hustvl/vitmatte-small-composition-1k`),
     `processor(images=img, trimaps=trimap)`, `model(**inputs).alphas`; crop the
     padded output back to original H/W (processor pads to `size_divisor`).
3. **Flip config**: `engine=torch`, `device=cuda`, `pipeline_mode=modular`,
   `model_name=ZhengPeng7/BiRefNet_HR`.
4. **Add ingest dual-res + pyvips tiling** for >~30 MP inputs (full-res RGB as
   edge truth, model-res resize for segmentation).
5. **Wrap, don't rewrite**: a FastAPI app loads both models once in a `lifespan`
   handler and exposes `POST /remove` (upload -> RGBA PNG) and `GET /health`.
   The handler calls the *same* `remove_background`. Dockerfile on an
   `nvidia/cuda` base; document NVIDIA Container Toolkit.
6. **Optional**: ONNX/TensorRT export for BiRefNet behind a flag.

Because `pipeline.py`, `trimap.py`, `defringe.py`, and `composite.py` depend only
on the seams and the array conventions, none of them are touched. That is the
whole point of the design.
