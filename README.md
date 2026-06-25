# White Background Removal

High-precision removal of solid white backgrounds with **no white halo** at the
edges. The edge fringe that normally survives background removal is eliminated
by recovering the true foreground colour (foreground estimation), not just the
alpha matte.

> **Phase 1 (this repo): CPU-only, importable library + CLI.** Segmentation runs
> BiRefNet as an ONNX model via [`rembg`](https://github.com/danielgatis/rembg)
> on ONNX Runtime — no torch, CUDA, or web service required. The design leaves
> clean seams so a GPU torch (BiRefNet + ViTMatte) engine and a FastAPI service
> drop in later as **phase 2** without a rewrite. See
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Why the halo dies

Every edge pixel of a subject shot on white is a blend:
`observed = alpha·F + (1−alpha)·white`. Compositing the *observed* RGB over a new
background leaves the `(1−alpha)·white` term behind — a glowing white fringe over
dark backgrounds. This project solves for the true foreground colour `F` (via
pymatting `estimate_foreground_ml`) and composites **that**, so edges carry the
subject's real colour and the white is gone. Full explanation in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §1.

## Requirements

- Python **3.11–3.13** (tested on 3.13).
- CPU only. No GPU, CUDA, or torch in phase 1.
- ~1–2 GB free disk for the downloaded ONNX model + dependencies.

## Setup

Create a virtual environment and install dependencies once. The CLI is the same
across platforms — only the venv-activation line differs.

```bash
# from the project root
python -m venv .venv

# activate the venv:
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\Activate.ps1       # Windows PowerShell
# .venv\Scripts\activate.bat       # Windows cmd.exe

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To rebuild from scratch, delete the `.venv` directory and repeat the steps above.

### Quick start

With the venv active, drop images in an input folder and batch them to
transparent PNGs. The first run also downloads the model (see below).

```bash
# default folders, with checkerboard previews
python -m app.cli input output --preview-dir preview

# custom folders, recurse into subfolders
python -m app.cli photos cutouts --recursive

# resume an interrupted batch: skip images already done
python -m app.cli photos cutouts --recursive --skip-existing

# faster model
python -m app.cli photos cutouts --model birefnet-general-lite
```

`--fill-holes` and `--fill-holes-restore-drops` are on by default. See
[CLI usage](#cli-usage) for the full flag table.

### Model download

The BiRefNet ONNX weights are fetched automatically on first use and cached in
`~/.u2net/` (rembg's cache). The default model is `birefnet-general`. The first
run is slower while it downloads; subsequent runs are offline.

If per-image latency is too high on your CPU, switch to the smaller
`birefnet-general-lite` model (faster, slightly lower edge quality) — see
configuration below.

## Library usage

The single public entrypoint is `remove_background`:

```python
from pathlib import Path

import numpy as np
from PIL import Image
from app import Config, remove_background
from app.composite import encode_png

# source may be a path, raw bytes, a PIL image, or a numpy RGB array
rgba = remove_background("subject.jpg")          # -> (H, W, 4) uint8, transparent
Path("out.png").write_bytes(encode_png(rgba))

# custom configuration
cfg = Config(
    model_name="birefnet-general-lite",          # faster CPU model
    pipeline_mode="modular",                      # explicit chain (see below)
    background_color=(0, 0, 0, 255),              # flat black background instead of transparent
)
rgba = remove_background("subject.jpg", cfg)
```

Repeated calls with the same model reuse one loaded session (the segmenter is
cached). For an explicit long-lived object, use `BackgroundRemover`:

```python
from app import BackgroundRemover, Config

remover = BackgroundRemover(Config())             # loads the model once
for path in paths:
    rgba = remover.process(path)
```

## Folder workflow

The CLI examples use three working folders:

```
input/                        <- input: drop your source images here (you create it)
output/                       <- transparent PNG results appear here (created on run)
preview/                      <- results over a transparency checkerboard (easy visual check)
```

Create the input folder, drop images in it, activate the venv, then run:

```bash
python -m app.cli input output --preview-dir preview
```

`output/<name>.png` is the transparent cutout; `preview/<name>.png` is the same
result composited over a checkerboard so you can eyeball the edges. Generated
image folders are not tracked — image outputs are gitignored. Use any folder
names you like; these are only the defaults.

## Two-step workflow with editable masks (`run.py`)

`run.py` splits the folder pipeline into two explicit steps so you can inspect
or hand-edit the masks between them and rerun only the cheap apply step. With no
`--step`, both run in sequence (the full pipeline):

```bash
python run.py                          # generate masks → apply masks → output/
python run.py --step generate_masks    # runs the model, writes masks/
# ...optionally edit masks (see edit_masks.py)...
python run.py --step apply_masks       # reads masks/, writes output/
```

`generate_masks` dispatches per folder exactly like `app.folder_pipeline`:
folders whose name contains **"pendant"** get a per-image AI mask; other folders
share one mask built from their **second image**. Every mask is saved as an
8-bit grayscale PNG (**white = keep, black = remove**) mirroring the input tree:

```
masks/
  object_A/
    img1_mask.png
    img2_mask.png
  object_B_pendant/
    img1_mask.png
    ...
```

`apply_masks` loads **no model**: it reads each `<name>_mask.png`, cuts the
native-resolution source (RGB kept inside the mask, transparent outside), and
resizes the mask to the source dimensions when they differ — so a hand-edited
mask of any size still fits.

| Flag | Meaning |
|------|---------|
| `--step {generate_masks,apply_masks}` | Which step to run (default: run both in sequence). |
| `--root DIR` | Input root (default `input`). |
| `--masks DIR` | Mask root (default `masks`). |
| `--out DIR` | Output root for `apply_masks` (default `output`). |
| `--model NAME` | rembg/ONNX model id (`generate_masks`). |
| `--mask-threshold 0-255` | Alpha cut for binarizing silhouettes (default 128). |
| `--preview-dir DIR` | Also write a checkerboard preview per image (`apply_masks`). |
| `--skip-existing` | Skip images whose mask (generate) or output (apply) already exists. |
| `--verbose` | Emit DEBUG logs and tracebacks. |

### Manual mask editor (`edit_masks.py`)

Opens a fullscreen OpenCV window per image (one at a time) to draw or correct a
mask, then saves it as the image's mask under `masks/` (overwriting):

```bash
python edit_masks.py --images input/object_A/img1.jpg input/object_A/img2.jpg
```

Controls (also shown on the bottom bar and in the window title):

The brush paints **directly into the mask** (left adds white = keep, right
removes), so every edit is visible in green and saved as-is on ENTER.

| Input | Action |
|-------|--------|
| Left-drag | Paint mask (add area). |
| SHIFT + left-click | Straight line from the last point to the click. |
| SHIFT + left-drag | Straight line constrained to 45° (H / V / diagonal). |
| Right-drag | Erase (same brush size, removes from the mask). |
| Scroll wheel | Zoom in/out centered on the cursor. |
| Middle-drag | Pan. (or press SPACE to toggle pan mode, then left-drag.) |
| `f` | Close gaps and fill enclosed holes in the mask. |
| `z` | Undo the last stroke. |
| `e` / `d` | Erase mode / draw mode. |
| `r` | Reset the mask to blank for this image. |
| `c` | Clear boundary + mask. |
| `[` / `]` | Brush size 1..50 px (default 3). |
| ENTER | Save the mask and go to the next image. |
| ESC | Cancel this image (no save) and go to the next. |
| `q` | Quit the whole session. |

The mask is edited at the source's **native resolution** (zoom in for precision)
and written straight back — no quality lost to a display downscale. The live
zoom level and mode appear in the window title (e.g. `img1.jpg - 320% | DRAW`).
`--root`/`--masks` control where the mask is written (defaults `input`/`masks`);
they must match the `run.py` step that reads them.

## CLI usage

Batch a folder of images to transparent PNGs:

```bash
python -m app.cli IN_DIR OUT_DIR [options]
```

The model loads once and is reused for the whole batch. Each file is processed
independently — a bad image is logged and skipped, never aborting the run.

```bash
# defaults: rembg one-shot, birefnet-general, transparent output
python -m app.cli photos cutouts

# faster model, explicit modular chain, recurse into subfolders
python -m app.cli photos cutouts --model birefnet-general-lite --mode modular --recursive

# flat white-to-black swap: composite onto solid black
python -m app.cli photos cutouts --bg-color 0,0,0
```

| Flag | Meaning |
|------|---------|
| `--recursive` | Descend into subdirectories (output mirrors the tree). |
| `--preview-dir DIR` | Also write an opaque checkerboard preview per image to `DIR`. |
| `--engine {rembg,torch}` | Segmentation engine (`torch` is phase 2). |
| `--device {cpu,cuda}` | Compute device (`cuda` is phase 2). |
| `--model NAME` | rembg/ONNX model id (e.g. `birefnet-general-lite`). |
| `--mode {rembg_oneshot,modular}` | Pipeline mode (see below). |
| `--band-pct FLOAT` | Trimap unknown-band width as a fraction of the longer edge. |
| `--bg-color R,G,B[,A]` | Composite onto a flat colour instead of transparency. |
| `--guided-filter` | Enable edge-aware guided-filter alpha refinement (modular). |
| `--premultiplied` | Emit premultiplied (associated) alpha. |
| `--restore-original-size` | Crop canonicalization padding off outputs so each PNG matches its original input dimensions instead of the canonical size. **On by default.** |
| `--fill-holes` | Keep central enclosed holes as in the original (restore source pixels, opaque). Off-centre holes stay transparent. **On by default.** |
| `--fill-holes-restore-drops` | Content-aware repair: hard-restore *saliency drops* (model dropping part of a solid reflective body) from the original at full opacity, while leaving real ring eyelets transparent. **On by default;** requires `--fill-holes`. |
| `--fill-holes-drop-edge-feather SIGMA` | Gaussian sigma (px) softening only the outer alpha edge of a restored drop. Default `0` = hard edge matching the true subject border. |
| `--skip-existing` | Skip images whose output PNG already exists (resume a batch instead of restarting from zero). |
| `--verbose` | Emit full tracebacks for per-file failures (logged at DEBUG) so a failing batch is debuggable. |

**Exit codes:** `0` all images succeeded · `1` at least one file failed · `2`
bad arguments, missing/empty input directory, or invalid colour.

## Pipeline modes

- **`rembg_oneshot`** (default) — a single `rembg` call with built-in alpha
  matting. rembg internally performs closed-form matting plus pymatting
  foreground estimation, so the halo-killer is on by default. Fastest path.
- **`modular`** — the engine-independent chain: segment → trimap → matting
  refine → defringe (`estimate_foreground_ml`) → composite. This is the exact
  chain phase 2 reuses with the GPU engine. Use it for explicit control of the
  trimap band or the optional guided filter.

Both modes share one model session; BiRefNet runs exactly once per image either
way.

## Interior holes (keep central holes as in the original)

Interior hole filling is **on by default** (`fill_holes=True`). A transparent
region *fully enclosed* by the subject is kept **as in the original image** when
it is **central** — its original pixels are restored and made opaque rather than
left see-through. To turn it off and keep all enclosed holes transparent, set
`Config(fill_holes=False)`.

A hole is filled when it is **central** — its centroid is within
`fill_holes_center_dist` of the image centre, distance normalized so `0` is the
centre and `1.0` is a corner (default `0.15`). Size does not matter: a large
central opening (a horseshoe gap) is filled too. Holes farther out stay
transparent, and background that reaches the image border is never enclosed, so
it is never touched.

```bash
python -m app.cli photos cutouts        # fill-holes already on
```

### Saliency-drop repair (content-aware)

Also on by default (`fill_holes_restore_drops=True`). Each enclosed hole is
classified: a hole whose alpha boundary is a soft semi-transparent ramp is a
**saliency drop** — the model under-segmented a solid reflective body (e.g. a
polished silver link). Drops are restored hard-edged from the original at full
opacity, covering the contaminated ramp ring that otherwise leaves a gray
halo/seam. Holes that look like a real **ring eyelet** (crisp see-through edge)
stay transparent. The model-uncertainty signal (soft vs. crisp ramp), not
colour, separates the two — both are near-white on a white-lightbox shot.

Disable with `Config(fill_holes_restore_drops=False)` to fall back to the
cosmetic feather. Tuning fields (`fill_holes_drop_ramp_min`,
`fill_holes_drop_max_grow`, `fill_holes_drop_blend_margin`,
`fill_holes_drop_edge_feather`) are documented in `app/config.py`.

Tune the radius via env var when needed:

```bash
export BGR_FILL_HOLES_CENTER_DIST=0.30          # Linux / macOS
# $env:BGR_FILL_HOLES_CENTER_DIST = "0.30"      # Windows PowerShell
python -m app.cli photos cutouts --fill-holes
```

## Configuration

All behaviour lives in `app/config.py` (`Config`, pydantic-settings). Override
any field via a `BGR_`-prefixed environment variable or a `.env` file:

```bash
# Linux / macOS
export BGR_MODEL_NAME=birefnet-general-lite
export BGR_PIPELINE_MODE=modular
# Windows PowerShell
#   $env:BGR_MODEL_NAME = "birefnet-general-lite"
#   $env:BGR_PIPELINE_MODE = "modular"
python -m app.cli photos cutouts
```

The full option table (thresholds, band width, guided-filter parameters,
output format) is documented in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §5.

## Project structure

```
app/
  config.py          Config (pydantic-settings) — the only place behaviour is tuned
  protocols.py       Segmenter, OneShotSegmenter & MattingRefiner seams
  ingest.py          decode -> canonical sRGB uint8 RGB array (EXIF, ICC)
  segmentation.py    RembgSegmenter (phase 1) + TorchBiRefNetSegmenter stub
  trimap.py          soft alpha -> trimap                       [engine-independent]
  matting.py         PyMattingRefiner (phase 1) + ViTMatteRefiner stub
  defringe.py        guided filter + estimate_foreground_ml     [engine-independent]
  holes.py           fill interior saliency-drop holes          [engine-independent]
  composite.py       premultiplied/straight RGBA, flat bg       [engine-independent]
  masks.py           grayscale mask I/O (save/load/resize, path mirror)
  pipeline.py        remove_background(...) — the public entrypoint
  cli.py             batch folder processing
  folder_pipeline.py folder dispatch + generate_masks / apply_masks steps
run.py               CLI: no --step = full pipeline; or generate_masks | apply_masks
edit_masks.py        interactive OpenCV mask editor
docs/ARCHITECTURE.md full system + phase-2 migration plan
```

