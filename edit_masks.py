"""Interactive OpenCV mask editor with zoom, pan, line tool, and undo.

Opens one fullscreen window per image so a mask can be drawn or corrected by
hand, then saved as the image's grayscale mask under ``masks/`` (overwriting any
existing one). It reads and writes the same mask format the pipeline uses:

    python run.py --step generate_masks      # AI masks -> masks/
    python edit_masks.py --images input/obj/img1.jpg input/obj/img2.jpg
    python run.py --step apply_masks         # cut with the edited masks

The brush paints **directly into the mask**: left adds (white = keep), right
removes (black). Edits are visible (green) and saved immediately on ENTER — there
is no hidden "boundary" layer to commit.

Controls (also shown on screen / window title):

* Left-drag           paint mask (add area).
* SHIFT + left-click  straight line from the last point to the click.
* SHIFT + left-drag   straight line constrained to 45° (H / V / diagonal).
* Right-drag          erase (same brush size, removes from the mask).
* Scroll wheel        zoom in/out centered on the cursor.
* Middle-drag         pan. (or press SPACE to toggle pan mode, then left-drag.)
* ``f``               close gaps and fill enclosed holes in the mask.
* ``z``               undo the last stroke.
* ``e`` / ``d``       erase mode / draw mode.
* ``r``               reset the mask to blank for this image.
* ``c``               clear boundary + mask.
* ``[`` / ``]``       brush size 1..50 px (default 3).
* ENTER               save the mask and go to the next image.
* ESC                 cancel this image (no save) and go to the next.
* ``q``               quit the whole session.

The mask is edited at the source's native resolution (zoom in for precision) and
written straight back, so no quality is lost to a display downscale.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.masks import (
    load_mask,
    mask_path,
    resize_mask,
    save_mask,
    shared_mask_path,
)

# Fallback screen size when it cannot be queried (no display server, etc.).
_DEFAULT_SCREEN = (1920, 1080)

# Brush size bounds and default (px, radius).
_BRUSH_MIN = 1
_BRUSH_MAX = 50
_BRUSH_DEFAULT = 3

# Zoom bounds and per-notch step.
_ZOOM_MIN = 0.05
_ZOOM_MAX = 32.0
_ZOOM_IN = 1.25
_ZOOM_OUT = 0.8

# Most recent strokes kept for undo (stroke-level, not pixel-level).
_MAX_UNDO = 30

# Kernel side (px) for closing small gaps in a freehand boundary before filling.
_FILL_CLOSE_KERNEL = 9

# Overlay colours (BGR).
_MASK_TINT = np.array([0, 200, 0], dtype=np.float32)  # green fill
_LINE_COLOR = (0, 0, 255)  # red boundary
_CURSOR_DRAW = (0, 255, 255)  # yellow brush ring (draw)
_CURSOR_ERASE = (60, 60, 255)  # red-ish brush ring (erase)
_LETTERBOX = (28, 28, 28)  # canvas background outside the image

_WINDOW = "edit_masks"


def _get_screen_size() -> Tuple[int, int]:
    """Return ``(width, height)`` of the primary screen in logical pixels.

    Uses ``tkinter`` from the standard library (no extra dependency). Falls back
    to a sensible default if no display is available.
    """
    try:
        import tkinter

        root = tkinter.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return _DEFAULT_SCREEN


def _imread_unicode(path: Path) -> Optional[np.ndarray]:
    """Read an image as BGR, tolerating non-ASCII paths on Windows.

    Args:
        path: Image path.

    Returns:
        BGR image array, or ``None`` if it cannot be decoded.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _wheel_delta(flags: int) -> int:
    """Extract the signed scroll delta from an OpenCV mouse-event ``flags``.

    ``cv2.getMouseWheelDelta`` is missing from some OpenCV builds, but the delta
    is always encoded in the high 16 bits of ``flags`` as a signed short
    (positive = wheel up / zoom in).

    Args:
        flags: The ``flags`` argument passed to the mouse callback.

    Returns:
        Signed wheel delta (``> 0`` up, ``< 0`` down).
    """
    delta = flags >> 16
    if delta > 0x7FFF:
        delta -= 0x10000
    return delta


def _snap_45(start: Tuple[int, int], end: Tuple[int, int]) -> Tuple[int, int]:
    """Snap the segment ``start -> end`` to the nearest 45° direction.

    The returned endpoint keeps the drag length but its direction is rounded to
    the nearest multiple of 45° (horizontal, vertical, or diagonal).

    Args:
        start: Anchor point ``(x, y)``.
        end: Raw cursor point ``(x, y)``.

    Returns:
        Snapped endpoint ``(x, y)``.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dist = math.hypot(dx, dy)
    if dist == 0:
        return end
    angle = round(math.atan2(dy, dx) / (math.pi / 4)) * (math.pi / 4)
    return (
        int(round(start[0] + dist * math.cos(angle))),
        int(round(start[1] + dist * math.sin(angle))),
    )


class _EditState:
    """Mutable drawing state for one image (native resolution)."""

    def __init__(self, image: np.ndarray, initial_mask: np.ndarray) -> None:
        """Initialize editor state at the image's native resolution.

        Args:
            image: BGR image, shape ``(H, W, 3)``.
            initial_mask: ``uint8`` mask (0/255), shape ``(H, W)``.
        """
        self.image = image
        self.mask = initial_mask.copy()
        self.initial_mask = initial_mask.copy()
        self.brush = _BRUSH_DEFAULT
        self.erase = False
        self.last_point: Optional[Tuple[int, int]] = None
        self.undo_stack: List[np.ndarray] = []
        # Set by edits; the GUI loop clears it after recompositing.
        self.dirty = True

    # --- stroke history -----------------------------------------------------

    def push_undo(self) -> None:
        """Snapshot the mask so the upcoming stroke can be undone."""
        self.undo_stack.append(self.mask.copy())
        if len(self.undo_stack) > _MAX_UNDO:
            self.undo_stack.pop(0)

    def undo(self) -> None:
        """Restore the most recent snapshot, if any."""
        if not self.undo_stack:
            return
        self.mask = self.undo_stack.pop()
        self.dirty = True

    # --- drawing primitives -------------------------------------------------

    def _thickness(self) -> int:
        """Brush stroke diameter in pixels."""
        return max(1, self.brush * 2)

    def paint(self, x: int, y: int, erase: bool) -> None:
        """Apply one round brush dab directly to the mask at ``(x, y)``.

        Args:
            x: X coordinate in image pixels.
            y: Y coordinate in image pixels.
            erase: When True, remove (black) under the brush; else add (white).
        """
        value = 0 if erase else 255
        cv2.circle(self.mask, (x, y), self.brush, value, -1)
        self.dirty = True

    def draw_segment(
        self, p0: Tuple[int, int], p1: Tuple[int, int], erase: bool
    ) -> None:
        """Paint a thick segment directly into the mask between two points.

        Args:
            p0: Start point ``(x, y)``.
            p1: End point ``(x, y)``.
            erase: When True, remove (black) along the segment; else add (white).
        """
        value = 0 if erase else 255
        cv2.line(self.mask, p0, p1, value, self._thickness(), cv2.LINE_8)
        self.dirty = True

    def fill(self) -> None:
        """Close gaps and fill enclosed holes in the current mask.

        Bridges small gaps in the painted outline, then fills the interior of
        every closed region so a hand-drawn outline becomes a solid mask.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (_FILL_CLOSE_KERNEL, _FILL_CLOSE_KERNEL)
        )
        closed = cv2.morphologyEx(self.mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(self.mask, contours, -1, 255, thickness=cv2.FILLED)
        self.dirty = True

    def clear(self) -> None:
        """Clear the mask."""
        self.mask[:] = 0
        self.dirty = True

    def reset(self) -> None:
        """Restore the mask that was loaded."""
        self.mask = self.initial_mask.copy()
        self.dirty = True


class _Viewport:
    """Zoom/pan transform mapping image coordinates to canvas pixels."""

    def __init__(self, image_wh: Tuple[int, int], screen_wh: Tuple[int, int]) -> None:
        """Center and fit the image inside the screen.

        Args:
            image_wh: Image ``(width, height)``.
            screen_wh: Screen ``(width, height)``.
        """
        self.image_wh = image_wh
        self.screen_wh = screen_wh
        self.zoom = _fit_scale(image_wh, screen_wh)
        sw, sh = screen_wh
        iw, ih = image_wh
        # Image coordinate shown at the canvas top-left, chosen to center it.
        self.view_x = iw / 2 - sw / (2 * self.zoom)
        self.view_y = ih / 2 - sh / (2 * self.zoom)

    def to_image(self, sx: float, sy: float) -> Tuple[int, int]:
        """Map a canvas point to integer image coordinates (unclamped)."""
        return (
            int(round(self.view_x + sx / self.zoom)),
            int(round(self.view_y + sy / self.zoom)),
        )

    def zoom_at(self, sx: float, sy: float, factor: float) -> None:
        """Zoom by ``factor`` keeping the image point under ``(sx, sy)`` fixed."""
        ix = self.view_x + sx / self.zoom
        iy = self.view_y + sy / self.zoom
        self.zoom = float(np.clip(self.zoom * factor, _ZOOM_MIN, _ZOOM_MAX))
        self.view_x = ix - sx / self.zoom
        self.view_y = iy - sy / self.zoom

    def pan(self, dsx: float, dsy: float) -> None:
        """Pan the view by a screen-space delta."""
        self.view_x -= dsx / self.zoom
        self.view_y -= dsy / self.zoom

    def matrix(self) -> np.ndarray:
        """Affine matrix mapping image -> canvas for ``cv2.warpAffine``."""
        z = self.zoom
        return np.array(
            [[z, 0, -self.view_x * z], [0, z, -self.view_y * z]], dtype=np.float32
        )


def _fit_scale(size_wh: Tuple[int, int], screen_wh: Tuple[int, int]) -> float:
    """Scale factor fitting an image inside the screen (small margin reserved)."""
    width, height = size_wh
    sw, sh = screen_wh
    return min((sw - 8) / width, (sh - 8) / height)


def _compose(state: _EditState) -> np.ndarray:
    """Composite the image with the mask tint and boundary at native res.

    Args:
        state: Current edit state.

    Returns:
        BGR image, shape ``(H, W, 3)``.
    """
    comp = state.image.copy()
    sel = state.mask > 0
    if sel.any():
        comp[sel] = (0.5 * comp[sel] + 0.5 * _MASK_TINT).astype(np.uint8)
        # Crisp red outline of the mask boundary for precise edge feedback.
        contours, _ = cv2.findContours(
            state.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(comp, contours, -1, _LINE_COLOR, 2)
    return comp


def _draw_hud(canvas: np.ndarray, state: _EditState, vp: _Viewport) -> None:
    """Draw a translucent control bar and cursor ring onto the canvas.

    Args:
        canvas: BGR canvas (modified in place).
        state: Current edit state.
        vp: Active viewport.
    """
    sw, sh = vp.screen_wh
    # Bottom translucent bar.
    bar = canvas[sh - 34 : sh, :].copy()
    cv2.rectangle(bar, (0, 0), (sw, 34), (20, 20, 20), -1)
    canvas[sh - 34 : sh, :] = cv2.addWeighted(canvas[sh - 34 : sh, :], 0.35, bar, 0.65, 0)

    mode = "ERASE" if state.erase else "DRAW"
    mode_color = _CURSOR_ERASE if state.erase else _CURSOR_DRAW
    cv2.putText(
        canvas, mode, (12, sh - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2,
        cv2.LINE_AA,
    )
    hint = (
        f"brush {state.brush}px  |  scroll:zoom  mid/space:pan  "
        "L:draw  shift+L:line  R:erase  f:fill  z:undo  r:reset  "
        "ENTER:save  ESC:skip  q:quit"
    )
    cv2.putText(
        canvas, hint, (120, sh - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
        (235, 235, 235), 1, cv2.LINE_AA,
    )


def _render(state: _EditState, vp: _Viewport, comp: np.ndarray,
            cursor: Tuple[int, int]) -> np.ndarray:
    """Build the full canvas for the current frame.

    Args:
        state: Current edit state.
        vp: Active viewport.
        comp: Cached native composite from :func:`_compose`.
        cursor: Last known cursor position in canvas pixels.

    Returns:
        BGR canvas, shape ``(screen_h, screen_w, 3)``.
    """
    sw, sh = vp.screen_wh
    interp = cv2.INTER_NEAREST if vp.zoom >= 1 else cv2.INTER_LINEAR
    canvas = cv2.warpAffine(
        comp, vp.matrix(), (sw, sh), flags=interp,
        borderMode=cv2.BORDER_CONSTANT, borderValue=_LETTERBOX,
    )
    radius = max(1, int(round(state.brush * vp.zoom)))
    ring = _CURSOR_ERASE if state.erase else _CURSOR_DRAW
    cv2.circle(canvas, cursor, radius, ring, 1, cv2.LINE_AA)
    _draw_hud(canvas, state, vp)
    return canvas


def edit_one(
    source: Path, root: Path, masks_dir: Path, screen_wh: Tuple[int, int]
) -> str:
    """Edit and (on ENTER) save the mask for one image.

    Args:
        source: Source image path.
        root: Input root (for mirroring the mask path).
        masks_dir: Mask root.
        screen_wh: Screen ``(width, height)`` for the fullscreen canvas.

    Returns:
        ``"saved"``, ``"cancelled"`` (ESC), or ``"quit"`` (q / window closed).
    """
    bgr = _imread_unicode(source)
    if bgr is None:
        print(f"  {source.name}: cannot read image, skipping", file=sys.stderr)
        return "cancelled"

    height, width = bgr.shape[:2]
    initial = _load_initial_mask(source, root, masks_dir, (width, height))
    state = _EditState(bgr, initial)
    vp = _Viewport((width, height), screen_wh)

    # Interaction flags shared with the mouse callback.
    ui = {
        "cursor": (screen_wh[0] // 2, screen_wh[1] // 2),
        "pan_mode": False,
        "panning": False,
        "pan_last": (0, 0),
        "drawing": False,
        "erasing": False,
        "last_img": (0, 0),
        "shift_line": False,
        "line_down": (0, 0),
        "line_erase": False,
        "moved": False,
        "rubber_end": (0, 0),
        "stroke_base": None,
        "view_dirty": True,
        "title_dirty": True,
    }

    def clamp(p: Tuple[int, int]) -> Tuple[int, int]:
        return (
            int(np.clip(p[0], 0, width - 1)),
            int(np.clip(p[1], 0, height - 1)),
        )

    def on_mouse(event: int, x: int, y: int, flags: int, _param) -> None:
        ui["cursor"] = (x, y)
        ui["view_dirty"] = True
        shift = bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        ip = clamp(vp.to_image(x, y))

        if event == cv2.EVENT_MOUSEWHEEL:
            factor = _ZOOM_IN if _wheel_delta(flags) > 0 else _ZOOM_OUT
            vp.zoom_at(x, y, factor)
            ui["title_dirty"] = True
            return

        # --- pan (middle button, or space-toggled left button) ---
        if event == cv2.EVENT_MBUTTONDOWN or (
            event == cv2.EVENT_LBUTTONDOWN and ui["pan_mode"]
        ):
            ui["panning"] = True
            ui["pan_last"] = (x, y)
            return
        if ui["panning"]:
            if event == cv2.EVENT_MOUSEMOVE:
                vp.pan(x - ui["pan_last"][0], y - ui["pan_last"][1])
                ui["pan_last"] = (x, y)
                return
            if event in (cv2.EVENT_MBUTTONUP, cv2.EVENT_LBUTTONUP):
                ui["panning"] = False
                return

        # --- erase (right button) ---
        if event == cv2.EVENT_RBUTTONDOWN:
            state.push_undo()
            ui["erasing"] = True
            ui["last_img"] = ip
            state.paint(ip[0], ip[1], True)
            return
        if event == cv2.EVENT_RBUTTONUP and ui["erasing"]:
            ui["erasing"] = False
            state.last_point = ip
            return

        # --- left button: draw / line ---
        if event == cv2.EVENT_LBUTTONDOWN:
            state.push_undo()
            if shift:
                ui["shift_line"] = True
                ui["moved"] = False
                ui["line_down"] = ip
                ui["line_erase"] = state.erase
                ui["stroke_base"] = state.mask.copy()
                anchor = state.last_point or ip
                state.draw_segment(anchor, ip, state.erase)
            else:
                ui["drawing"] = True
                ui["last_img"] = ip
                state.paint(ip[0], ip[1], state.erase)
            return

        if event == cv2.EVENT_MOUSEMOVE:
            if ui["shift_line"]:
                ui["moved"] = True
                state.mask = ui["stroke_base"].copy()
                end = clamp(_snap_45(ui["line_down"], ip))
                state.draw_segment(ui["line_down"], end, ui["line_erase"])
                ui["rubber_end"] = end
            elif ui["drawing"]:
                state.draw_segment(ui["last_img"], ip, state.erase)
                ui["last_img"] = ip
            elif ui["erasing"]:
                state.draw_segment(ui["last_img"], ip, True)
                ui["last_img"] = ip
            return

        if event == cv2.EVENT_LBUTTONUP:
            if ui["shift_line"]:
                state.last_point = ui["rubber_end"] if ui["moved"] else ip
                ui["shift_line"] = False
            elif ui["drawing"]:
                ui["drawing"] = False
                state.last_point = ip

    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(_WINDOW, on_mouse)
    print(f"  editing {source.name} ({width}x{height})")

    comp = _compose(state)
    last_title = ""
    while True:
        if state.dirty:
            comp = _compose(state)
            state.dirty = False
            ui["view_dirty"] = True
        if ui["view_dirty"]:
            cv2.imshow(_WINDOW, _render(state, vp, comp, ui["cursor"]))
            ui["view_dirty"] = False
        if ui["title_dirty"]:
            mode = "ERASE" if state.erase else "DRAW"
            title = f"{source.name} - {int(vp.zoom * 100)}% | {mode}"
            if title != last_title:
                cv2.setWindowTitle(_WINDOW, title)
                last_title = title
            ui["title_dirty"] = False

        key = cv2.waitKey(16) & 0xFF
        if cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return "quit"
        if key == 255:
            continue
        if key == 13:  # ENTER
            _save(state, source, root, masks_dir, (width, height))
            return "saved"
        if key == 27:  # ESC
            return "cancelled"
        if key == ord("q"):
            return "quit"
        _handle_key(key, state, ui)


def _handle_key(key: int, state: _EditState, ui: dict) -> None:
    """Apply a non-terminal keypress to the edit state.

    Args:
        key: The pressed key code (masked to a byte).
        state: Current edit state.
        ui: Interaction flags (for mode toggles needing a redraw/title update).
    """
    if key == ord("f"):
        state.push_undo()
        state.fill()
    elif key == ord("z"):
        state.undo()
    elif key == ord("c"):
        state.push_undo()
        state.clear()
    elif key == ord("r"):
        state.push_undo()
        state.reset()
    elif key == ord("e"):
        state.erase = True
        ui["view_dirty"] = ui["title_dirty"] = True
    elif key == ord("d"):
        state.erase = False
        ui["view_dirty"] = ui["title_dirty"] = True
    elif key == ord(" "):
        ui["pan_mode"] = not ui["pan_mode"]
        ui["view_dirty"] = True
    elif key == ord("["):
        state.brush = max(_BRUSH_MIN, state.brush - 1)
        ui["view_dirty"] = True
    elif key == ord("]"):
        state.brush = min(_BRUSH_MAX, state.brush + 1)
        ui["view_dirty"] = True


def _load_initial_mask(
    source: Path, root: Path, masks_dir: Path, size_wh: Tuple[int, int]
) -> np.ndarray:
    """Load any existing mask for ``source`` at ``size_wh``.

    Prefers the per-image mask; if absent, falls back to the folder's shared
    mask so a masked-folder silhouette can be corrected. Saving (ENTER) always
    writes a per-image mask, which then overrides the shared one for this image.

    Args:
        source: Source image path.
        root: Input root.
        masks_dir: Mask root.
        size_wh: Target ``(width, height)``.

    Returns:
        ``uint8`` mask (0/255) at the requested size; all-zero if none exists.
    """
    path = mask_path(source, root, masks_dir)
    width, height = size_wh
    if not path.exists():
        shared = shared_mask_path(source.parent, root, masks_dir)
        if shared.exists():
            path = shared
        else:
            return np.zeros((height, width), dtype=np.uint8)
    boolean = load_mask(path)
    if boolean.shape != (height, width):
        boolean = resize_mask(boolean, (width, height))
    return np.where(boolean, np.uint8(255), np.uint8(0))


def _save(
    state: _EditState,
    source: Path,
    root: Path,
    masks_dir: Path,
    native_wh: Tuple[int, int],
) -> None:
    """Write the edited mask, resizing to native size if it differs.

    Args:
        state: Current edit state.
        source: Source image path.
        root: Input root.
        masks_dir: Mask root.
        native_wh: Source native ``(width, height)``.
    """
    boolean = state.mask > 0
    if boolean.shape != (native_wh[1], native_wh[0]):
        boolean = resize_mask(boolean, native_wh)
    destination = mask_path(source, root, masks_dir)
    save_mask(boolean, destination)
    print(f"  {source.name} -> {destination}")


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Define and parse command-line arguments.

    Args:
        argv: Optional argument vector.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(
        prog="python edit_masks.py",
        description="Interactively draw/correct masks and save them to masks/.",
    )
    parser.add_argument(
        "--images", nargs="+", required=True, help="Source images to edit."
    )
    parser.add_argument(
        "--root", default="input", help="Input root (for mirroring mask paths)."
    )
    parser.add_argument(
        "--masks", default="masks", help="Mask root (mirrors the input tree)."
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``python edit_masks.py``.

    Args:
        argv: Optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` on a normal finish, ``2`` if no valid images were given.
    """
    args = _parse_args(argv)
    root = Path(args.root)
    masks_dir = Path(args.masks)
    images: List[Path] = [Path(p) for p in args.images]
    existing = [p for p in images if p.is_file()]
    for path in (p for p in images if not p.is_file()):
        print(f"warning: not a file, skipping: {path}", file=sys.stderr)
    if not existing:
        print("error: no valid images to edit", file=sys.stderr)
        return 2

    screen_wh = _get_screen_size()
    n_saved = n_cancelled = 0
    try:
        for index, source in enumerate(existing, start=1):
            print(f"[{index}/{len(existing)}] {source}")
            result = edit_one(source, root, masks_dir, screen_wh)
            if result == "saved":
                n_saved += 1
            elif result == "cancelled":
                n_cancelled += 1
            else:  # quit
                print("Session ended by user.")
                break
    finally:
        cv2.destroyAllWindows()

    print(f"\nDone: {n_saved} saved, {n_cancelled} skipped.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
