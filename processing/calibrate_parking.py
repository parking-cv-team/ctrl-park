"""
calibrate_parking.py - Single-camera parking lot calibration pipeline.

Replaces processing.draw_zones as the primary calibration tool.
Produces homography-based slot polygons stored in the DB and a top-down PNG.

Usage:
    python -m processing.calibrate_parking --uri <video_or_stream_uri>
    python -m processing.calibrate_parking --uri <uri> --recalibrate
    python -m processing.calibrate_parking --uri <uri> --visualize

Grid layout with files_per_row=2:
    Each logical row contains two facing files of slots (nose-to-nose, zero gap
    between them). row_gap separates consecutive logical rows (the aisle/road).

    logical row 0: [file 0 →][← file 1]
    --------------- row_gap -----------
    logical row 1: [file 0 →][← file 1]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOPDOWN_SCALE = 80          # pixels per metre
PANEL_W       = 340         # width of the side parameter panel (pixels)
PANEL_BG      = (45, 45, 45)


# ---------------------------------------------------------------------------
# URI → cam_name
# ---------------------------------------------------------------------------
def uri_to_cam_name(uri: str) -> str:
    """Derive a filesystem-safe camera name from a URI.

    Examples:
        "videos/lot_a.mp4"           → "lot_a"
        "rtsp://192.168.1.10/stream" → "192.168.1.10_stream"
    """
    if "://" in uri:
        s = re.sub(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", "", uri)
        p = Path(s)
        if p.suffix:
            s = str(p.with_suffix(""))
        s = s.replace("/", "_").replace("\\", "_")
        s = re.sub(r"[^a-zA-Z0-9._]", "_", s)
    else:
        s = Path(uri).stem
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# ---------------------------------------------------------------------------
# Calibration directory helpers
# ---------------------------------------------------------------------------
def get_calibration_dirs() -> tuple[Path, Path]:
    """Return (configs_dir, output_dir) from env vars."""
    configs_dir = Path(os.getenv("CALIBRATION_CONFIGS_DIR", "calibration/configs"))
    output_dir  = Path(os.getenv("CALIBRATION_OUTPUT_DIR",  "calibration/output"))
    configs_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return configs_dir, output_dir


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def extract_frame(uri: str, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {uri}", file=sys.stderr)
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        # RTSP streams often produce corrupted frames on initial connect; skip them.
    for _ in range(30):
        ok,frame = cap.read()
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# ---------------------------------------------------------------------------
# Grid geometry helpers
# ---------------------------------------------------------------------------
def total_grid_size(rows: int, cols: int,
                    slot_w: float, slot_h: float,
                    row_gap: float, files_per_row: int) -> tuple[float, float]:
    """Return (total_width_m, total_height_m) of the complete parking grid."""
    total_w = cols * slot_w
    # Each logical row contains `files_per_row` files stacked with no inner gap.
    # row_gap separates logical rows.
    logical_rows = rows // files_per_row
    if logical_rows == 0:
        logical_rows = 1
    total_h = logical_rows * (files_per_row * slot_h) + max(0, logical_rows - 1) * row_gap
    return total_w, total_h


def slot_y_origin(file_idx: int, logical_row: int,
                  slot_h: float, row_gap: float,
                  files_per_row: int) -> float:
    """Return the y-origin (metres) of a slot given its logical row and file index."""
    return logical_row * (files_per_row * slot_h + row_gap) + file_idx * slot_h


# ---------------------------------------------------------------------------
# Step 2 - DraggableRect  (homography calibration UI with side panel)
# ---------------------------------------------------------------------------
class DraggableRect:
    """Interactive 4-corner rectangle for parking area alignment.

    The window is composed of two side-by-side sections:
      • Left  (PANEL_W px): parameter panel with trackbars
      • Right (frame size): the camera frame + grid overlay

    All mouse coordinates on the right section are offset by -PANEL_W on X.
    """

    HANDLE_RADIUS = 8
    EDGE_HIT      = 12

    # Custom slider definitions: (key, lo, hi, default, scale, label, unit, color_bgr)
    # Displayed value = int_position / scale
    _SL_DEFS = [
        ("rows",    1,  40,   2,  1.0, "Rows",        "",   (100, 210, 120)),
        ("cols",    1,  40,   5,  1.0, "Cols",        "",   (100, 210, 120)),
        ("slot_w",  5, 150,  25, 10.0, "Slot Width",  "m",  ( 90, 160, 230)),
        ("slot_h",  5, 150,  50, 10.0, "Slot Height", "m",  ( 90, 160, 230)),
        ("row_gap", 0, 200,  60, 10.0, "Row Gap",     "m",  (180, 130, 230)),
        ("cushion", 0, 600, 150,  1.0, "Canvas Pad",  "px", (210, 160,  90)),
    ]

    def __init__(self, frame: np.ndarray, uri: str, initial_params: dict | None = None):
        self.uri = uri
        self.base_frame = frame.copy()
        fh, fw = frame.shape[:2]
        self.fh, self.fw = fh, fw

        # Canvas is fixed; frame is scaled+centered within the right area.
        self.panel_h       = fh + 300            # fixed canvas height
        self.canvas_w_base = fw + 300 + PANEL_W  # fixed canvas width
        # Display state updated each frame in _draw
        self.disp_scale    = 1.0
        self.frame_ox_disp = PANEL_W + 150
        self.frame_oy_disp = 150

        # Initial rectangle: 20% inset from frame edges (in frame-local coords)
        mx, my = int(fw * 0.2), int(fh * 0.2)
        self.pts = np.array([
            [mx,      my    ],
            [fw - mx, my    ],
            [fw - mx, fh - my],
            [mx,      fh - my],
        ], dtype=float)

        self.fpr         = int((initial_params or {}).get("files_per_row", 1))
        self.drag_idx    = None
        self.drag_edge   = None
        self.drag_origin = None
        self.pts_origin  = None
        self.confirmed   = False
        self.cancelled   = False

        # Cached params read from trackbars each frame
        self._params: dict = {}
        self._win: str = ""
        self._ui_scale = 1.0
        self._last_frame_idx = -1

        # If re-calibrating, pre-load existing values
        self._initial = initial_params or {}

        # Slider integer positions (initialized from initial_params or defaults)
        _cfg_map = {
            "rows": "rows", "cols": "cols", "slot_w": "slot_w",
            "slot_h": "slot_h", "row_gap": "row_gap", "cushion": "workspace_cushion",
        }
        self._slider_vals: dict[str, int] = {}
        for key, lo, hi, default, scale, *_ in self._SL_DEFS:
            cfg_key = _cfg_map.get(key)
            if cfg_key and cfg_key in self._initial:
                pos = int(round(self._initial[cfg_key] * scale))
                self._slider_vals[key] = max(lo, min(hi, pos))
            else:
                self._slider_vals[key] = default
        self._drag_slider: str | None = None

    def _read_params(self) -> dict:
        return {key: self._slider_vals.get(key, default) / scale
                for key, lo, hi, default, scale, *_ in self._SL_DEFS}

    def _params_changed(self, old: dict, new: dict) -> bool:
        return any(old.get(k) != new.get(k) for k, *_ in self._SL_DEFS)

    def _reset_rect(self):
        mx, my = int(self.fw * 0.2), int(self.fh * 0.2)
        self.pts = np.array([
            [mx,         my       ],
            [self.fw-mx, my       ],
            [self.fw-mx, self.fh-my],
            [mx,         self.fh-my],
        ], dtype=float)

    # ------------------------------------------------------------------
    # Grid geometry from current params
    # ------------------------------------------------------------------
    def _grid_dims(self, p: dict) -> tuple[int, int, float, float, float, int]:
        rows          = max(1, int(p["rows"]))
        cols          = max(1, int(p["cols"]))
        slot_w        = max(0.1, p["slot_w"])
        slot_h        = max(0.1, p["slot_h"])
        row_gap       = max(0.0, p["row_gap"])
        files_per_row = self.fpr
        return rows, cols, slot_w, slot_h, row_gap, files_per_row

    # ------------------------------------------------------------------
    # Homography (src = metric space → dst = frame pixel space)
    # ------------------------------------------------------------------
    def _live_homography(self, p: dict):
        rows, cols, slot_w, slot_h, row_gap, fpr = self._grid_dims(p)
        total_w, total_h = total_grid_size(rows, cols, slot_w, slot_h, row_gap, fpr)
        if total_w < 1e-6 or total_h < 1e-6:
            return None
        src = np.float32([
            (0.0,    0.0    ),
            (total_w, 0.0   ),
            (total_w, total_h),
            (0.0,    total_h),
        ])
        dst = np.float32(self.pts)
        H, _ = cv2.findHomography(src, dst)
        return H

    # ------------------------------------------------------------------
    # Grid overlay drawing
    # ------------------------------------------------------------------
    def _draw_grid_overlay(self, canvas: np.ndarray, H, p: dict,
                           s: float, frame_ox: int, frame_oy: int):
        rows, cols, slot_w, slot_h, row_gap, fpr = self._grid_dims(p)
        logical_rows = max(1, rows // fpr)
        offset = np.array([frame_ox, frame_oy], dtype=float)

        def to_canvas(rx, ry):
            pp = cv2.perspectiveTransform(np.float32([[[rx, ry]]]), H)[0][0]
            return (pp * s + offset).astype(int)

        overlay = canvas.copy()
        file_colors = [(100, 180, 100), (100, 130, 220)]

        for lr in range(logical_rows):
            for fi in range(fpr):
                for j in range(cols):
                    x0 = j * slot_w
                    y0 = slot_y_origin(fi, lr, slot_h, row_gap, fpr)
                    corners = [(x0, y0), (x0+slot_w, y0), (x0+slot_w, y0+slot_h), (x0, y0+slot_h)]
                    poly = np.array([to_canvas(rx, ry) for rx, ry in corners]).reshape(-1, 1, 2)
                    cv2.fillPoly(overlay, [poly], file_colors[fi % 2])

        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

        for lr in range(logical_rows):
            for fi in range(fpr):
                for j in range(cols):
                    x0 = j * slot_w
                    y0 = slot_y_origin(fi, lr, slot_h, row_gap, fpr)
                    corners = [(x0, y0), (x0+slot_w, y0), (x0+slot_w, y0+slot_h), (x0, y0+slot_h)]
                    poly = np.array([to_canvas(rx, ry) for rx, ry in corners]).reshape(-1, 1, 2)
                    cv2.polylines(canvas, [poly], True, (0, 200, 80), 1)

    # ------------------------------------------------------------------
    # Panel drawing
    # ------------------------------------------------------------------
    def _draw_panel(self, canvas: np.ndarray, p: dict):
        """Render parameter panel with custom sliders."""
        panel = canvas[:self.panel_h, :PANEL_W]
        panel[:] = PANEL_BG

        # ── Title ──────────────────────────────────────────────────────
        cv2.putText(panel, "CALIBRATION", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.line(panel, (12, 36), (PANEL_W - 12, 36), (75, 75, 75), 1)

        # ── Files / Row buttons ────────────────────────────────────────
        cv2.putText(panel, "Files / Row", (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 170), 1, cv2.LINE_AA)
        for val in [1, 2]:
            bx0 = 120 + (val - 1) * 60
            bx1 = bx0 + 50
            active = (self.fpr == val)
            cv2.rectangle(panel, (bx0, 44), (bx1, 70),
                          (65, 125, 75) if active else (60, 60, 60), -1)
            cv2.rectangle(panel, (bx0, 44), (bx1, 70), (120, 120, 120), 1)
            cv2.putText(panel, str(val), (bx0 + 18, 63),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255) if active else (150, 150, 150), 1, cv2.LINE_AA)

        # ── Sliders ────────────────────────────────────────────────────
        _TX0, _TX1 = 12, PANEL_W - 12   # track x bounds (full panel width)
        _TLN = _TX1 - _TX0              # track length in pixels
        _SH  = 46                       # height per slider row
        _SY0 = 86                       # y of first slider top
        _TRKH, _THMR = 6, 7            # track height, thumb radius

        for idx, (key, lo, hi, default, scale, label, unit, color) in enumerate(self._SL_DEFS):
            val_int  = self._slider_vals.get(key, default)
            t        = (val_int - lo) / max(1, hi - lo)
            thumb_x  = int(_TX0 + t * _TLN)
            label_y  = _SY0 + idx * _SH + 16
            track_cy = _SY0 + idx * _SH + _SH - 14

            # Label (left) + current value (right) above track
            disp = f"{val_int / scale:.1f} {unit}" if unit else str(int(val_int / scale))
            cv2.putText(panel, label, (_TX0, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (185, 185, 185), 1, cv2.LINE_AA)
            cv2.putText(panel, disp, (_TX0 + 120, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 210, 75), 1, cv2.LINE_AA)

            # Track background + border
            cv2.rectangle(panel,
                          (_TX0,   track_cy - _TRKH // 2),
                          (_TX1,   track_cy + _TRKH // 2), (58, 58, 58), -1)
            cv2.rectangle(panel,
                          (_TX0,   track_cy - _TRKH // 2),
                          (_TX1,   track_cy + _TRKH // 2), (88, 88, 88), 1)

            # Colored fill up to thumb
            if thumb_x > _TX0:
                cv2.rectangle(panel,
                              (_TX0,    track_cy - _TRKH // 2),
                              (thumb_x, track_cy + _TRKH // 2), color, -1)

            # Thumb: shadow ring + face
            cv2.circle(panel, (thumb_x, track_cy), _THMR + 1, (18, 18, 18), -1)
            cv2.circle(panel, (thumb_x, track_cy), _THMR, (225, 225, 225), -1)

        # ── Info ───────────────────────────────────────────────────────
        rows, cols, slot_w, slot_h, row_gap, fpr = self._grid_dims(p)
        logical_rows = max(1, rows // fpr)
        total_w, total_h = total_grid_size(rows, cols, slot_w, slot_h, row_gap, fpr)
        total_slots = logical_rows * fpr * cols

        info_y = _SY0 + len(self._SL_DEFS) * _SH + 8
        cv2.line(panel, (12, info_y), (PANEL_W - 12, info_y), (75, 75, 75), 1)
        info_y += 18
        for text, clr in [
            (f"Grid:   {total_w:.1f} x {total_h:.1f} m", (215, 195, 150)),
            (f"Slots:  {total_slots}",                    (215, 195, 150)),
        ]:
            cv2.putText(panel, text, (12, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, clr, 1, cv2.LINE_AA)
            info_y += 20

        # ── Controls ───────────────────────────────────────────────────
        info_y += 6
        cv2.line(panel, (12, info_y), (PANEL_W - 12, info_y), (75, 75, 75), 1)
        info_y += 18
        for text, clr in [
            ("Drag  corners / edges / body", (155, 155, 155)),
            ("Y   confirm",                  ( 95, 215,  95)),
            ("R   reset rectangle",          (215, 170,  90)),
            ("Esc  exit without saving",     (215,  90,  90)),
        ]:
            cv2.putText(panel, text, (12, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, clr, 1, cv2.LINE_AA)
            info_y += 18

    # ------------------------------------------------------------------
    # Full canvas draw
    # ------------------------------------------------------------------
    def _draw(self, p: dict) -> np.ndarray:
        canvas_h = self.panel_h
        canvas_w = self.canvas_w_base
        right_w  = canvas_w - PANEL_W

        # Scale frame to fit right area minus cushion margins, centered
        pad     = max(0, int(p.get("cushion", 150)))
        avail_w = max(10, right_w - 2 * pad)
        avail_h = max(10, canvas_h - 2 * pad)
        s       = max(0.25, min(avail_w / self.fw, avail_h / self.fh))
        sw, sh  = int(self.fw * s), int(self.fh * s)
        frame_ox = PANEL_W + (right_w - sw) // 2
        frame_oy = (canvas_h - sh) // 2

        # Store for mouse coordinate conversion
        self.disp_scale    = s
        self.frame_ox_disp = frame_ox
        self.frame_oy_disp = frame_oy

        canvas = np.full((canvas_h, canvas_w, 3), 50, dtype=np.uint8)

        # Draw scaled frame
        canvas[frame_oy:frame_oy + sh,
               frame_ox:frame_ox + sw] = cv2.resize(self.base_frame, (sw, sh))
        # Grid overlay
        H = self._live_homography(p)
        if H is not None:
            self._draw_grid_overlay(canvas, H, p, s, frame_ox, frame_oy)
        # Rectangle handles scaled to display coords
        pts_canvas = (self.pts * s + np.array([frame_ox, frame_oy])).astype(int)
        cv2.polylines(canvas, [pts_canvas], isClosed=True, color=(0, 255, 0), thickness=2)
        for pp in pts_canvas:
            cv2.circle(canvas, tuple(pp), self.HANDLE_RADIUS, (0, 200, 255), -1)
            cv2.circle(canvas, tuple(pp), self.HANDLE_RADIUS, (0, 100, 180), 2)
        # Panel
        self._draw_panel(canvas, p)
        # Vertical separator
        cv2.line(canvas, (PANEL_W, 0), (PANEL_W, canvas_h), (80, 80, 80), 1)
        return canvas

    # ------------------------------------------------------------------
    # Mouse (coordinates are in full canvas space)
    # ------------------------------------------------------------------
    def _frame_coords(self, x, y):
        """Convert canvas mouse coords to frame-local coords."""
        s = self.disp_scale or 1.0
        return (x - self.frame_ox_disp) / s, (y - self.frame_oy_disp) / s

    def _nearest_corner(self, fx, fy):
        dists = [np.hypot(fx - p[0], fy - p[1]) for p in self.pts]
        idx = int(np.argmin(dists))
        return idx if dists[idx] < self.HANDLE_RADIUS * 2 else None

    def _on_edge(self, fx, fy):
        p = self.pts
        edges = [(p[0], p[1]), (p[1], p[2]), (p[2], p[3]), (p[3], p[0])]
        for i, (a, b) in enumerate(edges):
            ab = b - a
            ap = np.array([fx, fy], dtype=float) - a
            t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
            proj = a + t * ab
            if np.hypot(fx - proj[0], fy - proj[1]) < self.EDGE_HIT:
                return i
        return None

    def _inside(self, fx, fy):
        contour = self.pts.reshape(-1, 1, 2).astype(np.float32)
        return cv2.pointPolygonTest(contour, (float(fx), float(fy)), False) >= 0

    def mouse_callback(self, event, x, y, flags, _param):
        if x < PANEL_W:
            _TX0, _TX1 = 12, PANEL_W - 12
            _SH, _SY0  = 46, 86
            if event == cv2.EVENT_LBUTTONDOWN:
                # Files/row buttons (drawn at y=44..70)
                if 44 < y < 70:
                    if 120 < x < 170:
                        self.fpr = 1
                    elif 180 < x < 230:
                        self.fpr = 2
                # Slider hit test
                for idx, (key, lo, hi, *_) in enumerate(self._SL_DEFS):
                    tcy = _SY0 + idx * _SH + _SH - 14
                    if abs(y - tcy) < 14 and _TX0 <= x <= _TX1:
                        self._drag_slider = key
                        break
            elif event == cv2.EVENT_MOUSEMOVE and self._drag_slider:
                for key, lo, hi, default, scale, *_ in self._SL_DEFS:
                    if key == self._drag_slider:
                        t = max(0.0, min(1.0,
                            (x - _TX0) / max(1, _TX1 - _TX0)))
                        self._slider_vals[key] = int(round(lo + t * (hi - lo)))
                        break
            elif event == cv2.EVENT_LBUTTONUP:
                self._drag_slider = None
            return
        fx, fy = self._frame_coords(x, y)

        if event == cv2.EVENT_LBUTTONDOWN:
            c = self._nearest_corner(fx, fy)
            if c is not None:
                self.drag_idx = c
            elif (e := self._on_edge(fx, fy)) is not None:
                self.drag_edge   = e
                self.drag_origin = np.array([fx, fy], dtype=float)
                self.pts_origin  = self.pts.copy()
            elif self._inside(fx, fy):
                self.drag_idx    = -1
                self.drag_origin = np.array([fx, fy], dtype=float)
                self.pts_origin  = self.pts.copy()

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_idx is not None and self.drag_idx >= 0:
                self.pts[self.drag_idx] = [fx, fy]
            elif self.drag_idx == -1:
                delta = np.array([fx, fy], dtype=float) - self.drag_origin
                self.pts = self.pts_origin + delta
            elif self.drag_edge is not None:
                delta = np.array([fx, fy], dtype=float) - self.drag_origin
                edge_corners = [(0, 1), (1, 2), (2, 3), (3, 0)]
                a_idx, b_idx = edge_corners[self.drag_edge]
                if self.drag_edge in (0, 2):
                    self.pts[a_idx][1] = self.pts_origin[a_idx][1] + delta[1]
                    self.pts[b_idx][1] = self.pts_origin[b_idx][1] + delta[1]
                else:
                    self.pts[a_idx][0] = self.pts_origin[a_idx][0] + delta[0]
                    self.pts[b_idx][0] = self.pts_origin[b_idx][0] + delta[0]

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_idx  = None
            self.drag_edge = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> dict | None:
        """Run the calibration UI.

        Returns the params dict on confirmation, or None if cancelled.
        """
        self._win = "Calibration - adjust trackbars and rectangle"
        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        # Show an initial frame to ensure the window is created/active (realized)
        # before we attempt to set callbacks or trackbars.
        cv2.imshow(self._win, self.base_frame)
        cv2.waitKey(50)

        cv2.setMouseCallback(self._win, self.mouse_callback)

        self._last_frame_idx = int(self._initial.get("frame_idx", 0))

        prev_params: dict = {}

        while True:
            cur_params = self._read_params()

            canvas = self._draw(cur_params)
            cv2.imshow(self._win, canvas)

            key = cv2.waitKey(16) & 0xFF
            if key in (ord("y"), ord("Y")):
                self.confirmed = True
                break
            elif key in (ord("r"), ord("R")):
                self._reset_rect()
            elif key == 27:
                self.cancelled = True
                break
            try:
                if cv2.getWindowProperty(self._win, cv2.WND_PROP_VISIBLE) < 1:
                    self.cancelled = True
                    break
            except cv2.error:
                self.cancelled = True
                break

        cv2.destroyWindow(self._win)

        if self.confirmed:
            rows, cols, slot_w, slot_h, row_gap, fpr = self._grid_dims(cur_params)
            return {
                "rows":          rows,
                "cols":          cols,
                "slot_w":        slot_w,
                "slot_h":        slot_h,
                "row_gap":       row_gap,
                "files_per_row": fpr,
                "frame_idx":     self._last_frame_idx,
            }
        return None


# ---------------------------------------------------------------------------
# Step 2 - Compute homography (updated signature)
# ---------------------------------------------------------------------------
def calibrate_homography(frame: np.ndarray, uri: str, initial_params: dict | None = None):
    """Show the interactive calibration UI.

    Returns (H 3×3, control_points [[x,y]×4], params_dict)
    or (None, None, None) if cancelled.
    """
    while True:
        rect = DraggableRect(frame, uri, initial_params=initial_params)
        params = rect.run()

        if rect.cancelled or params is None:
            print("[INFO] Calibration cancelled by user.")
            return None, None, None

        rows     = params["rows"]
        cols     = params["cols"]
        slot_w   = params["slot_w"]
        slot_h   = params["slot_h"]
        row_gap  = params["row_gap"]
        fpr      = params["files_per_row"]

        total_w, total_h = total_grid_size(rows, cols, slot_w, slot_h, row_gap, fpr)

        p_TL, p_TR, p_BR, p_BL = (rect.pts[i] for i in range(4))
        src = np.float32([
            (0.0,    0.0    ),
            (total_w, 0.0   ),
            (total_w, total_h),
            (0.0,    total_h),
        ])
        dst = np.float32([p_TL, p_TR, p_BR, p_BL])
        H, _ = cv2.findHomography(src, dst)
        if H is None:
            print("[ERROR] findHomography failed (collinear points?). "
                  "Please reposition the rectangle.", file=sys.stderr)
            continue

        ctrl = [p_TL.tolist(), p_TR.tolist(), p_BR.tolist(), p_BL.tolist()]
        return H, ctrl, params


# ---------------------------------------------------------------------------
# Step 3 - Grid projection (updated for files_per_row)
# ---------------------------------------------------------------------------
def project_point(x_real: float, y_real: float, H: np.ndarray):
    pt = np.float32([[[x_real, y_real]]])
    return cv2.perspectiveTransform(pt, H)[0][0]


def build_grid(rows: int, cols: int, slot_w: float, slot_h: float,
               H: np.ndarray, frame_w: int, frame_h: int,
               row_gap: float = 0.0, files_per_row: int = 1) -> list:
    """Return list of slot dicts.

    Each slot dict contains: row, col, file (0 or 1), logical_row,
    polygon_full_px, status, centroid.

    With files_per_row=2 each logical row has two files of slots.
    The 'row' field is the absolute file index (0-based across all files),
    so downstream code that uses row/col keys stays compatible.
    """
    slots = []
    logical_rows = max(1, rows // files_per_row)

    def inside(p):
        return 0 <= p[0] < frame_w and 0 <= p[1] < frame_h

    for lr in range(logical_rows):
        for fi in range(files_per_row):
            abs_row = lr * files_per_row + fi
            for j in range(cols):
                x0 = j * slot_w
                y0 = slot_y_origin(fi, lr, slot_h, row_gap, files_per_row)
                x1 = x0 + slot_w
                y1 = y0 + slot_h
                corners_real = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                poly_full    = [project_point(x, y, H) for x, y in corners_real]
                poly_full_int = [[int(round(p[0])), int(round(p[1]))] for p in poly_full]
                centroid = [
                    sum(p[0] for p in poly_full) / 4,
                    sum(p[1] for p in poly_full) / 4,
                ]

                all_in  = all(inside(p) for p in poly_full)
                cent_in = inside(centroid)
                if all_in:
                    status = "full"
                elif cent_in:
                    status = "partial"
                else:
                    status = "out"

                slots.append({
                    "row":          abs_row,
                    "col":          j,
                    "file":         fi,
                    "logical_row":  lr,
                    "polygon_full_px": poly_full_int,
                    "status":       status,
                    "centroid":     centroid,
                })
    return slots


# ---------------------------------------------------------------------------
# Step 4 - Clip polygon & approx to quad
# ---------------------------------------------------------------------------
def clip_polygon_to_frame(polygon_px, frame_w: int, frame_h: int):
    frame_rect = np.float32([
        [0,       0      ],
        [frame_w, 0      ],
        [frame_w, frame_h],
        [0,       frame_h],
    ])
    poly = np.float32(polygon_px)
    _, intersection = cv2.intersectConvexConvex(poly, frame_rect)
    if intersection is None or len(intersection) == 0:
        return None
    return intersection.reshape(-1, 2).astype(np.float32)


def approx_to_quad(polygon):
    """Reduce a convex polygon to exactly 4 vertices."""
    pts = np.float32(polygon).reshape(-1, 1, 2)
    epsilon = 0.05 * cv2.arcLength(pts, closed=True)
    approx  = cv2.approxPolyDP(pts, epsilon, closed=True)
    if len(approx) == 4:
        return approx.reshape(-1, 2).astype(int).tolist()
    hull = cv2.convexHull(pts).reshape(-1, 2)
    if len(hull) < 4:
        return None
    indices = [0]
    for _ in range(3):
        dists = [min(np.linalg.norm(hull[i] - hull[j]) for j in indices)
                 for i in range(len(hull))]
        indices.append(int(np.argmax(dists)))
    return hull[sorted(indices)].astype(int).tolist()


def compute_detection_polygons(slots, frame_w: int, frame_h: int):
    """Add polygon_detection_px to each slot (in-place)."""
    for slot in slots:
        if slot["status"] == "out":
            slot["polygon_detection_px"] = None
            continue
        clipped = clip_polygon_to_frame(slot["polygon_full_px"], frame_w, frame_h)
        if clipped is None or len(clipped) < 3:
            slot["polygon_detection_px"] = None
            slot["status"] = "out"
            continue
        quad = approx_to_quad(clipped)
        if quad is None:
            slot["polygon_detection_px"] = None
            slot["status"] = "out"
        else:
            slot["polygon_detection_px"] = quad
    return slots


# ---------------------------------------------------------------------------
# Metric polygon helper (updated for files_per_row)
# ---------------------------------------------------------------------------
def slot_metric_polygon(row: int, col: int,
                        slot_w: float, slot_h: float,
                        row_gap: float = 0.0,
                        files_per_row: int = 1) -> list:
    """Return the 4 corners of a slot in camera-local metric space (metres)."""
    fi  = row % files_per_row
    lr  = row // files_per_row
    x0  = col * slot_w
    y0  = slot_y_origin(fi, lr, slot_h, row_gap, files_per_row)
    return [
        [x0,          y0         ],
        [x0 + slot_w, y0         ],
        [x0 + slot_w, y0 + slot_h],
        [x0,          y0 + slot_h],
    ]


# ---------------------------------------------------------------------------
# Step 5 - Interactive slot selection
# ---------------------------------------------------------------------------
class SlotSelector:
    ALPHA = 0.35
    COL_ACTIVE_FILL    = (100, 220, 100)
    COL_ACTIVE_BORDER  = (0,   160,   0)
    COL_PARTIAL_FILL   = (0,   220, 220)
    COL_PARTIAL_BORDER = (0,   180, 180)
    COL_INACTIVE_FILL  = (160, 160, 160)
    COL_INACTIVE_BDR   = (100, 100, 100)

    def __init__(self, frame: np.ndarray, slots: list,
                 previous_active: dict = None):
        self.frame  = frame.copy()
        self.fh, self.fw = frame.shape[:2]
        self.slots  = [s for s in slots if s["status"] in ("full", "partial")]
        self.active = {}
        for s in self.slots:
            key = f"{s['row']}_{s['col']}"
            if previous_active and key in previous_active:
                self.active[key] = previous_active[key]
            else:
                self.active[key] = True
        self.saved     = False
        self.cancelled = False

    def _toggle(self, x, y):
        for s in self.slots:
            poly = np.array(s["polygon_full_px"], dtype=np.int32)
            if cv2.pointPolygonTest(poly.reshape(-1, 1, 2),
                                    (float(x), float(y)), False) >= 0:
                key = f"{s['row']}_{s['col']}"
                self.active[key] = not self.active[key]
                return

    def mouse_callback(self, event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._toggle(x, y)

    def _draw(self) -> np.ndarray:
        img     = self.frame.copy()
        overlay = img.copy()
        for s in self.slots:
            key        = f"{s['row']}_{s['col']}"
            poly       = np.array(s["polygon_full_px"], dtype=np.int32)
            is_active  = self.active[key]
            is_partial = s["status"] == "partial"
            if is_active:
                fill = self.COL_ACTIVE_FILL
            elif is_partial:
                fill = self.COL_PARTIAL_FILL
            else:
                fill = self.COL_INACTIVE_FILL
            cv2.fillPoly(overlay, [poly.reshape(-1, 1, 2)], fill)
        cv2.addWeighted(overlay, self.ALPHA, img, 1 - self.ALPHA, 0, img)
        for s in self.slots:
            key        = f"{s['row']}_{s['col']}"
            poly       = np.array(s["polygon_full_px"], dtype=np.int32)
            is_active  = self.active[key]
            is_partial = s["status"] == "partial"
            if is_active:
                border = self.COL_ACTIVE_BORDER
            elif is_partial:
                border = self.COL_PARTIAL_BORDER
            else:
                border = self.COL_INACTIVE_BDR
            if is_partial and not is_active:
                pts = poly.tolist()
                n   = len(pts)
                for k in range(n):
                    a     = tuple(pts[k])
                    b     = tuple(pts[(k + 1) % n])
                    total = int(np.hypot(b[0] - a[0], b[1] - a[1]))
                    dashes = max(1, total // 10)
                    for d in range(0, total, dashes * 2):
                        t0 = d / max(total, 1)
                        t1 = min((d + dashes) / max(total, 1), 1.0)
                        pa = (int(a[0] + t0*(b[0]-a[0])), int(a[1] + t0*(b[1]-a[1])))
                        pb = (int(a[0] + t1*(b[0]-a[0])), int(a[1] + t1*(b[1]-a[1])))
                        cv2.line(img, pa, pb, border, 2)
            else:
                cv2.polylines(img, [poly.reshape(-1, 1, 2)], True, border, 2)
            cx, cy = int(s["centroid"][0]), int(s["centroid"][1])
            label  = f"r{s['row']}_c{s['col']}"
            cv2.putText(img, label, (cx - 20, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (cx - 20, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        hud = "Click: toggle | A: activate all | Z: deactivate all | Q: save & exit | Esc: exit without saving"
        cv2.putText(img, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    def run(self):
        win = "Slot Selection"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        # Show initial frame to realize the window before setting the callback
        cv2.imshow(win, self.frame)
        cv2.waitKey(50)
        cv2.setMouseCallback(win, self.mouse_callback)
        while True:
            cv2.imshow(win, self._draw())
            key = cv2.waitKey(16) & 0xFF
            if key in (ord("q"), ord("Q")):
                self.saved = True
                break
            elif key in (ord("a"), ord("A")):
                for k in self.active:
                    self.active[k] = True
            elif key in (ord("z"), ord("Z")):
                for k in self.active:
                    self.active[k] = False
            elif key == 27:
                self.cancelled = True
                break
        cv2.destroyWindow(win)


# ---------------------------------------------------------------------------
# Save top-down map (updated for files_per_row)
# ---------------------------------------------------------------------------
def save_topdown(path: Path, cam_name: str, rows: int, cols: int,
                 slot_w: float, slot_h: float, row_gap: float,
                 slots: list, active: dict, files_per_row: int = 1):
    scale       = TOPDOWN_SCALE
    margin      = 0.0 # Margin is now retired
    margin_px   = 0
    fpr         = files_per_row
    logical_rows = max(1, rows // fpr)
    total_w_m   = cols * slot_w
    total_h_m   = logical_rows * fpr * slot_h + max(0, logical_rows - 1) * row_gap
    canvas_w    = int(total_w_m * scale) + 1
    canvas_h    = int(total_h_m * scale) + 1
    axis_pad    = 50
    title_pad   = 40
    img_w       = canvas_w + axis_pad
    img_h       = canvas_h + axis_pad + title_pad
    img         = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    ox          = axis_pad + margin_px
    oy          = title_pad + margin_px

    for s in slots:
        abs_row = s["row"]
        j       = s["col"]
        fi      = s.get("file", abs_row % fpr)
        lr      = s.get("logical_row", abs_row // fpr)
        status  = s["status"]
        key     = f"{abs_row}_{j}"
        is_active = active.get(key, False)

        x0 = ox + int(j * slot_w * scale)
        y0 = oy + int((lr * (fpr * slot_h + row_gap) + fi * slot_h) * scale)
        x1 = ox + int((j + 1) * slot_w * scale)
        y1 = oy + int((lr * (fpr * slot_h + row_gap) + (fi + 1) * slot_h) * scale)

        if status == "out":
            fill, border = (235, 235, 235), (200, 200, 200)
        elif is_active and status == "partial":
            fill, border = (200, 230, 200), (0, 180, 180)
        elif is_active:
            fill, border = (180, 230, 180), (0, 140, 0)
        else:
            fill, border = (210, 210, 210), (130, 130, 130)

        cv2.rectangle(img, (x0, y0), (x1, y1), fill,   -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), border,  2)
        if status == "out":
            cv2.line(img, (x0, y0), (x1, y1), (170, 170, 170), 1)
            cv2.line(img, (x1, y0), (x0, y1), (170, 170, 170), 1)

        cx_lbl = (x0 + x1) // 2
        cy_lbl = (y0 + y1) // 2
        cv2.putText(img, f"r{abs_row}_c{j}", (cx_lbl - 18, cy_lbl + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 60, 60), 1, cv2.LINE_AA)

    # Axes
    ax_y = title_pad + canvas_h + 10
    for tick_m in range(int(total_w_m) + 1):
        tx = axis_pad + int(tick_m * scale)
        cv2.line(img, (tx, ax_y - 5), (tx, ax_y + 5), (80, 80, 80), 1)
        cv2.putText(img, str(tick_m), (tx - 6, ax_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1)
    for tick_m in range(int(total_h_m) + 1):
        ty = title_pad + int(tick_m * scale)
        cv2.line(img, (axis_pad - 5, ty), (axis_pad + 5, ty), (80, 80, 80), 1)
        cv2.putText(img, str(tick_m), (2, ty + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1)
    cv2.putText(img, f"{cam_name} - {logical_rows}x{fpr}x{cols}", (axis_pad, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# Config file I/O
# ---------------------------------------------------------------------------
def load_config(configs_dir: Path, cam_name: str) -> dict | None:
    path = configs_dir / f"{cam_name}.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def save_config_file(path: Path, params: dict, ctrl_pts: list):
    """Save only the parameters needed to recalibrate (everything else is in the DB)."""
    data = {
        "frame_idx":            params.get("frame_idx", 0),
        "rows":                 params["rows"],
        "cols":                 params["cols"],
        "slot_w":               params["slot_w"],
        "slot_h":               params["slot_h"],
        "row_gap":              params.get("row_gap", 0.0),
        "files_per_row":        params.get("files_per_row", 1),
        "control_points_pixel": ctrl_pts,
    }
    path.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# DB I/O
# ---------------------------------------------------------------------------
def save_to_db(uri: str, cam_name: str, frame_w: int, frame_h: int,
               H: np.ndarray, slots: list, active: dict, params: dict):
    from db import SessionLocal
    from db.models import CameraSource, Zone

    db = SessionLocal()
    try:
        camera = db.query(CameraSource).filter(CameraSource.uri == uri).first()
        if not camera:
            camera = CameraSource(name=cam_name, uri=uri)
            db.add(camera)
            db.flush()
        else:
            camera.name = cam_name

        camera.frame_width  = frame_w
        camera.frame_height = frame_h
        camera.homography   = H.tolist()

        db.query(Zone).filter(Zone.camera_id == camera.id).delete(
            synchronize_session=False
        )

        slot_w       = params["slot_w"]
        slot_h       = params["slot_h"]
        row_gap      = params.get("row_gap", 0.0)
        files_per_row = params.get("files_per_row", 1)

        for s in sorted(slots, key=lambda x: (x["row"], x["col"])):
            key = f"{s['row']}_{s['col']}"
            if not active.get(key, False):
                continue
            poly_detect = s.get("polygon_detection_px")
            if poly_detect is None:
                continue
            poly_metric = slot_metric_polygon(
                s["row"], s["col"], slot_w, slot_h, row_gap, files_per_row
            )
            db.add(Zone(
                name=key,
                polygon=[[int(p[0]), int(p[1])] for p in poly_detect],
                polygon_metric=poly_metric,
                camera_id=camera.id,
                category="parking lot",
            ))

        db.commit()
        n_zones = sum(
            1 for s in slots
            if active.get(f"{s['row']}_{s['col']}", False)
            and s.get("polygon_detection_px") is not None
        )
        print(f"[calibrate_parking] Saved {n_zones} zone(s) to DB for URI: {uri}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Visualize mode
# ---------------------------------------------------------------------------
def run_visualize(uri: str):
    """Show the calibrated zones overlaid on a live video feed. Read-only."""
    from db import SessionLocal
    from db.models import CameraSource, Zone as ZoneModel

    db = SessionLocal()
    try:
        camera = db.query(CameraSource).filter(CameraSource.uri == uri).first()
        if not camera:
            print(f"[calibrate_parking] No camera found in DB for URI: {uri}")
            return
        zones = db.query(ZoneModel).filter(ZoneModel.camera_id == camera.id).all()
        polys = [(z.name, np.array(z.polygon, dtype=np.int32)) for z in zones]
    finally:
        db.close()

    window = "calibrate_parking - visualize"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {uri}", file=sys.stderr)
        return

    _COLORS = [
        (0, 255, 0), (0, 165, 255), (255, 0, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128),
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        overlay = frame.copy()
        for idx, (name, poly) in enumerate(polys):
            color = _COLORS[idx % len(_COLORS)]
            pts   = poly.reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        for idx, (name, poly) in enumerate(polys):
            color = _COLORS[idx % len(_COLORS)]
            pts   = poly.reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], True, color, 2)
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].mean())
            cv2.putText(frame, name, (cx - 15, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        cv2.imshow(window, frame)
        key = cv2.waitKey(20) & 0xFF
        try:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
        except cv2.error:
            break
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyWindow(window)


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Homography-based parking lot calibration - single camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uri", required=True,
                        help="Video file path or RTSP stream URI")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Redo interactive calibration even if config exists")
    parser.add_argument("--visualize", action="store_true",
                        help="Show calibrated zones on live video (no write)")
    args = parser.parse_args()

    uri      = args.uri
    cam_name = uri_to_cam_name(uri)
    configs_dir, output_dir = get_calibration_dirs()

    print(f"[calibrate_parking] URI:      {uri}")
    print(f"[calibrate_parking] cam_name: {cam_name}")

    # ------------------------------------------------------------------
    # --visualize: show existing zones and exit
    # ------------------------------------------------------------------
    if args.visualize:
        run_visualize(uri)
        return

    # ------------------------------------------------------------------
    # Load or create config file
    # ------------------------------------------------------------------
    cfg = load_config(configs_dir, cam_name)

    if cfg is not None and not args.recalibrate:
        print(f"\n[calibrate_parking] Existing config found for '{cam_name}'.")
        print(f"  rows={cfg['rows']}, cols={cfg['cols']}, "
              f"slot_w={cfg['slot_w']}, slot_h={cfg['slot_h']}, "
              f"row_gap={cfg.get('row_gap', 0.0)}, "
              f"files_per_row={cfg.get('files_per_row', 1)}")
        if "homography" in cfg:
            active_count = sum(1 for v in cfg.get("slots", {}).values() if v.get("active"))
            print(f"  Homography: calibrated  |  Active slots: {active_count}")
        args.recalibrate = True

    # ------------------------------------------------------------------
    # Build initial_params for the UI (from existing config if available)
    # ------------------------------------------------------------------
    if cfg is not None:
        initial_params = {
            "rows":          cfg["rows"],
            "cols":          cfg["cols"],
            "slot_w":        cfg["slot_w"],
            "slot_h":        cfg["slot_h"],
            "row_gap":       cfg.get("row_gap", 0.0),
            "files_per_row": cfg.get("files_per_row", 1),
            "frame_idx":     cfg.get("frame_idx", 0),
        }
        existing_active = None
        if args.recalibrate:
            print("[calibrate_parking] Recalibrating - existing slot states will be inherited.")
    else:
        initial_params  = None
        existing_active = None

    # ------------------------------------------------------------------
    # Extract a representative frame (use frame_idx=0 or from config)
    # ------------------------------------------------------------------
    frame_idx_init = (initial_params or {}).get("frame_idx", 0)
    frame = extract_frame(uri, frame_idx_init)
    if frame is None:
        print(f"[ERROR] Cannot read frame {frame_idx_init} from {uri}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 2 - Calibrate homography (UI includes param panel)
    # ------------------------------------------------------------------
    H, ctrl_pts, params = calibrate_homography(frame, uri, initial_params=initial_params)
    if H is None:
        print("[calibrate_parking] Exiting without saving.")
        sys.exit(0)

    # If the user changed frame_idx in the UI, re-extract the frame
    if params["frame_idx"] != frame_idx_init:
        new_frame = extract_frame(uri, params["frame_idx"])
        if new_frame is not None:
            frame = new_frame
        else:
            print(f"[WARN] Could not read frame {params['frame_idx']}, "
                  "keeping frame {frame_idx_init}.")

    fh, fw = frame.shape[:2]
    params["frame_width"]  = fw
    params["frame_height"] = fh

    # ------------------------------------------------------------------
    # Steps 3 & 4 - Build grid, clip polygons
    # ------------------------------------------------------------------
    slots = build_grid(
        params["rows"], params["cols"],
        params["slot_w"], params["slot_h"],
        H, fw, fh,
        row_gap=params.get("row_gap", 0.0),
        files_per_row=params.get("files_per_row", 1),
    )
    slots = compute_detection_polygons(slots, fw, fh)

    # ------------------------------------------------------------------
    # Step 5 - Interactive slot selection
    # ------------------------------------------------------------------
    selector = SlotSelector(frame, slots, previous_active=existing_active)
    selector.run()
    if selector.cancelled:
        print("[calibrate_parking] Exiting without saving.")
        sys.exit(0)

    active = selector.active
    for s in slots:
        key = f"{s['row']}_{s['col']}"
        if key not in active:
            active[key] = False

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    config_path = configs_dir / f"{cam_name}.json"
    save_config_file(config_path, params, ctrl_pts)
    print(f"[calibrate_parking] Config saved:    {config_path}")

    topdown_path = output_dir / f"{cam_name}_topdown.png"
    save_topdown(
        topdown_path, cam_name,
        params["rows"], params["cols"],
        params["slot_w"], params["slot_h"],
        params.get("row_gap", 0.0),
        slots, active,
        files_per_row=params.get("files_per_row", 1),
    )
    print(f"[calibrate_parking] Top-down saved:  {topdown_path}")

    save_to_db(uri, cam_name, fw, fh, H, slots, active, params)

    total      = len(slots)
    out_cnt    = sum(1 for s in slots if s["status"] == "out")
    active_slots = [s for s in slots
                    if active.get(f"{s['row']}_{s['col']}", False)
                    and s.get("polygon_detection_px") is not None]
    print(f"\n[calibrate_parking] Done.")
    print(f"  Total slots: {total}  |  Out-of-frame: {out_cnt}  |  Active: {len(active_slots)}")


if __name__ == "__main__":
    main()
