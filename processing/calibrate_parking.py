"""
calibrate_parking.py — Single-camera parking lot calibration pipeline.

Replaces processing.draw_zones as the primary calibration tool.
Produces homography-based slot polygons stored in the DB and a top-down PNG.

Usage:
    python -m processing.calibrate_parking --uri <video_or_stream_uri>
    python -m processing.calibrate_parking --uri <uri> --recalibrate
    python -m processing.calibrate_parking --uri <uri> --visualize
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
TOPDOWN_SCALE = 80  # pixels per metre


# ---------------------------------------------------------------------------
# URI → cam_name
# ---------------------------------------------------------------------------
def uri_to_cam_name(uri: str) -> str:
    """Derive a filesystem-safe camera name from a URI.

    Examples:
        "videos/lot_a.mp4"          → "lot_a"
        "rtsp://192.168.1.10/stream"→ "192.168.1.10_stream"
    """
    if "://" in uri:
        # Strip protocol
        s = re.sub(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", "", uri)
        # Remove extension from last path component if present
        p = Path(s)
        if p.suffix:
            s = str(p.with_suffix(""))
        # Replace slashes with underscores
        s = s.replace("/", "_").replace("\\", "_")
        # Replace characters that are not alphanumeric, dot, or underscore
        s = re.sub(r"[^a-zA-Z0-9._]", "_", s)
    else:
        s = Path(uri).stem
    # Collapse multiple underscores and strip leading/trailing ones
    s = re.sub(r"_+", "_", s).strip("_")
    return s


# ---------------------------------------------------------------------------
# Calibration directory helpers
# ---------------------------------------------------------------------------
def get_calibration_dirs() -> tuple[Path, Path]:
    """Return (configs_dir, output_dir) from env vars."""
    configs_dir = Path(os.getenv("CALIBRATION_CONFIGS_DIR", "calibration/configs"))
    output_dir = Path(os.getenv("CALIBRATION_OUTPUT_DIR", "calibration/output"))
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
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# ---------------------------------------------------------------------------
# Step 2 — DraggableRect (homography calibration UI)
# ---------------------------------------------------------------------------
class DraggableRect:
    """Interactive 4-corner rectangle for parking area alignment."""

    HANDLE_RADIUS = 8
    EDGE_HIT = 12
    PAD_FRAC = 0.8

    def __init__(self, frame: np.ndarray,
                 rows: int = 0, cols: int = 0,
                 slot_w: float = 0.0, slot_h: float = 0.0,
                 row_gap: float = 0.0):
        self.frame = frame.copy()
        h, w = frame.shape[:2]
        self.pad_x = int(w * self.PAD_FRAC)
        self.pad_y = int(h * self.PAD_FRAC)
        self.rows = rows
        self.cols = cols
        self.slot_w = slot_w
        self.slot_h = slot_h
        self.row_gap = row_gap
        mx, my = int(w * 0.2), int(h * 0.2)
        self.pts = np.array([
            [mx,     my    ],
            [w - mx, my    ],
            [w - mx, h - my],
            [mx,     h - my],
        ], dtype=float)
        self.drag_idx = None
        self.drag_edge = None
        self.drag_origin = None
        self.pts_origin = None
        self.confirmed = False
        self.cancelled = False

    def _nearest_corner(self, x, y):
        dists = [np.hypot(x - p[0], y - p[1]) for p in self.pts]
        idx = int(np.argmin(dists))
        return idx if dists[idx] < self.HANDLE_RADIUS * 2 else None

    def _on_edge(self, x, y):
        p = self.pts
        edges = [(p[0], p[1]), (p[1], p[2]), (p[2], p[3]), (p[3], p[0])]
        for i, (a, b) in enumerate(edges):
            ab = b - a
            ap = np.array([x, y], dtype=float) - a
            t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
            proj = a + t * ab
            if np.hypot(x - proj[0], y - proj[1]) < self.EDGE_HIT:
                return i
        return None

    def _inside(self, x, y):
        contour = self.pts.reshape(-1, 1, 2).astype(np.float32)
        return cv2.pointPolygonTest(contour, (float(x), float(y)), False) >= 0

    def mouse_callback(self, event, x, y, flags, _param):
        fx = x - self.pad_x
        fy = y - self.pad_y
        if event == cv2.EVENT_LBUTTONDOWN:
            c = self._nearest_corner(fx, fy)
            if c is not None:
                self.drag_idx = c
            elif (e := self._on_edge(fx, fy)) is not None:
                self.drag_edge = e
                self.drag_origin = np.array([fx, fy], dtype=float)
                self.pts_origin = self.pts.copy()
            elif self._inside(fx, fy):
                self.drag_idx = -1
                self.drag_origin = np.array([fx, fy], dtype=float)
                self.pts_origin = self.pts.copy()
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
            self.drag_idx = None
            self.drag_edge = None

    def _live_homography(self):
        if self.rows == 0 or self.cols == 0:
            return None
        total_h = self.rows * self.slot_h + max(0, self.rows - 1) * self.row_gap
        src = np.float32([
            (0.0,                 0.0     ),
            (self.cols * self.slot_w, 0.0 ),
            (self.cols * self.slot_w, total_h),
            (0.0,                 total_h ),
        ])
        dst = np.float32(self.pts)
        H, _ = cv2.findHomography(src, dst)
        return H

    def _draw_grid_overlay(self, canvas: np.ndarray, H, px: int, py: int):
        offset = np.array([px, py])
        overlay = canvas.copy()
        for i in range(self.rows):
            for j in range(self.cols):
                x0 = j * self.slot_w
                y0 = i * (self.slot_h + self.row_gap)
                x1 = (j + 1) * self.slot_w
                y1 = y0 + self.slot_h
                corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                pts_px = []
                for rx, ry in corners:
                    p = cv2.perspectiveTransform(np.float32([[[rx, ry]]]), H)[0][0]
                    pts_px.append((p + offset).astype(int))
                poly = np.array(pts_px).reshape(-1, 1, 2)
                cv2.fillPoly(overlay, [poly], (100, 180, 100))
        cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)
        for i in range(self.rows):
            for j in range(self.cols):
                x0 = j * self.slot_w
                y0 = i * (self.slot_h + self.row_gap)
                x1 = (j + 1) * self.slot_w
                y1 = y0 + self.slot_h
                corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                pts_px = []
                for rx, ry in corners:
                    p = cv2.perspectiveTransform(np.float32([[[rx, ry]]]), H)[0][0]
                    pts_px.append((p + offset).astype(int))
                poly = np.array(pts_px).reshape(-1, 1, 2)
                cv2.polylines(canvas, [poly], True, (0, 200, 80), 1)

    def _draw(self) -> np.ndarray:
        fh, fw = self.frame.shape[:2]
        px, py = self.pad_x, self.pad_y
        canvas = np.full((fh + 2 * py, fw + 2 * px, 3), 60, dtype=np.uint8)
        canvas[py:py + fh, px:px + fw] = self.frame
        H = self._live_homography()
        if H is not None:
            self._draw_grid_overlay(canvas, H, px, py)
        pts_canvas = (self.pts + np.array([px, py])).astype(int)
        cv2.polylines(canvas, [pts_canvas], isClosed=True, color=(0, 255, 0), thickness=2)
        for p in pts_canvas:
            cv2.circle(canvas, tuple(p), self.HANDLE_RADIUS, (0, 200, 255), -1)
            cv2.circle(canvas, tuple(p), self.HANDLE_RADIUS, (0, 100, 180), 2)
        hud_lines = [
            "Drag corners/edges/body to fit the parking area",
            "Y: confirm | R: reset | Esc: exit without saving",
        ]
        for i, line in enumerate(hud_lines):
            y = py + 30 + i * 28
            cv2.putText(canvas, line, (px + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(canvas, line, (px + 10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return canvas

    def _reset(self):
        h, w = self.frame.shape[:2]
        mx, my = int(w * 0.2), int(h * 0.2)
        self.pts = np.array([
            [mx,     my    ],
            [w - mx, my    ],
            [w - mx, h - my],
            [mx,     h - my],
        ], dtype=float)

    def run(self):
        win = "Calibration - adjust rectangle to cover parking area"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.waitKey(1)
        cv2.setMouseCallback(win, self.mouse_callback)
        while True:
            cv2.imshow(win, self._draw())
            key = cv2.waitKey(16) & 0xFF
            if key in (ord("y"), ord("Y")):
                self.confirmed = True
                break
            elif key in (ord("r"), ord("R")):
                self._reset()
            elif key == 27:
                self.cancelled = True
                break
        cv2.destroyWindow(win)


# ---------------------------------------------------------------------------
# Step 2 — Compute homography
# ---------------------------------------------------------------------------
def calibrate_homography(frame: np.ndarray, rows: int, cols: int,
                         slot_w: float, slot_h: float, row_gap: float = 0.0):
    """
    Returns (H 3×3, control_points [[x,y]×4]) or (None, None) if cancelled.
    """
    total_h = rows * slot_h + max(0, rows - 1) * row_gap
    while True:
        rect = DraggableRect(frame, rows=rows, cols=cols,
                             slot_w=slot_w, slot_h=slot_h, row_gap=row_gap)
        rect.run()
        if rect.cancelled:
            print("[INFO] Calibration cancelled by user.")
            return None, None

        p_TL, p_TR, p_BR, p_BL = (rect.pts[i] for i in range(4))
        src = np.float32([(0.0, 0.0), (cols * slot_w, 0.0),
                          (cols * slot_w, total_h), (0.0, total_h)])
        dst = np.float32([p_TL, p_TR, p_BR, p_BL])
        H, _ = cv2.findHomography(src, dst)
        if H is None:
            print("[ERROR] findHomography failed (collinear points?). "
                  "Please reposition the rectangle.", file=sys.stderr)
            continue
        ctrl = [p_TL.tolist(), p_TR.tolist(), p_BR.tolist(), p_BL.tolist()]
        return H, ctrl


# ---------------------------------------------------------------------------
# Step 3 — Grid projection
# ---------------------------------------------------------------------------
def project_point(x_real: float, y_real: float, H: np.ndarray):
    pt = np.float32([[[x_real, y_real]]])
    return cv2.perspectiveTransform(pt, H)[0][0]


def build_grid(rows: int, cols: int, slot_w: float, slot_h: float,
               H: np.ndarray, frame_w: int, frame_h: int, row_gap: float = 0.0):
    """Return list of slot dicts: row, col, polygon_full_px, status, centroid."""
    slots = []
    for i in range(rows):
        for j in range(cols):
            x0 = j * slot_w
            y0 = i * (slot_h + row_gap)
            x1 = (j + 1) * slot_w
            y1 = y0 + slot_h
            corners_real = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            poly_full = [project_point(x, y, H) for x, y in corners_real]
            poly_full_int = [[int(round(p[0])), int(round(p[1]))] for p in poly_full]
            centroid = [
                sum(p[0] for p in poly_full) / 4,
                sum(p[1] for p in poly_full) / 4,
            ]

            def inside(p):
                return 0 <= p[0] < frame_w and 0 <= p[1] < frame_h

            all_in = all(inside(p) for p in poly_full)
            cent_in = inside(centroid)
            if all_in:
                status = "full"
            elif cent_in:
                status = "partial"
            else:
                status = "out"

            slots.append({
                "row": i, "col": j,
                "polygon_full_px": poly_full_int,
                "status": status,
                "centroid": centroid,
            })
    return slots


# ---------------------------------------------------------------------------
# Step 4 — Clip polygon & approx to quad
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
    approx = cv2.approxPolyDP(pts, epsilon, closed=True)
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
# Metric polygon helper
# ---------------------------------------------------------------------------
def slot_metric_polygon(row: int, col: int,
                        slot_w: float, slot_h: float,
                        row_gap: float = 0.0) -> list:
    """Return the 4 corners of a slot in camera-local metric space (metres)."""
    x0 = col * slot_w
    y0 = row * (slot_h + row_gap)
    return [
        [x0,          y0         ],
        [x0 + slot_w, y0         ],
        [x0 + slot_w, y0 + slot_h],
        [x0,          y0 + slot_h],
    ]


# ---------------------------------------------------------------------------
# Step 5 — Interactive slot selection
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
        self.frame = frame.copy()
        self.fh, self.fw = frame.shape[:2]
        self.slots = [s for s in slots if s["status"] in ("full", "partial")]
        self.active = {}
        for s in self.slots:
            key = f"{s['row']}_{s['col']}"
            if previous_active and key in previous_active:
                self.active[key] = previous_active[key]
            else:
                self.active[key] = True
        self.saved = False
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
        img = self.frame.copy()
        overlay = img.copy()
        for s in self.slots:
            key = f"{s['row']}_{s['col']}"
            poly = np.array(s["polygon_full_px"], dtype=np.int32)
            is_active = self.active[key]
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
            key = f"{s['row']}_{s['col']}"
            poly = np.array(s["polygon_full_px"], dtype=np.int32)
            is_active = self.active[key]
            is_partial = s["status"] == "partial"
            if is_active:
                border = self.COL_ACTIVE_BORDER
            elif is_partial:
                border = self.COL_PARTIAL_BORDER
            else:
                border = self.COL_INACTIVE_BDR
            if is_partial and not is_active:
                pts = poly.tolist()
                n = len(pts)
                for k in range(n):
                    a = tuple(pts[k])
                    b = tuple(pts[(k + 1) % n])
                    total = int(np.hypot(b[0] - a[0], b[1] - a[1]))
                    dashes = max(1, total // 10)
                    for d in range(0, total, dashes * 2):
                        t0 = d / max(total, 1)
                        t1 = min((d + dashes) / max(total, 1), 1.0)
                        pa = (int(a[0] + t0 * (b[0] - a[0])), int(a[1] + t0 * (b[1] - a[1])))
                        pb = (int(a[0] + t1 * (b[0] - a[0])), int(a[1] + t1 * (b[1] - a[1])))
                        cv2.line(img, pa, pb, border, 2)
            else:
                cv2.polylines(img, [poly.reshape(-1, 1, 2)], True, border, 2)
            cx, cy = int(s["centroid"][0]), int(s["centroid"][1])
            label = f"r{s['row']}_c{s['col']}"
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
        cv2.waitKey(1)
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
# Save top-down map
# ---------------------------------------------------------------------------
def save_topdown(path: Path, cam_name: str, rows: int, cols: int,
                 slot_w: float, slot_h: float, row_gap: float, margin: float,
                 slots: list, active: dict):
    scale = TOPDOWN_SCALE
    margin_px = int(margin * scale)
    total_w_m = cols * slot_w + 2 * margin
    total_h_m = rows * slot_h + max(0, rows - 1) * row_gap + 2 * margin
    canvas_w = int(total_w_m * scale) + 1
    canvas_h = int(total_h_m * scale) + 1
    axis_pad = 50
    title_pad = 40
    img_w = canvas_w + axis_pad
    img_h = canvas_h + axis_pad + title_pad
    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255
    ox = axis_pad + margin_px
    oy = title_pad + margin_px
    for s in slots:
        i, j = s["row"], s["col"]
        status = s["status"]
        key = f"{i}_{j}"
        is_active = active.get(key, False)
        x0 = ox + int(j * slot_w * scale)
        y0 = oy + int(i * (slot_h + row_gap) * scale)
        x1 = ox + int((j + 1) * slot_w * scale)
        y1 = oy + int((i * (slot_h + row_gap) + slot_h) * scale)
        if status == "out":
            fill, border = (235, 235, 235), (200, 200, 200)
        elif is_active and status == "partial":
            fill, border = (200, 230, 200), (0, 180, 180)
        elif is_active:
            fill, border = (180, 230, 180), (0, 140, 0)
        else:
            fill, border = (210, 210, 210), (130, 130, 130)
        cv2.rectangle(img, (x0, y0), (x1, y1), fill, -1)
        cv2.rectangle(img, (x0, y0), (x1, y1), border, 2)
        if status == "out":
            cv2.line(img, (x0, y0), (x1, y1), (170, 170, 170), 1)
            cv2.line(img, (x1, y0), (x0, y1), (170, 170, 170), 1)
        cx_lbl = (x0 + x1) // 2
        cy_lbl = (y0 + y1) // 2
        cv2.putText(img, f"r{i}_c{j}", (cx_lbl - 18, cy_lbl + 5),
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
    cv2.putText(img, f"{cam_name} — {rows}x{cols}", (axis_pad, 28),
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


def prompt_params() -> dict:
    """Ask the user for grid parameters interactively."""
    print("\n[calibrate_parking] Enter parking grid parameters:")
    rows = int(input("  rows (number of rows): "))
    cols = int(input("  cols (number of columns): "))
    slot_w_s = input("  slot_w in metres [2.5]: ").strip()
    slot_h_s = input("  slot_h in metres [5.0]: ").strip()
    row_gap_s = input("  row_gap in metres [6.0]: ").strip()
    margin_s = input("  margin in metres [1.0]: ").strip()
    frame_idx_s = input("  frame_idx [0]: ").strip()
    return {
        "rows": rows,
        "cols": cols,
        "slot_w": float(slot_w_s) if slot_w_s else 2.5,
        "slot_h": float(slot_h_s) if slot_h_s else 5.0,
        "row_gap": float(row_gap_s) if row_gap_s else 6.0,
        "margin": float(margin_s) if margin_s else 1.0,
        "frame_idx": int(frame_idx_s) if frame_idx_s else 0,
    }


def save_config_file(path: Path, uri: str, params: dict,
                     H: np.ndarray, ctrl_pts: list,
                     slots: list, active: dict):
    slots_dict = {}
    for s in slots:
        key = f"{s['row']}_{s['col']}"
        slots_dict[key] = {
            "status": s["status"],
            "polygon_full_px": s["polygon_full_px"],
            "polygon_detection_px": s.get("polygon_detection_px"),
            "active": active.get(key, False),
        }
    data = {
        "uri": uri,
        "frame_width": params.get("frame_width", 0),
        "frame_height": params.get("frame_height", 0),
        "frame_idx": params.get("frame_idx", 0),
        "rows": params["rows"],
        "cols": params["cols"],
        "slot_w": params["slot_w"],
        "slot_h": params["slot_h"],
        "row_gap": params.get("row_gap", 0.0),
        "margin": params.get("margin", 1.0),
        "homography": H.tolist(),
        "control_points_pixel": ctrl_pts,
        "slots": slots_dict,
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

        camera.frame_width = frame_w
        camera.frame_height = frame_h
        camera.homography = H.tolist()

        # Delete existing zones for this camera
        db.query(Zone).filter(Zone.camera_id == camera.id).delete(
            synchronize_session=False
        )

        slot_w = params["slot_w"]
        slot_h = params["slot_h"]
        row_gap = params.get("row_gap", 0.0)

        for s in sorted(slots, key=lambda x: (x["row"], x["col"])):
            key = f"{s['row']}_{s['col']}"
            if not active.get(key, False):
                continue
            poly_detect = s.get("polygon_detection_px")
            if poly_detect is None:
                continue
            poly_metric = slot_metric_polygon(s["row"], s["col"], slot_w, slot_h, row_gap)
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

    window = "calibrate_parking — visualize"
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
            pts = poly.reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        for idx, (name, poly) in enumerate(polys):
            color = _COLORS[idx % len(_COLORS)]
            pts = poly.reshape(-1, 1, 2)
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
        description="Homography-based parking lot calibration — single camera.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--uri", required=True,
                        help="Video file path or RTSP stream URI")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Redo interactive calibration even if config exists")
    parser.add_argument("--visualize", action="store_true",
                        help="Show calibrated zones on live video (no write)")
    args = parser.parse_args()

    uri = args.uri
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
        # Camera already calibrated — show summary and offer recalibration
        print(f"\n[calibrate_parking] Existing config found for '{cam_name}'.")
        print(f"  rows={cfg['rows']}, cols={cfg['cols']}, "
              f"slot_w={cfg['slot_w']}, slot_h={cfg['slot_h']}, "
              f"row_gap={cfg.get('row_gap', 0.0)}, margin={cfg.get('margin', 1.0)}")
        if "homography" in cfg:
            active_count = sum(1 for v in cfg.get("slots", {}).values() if v.get("active"))
            print(f"  Homography: calibrated  |  Active slots: {active_count}")
        answer = input("Re-run calibration? [y/N]: ").strip().lower()
        if answer != "y":
            print("[calibrate_parking] Exiting without changes.")
            return
        # User wants to recalibrate
        args.recalibrate = True

    # ------------------------------------------------------------------
    # Load or prompt parameters
    # ------------------------------------------------------------------
    if cfg is not None:
        params = {
            "rows": cfg["rows"],
            "cols": cfg["cols"],
            "slot_w": cfg["slot_w"],
            "slot_h": cfg["slot_h"],
            "row_gap": cfg.get("row_gap", 0.0),
            "margin": cfg.get("margin", 1.0),
            "frame_idx": cfg.get("frame_idx", 0),
        }
        existing_active = {k: v["active"] for k, v in cfg.get("slots", {}).items()}
        if args.recalibrate:
            print("[calibrate_parking] Recalibrating — existing slot states will be inherited.")
    else:
        params = prompt_params()
        existing_active = None
        # Save initial config without calibration data
        config_path = configs_dir / f"{cam_name}.json"
        config_path.write_text(json.dumps({"uri": uri, **params}, indent=2))
        print(f"[calibrate_parking] Config saved to {config_path}")

    # ------------------------------------------------------------------
    # Extract frame
    # ------------------------------------------------------------------
    frame_idx = params.get("frame_idx", 0)
    frame = extract_frame(uri, frame_idx)
    if frame is None:
        print(f"[ERROR] Cannot read frame {frame_idx} from {uri}", file=sys.stderr)
        sys.exit(1)
    fh, fw = frame.shape[:2]
    params["frame_width"] = fw
    params["frame_height"] = fh

    # ------------------------------------------------------------------
    # Step 2 — Calibrate homography
    # ------------------------------------------------------------------
    H, ctrl_pts = calibrate_homography(
        frame,
        params["rows"], params["cols"],
        params["slot_w"], params["slot_h"],
        params.get("row_gap", 0.0),
    )
    if H is None:
        print("[calibrate_parking] Exiting without saving.")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Steps 3 & 4 — Build grid, clip polygons
    # ------------------------------------------------------------------
    slots = build_grid(
        params["rows"], params["cols"],
        params["slot_w"], params["slot_h"],
        H, fw, fh,
        params.get("row_gap", 0.0),
    )
    slots = compute_detection_polygons(slots, fw, fh)

    # ------------------------------------------------------------------
    # Step 5 — Interactive slot selection
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
    # Config file
    config_path = configs_dir / f"{cam_name}.json"
    save_config_file(config_path, uri, params, H, ctrl_pts, slots, active)
    print(f"[calibrate_parking] Config saved:    {config_path}")

    # Top-down PNG
    topdown_path = output_dir / f"{cam_name}_topdown.png"
    save_topdown(
        topdown_path, cam_name,
        params["rows"], params["cols"],
        params["slot_w"], params["slot_h"],
        params.get("row_gap", 0.0), params.get("margin", 1.0),
        slots, active,
    )
    print(f"[calibrate_parking] Top-down saved:  {topdown_path}")

    # DB
    save_to_db(uri, cam_name, fw, fh, H, slots, active, params)

    # Summary
    total = len(slots)
    out_cnt = sum(1 for s in slots if s["status"] == "out")
    active_slots = [s for s in slots
                    if active.get(f"{s['row']}_{s['col']}", False)
                    and s.get("polygon_detection_px") is not None]
    print(f"\n[calibrate_parking] Done.")
    print(f"  Total slots: {total}  |  Out-of-frame: {out_cnt}  |  Active: {len(active_slots)}")


if __name__ == "__main__":
    main()
