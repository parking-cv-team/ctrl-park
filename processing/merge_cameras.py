"""
merge_cameras.py — Merge multiple camera calibrations into a unified parking map.

Reads all cameras with homography from the DB (populated by calibrate_parking),
lets the user pair corresponding slots between cameras, then:
  - Writes mapped_zones rows to DB (one per physical parking slot)
  - Updates zones.mapped_zone_id for all paired zones
  - Saves merged_topdown.png to CALIBRATION_OUTPUT_DIR

Usage:
    python -m processing.merge_cameras
    python -m processing.merge_cameras --recalibrate
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOPDOWN_SCALE = 80  # px/m
PANEL_W       = 340         # width of the side parameter panel (pixels)
PANEL_BG      = (45, 45, 45)

CAM_COLORS = [          # BGR
    (220,  80,  80),    # cam 0 — blue
    (0,   165, 255),    # cam 1 — orange
    ( 80,  80, 220),    # cam 2 — red
    (200, 200,  60),    # cam 3 — cyan
    (200,  60, 200),    # cam 4 — magenta
    ( 60, 200, 200),    # cam 5 — yellow
]

IOU_MATCH_THRESHOLD = 0.3   # minimum IoU for two slots to be considered the same physical spot


def cam_color(idx):
    return CAM_COLORS[idx % len(CAM_COLORS)]


# ---------------------------------------------------------------------------
# URI → cam_name  (same logic as calibrate_parking)
# ---------------------------------------------------------------------------
def uri_to_cam_name(uri: str) -> str:
    import re
    if "://" in uri:
        s = re.sub(r"^[a-zA-Z][a-zA-Z0-9+\-.]*://", "", uri)
        p = Path(s)
        if p.suffix:
            s = str(p.with_suffix(""))
        s = s.replace("/", "_").replace("\\", "_")
        s = re.sub(r"[^a-zA-Z0-9._]", "_", s)
    else:
        s = Path(uri).stem
    import re as _re
    s = _re.sub(r"_+", "_", s).strip("_")
    return s


# ---------------------------------------------------------------------------
# Affine helpers
# ---------------------------------------------------------------------------
def to_3x3(M2x3):
    out = np.eye(3)
    out[:2, :] = M2x3
    return out


def apply_affine_pts(M2x3, pts):
    """pts: Nx2 → Nx2"""
    pts = np.asarray(pts, dtype=float)
    return (M2x3 @ np.hstack([pts, np.ones((len(pts), 1))]).T).T[:, :2]


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def extract_frame(uri: str, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(uri)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


# ---------------------------------------------------------------------------
# Load cameras from DB
# ---------------------------------------------------------------------------
def load_cameras_from_db(configs_dir: Path) -> list:
    """Load all cameras with homography and polygon_metric zones from DB."""
    from db import SessionLocal
    from db.models import CameraSource, Zone

    db = SessionLocal()
    try:
        cameras = (
            db.query(CameraSource)
            .filter(CameraSource.homography.isnot(None))
            .order_by(CameraSource.id)
            .all()
        )
        if not cameras:
            print("[ERROR] No cameras with homography found in DB.")
            print("        Run python -m processing.calibrate_parking --uri <uri> first.")
            sys.exit(1)

        cam_data_list = []
        for camera in cameras:
            cam_name = uri_to_cam_name(camera.uri)
            zones = (
                db.query(Zone)
                .filter(
                    Zone.camera_id == camera.id,
                    Zone.polygon_metric.isnot(None),
                )
                .all()
            )
            if not zones:
                print(f"[WARN] Camera '{cam_name}' has no zones with polygon_metric — skipping.")
                continue

            # Read frame_idx from config file (default 0)
            config_path = configs_dir / f"{cam_name}.json"
            frame_idx = 0
            if config_path.exists():
                with config_path.open() as f:
                    cfg_file = json.load(f)
                frame_idx = cfg_file.get("frame_idx", 0)

            # Extract representative frame for visualization
            frame = extract_frame(camera.uri, frame_idx)
            if frame is None:
                print(f"[WARN] Cannot read frame from {camera.uri}; using blank.")
                frame = np.zeros((480, 640, 3), np.uint8)

            # Build slot list
            slots = []
            for zone in zones:
                try:
                    row, col = map(int, zone.name.split("_"))
                except ValueError:
                    print(f"[WARN] Cannot parse row/col from zone name '{zone.name}' — skipping.")
                    continue
                poly_metric = np.array(zone.polygon_metric, dtype=float)
                centroid = poly_metric.mean(axis=0).tolist()
                slots.append({
                    "key": zone.name,
                    "row": row,
                    "col": col,
                    "polygon_detection_px": zone.polygon,
                    "polygon_metric": zone.polygon_metric,
                    "metric_centroid": centroid,
                    "zone_id": zone.id,
                })

            cam_data_list.append({
                "cam_idx": len(cam_data_list),
                "camera_id": camera.id,
                "uri": camera.uri,
                "cam_name": cam_name,
                "frame": frame,
                "slots": slots,
            })

        return cam_data_list
    finally:
        db.close()


# ---------------------------------------------------------------------------
# CameraView
# ---------------------------------------------------------------------------
class CameraView:
    ALPHA = 0.30

    def __init__(self, cam_idx: int, cam_data: dict):
        self.cam_idx  = cam_idx
        self.cam_name = cam_data["cam_name"]
        self.uri      = cam_data["uri"]
        self.frame    = (cam_data["frame"].copy()
                         if cam_data["frame"] is not None
                         else np.zeros((480, 640, 3), np.uint8))
        self.slots    = cam_data["slots"]
        self.color    = cam_color(cam_idx)
        self.selected = None
        self.paired   = {}   # slot key → partner key

    def find_slot_at(self, x, y):
        for s in self.slots:
            poly = s.get("polygon_detection_px")
            if not poly:
                continue
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            if cv2.pointPolygonTest(pts, (float(x), float(y)), False) >= 0:
                return s["key"]
        return None

    def draw(self, active=False, pair_count=0, need_pairs=2):
        img = self.frame.copy()
        overlay = img.copy()
        for s in self.slots:
            poly = s.get("polygon_detection_px")
            if not poly:
                continue
            key = s["key"]
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            if key in self.paired:
                fill = (60, 200, 60)
            elif key == self.selected:
                fill = (60, 220, 220)
            else:
                fill = self.color
            cv2.fillPoly(overlay, [pts], fill)
        cv2.addWeighted(overlay, self.ALPHA, img, 1 - self.ALPHA, 0, img)
        for s in self.slots:
            poly = s.get("polygon_detection_px")
            if not poly:
                continue
            key = s["key"]
            pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
            if key in self.paired:
                border = (0, 150, 0)
            elif key == self.selected:
                border = (0, 160, 160)
            else:
                border = tuple(max(0, v - 60) for v in self.color)
            cv2.polylines(img, [pts], True, border, 2)
            cx = int(sum(p[0] for p in poly) / len(poly))
            cy = int(sum(p[1] for p in poly) / len(poly))
            lbl = f"r{s['row']}_c{s['col']}"
            cv2.putText(img, lbl, (cx - 20, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, lbl, (cx - 20, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Camera label in top-left corner
        hud = f"Cam {self.cam_idx}: {self.cam_name}"
        if active:
            color = (0, 220, 220)
            hud += " [ACTIVE]"
        else:
            color = (200, 200, 200)
        cv2.putText(img, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.50, color, 1, cv2.LINE_AA)

        # Active border highlight
        if active:
            cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1),
                          (0, 220, 220), 3)
        return img


# ---------------------------------------------------------------------------
# PairingState
# ---------------------------------------------------------------------------
class PairingState:
    MIN_PAIRS = 2

    def __init__(self, view_a: CameraView, view_b: CameraView):
        self.view_a  = view_a
        self.view_b  = view_b
        self.pending = {view_a.cam_idx: None, view_b.cam_idx: None}
        self.pairs   = []   # [(key_a, key_b)]

    def click(self, cam_idx, x, y):
        view = self.view_a if cam_idx == self.view_a.cam_idx else self.view_b
        key = view.find_slot_at(x, y)
        if key is None:
            return
        self.pending[cam_idx] = key
        view.selected = key
        ka = self.pending[self.view_a.cam_idx]
        kb = self.pending[self.view_b.cam_idx]
        if ka and kb:
            already = any(p[0] == ka or p[1] == kb for p in self.pairs)
            if not already:
                self.pairs.append((ka, kb))
                self.view_a.paired[ka] = kb
                self.view_b.paired[kb] = ka
                print(f"  Pair {len(self.pairs)}: {ka} ↔ {kb}")
            self.pending[self.view_a.cam_idx] = None
            self.pending[self.view_b.cam_idx] = None
            self.view_a.selected = None
            self.view_b.selected = None

    def undo(self):
        if not self.pairs:
            return
        ka, kb = self.pairs.pop()
        self.view_a.paired.pop(ka, None)
        self.view_b.paired.pop(kb, None)
        print(f"  Undo: {ka} ↔ {kb}")

    @property
    def ready(self):
        return len(self.pairs) >= self.MIN_PAIRS


# ---------------------------------------------------------------------------
# Unified merge UI — single window with sidebar + tiled cameras
# ---------------------------------------------------------------------------
class MergeUI:
    """Single-window merge interface with a sidebar panel and tiled camera views."""

    WIN_NAME = "Merge Cameras"

    def __init__(self, cam_views: list[CameraView]):
        self.cam_views = cam_views
        self.n = len(cam_views)

        # Layout: compute grid for tiling cameras
        self.grid_cols = int(np.ceil(np.sqrt(self.n)))
        self.grid_rows = int(np.ceil(self.n / self.grid_cols))

        # Spanning tree state
        self.linked = {0}
        self.unlinked = set(range(1, self.n))
        self.step_results = []

        # Current pairing state
        self.state: PairingState | None = None
        self.active_a: int | None = None
        self.active_b: int | None = None

        # Pair selection via sidebar buttons
        self.sel_a: int | None = None
        self.sel_b: int | None = None

        # UI phase: "select_pair" or "pairing"
        self.phase = "select_pair"
        self.aborted = False
        self.finished = False

        # Message to display in sidebar (feedback)
        self.message = ""
        self.message_color = (200, 200, 200)

        # Camera tile layout (computed in _draw)
        self.tile_w = 0
        self.tile_h = 0
        self.canvas_h = 0
        self.canvas_w = 0

    # ------------------------------------------------------------------
    # Sidebar panel drawing
    # ------------------------------------------------------------------
    def _draw_panel(self, canvas: np.ndarray):
        panel = canvas[:, :PANEL_W]
        panel[:] = PANEL_BG

        y = 28
        cv2.putText(panel, "MERGE CAMERAS", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 220, 220), 2, cv2.LINE_AA)
        y += 10
        cv2.line(panel, (12, y), (PANEL_W - 12, y), (75, 75, 75), 1)
        y += 20

        # ── Phase: select_pair ─────────────────────────────────────────
        if self.phase == "select_pair":
            # Show linked / unlinked cameras
            cv2.putText(panel, "Linked:", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 210, 120), 1, cv2.LINE_AA)
            linked_str = ", ".join(str(i) for i in sorted(self.linked))
            cv2.putText(panel, linked_str, (80, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
            y += 20

            cv2.putText(panel, "Unlinked:", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (215, 90, 90), 1, cv2.LINE_AA)
            if self.unlinked:
                unlinked_str = ", ".join(str(i) for i in sorted(self.unlinked))
            else:
                unlinked_str = "none"
            cv2.putText(panel, unlinked_str, (90, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)
            y += 24

            cv2.line(panel, (12, y), (PANEL_W - 12, y), (75, 75, 75), 1)
            y += 16

            # Camera A selector
            cv2.putText(panel, "Camera A (linked):", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
            y += 22
            self._cam_a_buttons_y = y
            btn_x = 12
            for idx in sorted(self.linked):
                bw = 44
                active = (self.sel_a == idx)
                cv2.rectangle(panel, (btn_x, y - 14), (btn_x + bw, y + 8),
                              (65, 125, 75) if active else (60, 60, 60), -1)
                cv2.rectangle(panel, (btn_x, y - 14), (btn_x + bw, y + 8),
                              (120, 120, 120), 1)
                cv2.putText(panel, str(idx), (btn_x + 14, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (255, 255, 255) if active else (150, 150, 150), 1, cv2.LINE_AA)
                btn_x += bw + 6
            y += 22

            # Camera B selector
            cv2.putText(panel, "Camera B (to link):", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170, 170, 170), 1, cv2.LINE_AA)
            y += 22
            self._cam_b_buttons_y = y
            btn_x = 12
            cams_for_b = sorted(self.unlinked) if self.unlinked else sorted(set(range(self.n)) - self.linked)
            self._cam_b_list = cams_for_b
            for idx in cams_for_b:
                bw = 44
                active = (self.sel_b == idx)
                cv2.rectangle(panel, (btn_x, y - 14), (btn_x + bw, y + 8),
                              (65, 125, 75) if active else (60, 60, 60), -1)
                cv2.rectangle(panel, (btn_x, y - 14), (btn_x + bw, y + 8),
                              (120, 120, 120), 1)
                cv2.putText(panel, str(idx), (btn_x + 14, y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (255, 255, 255) if active else (150, 150, 150), 1, cv2.LINE_AA)
                btn_x += bw + 6
            y += 26

            # Start pairing button
            can_start = (self.sel_a is not None and self.sel_b is not None
                         and self.sel_a != self.sel_b)
            self._start_btn_y = y
            cv2.rectangle(panel, (12, y - 14), (PANEL_W - 12, y + 12),
                          (45, 120, 55) if can_start else (50, 50, 50), -1)
            cv2.rectangle(panel, (12, y - 14), (PANEL_W - 12, y + 12),
                          (100, 200, 110) if can_start else (80, 80, 80), 1)
            cv2.putText(panel, "Start Pairing (S)", (70, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255) if can_start else (120, 120, 120), 1, cv2.LINE_AA)
            y += 30

        # ── Phase: pairing ─────────────────────────────────────────────
        elif self.phase == "pairing":
            cv2.putText(panel, f"Pairing: Cam {self.active_a} <-> Cam {self.active_b}",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 220, 220), 1, cv2.LINE_AA)
            y += 24

            # Pair count
            if self.state:
                cnt = len(self.state.pairs)
                needed = self.state.MIN_PAIRS
                color = (100, 210, 120) if cnt >= needed else (255, 210, 75)
                cv2.putText(panel, f"Pairs: {cnt}/{needed}", (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1, cv2.LINE_AA)
                y += 24

                # List current pairs
                for i, (ka, kb) in enumerate(self.state.pairs):
                    pair_txt = f"  {i+1}. {ka} <-> {kb}"
                    cv2.putText(panel, pair_txt, (12, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                                (180, 220, 180), 1, cv2.LINE_AA)
                    y += 16

            y += 8
            cv2.line(panel, (12, y), (PANEL_W - 12, y), (75, 75, 75), 1)
            y += 16

            # Confirm button
            can_confirm = self.state and self.state.ready
            self._confirm_btn_y = y
            cv2.rectangle(panel, (12, y - 14), (PANEL_W - 12, y + 12),
                          (45, 120, 55) if can_confirm else (50, 50, 50), -1)
            cv2.rectangle(panel, (12, y - 14), (PANEL_W - 12, y + 12),
                          (100, 200, 110) if can_confirm else (80, 80, 80), 1)
            cv2.putText(panel, "Confirm (C)", (100, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255) if can_confirm else (120, 120, 120), 1, cv2.LINE_AA)
            y += 28

            # Undo button
            self._undo_btn_y = y
            cv2.rectangle(panel, (12, y - 14), (PANEL_W - 12, y + 12),
                          (60, 60, 60), -1)
            cv2.rectangle(panel, (12, y - 14), (PANEL_W - 12, y + 12),
                          (100, 100, 100), 1)
            cv2.putText(panel, "Undo Last (U)", (90, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (215, 170, 90), 1, cv2.LINE_AA)
            y += 28

            # Cancel button
            self._cancel_btn_y = y
            cv2.rectangle(panel, (12, y - 14), (PANEL_W - 12, y + 12),
                          (60, 45, 45), -1)
            cv2.rectangle(panel, (12, y - 14), (PANEL_W - 12, y + 12),
                          (120, 80, 80), 1)
            cv2.putText(panel, "Cancel Pairing", (85, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (215, 90, 90), 1, cv2.LINE_AA)
            y += 28

        # ── Steps completed ────────────────────────────────────────────
        y += 8
        cv2.line(panel, (12, y), (PANEL_W - 12, y), (75, 75, 75), 1)
        y += 18
        cv2.putText(panel, "Completed links:", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 170, 170), 1, cv2.LINE_AA)
        y += 18
        if self.step_results:
            for step in self.step_results:
                txt = f"  Cam {step['cam_a']} <-> Cam {step['cam_b']}: {len(step['pairs'])} pairs"
                cv2.putText(panel, txt, (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (180, 220, 180), 1, cv2.LINE_AA)
                y += 16
        else:
            cv2.putText(panel, "  (none yet)", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (120, 120, 120), 1, cv2.LINE_AA)
            y += 16

        # ── Message ────────────────────────────────────────────────────
        if self.message:
            y += 12
            cv2.putText(panel, self.message, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        self.message_color, 1, cv2.LINE_AA)
            y += 16

        # ── Controls ───────────────────────────────────────────────────
        y = self.canvas_h - 100
        cv2.line(panel, (12, y), (PANEL_W - 12, y), (75, 75, 75), 1)
        y += 18
        controls = [
            ("Click slots to pair them", (155, 155, 155)),
            ("S  start pairing",         ( 95, 215,  95)),
            ("C  confirm pairs",         ( 95, 215,  95)),
            ("U  undo last pair",        (215, 170,  90)),
            ("Esc  exit",                (215,  90,  90)),
        ]
        for text, clr in controls:
            cv2.putText(panel, text, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, clr, 1, cv2.LINE_AA)
            y += 16

    # ------------------------------------------------------------------
    # Full canvas draw with tiled cameras
    # ------------------------------------------------------------------
    def _draw(self) -> np.ndarray:
        # Determine tile size from first camera frame
        ref_h, ref_w = self.cam_views[0].frame.shape[:2]

        # Target tile size: fit all cameras in available space
        target_h = max(300, 600 // self.grid_rows)
        scale = target_h / ref_h
        self.tile_w = int(ref_w * scale)
        self.tile_h = int(ref_h * scale)

        right_w = self.grid_cols * self.tile_w
        self.canvas_w = PANEL_W + right_w
        self.canvas_h = max(self.grid_rows * self.tile_h, 500)

        canvas = np.full((self.canvas_h, self.canvas_w, 3), 50, dtype=np.uint8)

        # Draw tiled camera views
        for i, view in enumerate(self.cam_views):
            row = i // self.grid_cols
            col = i % self.grid_cols
            x0 = PANEL_W + col * self.tile_w
            y0 = row * self.tile_h

            is_active = (self.phase == "pairing" and
                         view.cam_idx in (self.active_a, self.active_b))

            cam_img = view.draw(
                active=is_active,
                pair_count=len(self.state.pairs) if self.state else 0,
                need_pairs=PairingState.MIN_PAIRS,
            )
            resized = cv2.resize(cam_img, (self.tile_w, self.tile_h))
            canvas[y0:y0 + self.tile_h, x0:x0 + self.tile_w] = resized

        # Draw sidebar panel
        self._draw_panel(canvas)

        # Vertical separator
        cv2.line(canvas, (PANEL_W, 0), (PANEL_W, self.canvas_h), (80, 80, 80), 1)

        return canvas

    # ------------------------------------------------------------------
    # Mouse callback
    # ------------------------------------------------------------------
    def _mouse_cb(self, event, x, y, flags, _param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # Panel clicks
        if x < PANEL_W:
            self._handle_panel_click(x, y)
            return

        # Camera tile clicks (only during pairing)
        if self.phase == "pairing" and self.state:
            self._handle_tile_click(x, y)

    def _handle_panel_click(self, x, y):
        if self.phase == "select_pair":
            # Camera A buttons
            btn_y = getattr(self, '_cam_a_buttons_y', 0)
            if btn_y - 14 <= y <= btn_y + 8:
                btn_x = 12
                for idx in sorted(self.linked):
                    bw = 44
                    if btn_x <= x <= btn_x + bw:
                        self.sel_a = idx
                        self.message = f"Selected Camera A = {idx}"
                        self.message_color = (200, 200, 200)
                        return
                    btn_x += bw + 6

            # Camera B buttons
            btn_y = getattr(self, '_cam_b_buttons_y', 0)
            if btn_y - 14 <= y <= btn_y + 8:
                btn_x = 12
                for idx in getattr(self, '_cam_b_list', []):
                    bw = 44
                    if btn_x <= x <= btn_x + bw:
                        self.sel_b = idx
                        self.message = f"Selected Camera B = {idx}"
                        self.message_color = (200, 200, 200)
                        return
                    btn_x += bw + 6

            # Start pairing button
            start_y = getattr(self, '_start_btn_y', 0)
            if start_y - 14 <= y <= start_y + 12 and 12 <= x <= PANEL_W - 12:
                self._try_start_pairing()

        elif self.phase == "pairing":
            # Confirm button
            confirm_y = getattr(self, '_confirm_btn_y', 0)
            if confirm_y - 14 <= y <= confirm_y + 12 and 12 <= x <= PANEL_W - 12:
                self._try_confirm()
                return

            # Undo button
            undo_y = getattr(self, '_undo_btn_y', 0)
            if undo_y - 14 <= y <= undo_y + 12 and 12 <= x <= PANEL_W - 12:
                if self.state:
                    self.state.undo()
                return

            # Cancel button
            cancel_y = getattr(self, '_cancel_btn_y', 0)
            if cancel_y - 14 <= y <= cancel_y + 12 and 12 <= x <= PANEL_W - 12:
                self._cancel_pairing()
                return

    def _handle_tile_click(self, x, y):
        """Translate click on the tiled camera area to a slot click."""
        rel_x = x - PANEL_W
        col = rel_x // self.tile_w
        row = y // self.tile_h
        cam_idx_clicked = row * self.grid_cols + col

        if cam_idx_clicked >= self.n:
            return
        if cam_idx_clicked not in (self.active_a, self.active_b):
            return

        # Convert to original frame coordinates
        ref_h, ref_w = self.cam_views[cam_idx_clicked].frame.shape[:2]
        local_x = rel_x - col * self.tile_w
        local_y = y - row * self.tile_h
        frame_x = local_x * ref_w / self.tile_w
        frame_y = local_y * ref_h / self.tile_h

        self.state.click(cam_idx_clicked, frame_x, frame_y)

    # ------------------------------------------------------------------
    # Pairing control
    # ------------------------------------------------------------------
    def _try_start_pairing(self):
        if self.sel_a is None or self.sel_b is None:
            self.message = "Select both Camera A and B first"
            self.message_color = (215, 90, 90)
            return
        if self.sel_a == self.sel_b:
            self.message = "A and B must be different"
            self.message_color = (215, 90, 90)
            return
        if self.sel_a not in self.linked:
            self.message = f"Cam {self.sel_a} not in linked set"
            self.message_color = (215, 90, 90)
            return

        self.active_a = self.sel_a
        self.active_b = self.sel_b

        va = self.cam_views[self.active_a]
        vb = self.cam_views[self.active_b]
        # Reset view state
        for v in self.cam_views:
            v.selected = None
            v.paired = {}

        self.state = PairingState(va, vb)
        self.phase = "pairing"
        self.message = f"Click slots in Cam {self.active_a} and Cam {self.active_b}"
        self.message_color = (200, 200, 200)
        print(f"Linking cam {self.active_a} ({va.cam_name})"
              f" ↔ cam {self.active_b} ({vb.cam_name})")

    def _try_confirm(self):
        if not self.state or not self.state.ready:
            self.message = f"Need >= {PairingState.MIN_PAIRS} pairs"
            self.message_color = (215, 90, 90)
            return

        print(f"  ✓ {len(self.state.pairs)} pairs confirmed")
        self.step_results.append({
            "cam_a": self.active_a,
            "cam_b": self.active_b,
            "pairs": [(ka, kb) for ka, kb in self.state.pairs],
        })
        self.linked.add(self.active_b)
        self.unlinked.discard(self.active_b)

        # Reset for next pair
        self.state = None
        self.active_a = None
        self.active_b = None
        self.sel_a = None
        self.sel_b = None

        if not self.unlinked:
            self.finished = True
            self.message = "All cameras linked!"
            self.message_color = (100, 210, 120)
        else:
            self.phase = "select_pair"
            self.message = "Pair confirmed. Select next pair."
            self.message_color = (100, 210, 120)

    def _cancel_pairing(self):
        self.state = None
        self.active_a = None
        self.active_b = None
        self.phase = "select_pair"
        self.message = "Pairing cancelled"
        self.message_color = (215, 170, 90)
        for v in self.cam_views:
            v.selected = None
            v.paired = {}

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> list:
        cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_NORMAL)
        # Show initial frame to realize window
        cv2.imshow(self.WIN_NAME, np.zeros((400, 600, 3), dtype=np.uint8))
        cv2.waitKey(50)
        cv2.setMouseCallback(self.WIN_NAME, self._mouse_cb)

        while True:
            canvas = self._draw()
            cv2.imshow(self.WIN_NAME, canvas)

            key = cv2.waitKey(16) & 0xFF

            if key == 27:  # Esc
                self.aborted = True
                break

            if self.phase == "select_pair":
                if key in (ord("s"), ord("S")):
                    self._try_start_pairing()
            elif self.phase == "pairing":
                if key in (ord("c"), ord("C")):
                    self._try_confirm()
                elif key in (ord("u"), ord("U")):
                    if self.state:
                        self.state.undo()

            if self.finished:
                # Show final state briefly
                canvas = self._draw()
                cv2.imshow(self.WIN_NAME, canvas)
                cv2.waitKey(500)
                break

            try:
                if cv2.getWindowProperty(self.WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    self.aborted = True
                    break
            except cv2.error:
                self.aborted = True
                break

        cv2.destroyAllWindows()

        if self.aborted:
            print("[INFO] Aborted.")
            sys.exit(0)

        return self.step_results


# ---------------------------------------------------------------------------
# Transform estimation
# ---------------------------------------------------------------------------
def estimate_transforms(step_results: list, cam_data_list: list):
    """Compute affine_M for each step using polygon_metric centroids from DB."""
    # Build centroid lookup: cam_idx → slot_key → centroid (np array)
    centroids = {}
    for cam_data in cam_data_list:
        idx = cam_data["cam_idx"]
        centroids[idx] = {}
        for slot in cam_data["slots"]:
            poly = np.array(slot["polygon_metric"], dtype=float)
            centroids[idx][slot["key"]] = poly.mean(axis=0)

    for step in step_results:
        a_idx = step["cam_a"]
        b_idx = step["cam_b"]
        dst_pts, src_pts = [], []
        for ka, kb in step["pairs"]:
            if ka in centroids.get(a_idx, {}) and kb in centroids.get(b_idx, {}):
                dst_pts.append(centroids[a_idx][ka])
                src_pts.append(centroids[b_idx][kb])

        if len(src_pts) < 2:
            print(f"[ERROR] Not enough valid pairs for step {a_idx}↔{b_idx}")
            sys.exit(1)

        M, inliers = cv2.estimateAffinePartial2D(
            np.float32(src_pts), np.float32(dst_pts),
            method=cv2.RANSAC, ransacReprojThreshold=0.5,
            confidence=0.99, maxIters=2000, refineIters=10,
        )
        if M is None:
            print(f"[ERROR] estimateAffinePartial2D failed for {a_idx}↔{b_idx}")
            sys.exit(1)
        step["affine_M"] = M
        n_in = int(inliers.sum()) if inliers is not None else len(step["pairs"])
        print(f"  Cam {b_idx} → Cam {a_idx}: {n_in}/{len(step['pairs'])} inliers")


def compute_global_transforms(n_cams: int, step_results: list) -> list:
    """BFS from cam 0 — returns list of 2×3 affine matrices (local → global)."""
    edges = {}   # cam_b → (cam_a, M_3x3)
    for step in step_results:
        edges[step["cam_b"]] = (step["cam_a"], to_3x3(step["affine_M"]))

    T3 = {0: np.eye(3)}
    queue = [0]
    while queue:
        node = queue.pop(0)
        for b, (a, M) in edges.items():
            if a == node and b not in T3:
                T3[b] = T3[a] @ M
                queue.append(b)

    return [T3.get(i, np.eye(3))[:2, :] for i in range(n_cams)]


# ---------------------------------------------------------------------------
# Save merged top-down PNG
# ---------------------------------------------------------------------------
def save_merged_topdown(path: Path, cam_data_list: list, global_transforms: list):
    scale = TOPDOWN_SCALE

    per_cam = []
    all_pts = []
    for cam_data, M in zip(cam_data_list, global_transforms):
        cam_slots = []
        for slot in cam_data["slots"]:
            poly_local = np.array(slot["polygon_metric"], dtype=float)
            poly_global = apply_affine_pts(M, poly_local)
            cam_slots.append({
                "key": slot["key"],
                "row": slot["row"], "col": slot["col"],
                "poly_global": poly_global,
            })
            all_pts.extend(poly_global.tolist())
        per_cam.append(cam_slots)

    if not all_pts:
        print("[WARN] No active slots — nothing to draw.")
        return

    pts_arr = np.array(all_pts)
    margin = 2.0
    x_min = pts_arr[:, 0].min() - margin
    y_min = pts_arr[:, 1].min() - margin
    x_max = pts_arr[:, 0].max() + margin
    y_max = pts_arr[:, 1].max() + margin

    canvas_w = int((x_max - x_min) * scale) + 1
    canvas_h = int((y_max - y_min) * scale) + 1
    axis_pad = 60
    title_pad = 40
    legend_h = 26 * len(cam_data_list) + 10

    img_w = canvas_w + axis_pad
    img_h = canvas_h + axis_pad + title_pad + legend_h

    def to_px(gx, gy):
        return (
            axis_pad + int((gx - x_min) * scale),
            title_pad + legend_h + int((gy - y_min) * scale),
        )

    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 255

    # Filled polygons per camera
    for cam_idx, cam_slots in enumerate(per_cam):
        color = cam_color(cam_idx)
        fill = tuple(min(255, c + 70) for c in color)
        layer = img.copy()
        for slot in cam_slots:
            pts_px = np.array([to_px(x, y) for x, y in slot["poly_global"]],
                              dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(layer, [pts_px], fill)
        cv2.addWeighted(layer, 0.45, img, 0.55, 0, img)

    # Borders + labels
    for cam_idx, cam_slots in enumerate(per_cam):
        color = cam_color(cam_idx)
        for slot in cam_slots:
            pts_px = np.array([to_px(x, y) for x, y in slot["poly_global"]],
                              dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts_px], True, color, 2)
            cx = int(pts_px[:, 0, 0].mean())
            cy = int(pts_px[:, 0, 1].mean())
            lbl = f"{slot['row']},{slot['col']}"
            cv2.putText(img, lbl, (cx - 11, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)

    # X axis
    grid_bottom_y = title_pad + legend_h + canvas_h
    for tick in range(int(x_min), int(x_max) + 1):
        tx, _ = to_px(float(tick), y_min)
        cv2.line(img, (tx, grid_bottom_y + 3), (tx, grid_bottom_y + 9), (80, 80, 80), 1)
        cv2.putText(img, str(tick), (tx - 6, grid_bottom_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 80, 80), 1)

    # Y axis
    for tick in range(int(y_min), int(y_max) + 1):
        _, ty = to_px(x_min, float(tick))
        cv2.line(img, (axis_pad - 6, ty), (axis_pad + 2, ty), (80, 80, 80), 1)
        cv2.putText(img, str(tick), (2, ty + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 80, 80), 1)

    # Legend
    for cam_idx, cam_data in enumerate(cam_data_list):
        color = cam_color(cam_idx)
        ly = title_pad + 12 + cam_idx * 26
        cv2.rectangle(img, (axis_pad, ly - 9), (axis_pad + 18, ly + 7), color, -1)
        lbl = f"Cam {cam_idx}: {cam_data['cam_name']}"
        cv2.putText(img, lbl, (axis_pad + 24, ly + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)

    # Title
    cv2.putText(img, f"Merged parking map — {len(cam_data_list)} cameras",
                (axis_pad, title_pad - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.70, (30, 30, 30), 2, cv2.LINE_AA)

    cv2.imwrite(str(path), img)
    print(f"[merge_cameras] Merged top-down saved: {path}")


# ---------------------------------------------------------------------------
# Save individual top-down if missing
# ---------------------------------------------------------------------------
def ensure_topdown(output_dir: Path, cam_data: dict, global_transforms: list):
    """Regenerate {cam_name}_topdown.png if it does not exist."""
    from processing.calibrate_parking import save_topdown, load_config

    cam_name = cam_data["cam_name"]
    topdown_path = output_dir / f"{cam_name}_topdown.png"
    if topdown_path.exists():
        return

    configs_dir = Path(os.getenv("CALIBRATION_CONFIGS_DIR", "calibration/configs"))
    cfg = load_config(configs_dir, cam_name)
    if cfg is None:
        print(f"[WARN] Cannot regenerate top-down for {cam_name}: no config file found.")
        return

    slots_list = [
        {
            "row": int(k.split("_")[0]),
            "col": int(k.split("_")[1]),
            "status": v.get("status", "full"),
        }
        for k, v in cfg.get("slots", {}).items()
    ]
    active = {k: v.get("active", False) for k, v in cfg.get("slots", {}).items()}
    save_topdown(
        topdown_path, cam_name,
        cfg["rows"], cfg["cols"],
        cfg["slot_w"], cfg["slot_h"],
        cfg.get("row_gap", 0.0),
        slots_list, active,
    )
    print(f"[merge_cameras] Regenerated top-down: {topdown_path}")


# ---------------------------------------------------------------------------
# Polygon vertex-order canonicalization
# ---------------------------------------------------------------------------
def _canonical_quad(pts: np.ndarray) -> np.ndarray:
    """
    Sort a 4-vertex polygon into TL → TR → BR → BL order so that
    corresponding corners align across cameras before averaging.

    Coordinate system: x increases right, y increases down (metric space).
      - TL = vertex with smallest  x + y
      - BR = vertex with largest   x + y
      - TR = remaining vertex with larger  x
      - BL = remaining vertex with smaller x
    """
    pts = np.asarray(pts, dtype=float)
    sums = pts[:, 0] + pts[:, 1]
    order = np.argsort(sums)          # [tl_idx, ?, ?, br_idx]
    tl = pts[order[0]]
    br = pts[order[3]]
    mid = pts[order[[1, 2]]]
    if mid[0, 0] <= mid[1, 0]:
        bl, tr = mid[0], mid[1]
    else:
        tr, bl = mid[0], mid[1]
    return np.array([tl, tr, br, bl])


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------
def polygon_iou(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    """Intersection-over-Union between two convex polygons (Nx2 float arrays).

    Uses cv2.intersectConvexConvex whose return value is the intersection area.
    Returns 0.0 if either polygon is degenerate or they do not overlap.
    """
    pa = np.asarray(poly_a, dtype=np.float32).reshape(-1, 1, 2)
    pb = np.asarray(poly_b, dtype=np.float32).reshape(-1, 1, 2)
    inter_area, _ = cv2.intersectConvexConvex(pa, pb)
    if inter_area <= 0.0:
        return 0.0
    area_a = cv2.contourArea(pa)
    area_b = cv2.contourArea(pb)
    union_area = area_a + area_b - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


# ---------------------------------------------------------------------------
# Save merge results to DB
# ---------------------------------------------------------------------------
def save_merge_to_db(step_results: list, cam_data_list: list,
                     global_transforms: list) -> int:
    """Auto-match all slots via IoU in global metric space, then write to DB.

    Algorithm:
      1. Clear existing mapped_zone_id for every zone of the cameras being processed.
      2. Project each slot's polygon_metric into global metric space using global_transforms.
      3. For every pair of cameras (i, j), find mutual best-IoU matches above
         IOU_MATCH_THRESHOLD and feed them into Union-Find.
      4. Every slot (matched or singleton) ends up in a group.
      5. For each group write one MappedZone (mean global polygon) and update
         mapped_zone_id on all member zones.

    step_results is accepted for API compatibility but not used by this function;
    the matching is driven entirely by geometry.
    """
    from db import SessionLocal
    from db.models import MappedZone, Zone

    # ------------------------------------------------------------------
    # Step 1 & 2 — project all slots to global metric space
    # ------------------------------------------------------------------
    # global_slots[cam_idx][key] = {"slot": slot_dict, "poly_global": ndarray}
    global_slots: dict[int, dict[str, dict]] = {}
    all_nodes: list[tuple] = []
    for cam_data in cam_data_list:
        cam_idx = cam_data["cam_idx"]
        M = global_transforms[cam_idx]
        global_slots[cam_idx] = {}
        for slot in cam_data["slots"]:
            poly_local = np.array(slot["polygon_metric"], dtype=float)
            poly_global = apply_affine_pts(M, poly_local)
            global_slots[cam_idx][slot["key"]] = {
                "slot": slot,
                "poly_global": poly_global,
            }
            all_nodes.append((cam_idx, slot["key"]))

    # ------------------------------------------------------------------
    # Step 3 — Union-Find initialisation (every slot is its own group)
    # ------------------------------------------------------------------
    parent: dict = {node: node for node in all_nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression (halving)
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # ------------------------------------------------------------------
    # Step 4 — Mutual best-IoU matching across every camera pair
    # ------------------------------------------------------------------
    cam_indices = [d["cam_idx"] for d in cam_data_list]
    n_pairs_found = 0
    for ii, ci in enumerate(cam_indices):
        for cj in cam_indices[ii + 1:]:
            items_i = list(global_slots[ci].items())   # [(key, entry), ...]
            items_j = list(global_slots[cj].items())
            if not items_i or not items_j:
                continue

            # best match for each slot in cam_i → cam_j
            best_of_i: dict[str, tuple[float, str | None]] = {}
            for ki, vi in items_i:
                best_iou, best_kj = 0.0, None
                for kj, vj in items_j:
                    iou = polygon_iou(vi["poly_global"], vj["poly_global"])
                    if iou > best_iou:
                        best_iou, best_kj = iou, kj
                best_of_i[ki] = (best_iou, best_kj)

            # best match for each slot in cam_j → cam_i
            best_of_j: dict[str, tuple[float, str | None]] = {}
            for kj, vj in items_j:
                best_iou, best_ki = 0.0, None
                for ki, vi in items_i:
                    iou = polygon_iou(vj["poly_global"], vi["poly_global"])
                    if iou > best_iou:
                        best_iou, best_ki = iou, ki
                best_of_j[kj] = (best_iou, best_ki)

            # Accept only mutual best matches above threshold
            for ki, (iou_ij, kj) in best_of_i.items():
                if kj is None or iou_ij < IOU_MATCH_THRESHOLD:
                    continue
                iou_ji, ki_back = best_of_j.get(kj, (0.0, None))
                if ki_back == ki:   # mutual agreement
                    union((ci, ki), (cj, kj))
                    n_pairs_found += 1

    print(f"[merge_cameras] IoU auto-match: {n_pairs_found} cross-camera pair(s) "
          f"(threshold={IOU_MATCH_THRESHOLD})")

    # ------------------------------------------------------------------
    # Step 5 — Collect groups (singletons included)
    # ------------------------------------------------------------------
    groups: dict = {}
    for node in all_nodes:
        groups.setdefault(find(node), []).append(node)
    all_groups = list(groups.values())

    # ------------------------------------------------------------------
    # Step 6 — Write to DB
    # ------------------------------------------------------------------
    db = SessionLocal()
    try:
        # Clear existing mapped_zone_id for every zone of the cameras in play
        camera_ids = [d["camera_id"] for d in cam_data_list]
        for cam_id in camera_ids:
            db.query(Zone).filter(Zone.camera_id == cam_id).update(
                {"mapped_zone_id": None}, synchronize_session=False
            )
        db.flush()

        n_saved = 0
        for group in all_groups:
            polygons = []
            zone_ids = []
            for cam_idx, slot_key in group:
                entry = global_slots.get(cam_idx, {}).get(slot_key)
                if entry is None:
                    continue
                polygons.append(entry["poly_global"])
                zone_ids.append(entry["slot"]["zone_id"])

            if not polygons:
                continue

            # Canonicalize vertex order (TL→TR→BR→BL) before averaging so that
            # vertex[i] of cam-A's polygon corresponds to vertex[i] of cam-B's.
            canonical = [_canonical_quad(p) for p in polygons]
            mean_poly = np.mean(canonical, axis=0).tolist()
            mz = MappedZone(polygon_global_metric=mean_poly)
            db.add(mz)
            db.flush()

            for zid in zone_ids:
                z = db.query(Zone).filter(Zone.id == zid).first()
                if z:
                    z.mapped_zone_id = mz.id

            n_saved += 1

        db.commit()
        n_multi = sum(1 for g in all_groups if len(g) > 1)
        print(f"[merge_cameras] DB: {n_saved} MappedZone(s) written "
              f"({n_multi} multi-camera, {n_saved - n_multi} singleton)")
        return n_saved
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple camera calibrations into a unified parking map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--recalibrate", action="store_true",
                        help="Redo interactive pairing even if merge config exists")
    args = parser.parse_args()

    configs_dir = Path(os.getenv("CALIBRATION_CONFIGS_DIR", "calibration/configs"))
    output_dir = Path(os.getenv("CALIBRATION_OUTPUT_DIR", "calibration/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    topdown_path = output_dir / "merged_topdown.png"

    # ------------------------------------------------------------------
    # Load cameras from DB
    # ------------------------------------------------------------------
    cam_data_list = load_cameras_from_db(configs_dir)
    n = len(cam_data_list)
    print(f"[merge_cameras] Found {n} calibrated camera(s): "
          + ", ".join(d["cam_name"] for d in cam_data_list))

    if n == 1:
        print("[merge_cameras] Only one camera — generating single-cam top-down.")
        M_id = np.array([[1., 0., 0.], [0., 1., 0.]])
        save_merged_topdown(topdown_path, cam_data_list, [M_id])
        return

    # ------------------------------------------------------------------
    # Single-window merge UI with sidebar
    # ------------------------------------------------------------------
    cam_views = [CameraView(d["cam_idx"], d) for d in cam_data_list]
    ui = MergeUI(cam_views)
    step_results = ui.run()

    # ------------------------------------------------------------------
    # Estimate transforms
    # ------------------------------------------------------------------
    print("\n[merge_cameras] Estimating transforms…")
    estimate_transforms(step_results, cam_data_list)
    global_transforms = compute_global_transforms(n, step_results)

    # ------------------------------------------------------------------
    # Ensure per-camera top-downs exist
    # ------------------------------------------------------------------
    for cam_data, M in zip(cam_data_list, global_transforms):
        ensure_topdown(output_dir, cam_data, [M])

    # ------------------------------------------------------------------
    # Save merged top-down PNG
    # ------------------------------------------------------------------
    save_merged_topdown(topdown_path, cam_data_list, global_transforms)

    # ------------------------------------------------------------------
    # Save merge results to DB
    # ------------------------------------------------------------------
    n_mapped = save_merge_to_db(step_results, cam_data_list, global_transforms)
    print(f"[merge_cameras] {n_mapped} physical slot(s) mapped in DB.")

    # Summary
    print(f"\n[merge_cameras] Done.")
    for step in step_results:
        a, b = step["cam_a"], step["cam_b"]
        print(f"  Cam {a} ↔ Cam {b}: {len(step['pairs'])} pairs")
    print(f"  Merged top-down: {topdown_path}")


if __name__ == "__main__":
    main()
