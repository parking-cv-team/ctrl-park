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

CAM_COLORS = [          # BGR
    (220,  80,  80),    # cam 0 — blue
    ( 80, 200,  80),    # cam 1 — green
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
        self.win      = f"Cam {cam_idx}: {cam_data['cam_name']}"
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
        if active:
            status = f"{pair_count}/{need_pairs} pairs"
            hud = f"Cam {self.cam_idx} [{self.cam_name}] ACTIVE — {status} | C: confirm | U: undo | Esc: abort"
            color = (0, 220, 220)
        else:
            hud = f"Cam {self.cam_idx}: {self.cam_name}"
            color = (200, 200, 200)
        cv2.putText(img, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2, cv2.LINE_AA)
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


def _make_cb(state: PairingState, cam_idx: int):
    def cb(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            state.click(cam_idx, x, y)
    return cb


# ---------------------------------------------------------------------------
# Interactive pairing loop
# ---------------------------------------------------------------------------
def run_pairing(cam_views: list) -> list:
    """Spanning-tree pairing: user freely chooses which cam pair to link."""
    n = len(cam_views)
    step_results = []
    linked = {0}
    unlinked = set(range(1, n))

    while unlinked:
        print(f"\n{'='*60}")
        print(f"Cameras in tree:   {sorted(linked)}")
        print(f"Cameras to link:   {sorted(unlinked)}")
        print("Type two camera indices to link (e.g.  0 1 ): ", end="", flush=True)
        while True:
            for v in cam_views:
                cv2.imshow(v.win, v.draw(active=False))
            cv2.waitKey(16)
            try:
                raw = input().strip().split()
                a_idx, b_idx = int(raw[0]), int(raw[1])
            except Exception:
                print("  Invalid input. Try again: ", end="", flush=True)
                continue
            if not (0 <= a_idx < n and 0 <= b_idx < n and a_idx != b_idx):
                print(f"  Indices must be 0..{n-1} and different. Try again: ",
                      end="", flush=True)
                continue
            if a_idx not in linked and b_idx not in linked:
                print("  At least one cam must already be linked. Try again: ",
                      end="", flush=True)
                continue
            if a_idx not in linked:
                a_idx, b_idx = b_idx, a_idx
            break

        va, vb = cam_views[a_idx], cam_views[b_idx]
        for v in cam_views:
            v.selected = None
            v.paired = {}

        state = PairingState(va, vb)
        cv2.waitKey(1)
        cv2.setMouseCallback(va.win, _make_cb(state, a_idx))
        cv2.setMouseCallback(vb.win, _make_cb(state, b_idx))

        print(f"Linking cam {a_idx} ({va.cam_name})"
              f" ↔ cam {b_idx} ({vb.cam_name})")
        print(f"  C: confirm (≥{state.MIN_PAIRS} pairs) | U: undo | Esc: abort")

        while True:
            for v in cam_views:
                is_active = v.cam_idx in (a_idx, b_idx)
                cv2.imshow(v.win, v.draw(
                    active=is_active,
                    pair_count=len(state.pairs),
                    need_pairs=state.MIN_PAIRS,
                ))
            key = cv2.waitKey(16) & 0xFF
            if key in (ord("c"), ord("C")):
                if state.ready:
                    break
                print(f"  [!] Need ≥{state.MIN_PAIRS} pairs (have {len(state.pairs)})")
            elif key in (ord("u"), ord("U")):
                state.undo()
            elif key == 27:
                print("[INFO] Aborted.")
                cv2.destroyAllWindows()
                sys.exit(0)

        print(f"  ✓ {len(state.pairs)} pairs confirmed")
        step_results.append({
            "cam_a": a_idx,
            "cam_b": b_idx,
            "pairs": [(ka, kb) for ka, kb in state.pairs],
        })
        linked.add(b_idx)
        unlinked.discard(b_idx)

    cv2.destroyAllWindows()
    return step_results


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
        cfg.get("row_gap", 0.0), cfg.get("margin", 1.0),
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
    # Open camera windows
    # ------------------------------------------------------------------
    cam_views = []
    for cam_data in cam_data_list:
        view = CameraView(cam_data["cam_idx"], cam_data)
        cv2.namedWindow(view.win, cv2.WINDOW_NORMAL)
        cv2.waitKey(1)
        cam_views.append(view)
    for view in cam_views:
        cv2.imshow(view.win, view.draw())
    cv2.waitKey(1)

    # ------------------------------------------------------------------
    # Interactive pairing
    # ------------------------------------------------------------------
    step_results = run_pairing(cam_views)

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
