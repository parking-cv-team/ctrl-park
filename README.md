# Ctrl+Park

Multi-camera parking occupancy pipeline for angled CCTV footage. Detects, tracks, and classifies vehicles and pedestrians in real time, persists structured data to MySQL, and exposes it through a FastAPI backend and Streamlit dashboard.

Built for **Beantech Spring Hackathon 2026 — Challenge 2: AI Video Analytics for Smart Parking**.

---

## Features

- Real-time vehicle and pedestrian detection using fine-tuned YOLO26s models
- Per-slot occupancy tracking with dwell time, entry/exit counts, and departure events
- Multi-camera support with homography-based cross-camera slot deduplication (`global_id`)
- Trajectory tracking with direction arrows and occupancy heatmaps
- Interactive Streamlit dashboard (Overview, KPIs, Tracking, Reports, 3D Simulation)
- FastAPI REST backend with structured analytics endpoints
- Designed for angled CCTV cameras — the dominant real-world configuration

---

## Architecture

```
Video Sources (MP4 / RTSP)
        │
        ▼  frame queue (3 FPS, backpressure)
┌──────────────────────────────────────────┐
│         Per-camera Worker Thread         │
│  YOLO26s (cars, conf=0.70)               │
│  YOLO26s (peds, conf=0.50)               │
│  ByteTrack → zone assignment             │
│  Cross-camera reconciliation (global_id) │
└───────────────────┬──────────────────────┘
                    │
             MySQL Database
          (detections, zone_occupancy,
           zones, mapped_zones)
                    │
          FastAPI REST API (:8000)
                    │
        Streamlit Dashboard (:8501)
```

| Stage | File | Output |
|-------|------|--------|
| Frame ingest | `processing/camera_ingest.py` | Frames @ 3 FPS → bounded Queue |
| Calibration | `processing/calibrate_parking.py` | Homography + zone polygons → DB |
| Multi-cam merge | `processing/merge_cameras.py` | `MappedZone` records → DB |
| Detection | `processing/detect_frame.py` | YOLO26 bboxes (cars + peds) |
| Tracking | `processing/tracking.py` | Stable `tracker_id` per object (ByteTrack) |
| Zone assignment | `processing/worker.py` | `zone_id` per detection |
| Cross-cam dedup | `processing/worker.py` | `global_id` per physical slot |
| Persistence | `processing/worker.py` | Detection rows + ZoneOccupancy → DB |
| API | `backend/main.py` | JSON analytics endpoints |
| Dashboard | `dashboard/app.py` | Interactive Streamlit UI |

---

## Models

Two separate fine-tuned YOLO26s instances (one per class):

| Weight file | Class | Conf | Training |
|-------------|-------|------|----------|
| `best_retrained.pt` | Car / Van | 0.70 | Atlas Car Dataset v1 (Roboflow), 50 epochs, Colab T4 |
| `yolo26s_pedestriantuned.pt` | Pedestrian | 0.50 | People CV Model (Roboflow), 50 epochs, Colab T4 |

**Baseline vs fine-tuned (car detection, mAP@50):** 0.7% → **94.0%**. The COCO baseline collapses on angled parking footage due to domain shift; fine-tuning is required.

Weights are committed to `processing/models_weights/`. To retrain, run the notebooks in `benchmarking/` on Google Colab.

---

## Installation

**Prerequisites:** Python 3.11+, MySQL 8.0+

```bash
git clone <repo-url>
cd ctrl-park
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### MySQL setup

No manual schema creation needed — `init_db()` in `db/database.py` runs automatically on first API startup.

```sql
CREATE USER 'ctrlpark'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON ctrl_park.* TO 'ctrlpark'@'localhost';
FLUSH PRIVILEGES;
```

### Environment

Create a `.env` file in the project root (see `.env.example`):

```dotenv
DB_USER=root
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=ctrl_park
API_BASE_URL=http://localhost:8000
CAMERA_URI=path/to/your/video.mp4
# Multi-camera: CAMERA_URIS=cam1.mp4,cam2.mp4
TARGET_FPS=3
OCCUPANT_ABSENCE_THRESHOLD_SECONDS=120
```

---

## Usage

### Start the backend

```bash
python -m uvicorn backend.main:app --reload
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### Start the dashboard

```bash
python -m streamlit run dashboard/app.py
```

### Run the pipeline

```bash
# Single camera
python -m processing.run --video path/to/video.mp4

# Multiple cameras
python -m processing.run --video cam1.mp4 cam2.mp4

# With debug window (live annotated feed)
python -m processing.run --video cam1.mp4 --debug

# From .env (CAMERA_URI / CAMERA_URIS)
python -m processing.run
```

### Calibrate a camera

```bash
python -m processing.calibrate_parking --uri path/to/video.mp4
python -m processing.calibrate_parking --uri path/to/video.mp4 --recalibrate
```

### Merge multiple cameras

```bash
python -m processing.merge_cameras
```

Or use the dashboard: **Add Camera Source** → calibrate → **Merge** overlapping slots.

### Simulate a live RTSP stream

```bash
docker run --rm -it -v $PWD/rtsp-simple-server.yml:/rtsp-simple-server.yml \
  -p 8554:8554 aler9/rtsp-simple-server:v1.3.0

ffmpeg -re -stream_loop -1 -i path/to/video.mp4 \
  -f rtsp -rtsp_transport tcp rtsp://localhost:8554/live.stream
```

Then set `CAMERA_URI=rtsp://localhost:8554/live.stream`.

---

## Dashboard

| Tab | Contents |
|-----|----------|
| **Overview** | Vehicle count, per-zone occupancy (free/occupied), live video with zone overlay |
| **KPIs** | Tracked vehicles/pedestrians, avg confidence, dwell time, occupancy stats |
| **Tracking** | Trajectory scatterplot (parked/moving cars, pedestrians), occupancy heatmaps |
| **Reports** | Time-range KPI export, time-series charts |
| **3D Simulation** | WebGL top-down parking map with live occupancy colouring |

---

## Known Limitations

- **Van detection:** mAP@50 = 0.448 (vs 0.940 for cars) — vans underrepresented in training data
- **Departure latency:** Slot marked free only after 120 s of tracker absence (configurable)
- **3 FPS processing:** Fast-moving vehicles may traverse a zone between frames; sufficient for parked-car detection
- **Manual calibration:** First-time cross-camera merge requires an operator to pair corresponding slots in the OpenCV UI
- **No automatic data retention policy** — implement a deletion scheduler before production use

---

## Datasets

| Purpose | Dataset | License |
|---------|---------|---------|
| Car detector fine-tuning | Atlas Car Dataset v1 (Roboflow) | Roboflow Public |
| Pedestrian fine-tuning | People Computer Vision Model (Roboflow) | Roboflow Public |
| Pipeline demo | CHAD Dataset (TeCSAR-UNCC) | Research use |

---

## License

MIT License — see [LICENSE](./LICENSE).

Copyright © 2026 parking-cv-team

---

## Team

| Name | Role |
|------|------|
| **Alessio Flego** | Single-camera and cross-camera tracking logic |
| **Dino Meng** | Model training and fine-tuning, dashboard, testing |
| **Joel Bosio** | System architecture, database schema, FastAPI backend, 3D viewer |
| **Lorenzo Raffin** | Dashboard development and integration |
| **Sabrina Ciccolo** | Homographic mapping, calibration UI, multi-camera merge algorithm |
