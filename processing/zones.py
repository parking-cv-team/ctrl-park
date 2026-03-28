import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from db import SessionLocal, CameraSource, Zone


def _load_zone_config_from_db(source: str) -> CameraSource:
    db = SessionLocal()
    try:
        camera = db.query(CameraSource).filter(CameraSource.uri == source).first()
        if not camera:
            raise FileNotFoundError(f"Zone config not found in DB for source: {source}")

        zones: list[Zone] = []
        for z in camera.zones:
            polygon = np.array(z.polygon, dtype=np.int32)
            zones.append(Zone(name=z.name, polygon=polygon))

        return CameraSource(
            name=camera.name,
            uri=camera.uri,
            frame_width=camera.frame_width or 0,
            frame_height=camera.frame_height or 0,
            zones=zones,
        )
    finally:
        db.close()


def _save_zone_config_to_db(config: CameraSource) -> None:
    db = SessionLocal()
    try:
        camera = db.query(CameraSource).filter(CameraSource.uri == config.uri).first()
        if not camera:
            camera = CameraSource(name=config.name, uri=config.uri)
            db.add(camera)
            db.flush()

        camera.frame_width = config.frame_width
        camera.frame_height = config.frame_height

        # replace existing camera zones with updated list
        db.query(Zone).filter(Zone.camera_id == camera.id).delete(
            synchronize_session=False
        )

        for z in config.zones:
            db.add(
                Zone(
                    name=z.name,
                    polygon=z.polygon.tolist(),
                    camera_id=camera.id,
                )
            )

        db.commit()
    finally:
        db.close()
