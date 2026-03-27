import supervision as sv 
import cv2 
from ultralytics import YOLO
from rfdetr import RFDETRMedium
from typing import Tuple

def detect_frame_dual(
        frame,
        mcar_path: str = r'processing\models_weights\best.pt', # path to the car detection model
        mped_path: str = r'processing\models_weights\yolov5su.pt', # path to the pedestrian detection model
        arg_car: dict = {'conf': 0.75}, # arguments to the car detector
        arg_ped: dict = {'conf': 0.4}, # arguments to the pedestrian detector
    ) -> Tuple[sv.Detections, sv.Detections]:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    model_car = YOLO(mcar_path)
    model_pedestrian = YOLO(mped_path)

    detections_car = model_car.predict(frame_rgb, **arg_car)[0]
    detections_pedestrians = model_pedestrian.predict(frame_rgb, classes=0, **arg_ped)[0]
    
    return (sv.Detections.from_ultralytics(detections_car), sv.Detections.from_ultralytics(detections_pedestrians))