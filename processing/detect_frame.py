import supervision as sv 
import cv2 
from ultralytics import YOLO
from typing import Tuple, Dict

def load_models(
        mcar_path: str = r'processing/models_weights/best_retrained.pt', # path to the car detection model
        mped_path: str = r'processing/models_weights/yolo26s_pedestriantuned.pt', # path to the pedestrian detection model
) -> Tuple[YOLO, YOLO]:
    # Load models to optimize time and memory efficiency
    return (YOLO(mcar_path), YOLO(mped_path))


def detect_frame_dual(
        frame,
        model_car: YOLO,
        model_ped: YOLO, # models loaded with load_models
        arg_car: dict = {'conf': 0.7}, # arguments to the car detector
        arg_ped: dict = {'conf': 0.5}, # arguments to the pedestrian detector
    ) -> Dict[str, sv.Detections]:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detections_car = model_car.predict(frame_rgb, **arg_car)[0]
    detections_pedestrians = model_ped.predict(frame_rgb, **arg_ped)[0]

    return {
        "cars": sv.Detections.from_ultralytics(detections_car),
        "pedestrian": sv.Detections.from_ultralytics(detections_pedestrians),
    }

def load_baseline_model(
        model_path: str = "processing/models_weights/yolo26s.pt"
)-> YOLO:
        return YOLO(model_path)


def detect_frame_baseline(
    frame,
    model: YOLO,
    args: dict = {'conf': 0.4}, # arguments to the pedestrian detector
) -> Dict[str, sv.Detections]:
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    detections_car = model.predict(frame_rgb, classes=[2, 6],**args)[0]
    detections_pedestrians = model.predict(frame_rgb, classes=0, **args)[0]

    return  {
        "cars": sv.Detections.from_ultralytics(detections_car),
        "pedestrian": sv.Detections.from_ultralytics(detections_pedestrians),
    }
