import argparse
import cv2
import supervision as sv

BASELINE_ = False

if BASELINE_:
    from .detect_frame import detect_frame_baseline as detect
    from .detect_frame import load_baseline_model as load_model
else:
    from .detect_frame import detect_frame_dual as detect
    from .detect_frame import load_models as load_models



# RUN THIS BY WRITING THE FOLLOWING ON COMMAND PROMPT:
# python -m processing.SCRIPT_detect_video --video <insert_video_path>

def main():
    parser = argparse.ArgumentParser(description="Run detection on a video file.")
    parser.add_argument("--video", help="Path to the input video file")
    parser.add_argument("--output", "-o", help="Path to save annotated output video (optional)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: cannot open video '{args.video}'")
        return

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))  # type: ignore

    frame_idx = 0

    if BASELINE_:
        m1 = load_model()
    else:
        m1, m2 = load_models()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if BASELINE_:
            raw = detect(frame, m1)
        else:
            raw = detect(frame, m1, m2)

        cars = raw["cars"]
        peds = raw["pedestrian"]

        car_labels = [f"car {conf:.2f}" for conf in cars.confidence]
        ped_labels = [f"pedestrian {conf:.2f}" for conf in peds.confidence]

        annotated = box_annotator.annotate(frame.copy(), detections=cars)
        annotated = box_annotator.annotate(annotated, detections=peds)

        annotated = label_annotator.annotate(annotated, detections=cars, labels=car_labels)
        annotated = label_annotator.annotate(annotated, detections=peds, labels=ped_labels)

        cv2.putText(
            annotated,
            f"Cars: {len(cars)}  Pedestrians: {len(peds)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if writer:
            writer.write(annotated)

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    if writer:
        writer.release()
        print(f"Saved annotated video to '{args.output}'")

if __name__ == "__main__":
    main()