import argparse
import cv2
import supervision as sv
from detect_frame import detect_frame_dual as detect

# RUN THIS BY WRITING THE FOLLOWING ON COMMAND PROMPT:
# python SCRIPT_detect_video.py --video <insert_video_path>

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

    writer = None
    if args.output:
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)) # type: ignore

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cars, peds = detect(frame)

        annotated = box_annotator.annotate(frame.copy(), detections=cars)
        annotated = box_annotator.annotate(annotated, detections=peds)

        cv2.putText(annotated, f"Cars: {len(cars)}  Pedestrians: {len(peds)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
