"""Real-time soccer-ball detection from a webcam using Ultralytics YOLO26.

The COCO ``sports ball`` class (id 32) is used as the proxy for "soccer ball"
since the pretrained ``yolo26n.pt`` checkpoint was not trained on a
soccer-specific class. See README for the caveat and fine-tuning pointer.

Run:
    python detect.py
    python detect.py --source 1 --conf 0.4
    python detect.py --source path/to/clip.mp4
    python detect.py --source rtsp://example/stream
"""

from __future__ import annotations

import argparse
import logging
import sys

import cv2

from soccer_ball.detector import (
    COCO_SPORTS_BALL_ID,
    annotate_frame,
    filter_sports_ball,
)

log = logging.getLogger("soccer_ball")


def _parse_source(raw: str) -> int | str:
    """Convert numeric strings to int (camera index); leave URLs/paths as str."""
    try:
        return int(raw)
    except ValueError:
        return raw


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--source",
        default="0",
        help='Camera index (e.g. "0"), video file path, or RTSP/HTTP URL.',
    )
    p.add_argument(
        "--model",
        default="yolo26n.pt",
        help="Ultralytics model checkpoint (downloaded on first run).",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for keeping detections.",
    )
    p.add_argument(
        "--ball-class",
        type=int,
        default=COCO_SPORTS_BALL_ID,
        help="COCO class id treated as the ball. Default: 32 (sports ball).",
    )
    p.add_argument(
        "--no-display",
        action="store_true",
        help="Process frames without opening a window. Useful for headless smoke tests.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)

    # Imported lazily so `python detect.py --help` works without ultralytics installed.
    from ultralytics import YOLO

    log.info("Loading model %s", args.model)
    model = YOLO(args.model)

    source = _parse_source(args.source)
    log.info("Opening source %r", source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error("Failed to open source %r", source)
        return 1

    window = "Soccer Ball Detector (q to quit)"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.info("End of stream.")
                break

            results = model.predict(frame, verbose=False)
            boxes = results[0].boxes
            if boxes is None or len(boxes) == 0:
                annotated = frame
            else:
                cls = boxes.cls.cpu().numpy().astype(int)
                conf = boxes.conf.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy()
                kept_xyxy, kept_conf = filter_sports_ball(
                    cls, conf, xyxy,
                    ball_class_id=args.ball_class,
                    conf_threshold=args.conf,
                )
                annotated = annotate_frame(frame, kept_xyxy, kept_conf)

            if args.no_display:
                continue
            cv2.imshow(window, annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
