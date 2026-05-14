"""Real-time soccer-ball detection using Ultralytics YOLO26.

Defaults are tuned for an Apple Silicon (M-series) MacBook:
- ``yolo26l.pt`` (large) for higher accuracy
- ``--device auto`` resolves to MPS on macOS, CUDA elsewhere, CPU as fallback
- Threaded frame grabber + MJPG fourcc to keep the camera off its low-FPS default
- Interactive camera picker if ``--source`` is omitted

The COCO ``sports ball`` class (id 32) is the proxy for "soccer ball" since
the pretrained YOLO26 checkpoints are trained on COCO. See README.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2

from soccer_ball.cameras import pick_camera_interactive, probe_cameras
from soccer_ball.capture import FpsMeter, ThreadedGrabber
from soccer_ball.detector import (
    COCO_SPORTS_BALL_ID,
    annotate_frame,
    filter_sports_ball,
    overlay_fps,
)
from soccer_ball.devices import auto_device

log = logging.getLogger("soccer_ball")


def _parse_source(raw: str) -> int | str:
    try:
        return int(raw)
    except ValueError:
        return raw


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--source",
        default=None,
        help='Camera index, video file, or stream URL. Omit to launch the camera picker.',
    )
    p.add_argument("--model", default="yolo26l.pt", help="Ultralytics checkpoint.")
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "mps", "cuda", "cpu"],
        help="Inference device. 'auto' picks MPS > CUDA > CPU.",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    p.add_argument("--ball-class", type=int, default=COCO_SPORTS_BALL_ID, help="COCO class id treated as the ball.")
    p.add_argument("--width", type=int, default=None, help="Optional camera width hint.")
    p.add_argument("--height", type=int, default=None, help="Optional camera height hint.")
    p.add_argument("--probe-max", type=int, default=5, help="Max camera index to probe in the picker.")
    p.add_argument("--list-cameras", action="store_true", help="Probe and print available cameras, then exit.")
    p.add_argument("--no-display", action="store_true", help="Skip cv2.imshow (headless smoke test).")
    p.add_argument("--no-fps", action="store_true", help="Disable the FPS overlay.")
    return p.parse_args(argv)


def _open_capture(source: int | str, width: int | None, height: int | None) -> cv2.VideoCapture:
    backend = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    cap = cv2.VideoCapture(source, backend) if isinstance(source, int) else cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def _resolve_source(args: argparse.Namespace) -> int | str:
    if args.source is not None:
        return _parse_source(args.source)

    cams = probe_cameras(max_index=args.probe_max)
    if not cams:
        raise SystemExit("No cameras found. Pass --source explicitly.")
    if not sys.stdin.isatty():
        log.info("stdin is not a TTY; defaulting to first available camera.")
        return cams[0].index
    return pick_camera_interactive(cams)


def _run_loop(args: argparse.Namespace, source: int | str, device: str) -> int:
    from ultralytics import YOLO

    log.info("Loading model %s on device=%s", args.model, device)
    model = YOLO(args.model)

    cap = _open_capture(source, args.width, args.height)
    if not cap.isOpened():
        log.error("Failed to open source %r", source)
        return 1

    fps = FpsMeter()
    window = "Soccer Ball Detector (q to quit)"

    try:
        with ThreadedGrabber(cap) as grabber:
            # Wait briefly for the first frame so we don't burn CPU before the camera warms up.
            deadline = time.monotonic() + 5.0
            while grabber.read() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            if grabber.read() is None:
                log.error("Camera did not produce a frame within 5s.")
                return 1

            while True:
                frame = grabber.read()
                if frame is None:
                    continue

                results = model.predict(frame, verbose=False, device=device)
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

                fps.tick()
                if not args.no_fps:
                    annotated = overlay_fps(annotated, fps.value)

                if args.no_display:
                    continue
                cv2.imshow(window, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        cv2.destroyAllWindows()

    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)

    if args.list_cameras:
        from soccer_ball.cameras import format_camera_menu
        cams = probe_cameras(max_index=args.probe_max)
        print(format_camera_menu(cams) if cams else "No cameras found.")
        return 0

    device = auto_device() if args.device == "auto" else args.device
    source = _resolve_source(args)
    return _run_loop(args, source, device)


if __name__ == "__main__":
    sys.exit(main())
