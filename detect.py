"""Real-time soccer-ball detection using Ultralytics YOLO26.

Defaults are tuned for an Apple Silicon (M-series) MacBook:
- ``yolo26l.pt`` (large) for higher accuracy
- ``--device auto`` resolves to MPS on macOS, CUDA elsewhere, CPU as fallback
- Threaded frame grabber + MJPG fourcc to keep the camera off its low-FPS default
- Interactive TUI for launch config + cv2 trackbar panel for live tuning
- JSON presets persist your favourite settings between runs

Precedence (low → high): builtin defaults → preset → CLI flags → TUI → trackbars.

The COCO ``sports ball`` class (id 32) is the proxy for "soccer ball".
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace

import cv2

from soccer_ball.cameras import (
    format_camera_menu,
    pick_camera_interactive,
    probe_cameras,
)
from soccer_ball.capture import FpsMeter, ThreadedGrabber
from soccer_ball.detector import (
    annotate_frame,
    filter_sports_ball,
    overlay_fps,
    overlay_settings,
)
from soccer_ball.devices import auto_device, resolve_half
from soccer_ball.settings import (
    DEFAULT_PRESETS_DIR,
    LaunchConfig,
    RuntimeSettings,
    load_preset,
    merge,
    save_preset,
)
from soccer_ball.trackbars import TrackbarPanel
from soccer_ball.tui import prompt_launch_config

log = logging.getLogger("soccer_ball")

_LAUNCH_FIELDS = {"source", "model", "device", "width", "height", "half", "preset"}
_RUNTIME_FIELDS = {
    "conf", "iou", "max_det", "imgsz", "ball_class",
    "agnostic_nms", "show_fps", "show_label",
}


def _parse_source(raw: str) -> int | str:
    try:
        return int(raw)
    except ValueError:
        return raw


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    # Launch config — defaults shown in --help text, actual default=None so we
    # can detect "user did not pass this".
    p.add_argument("--source", default=None, help="Camera index, video, or URL. Default: camera picker.")
    p.add_argument("--model", default=None, help="Ultralytics checkpoint. Default: yolo26l.pt.")
    p.add_argument("--device", default=None, choices=["auto", "mps", "cuda", "cpu"],
                   help="Inference device. Default: auto (mps > cuda > cpu).")
    p.add_argument("--width", type=int, default=None, help="Camera width hint.")
    p.add_argument("--height", type=int, default=None, help="Camera height hint.")
    p.add_argument("--half", action=argparse.BooleanOptionalAction, default=None,
                   help="Use FP16 half-precision. Default: True on mps/cuda, forced False on cpu. Pass --no-half to disable.")
    p.add_argument("--preset", default="default", help="Preset name to load and save. Default: default.")

    # Runtime settings
    p.add_argument("--conf", type=float, default=None, help="Confidence threshold. Default: 0.25.")
    p.add_argument("--iou", type=float, default=None, help="NMS IoU threshold. Default: 0.7.")
    p.add_argument("--max-det", type=int, default=None, dest="max_det", help="Max detections. Default: 300.")
    p.add_argument("--imgsz", type=int, default=None, help="Inference image size (multiple of 32). Default: 640.")
    p.add_argument("--ball-class", type=int, default=None, dest="ball_class",
                   help="COCO class id treated as the ball. Default: 32 (sports ball).")
    p.add_argument("--agnostic-nms", action="store_true", default=None, dest="agnostic_nms",
                   help="Use class-agnostic NMS.")

    # Workflow toggles
    p.add_argument("--no-tui", action="store_true", help="Skip the pre-loop launch TUI.")
    p.add_argument("--no-trackbars", action="store_true", help="Skip the live trackbar panel.")
    p.add_argument("--no-display", action="store_true", help="Skip cv2.imshow (headless).")
    p.add_argument("--no-fps", action="store_true", help="Disable the FPS overlay at startup.")
    p.add_argument("--list-cameras", action="store_true", help="Probe cameras, print, exit.")
    p.add_argument("--probe-max", type=int, default=3, help="Max camera index to probe. Default 3 keeps macOS quiet; raise to 5+ if you have more cameras.")

    return p.parse_args(argv)


def _build_launch_config(args: argparse.Namespace) -> LaunchConfig:
    cfg = LaunchConfig()
    overrides = {f: getattr(args, f) for f in _LAUNCH_FIELDS if getattr(args, f, None) is not None}
    return replace(cfg, **overrides) if overrides else cfg


def _build_runtime_settings(args: argparse.Namespace, preset_name: str) -> RuntimeSettings:
    settings = RuntimeSettings()
    try:
        settings = load_preset(preset_name)
        log.info("Loaded preset %r", preset_name)
    except FileNotFoundError:
        log.info("No preset %r found; using built-in defaults.", preset_name)

    cli_overrides = {f: getattr(args, f) for f in _RUNTIME_FIELDS if getattr(args, f, None) is not None}
    if args.no_fps:
        cli_overrides["show_fps"] = False
    return merge(settings, cli_overrides)


def _resolve_source(launch_cfg: LaunchConfig, probe_max: int) -> int | str:
    if launch_cfg.source is not None:
        return _parse_source(launch_cfg.source)
    cams = probe_cameras(max_index=probe_max)
    if not cams:
        raise SystemExit("No cameras found. Pass --source explicitly.")
    if not sys.stdin.isatty():
        log.info("stdin is not a TTY; defaulting to first available camera.")
        return cams[0].index
    return pick_camera_interactive(cams)


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


def _settings_lines(launch_cfg: LaunchConfig, live: RuntimeSettings, device: str, model) -> list[str]:
    class_name = "?"
    if hasattr(model, "names"):
        class_name = model.names.get(live.ball_class, "?") if isinstance(model.names, dict) else "?"
    return [
        f"model:    {launch_cfg.model}",
        f"device:   {device}   half:{'y' if launch_cfg.half else 'n'}",
        f"conf:     {live.conf:.2f}   iou:{live.iou:.2f}",
        f"imgsz:    {live.imgsz}   max_det:{live.max_det}",
        f"agnostic: {'y' if live.agnostic_nms else 'n'}",
        f"class:    {live.ball_class} {class_name}",
    ]


def _try_create_panel(initial: RuntimeSettings, enabled: bool) -> TrackbarPanel | None:
    if not enabled:
        return None
    try:
        return TrackbarPanel().create(initial)
    except cv2.error as exc:
        log.warning("Trackbar panel unavailable (%s); continuing without live tuning.", exc)
        return None


def _run_loop(
    launch_cfg: LaunchConfig,
    settings: RuntimeSettings,
    args: argparse.Namespace,
    device: str,
    source: int | str,
) -> int:
    from ultralytics import YOLO

    log.info("Loading model %s on device=%s (half=%s)", launch_cfg.model, device, launch_cfg.half)
    model = YOLO(launch_cfg.model)

    cap = _open_capture(source, launch_cfg.width, launch_cfg.height)
    if not cap.isOpened():
        log.error("Failed to open source %r", source)
        return 1

    fps = FpsMeter()
    main_window = "Soccer Ball Detector (q quit, s save preset)"
    panel = _try_create_panel(settings, enabled=not args.no_trackbars and not args.no_display)
    last_ball_class: int | None = None

    try:
        with ThreadedGrabber(cap) as grabber:
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

                live = panel.current() if panel is not None else settings
                if live.ball_class != last_ball_class:
                    name = model.names.get(live.ball_class, "?") if hasattr(model, "names") else "?"
                    log.info("ball_class -> %d (%s)", live.ball_class, name)
                    last_ball_class = live.ball_class

                results = model.predict(
                    frame,
                    verbose=False,
                    device=device,
                    conf=live.conf,
                    iou=live.iou,
                    max_det=live.max_det,
                    imgsz=live.imgsz,
                    half=launch_cfg.half,
                    agnostic_nms=live.agnostic_nms,
                )
                boxes = results[0].boxes
                if boxes is None or len(boxes) == 0:
                    annotated = frame
                else:
                    cls = boxes.cls.cpu().numpy().astype(int)
                    conf = boxes.conf.cpu().numpy()
                    xyxy = boxes.xyxy.cpu().numpy()
                    kept_xyxy, kept_conf = filter_sports_ball(
                        cls, conf, xyxy,
                        ball_class_id=live.ball_class,
                        conf_threshold=live.conf,
                    )
                    annotated = annotate_frame(frame, kept_xyxy, kept_conf, show_label=live.show_label)

                fps.tick()
                if live.show_fps:
                    annotated = overlay_fps(annotated, fps.value)
                if live.show_settings:
                    annotated = overlay_settings(annotated, _settings_lines(launch_cfg, live, device, model))

                if args.no_display:
                    continue
                cv2.imshow(main_window, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    path = save_preset(launch_cfg.preset, live, presets_dir=DEFAULT_PRESETS_DIR)
                    log.info("Saved preset to %s", path)
    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    finally:
        if panel is not None:
            panel.close()
        cv2.destroyAllWindows()

    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args(argv)

    if args.list_cameras:
        cams = probe_cameras(max_index=args.probe_max)
        print(format_camera_menu(cams) if cams else "No cameras found.")
        return 0

    launch_cfg = _build_launch_config(args)
    runtime = _build_runtime_settings(args, launch_cfg.preset)

    if not args.no_tui and sys.stdin.isatty():
        launch_cfg = prompt_launch_config(launch_cfg)
        # Preset name may have changed in the TUI — reload runtime if so.
        if launch_cfg.preset != args.preset:
            runtime = _build_runtime_settings(args, launch_cfg.preset)

    device = auto_device() if launch_cfg.device == "auto" else launch_cfg.device
    resolved_half = resolve_half(launch_cfg.half, device)
    if resolved_half != launch_cfg.half:
        log.info("Forcing half=False on device=%s (FP16 is unsupported / slower on CPU).", device)
        launch_cfg = replace(launch_cfg, half=resolved_half)

    source = _resolve_source(launch_cfg, args.probe_max)
    return _run_loop(launch_cfg, runtime, args, device, source)


if __name__ == "__main__":
    sys.exit(main())
