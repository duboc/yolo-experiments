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
import numpy as np

from soccer_ball.audio import SoundPlayer
from soccer_ball.cameras import (
    format_camera_menu,
    pick_camera_interactive,
    probe_cameras,
)
from soccer_ball.capture import (
    CAPTURE_PRESETS,
    CaptureConfig,
    FpsMeter,
    ThreadedGrabber,
    apply_capture_config,
    merge_capture_config,
)
from soccer_ball.detector import (
    annotate_frame,
    filter_by_area,
    filter_sports_ball,
    overlay_debug,
    overlay_fps,
    overlay_kickup,
    overlay_pose,
    overlay_settings,
    overlay_trail,
)
from soccer_ball.kickup import KickupCounter, MotionTrail
from soccer_ball.devices import auto_device, resolve_half
from soccer_ball.pose import (
    BodyPart,
    decide_bounce_credit,
    extract_body_keypoints,
)
from soccer_ball.tracking import SingleBallTracker
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
    "agnostic_nms", "show_fps", "show_label", "proximity_px",
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
    p.add_argument("--fps", type=int, default=None, help="Target capture FPS hint.")
    p.add_argument("--capture-preset", choices=list(CAPTURE_PRESETS), default=None,
                   dest="capture_preset",
                   help="Bundle of width/height/fps. low=640x480@60, balanced=1280x720@60, high=1920x1080@30.")
    p.add_argument("--exposure", type=float, default=None,
                   help="Manual exposure value (camera-specific scale; e.g. -7 on most macOS USB cams).")
    p.add_argument("--no-auto-exposure", action="store_true", dest="no_auto_exposure",
                   help="Disable camera auto-exposure (best-effort).")
    p.add_argument("--focus", type=float, default=None,
                   help="Manual focus distance (camera-specific scale).")
    p.add_argument("--no-auto-focus", action="store_true", dest="no_auto_focus",
                   help="Disable camera auto-focus (best-effort).")
    p.add_argument("--wb-temp", type=float, default=None, dest="wb_temp",
                   help="Manual white-balance temperature in Kelvin.")
    p.add_argument("--no-auto-wb", action="store_true", dest="no_auto_wb",
                   help="Disable camera auto white-balance (best-effort).")
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
    p.add_argument("--proximity-px", type=int, default=None, dest="proximity_px",
                   help="Foot/knee/head proximity radius for kickup gating. Default: 80.")

    # Pose model (Stage 2)
    p.add_argument("--pose-model", default="yolo26n-pose.pt", dest="pose_model",
                   help="Pose checkpoint for body-part gating. Default: yolo26n-pose.pt.")
    p.add_argument("--no-pose", action="store_true",
                   help="Disable pose-based kickup gating (Stage 1 only).")

    # Audio
    p.add_argument("--sound-path", default=None, dest="sound_path",
                   help="Path to a sound played on each counted kickup. macOS default: /System/Library/Sounds/Pop.aiff.")
    p.add_argument("--no-sound", action="store_true", dest="no_sound",
                   help="Disable kickup sound effects.")

    # Kickup tuning
    p.add_argument("--acceleration-threshold", type=float, default=0.5,
                   dest="acceleration_threshold",
                   help="Min |Δsmoothed-velocity| to count a bounce. Lower = more permissive. Default: 0.5.")

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


def _build_capture_config(args: argparse.Namespace, launch_cfg: LaunchConfig) -> CaptureConfig:
    base = CAPTURE_PRESETS.get(args.capture_preset, CaptureConfig()) if args.capture_preset else CaptureConfig()
    # width/height come through LaunchConfig (CLI + TUI both write there);
    # other capture-only fields are CLI-only so we read them from args.
    overrides = CaptureConfig(
        width=launch_cfg.width,
        height=launch_cfg.height,
        fps=args.fps,
        exposure=args.exposure,
        focus=args.focus,
        wb_temp=args.wb_temp,
        auto_exposure=False if args.no_auto_exposure else None,
        auto_focus=False if args.no_auto_focus else None,
        auto_wb=False if args.no_auto_wb else None,
    )
    return merge_capture_config(base, overrides)


def _open_capture(source: int | str, capture_config: CaptureConfig) -> cv2.VideoCapture:
    backend = getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    cap = cv2.VideoCapture(source, backend) if isinstance(source, int) else cv2.VideoCapture(source)
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        results = apply_capture_config(cap, capture_config)
        for prop_name, success in results.items():
            level = log.info if success else log.warning
            level("capture[%s] applied: %s", prop_name, success)
    return cap


def _settings_lines(
    launch_cfg: LaunchConfig,
    live: RuntimeSettings,
    device: str,
    model,
    counter: KickupCounter,
    pose_model_name: str | None,
    pose_keypoints_seen: int,
) -> list[str]:
    class_name = "?"
    if hasattr(model, "names"):
        class_name = model.names.get(live.ball_class, "?") if isinstance(model.names, dict) else "?"
    parts = counter.counts_by_part
    rej = counter.rejected_count
    breakdown = (
        f"foot:{parts[BodyPart.FOOT]} knee:{parts[BodyPart.KNEE]} head:{parts[BodyPart.HEAD]}  rej:{rej}"
        if pose_model_name is not None
        else f"pose off  rej:{rej}"
    )
    pose_line = (
        f"pose:     {pose_model_name}  ({pose_keypoints_seen} kpts)"
        if pose_model_name is not None
        else "pose:     off"
    )
    return [
        f"model:    {launch_cfg.model}",
        pose_line,
        f"device:   {device}   half:{'y' if launch_cfg.half else 'n'}",
        f"conf:     {live.conf:.2f}   iou:{live.iou:.2f}",
        f"imgsz:    {live.imgsz}   max_det:{live.max_det}",
        f"min_area: {live.min_area_pct:.1f}%   prox:{live.proximity_px}px",
        f"class:    {live.ball_class} {class_name}",
        f"kickups:  {counter.count}  ({breakdown})",
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
    capture_config: CaptureConfig,
) -> int:
    from ultralytics import YOLO

    log.info("Loading model %s on device=%s (half=%s)", launch_cfg.model, device, launch_cfg.half)
    model = YOLO(launch_cfg.model)

    cap = _open_capture(source, capture_config)
    if not cap.isOpened():
        log.error("Failed to open source %r", source)
        return 1

    pose_model = None
    pose_model_name = None if args.no_pose else args.pose_model
    if pose_model_name is not None:
        log.info("Loading pose model %s on device=%s", pose_model_name, device)
        pose_model = YOLO(pose_model_name)

    sound_player = SoundPlayer(sound_path=args.sound_path, enabled=not args.no_sound)
    log.info("Sound: %s (path=%r)", "enabled" if sound_player.enabled and sound_player._cmd else "disabled", sound_player.sound_path)

    fps = FpsMeter()
    main_window = "Soccer Ball Detector (q quit, s save preset, r reset kickups)"
    panel = _try_create_panel(settings, enabled=not args.no_trackbars and not args.no_display)
    last_ball_class: int | None = None
    # Resolution-aware velocity floor + acceleration gate are sized to frame height
    # at runtime. Acceleration threshold is permissive by default (0.5) so the
    # counter actually counts on slow / smooth kickups; raise via CLI if
    # false-positives appear in your scene.
    kickup = KickupCounter(
        min_velocity=1.0,
        min_velocity_pct=0.005,
        acceleration_threshold=args.acceleration_threshold,
    )
    trail = MotionTrail(max_len=30)
    ball_tracker = SingleBallTracker(lose_after_frames=15)

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

                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
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
                centroid: tuple[int, int] | None = None
                kept_xyxy = np.zeros((0, 4), dtype=np.float32)
                kept_conf = np.zeros((0,), dtype=np.float32)
                kept_ids: np.ndarray | None = None

                if boxes is not None and len(boxes) > 0:
                    cls = boxes.cls.cpu().numpy().astype(int)
                    conf = boxes.conf.cpu().numpy()
                    xyxy = boxes.xyxy.cpu().numpy()
                    ids_full = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

                    class_mask = (cls == live.ball_class) & (conf >= live.conf)
                    stage_xyxy = xyxy[class_mask]
                    stage_conf = conf[class_mask]
                    stage_ids = ids_full[class_mask] if ids_full is not None else None

                    # Inline area filter so we can keep IDs aligned with boxes.
                    if stage_xyxy.shape[0] > 0 and live.min_area_pct > 0:
                        h, w = frame.shape[:2]
                        min_pixels = (live.min_area_pct / 100.0) * h * w
                        widths = stage_xyxy[:, 2] - stage_xyxy[:, 0]
                        heights = stage_xyxy[:, 3] - stage_xyxy[:, 1]
                        area_mask = (widths * heights) >= min_pixels
                        kept_xyxy = stage_xyxy[area_mask]
                        kept_conf = stage_conf[area_mask]
                        kept_ids = stage_ids[area_mask] if stage_ids is not None else None
                    else:
                        kept_xyxy = stage_xyxy
                        kept_conf = stage_conf
                        kept_ids = stage_ids

                # SingleBallTracker locks onto one ID across frames.
                _, locked_box = ball_tracker.update(kept_ids, kept_xyxy)
                if locked_box is not None:
                    x1, y1, x2, y2 = locked_box
                    centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                annotated = annotate_frame(frame, kept_xyxy, kept_conf, show_label=live.show_label)

                # Stage 1: update bounce counter with frame-height-aware threshold.
                kickup.update(
                    y=float(centroid[1]) if centroid else None,
                    frame_height=frame.shape[0],
                )
                trail.append(centroid)

                # Stage 2: pose inference + body-part gating.
                body_parts: dict[BodyPart, list[tuple[int, int]]] = {bp: [] for bp in BodyPart}
                if pose_model is not None:
                    pose_results = pose_model.predict(
                        frame,
                        verbose=False,
                        device=device,
                        half=launch_cfg.half,
                        classes=[0],  # person only
                    )
                    pose_kp = pose_results[0].keypoints
                    if pose_kp is not None and pose_kp.xy is not None:
                        body_parts = extract_body_keypoints(
                            pose_kp.xy.cpu().numpy(),
                            pose_kp.conf.cpu().numpy() if pose_kp.conf is not None else np.ones(pose_kp.xy.shape[:2]),
                        )
                if kickup.just_kicked:
                    if pose_model is None or centroid is None:
                        # Stage 1 only: credit FOOT by convention so per-part
                        # totals still add up to the aggregate count.
                        kickup.credit_last(BodyPart.FOOT)
                        log.info("kickup #%d: kept (no pose)", kickup.count)
                    else:
                        part = decide_bounce_credit(body_parts, centroid, live.proximity_px)
                        kickup.credit_last(part)
                        if part is None:
                            log.info(
                                "kickup rejected: no body part within %dpx of ball at %s",
                                live.proximity_px, centroid,
                            )
                        else:
                            log.info("kickup #%d: kept (%s)", kickup.count, part.value)
                    # If the bounce was credited (just_kicked still True after credit_last),
                    # play the sound. Rejected bounces clear just_kicked, so they stay silent.
                    if kickup.just_kicked:
                        sound_player.play()

                annotated = overlay_trail(annotated, list(trail))
                if pose_model is not None and live.show_pose:
                    annotated = overlay_pose(annotated, body_parts, live.proximity_px)
                annotated = overlay_kickup(
                    annotated, kickup.count, kickup.just_kicked, kickup.last_part
                )
                if live.show_debug:
                    threshold = max(1.0, frame.shape[0] * 0.005)
                    debug_info = {
                        "state":    kickup.state,
                        "v":        f"{kickup.smoothed_velocity:+6.1f} / {threshold:4.1f}",
                        "a":        f"{kickup.last_acceleration:+6.2f} / {args.acceleration_threshold:4.1f}",
                        "ball":     "yes" if centroid is not None else "no",
                        "lock":     f"id={ball_tracker.locked_id}" if ball_tracker.locked_id is not None else "unlocked",
                        "pose":     f"{sum(len(p) for p in body_parts.values())} kpts",
                        "kickups":  f"{kickup.count}  rej:{kickup.rejected_count}",
                    }
                    annotated = overlay_debug(annotated, debug_info)

                fps.tick()
                if live.show_fps:
                    cam_fps = grabber.capture_fps if grabber.capture_fps > 0 else None
                    drop = grabber.stats.drop_rate if grabber.stats.captured >= 10 else None
                    annotated = overlay_fps(annotated, fps.value, cam_fps=cam_fps, drop_rate=drop)
                if live.show_settings:
                    pose_kp_count = sum(len(pts) for pts in body_parts.values())
                    annotated = overlay_settings(
                        annotated,
                        _settings_lines(
                            launch_cfg, live, device, model, kickup, pose_model_name, pose_kp_count
                        ),
                    )

                if args.no_display:
                    continue
                cv2.imshow(main_window, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    path = save_preset(launch_cfg.preset, live, presets_dir=DEFAULT_PRESETS_DIR)
                    log.info("Saved preset to %s", path)
                if key == ord("r"):
                    kickup.reset()
                    trail.clear()
                    ball_tracker.reset()
                    log.info("Kickup counter reset.")
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
    capture_config = _build_capture_config(args, launch_cfg)
    return _run_loop(launch_cfg, runtime, args, device, source, capture_config)


if __name__ == "__main__":
    sys.exit(main())
