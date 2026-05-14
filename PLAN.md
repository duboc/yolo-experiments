# Plan: YOLO26 Soccer Ball Webcam Detector

## Context

- **Model**: `yolo26n.pt` (pretrained on COCO, 80 classes). COCO class index `32` is `sports ball` — soccer balls fall under this label.
- **Source**: Webcam (default `cv2.VideoCapture(0)`), with a CLI flag to switch to a different camera index or RTSP/file URL.
- **Packaging**: One script (`detect.py`), one `requirements.txt`, one `README.md`. Tests live in `tests/`.
- **TDD**: Pure helper functions (class filtering, label formatting, frame annotation contract) get tests first. The webcam loop itself is integration glue and is exercised manually.
- **Class filtering rationale**: The COCO `sports ball` label covers soccer balls plus basketballs, baseballs, etc. We accept this as the "soccer ball" proxy for the pretrained-COCO approach the user chose. Document this honestly in the README.

## Architecture

```
yolo-experiments/
├── PLAN.md
├── README.md
├── requirements.txt
├── .gitignore
├── detect.py                # CLI entrypoint + webcam loop
├── soccer_ball/
│   ├── __init__.py
│   └── detector.py          # pure helpers: filter_sports_ball, annotate_frame
└── tests/
    └── test_detector.py     # pytest, no webcam, no real weights
```

Splitting helpers into `soccer_ball/detector.py` keeps `detect.py` thin and lets pytest import without triggering OpenCV/webcam initialization.

## Checklist

- [x] Create `.gitignore` ignoring `__pycache__/`, `*.pyc`, `*.pt`, `runs/`, `.venv/`, `.pytest_cache/`.
- [x] Create `requirements.txt` pinning `ultralytics>=8.4.0` (YOLO26 weights are released under v8.4.0 assets) and `opencv-python>=4.8`. Add `pytest` for tests.
- [x] Create `soccer_ball/__init__.py` (empty).
- [x] In `tests/test_detector.py`, write a failing test for `soccer_ball.detector.filter_sports_ball(boxes_cls, boxes_conf, boxes_xyxy, ball_class_id=32, conf_threshold=0.25)` that returns only rows where `cls == ball_class_id` AND `conf >= conf_threshold`. Cover: all matches, no matches, mixed, empty input, sub-threshold filtered out.
- [x] In `soccer_ball/detector.py`, implement `filter_sports_ball` using NumPy boolean indexing. Make the test pass.
- [x] In `tests/test_detector.py`, add a failing test for `format_label(conf)` returning `"soccer ball 0.87"` for `conf=0.8732`. Cover: rounding to 2 decimals, edge case `conf=1.0`.
- [x] In `soccer_ball/detector.py`, implement `format_label`. Make the test pass.
- [x] In `tests/test_detector.py`, add a failing test for `annotate_frame(frame, xyxy, confs)` that: returns a new array of the same shape and dtype as input, leaves input unmodified (no in-place mutation), and handles empty detection list (returns a copy unchanged). Use a synthetic `np.zeros((480, 640, 3), dtype=np.uint8)` frame.
- [x] In `soccer_ball/detector.py`, implement `annotate_frame` using `cv2.rectangle` + `cv2.putText` on a `frame.copy()`. Make the test pass.
- [x] In `detect.py`, add `argparse` CLI: `--source` (default `"0"`, accepts int-like or string), `--model` (default `"yolo26n.pt"`), `--conf` (default `0.25`), `--ball-class` (default `32`), `--no-display` (flag, useful for headless smoke tests).
- [x] In `detect.py`, implement `main()`: load `YOLO(args.model)`, open `cv2.VideoCapture`, loop frames, call `model.predict(frame, verbose=False)`, extract `boxes.cls/conf/xyxy` as NumPy, pass through `filter_sports_ball`, call `annotate_frame`, `cv2.imshow` unless `--no-display`, break on `q` key. Handle `KeyboardInterrupt` and always release the capture + destroy windows in a `finally` block.
- [x] Write `README.md`: prerequisites (Python 3.10+, webcam), install (`pip install -r requirements.txt`), run (`python detect.py`), CLI flags, **explicit caveat** that COCO `sports ball` matches any sports ball (not soccer-specific), and a "Next steps" pointer to fine-tuning if soccer-only detection is needed.
- [x] Run `pytest -q` from the project root and confirm all tests pass.
- [x] Commit in logical chunks (scaffolding, helpers+tests, CLI, docs).

## Self-review gates

- **Security**: No secrets. `--source` is passed straight to OpenCV; document that an attacker-controlled RTSP URL is the user's responsibility.
- **Performance**: `model.predict` is called per-frame; no batching needed for webcam. `verbose=False` to avoid stdout spam.
- **Correctness**: COCO class 32 verified at runtime by reading `model.names`. Webcam release happens in `finally`.
- **Style**: Functions under 50 lines. Pure helpers separated from I/O loop. No `print` in library code — use `logging` if anything needs to surface.

---

# Plan: Real-time enhancements (round 2)

## Context

User asked for: large model default, M4 GPU acceleration (Apple MPS), improved frame capture pipeline, and an interactive camera picker before the loop starts.

## Design decisions

- **Default model**: `yolo26l.pt` (the "L" / large tier). `--model yolo26x.pt` for extra-large.
- **Device selection**: `--device auto` resolves to `mps` if available (M-series Macs), else `cuda` if available, else `cpu`. `--device {auto,mps,cuda,cpu}` override. We propagate errors instead of silent fallback so the user knows when MPS isn't kicking in.
- **Frame capture**: A `ThreadedGrabber` runs `cap.read()` in a background daemon thread and atomically stores the latest frame. Main loop pulls from `grabber.read()` and never blocks waiting on the camera. On macOS the AVFoundation backend ignores `CAP_PROP_BUFFERSIZE`, which is exactly why a threaded grabber matters there. We also set MJPG fourcc to push the camera off its default low-FPS YUYV mode.
- **FPS overlay**: A small EMA over recent frame deltas, rendered top-left.
- **Camera picker**: Probe indices 0-N (default N=5), open each briefly, record resolution and FPS, present a numbered list, read choice via `input()`. Skipped if `--source` was supplied or if stdin is not a TTY (so headless runs still work).
- **Testability**: All non-I/O logic lives in pure helpers with injected dependencies. The `ThreadedGrabber` accepts any object exposing `read()`/`release()`, so tests use a fake capture. The picker takes the camera list and an `input_fn` for the prompt.

## Architecture additions

```
soccer_ball/
├── devices.py     # auto_device(), with optional torch_module injection for tests
├── cameras.py     # CameraInfo, probe_cameras(), pick_camera_interactive()
└── capture.py     # ThreadedGrabber (context manager), FpsMeter
```

`detect.py` keeps its role as the I/O glue: parse args → pick device → pick camera → open grabber → loop.

## Checklist

- [x] In `tests/test_devices.py`, write failing tests for `auto_device(torch_module)` covering: mps available → "mps"; cuda available, mps not → "cuda"; neither → "cpu". Use a `SimpleNamespace` fake torch.
- [x] In `soccer_ball/devices.py`, implement `auto_device(torch_module=None)` that imports torch lazily when the arg is None. Make tests pass.
- [x] In `tests/test_cameras.py`, write failing tests for `pick_camera_interactive(cameras, input_fn)`: returns the chosen index for a valid number, re-prompts on invalid input, raises `KeyboardInterrupt` cleanly on EOF. Also test that `format_camera_menu(cameras)` renders the expected lines.
- [x] In `soccer_ball/cameras.py`, implement `CameraInfo`, `format_camera_menu`, `pick_camera_interactive`. Implement `probe_cameras(max_index, capture_factory)` with an injectable factory so tests don't open real cameras. Make tests pass.
- [x] In `tests/test_capture.py`, write failing tests for `ThreadedGrabber` using a fake capture: `read()` returns the latest frame produced by the fake; `stop()` joins the thread; using as a context manager auto-releases. Also test `FpsMeter.tick()` returns a sensible EMA after several ticks.
- [x] In `soccer_ball/capture.py`, implement `ThreadedGrabber` (daemon thread, threading.Lock, `started`/`stopped` events) and `FpsMeter`. Make tests pass.
- [x] Update `detect.py`:
  - Change `--model` default to `yolo26l.pt`.
  - Add `--device {auto,mps,cuda,cpu}` (default `auto`).
  - Add `--probe-max` (default `5`) for picker breadth.
  - Add `--width`, `--height` (optional camera resolution hints).
  - When `--source` is omitted, run the camera picker (only if stdin is a TTY).
  - Use `ThreadedGrabber` instead of direct `cap.read()`.
  - Pass `device=` to `model.predict()`.
  - Overlay FPS via `FpsMeter` on each annotated frame.
- [x] Add `_overlay_fps(frame, fps)` helper in `soccer_ball/detector.py` with a test (immutability, draws something).
- [x] Update `README.md`: new defaults, `--device`, picker UX, FPS overlay, M-series MPS note, threaded-grabber rationale.
- [x] Run `pytest -q` — all old tests still pass, new tests added.
- [x] Commit in coherent chunks.

## Critique pass

- **Risk: macOS-only camera picker on Linux test box**: I can't actually open a camera here. Mitigated by injectable `capture_factory` so tests use a fake.
- **Risk: MPS not available in CI / Linux**: `auto_device` fake-tested via injected torch module. Real device confirmed manually on the user's Mac.
- **Risk: ThreadedGrabber thread leak**: Always cleanup via context manager; tests assert thread is non-alive after `stop()`.
- **Risk: MJPG fourcc not supported by some cameras**: Wrap the `set` calls in try/log; OpenCV `set` returns False on failure but doesn't raise.
- **Risk: --source 0 vs picker ambiguity**: Picker only runs when `--source` was *not* passed. `argparse` default sentinel (`None`) distinguishes "user typed 0" from "user typed nothing".
- **Risk: FPS overlay covers detections at top-left**: Acceptable; small font, fixed position, user can disable via `--no-fps`.
