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

---

# Plan: Full param exposure (round 3)

## Context

User wants every param tunable. Hybrid approach:
- **Init-time** params (need a model reload or capture restart) → TUI prompts before the loop.
- **Live-tunable** params (re-applied each frame) → OpenCV trackbar panel inside the running loop.
- **Persistence** → JSON presets in `./presets/<name>.json`. `--preset NAME` loads, `s` key saves to the current preset name. A `default` preset auto-loads if present.

## Parameter inventory

**Init-time (TUI + CLI flags, fixed for the run):**
| Param | Default | Values |
|---|---|---|
| `model` | `yolo26l.pt` | any ultralytics name/path |
| `device` | `auto` | `auto`/`mps`/`cuda`/`cpu` |
| `source` | (picker) | int / path / URL |
| `width` | unset | int |
| `height` | unset | int |
| `half` | `False` | bool — FP16 on MPS/CUDA |
| `preset` | `default` | preset name |

**Live-tunable (trackbars, re-read each frame):**
| Param | Range | Encoding |
|---|---|---|
| `conf` | 0.0-1.0 | trackbar 0-100 → /100 |
| `iou` | 0.0-1.0 | trackbar 0-100 → /100 |
| `max_det` | 1-300 | direct int |
| `imgsz` | 320-1280 (×32) | trackbar 10-40 → ×32 |
| `ball_class` | 0-79 | direct int (COCO classes) |
| `agnostic_nms` | bool | trackbar 0-1 → bool |
| `show_fps` | bool | trackbar 0-1 → bool |
| `show_label` | bool | trackbar 0-1 → bool |

## Precedence

`builtin defaults` ← `preset (if loaded)` ← `CLI flags` ← `TUI overrides` ← `trackbar live changes`

Each layer overlays the previous. `--no-tui` skips the TUI layer; `--no-trackbars` skips the trackbar layer.

## New modules

```
soccer_ball/
├── settings.py    # LaunchConfig + RuntimeSettings dataclasses, preset I/O (pure)
├── tui.py         # prompt_launch_config(input_fn) — testable via input injection
└── trackbars.py   # encode/decode helpers (pure) + TrackbarPanel cv2 wrapper (thin, untested)
```

## Checklist

- [x] In `tests/test_settings.py`, write failing tests for: `LaunchConfig` and `RuntimeSettings` dataclass defaults; `RuntimeSettings.to_dict()` / `from_dict()` round-trip; `merge(base, override)` overlay where `override` fields with `None` are skipped; `save_preset` / `load_preset` round-trip in a tmp_path; `load_preset` raises `FileNotFoundError` for unknown name; `list_presets` returns sorted names.
- [x] In `soccer_ball/settings.py`, implement `LaunchConfig`, `RuntimeSettings`, `merge`, `save_preset`, `load_preset`, `list_presets`. Default presets dir = `./presets`. Make tests pass.
- [x] In `tests/test_trackbars.py`, write failing tests for the pure encode/decode helpers: `decode_conf(50) == 0.5`, `decode_iou(70) == 0.7`, `decode_imgsz(20) == 640`, `decode_imgsz(11) == 352` (rounding), `encode_settings(RuntimeSettings)` returns the expected dict of trackbar ints, and `decode_trackbars({...}) -> RuntimeSettings` round-trips.
- [x] In `soccer_ball/trackbars.py`, implement pure encode/decode helpers. Add a `TrackbarPanel` class with `create_in_window(name)`, `current() -> RuntimeSettings`, `apply(settings)`. The cv2 calls live only in the wrapper methods. Make pure tests pass.
- [x] In `tests/test_tui.py`, write failing tests for `prompt_launch_config(defaults, input_fn)`: blank input keeps the default; provided value overrides; invalid device re-prompts; "y"/"n" handled for `half`; EOF maps to KeyboardInterrupt.
- [x] In `soccer_ball/tui.py`, implement `prompt_launch_config(defaults, input_fn=input)`. Make tests pass.
- [x] Extend `detect.py`:
  - New CLI flags: `--preset NAME` (default `"default"`), `--no-tui`, `--no-trackbars`, `--iou`, `--max-det`, `--imgsz`, `--half`, `--agnostic-nms`.
  - On startup: load `defaults` → if `--preset` exists, overlay it → overlay parsed CLI flags (only those the user actually set) → if `--no-tui` is off, run TUI to confirm/edit → split into `LaunchConfig` and `RuntimeSettings`.
  - Build `TrackbarPanel` (unless `--no-trackbars`), seed it from `RuntimeSettings`.
  - In the loop: read live settings from the panel each frame, pass `conf=`/`iou=`/`max_det=`/`imgsz=`/`agnostic_nms=` into `model.predict`, use `settings.ball_class` in the filter, honour `show_fps`/`show_label` overlays.
  - Key bindings: `q` quit, `s` save current settings to the active preset name (log the path).
  - Whenever `ball_class` changes, log `model.names[class_id]` so the user sees what they switched to.
- [x] Add a small `format_label_with_class(name, conf)` helper in `detector.py` and toggle it via `show_label` (when off, draw box without text).
- [x] Update `README.md`: TUI walk-through, trackbar list, preset workflow, `s` key, `--preset`/`--no-tui`/`--no-trackbars` flags, precedence rules.
- [x] Add `presets/` to `.gitignore` (presets are user-local) but commit a `presets/.gitkeep`.
- [x] Run `pytest -q`. Aim: all green, ~50 tests total.

## Critique pass

- **Risk: cv2 trackbars are flaky on macOS** — `--no-trackbars` opt-out + log a hint when the platform is `darwin` and the panel fails to create.
- **Risk: model.predict per-call kwargs may be slow if imgsz changes often** — Ultralytics handles dynamic imgsz fine but may re-warmup. Document the slight hitch when dragging the imgsz slider.
- **Risk: trackbar UX is ugly** — accept it; this is a dev tool. Class name shown in the log on change, not in the trackbar (cv2 limitation).
- **Risk: precedence order is confusing** — README spells it out, and TUI prompts show the resolved default in brackets so the user always sees the effective value before confirming.
- **Risk: preset file accidentally checked in** — `.gitignore`s `presets/*.json` (keep `.gitkeep`).
- **Risk: legacy --conf and --ball-class flags** — keep them as the canonical CLI surface, route them through the same overlay machinery as the new flags. No breaking change.

---

# Plan: On-screen settings overlay (round 4)

## Context

User wants to see which settings the detector is actually using, rendered on the frame itself. Adds a top-right semi-transparent panel listing model/device/conf/iou/imgsz/max_det/agnostic_nms/class. Toggleable, on by default for visibility, consistent with the existing `show_fps` / `show_label` toggles.

## Checklist

- [x] In `tests/test_settings.py`, add a failing test that `RuntimeSettings.show_settings` defaults to True.
- [x] In `soccer_ball/settings.py`, add `show_settings: bool = True` to `RuntimeSettings`. Make tests pass; existing round-trip and merge tests must still pass.
- [x] In `tests/test_trackbars.py`, add a failing test that `encode_settings` includes `show_settings: 1` for defaults and that `decode_trackbars` round-trips the new field.
- [x] In `soccer_ball/trackbars.py`, add `show_settings` to `_NAMES`, `_TRACKBAR_MAX`, `encode_settings`, `decode_trackbars`. Make tests pass.
- [x] In `tests/test_detector.py`, add failing tests for `overlay_settings(frame, lines)`: returns a new array, leaves input unchanged, draws something for non-empty lines, returns unchanged copy for empty list, preserves shape/dtype.
- [x] In `soccer_ball/detector.py`, implement `overlay_settings` rendering top-right with a semi-transparent black background and white text. Handles right-edge clipping cleanly. Make tests pass.
- [x] In `detect.py`, build a `_settings_lines(launch_cfg, live, model)` helper that returns the list of strings shown in the overlay. Call `overlay_settings` after `overlay_fps` when `live.show_settings`.
- [x] Update `README.md` trackbar table and add a screenshot-style block describing the overlay.
- [x] Run `pytest -q` — all green.

## Critique

- **Risk: overlay covers detections in top-right** — accept; user can disable via the trackbar.
- **Risk: text wrap on small frames** — choose a small font (0.5 scale) and short labels; for very narrow frames the background may overflow but cv2 will just clip. Acceptable for a dev tool.
- **Risk: model.names lookup throws if unfamiliar class id** — use `.get(id, "?")` style guard, already done in detect.py for ball_class logging.

---

# Plan: M4 Pro defaults (round 5)

## Context

User has a MacBook Pro M4 Pro (10P + 4E cores, 16-core Apple GPU, 48 GB unified memory). MPS is the right device. FP16 half-precision gives a meaningful speedup on MPS/CUDA but is unstable/slow on CPU. Probe noise on macOS comes from probing camera indices that don't exist.

## Checklist

- [x] In `tests/test_devices.py`, add failing tests for `resolve_half(True, "mps") == True`, `(True, "cuda") == True`, `(True, "cpu") == False`, `(False, "mps") == False`.
- [x] In `soccer_ball/devices.py`, implement `resolve_half(requested, device)`. Make tests pass.
- [x] In `tests/test_settings.py`, update the `LaunchConfig.half` default expectation to `True`.
- [x] In `soccer_ball/settings.py`, change `LaunchConfig.half` default to `True`.
- [x] In `detect.py`:
  - Switch `--half` to `argparse.BooleanOptionalAction` so users can pass `--no-half` to opt out.
  - After resolving device, call `resolve_half`; if it overrode `True → False` on CPU, log a warning.
  - Lower `--probe-max` default from 5 to 3 to quiet the macOS index-out-of-bounds noise.
- [x] Update `README.md`: a "Device guide" section with M-series guidance and the `--no-half` flag.
- [x] Run `pytest -q`. Commit + push.

## Critique

- **Risk: changing dataclass default to `half=True` breaks CPU users silently** — mitigated by `resolve_half`, which forces False on CPU and logs the override.
- **Risk: `BooleanOptionalAction` is Python 3.9+** — README already requires 3.10+, so this is safe.
- **Risk: lowering probe-max from 5 to 3 hides cameras 3-4 from auto-probe** — accept; users with more cameras pass `--probe-max 5` (documented).
- **Risk: half=True on MPS may break for some YOLO export paths** — only affects predict, which Ultralytics handles cleanly.

---

# Plan: Kickup detector (round 6)

## Context

User wants the detector tuned for soccer kickups (juggling): close-up balls, fast vertical motion, motion blur, real-time count. Three pillars: focus on the larger (closer) ball, persist tracking through brief blur, and recognise vertical bounces.

## Design

- **Tracking**: switch from `model.predict()` to `model.track(persist=True, tracker="bytetrack.yaml")`. ByteTrack (Ultralytics' default) handles brief detection drops by carrying IDs forward, which is exactly what motion blur needs.
- **Larger-ball focus**: new `min_area_pct` runtime setting + trackbar (range 0–10%, default 1%). Detections whose box area is smaller than `min_area_pct * frame_area` are dropped before the kickup logic looks at them.
- **Kickup detection**: a `KickupCounter` watches the centroid y of the largest passing detection. EMA-smoothed vertical velocity; a sign change from positive (falling) to negative (rising) increments the count. Auto-reset after N frames with no detection.
- **Visualisation**: motion trail (last N centroids, fading) + a center-top "KICKUPS: N" overlay that flashes briefly on each new kick.

## New modules and helpers

```
soccer_ball/
├── kickup.py                  # KickupCounter, MotionTrail (pure)
└── detector.py                # + filter_by_area, overlay_trail, overlay_kickup
```

## Checklist

- [x] Tests for `RuntimeSettings.min_area_pct` (default 1.0); update encode/decode round-trip.
- [x] Add `min_area_pct: float = 1.0` to `RuntimeSettings`. Add `min_area_pct` to trackbar names / max / encode / decode (encoding: trackbar 0-100 → /10 = 0.0–10.0%).
- [x] Tests for `filter_by_area(xyxy, conf, min_area_pct, frame_shape)`: drops below-threshold boxes; keeps above; empty input; zero threshold passes everything.
- [x] Implement `filter_by_area` in `detector.py`.
- [x] Tests for `KickupCounter`:
  - constant y → count 0
  - down-only or up-only → count 0
  - one full down→up cycle → count 1
  - two cycles → count 2
  - tiny noise (sub-threshold velocity) → count 0
  - long absence → resets count to 0
  - `update(None)` (no detection that frame) advances the absence counter
- [x] Tests for `MotionTrail`: bounded length, returns insertion order, empty initially, `clear()` empties.
- [x] Implement `KickupCounter` and `MotionTrail` in `soccer_ball/kickup.py`.
- [x] Tests for `overlay_trail(frame, centroids)`: immutability, empty no-op, draws something for non-empty.
- [x] Tests for `overlay_kickup(frame, count, just_kicked)`: immutability, draws something, count text appears (assert non-blank), flash color changes when `just_kicked=True`.
- [x] Implement `overlay_trail` and `overlay_kickup` in `detector.py`.
- [x] Wire `detect.py`:
  - Replace `model.predict(...)` with `model.track(persist=True, tracker="bytetrack.yaml", ...)`.
  - After existing class+conf filter, run `filter_by_area` using `live.min_area_pct` and the frame shape.
  - Pick the largest remaining box (max area); compute centroid y; feed to `KickupCounter`.
  - Update `MotionTrail` with the centroid (or `None` to advance absence).
  - Render `overlay_trail` and `overlay_kickup` (after FPS/settings overlays so the count sits on top).
  - Add a `r` key binding to reset the kickup counter manually.
- [x] Update settings overlay to include `min_area:` and `kickups:` lines.
- [x] README: new "Kickup mode" section explaining the trail, the count logic, and the `r` key.
- [x] Run pytest — aim for ~85+ green. Commit + push.

## Critique

- **Risk: model.track() returns the same `boxes` shape but adds `.id` — code already only uses `cls/conf/xyxy`, so the swap is mechanical.** Tracker config file is bundled with ultralytics; no extra setup.
- **Risk: kickup logic over-counts when ball passes near top of frame** — EMA smoothing with `alpha=0.3` and a minimum velocity-change magnitude threshold filters small wiggles.
- **Risk: largest-box assumption breaks in multi-ball scenes** — accept; this is a kickup detector, single ball is the use case. Document.
- **Risk: model.track() may be slower than predict** — empirically a few percent overhead. Worth it for blur robustness.
- **Risk: trail clutter** — fixed length (30 frames), fading alpha; user can disable via `show_label` toggle (no, that's wrong). Add a separate `show_trail` toggle? Skip for now to keep trackbar count manageable; trail is small visual cost.

---

# Plan: Capture pipeline upgrades (round 8)

## Context

Capture today is doing the bare minimum — AVFoundation backend + MJPG fourcc + ThreadedGrabber's "always latest" semantics. For kickup detection the two real bottlenecks left are motion blur (auto-exposure picks 1/30s shutter and smears the ball) and sample density per bounce (default 30 FPS gives ~6 frames per kickup arc). User selected three upgrades:

1. **Manual exposure + AF/WB lock** — best-effort property setting with success/failure logging. Works on most external USB cams; built-in MacBook cameras ignore some properties.
2. **Capture presets + 60fps** — `--capture-preset {low,balanced,high}` bundles resolution + target FPS. Defaults: low=640×480@60, balanced=1280×720@60, high=1920×1080@30.
3. **Dropped-frames diagnostic** — overlay shows effective inference FPS vs camera FPS so the user can see when YOLO is throttling them below camera potential.

Skipped this round: `--record path.mp4` (deferred).

## Design

- **`CaptureConfig` dataclass** in `capture.py` — width, height, fps, exposure, focus, auto_exposure, auto_focus, auto_wb, wb_temp. All optional; `None` = leave camera default.
- **`apply_capture_config(cap, config) -> dict[str, bool]`** — applies each non-None field, returns a map of `prop_name → success` so the loop can log which properties the camera actually accepted.
- **`CAPTURE_PRESETS` constant** — `{low, balanced, high}` → `CaptureConfig`. CLI overlays on top.
- **`GrabberStats`** — pure class tracking `captured` / `unique_reads`, exposing `drop_rate`. ThreadedGrabber instantiates one and updates it on capture / read. No threading in the stats class itself, so it tests cleanly.
- **`overlay_fps(frame, fps, cam_fps=None, drop_rate=None)`** — extended signature appends `(cam 60, drop 35%)` when the new args are non-None.

## Checklist

- [x] TDD: `CaptureConfig` defaults + `apply_capture_config(fake_cap, config)` returns dict-of-bool. Cover: all fields None → no calls; auto_exposure=False → manual mode prop set; failed `cap.set()` → False in returned dict.
- [x] TDD: `CAPTURE_PRESETS` lookup + `merge_capture_config(base, override)` overlays non-None fields.
- [x] TDD: `GrabberStats` — `on_capture` increments captured; `on_read` increments unique_reads only when captured > last_read; `drop_rate` formula correct; zero-state returns 0.0.
- [x] Implement `CaptureConfig`, `apply_capture_config`, `CAPTURE_PRESETS`, `merge_capture_config`, `GrabberStats` in `soccer_ball/capture.py`.
- [x] Wire `GrabberStats` into `ThreadedGrabber`: tick on capture, tick on read; expose `stats` and `capture_fps` (via internal FpsMeter).
- [x] TDD: `overlay_fps` extended signature — original 1-arg call still works; passing cam_fps/drop_rate changes output; defaults to bare FPS string.
- [x] Implement extended `overlay_fps` in `soccer_ball/detector.py`.
- [x] `detect.py`:
  - Add CLI flags: `--fps`, `--exposure`, `--focus`, `--no-auto-exposure`, `--no-auto-focus`, `--no-auto-wb`, `--wb-temp`, `--capture-preset {low,balanced,high}`.
  - Replace inline `cap.set(...)` block in `_open_capture` with `apply_capture_config`. Log the success-map.
  - Pass `grabber.capture_fps` and `grabber.stats.drop_rate` into `overlay_fps`.
- [x] Update README — capture preset table, exposure/focus flags, drop-rate overlay.

## Critique

- **Risk: built-in MacBook camera ignores manual exposure / focus / WB** — mitigated by best-effort + per-property success logging so the user knows which knobs took effect.
- **Risk: 60fps preset asks for more than camera supports** — `cap.set(CAP_PROP_FPS, 60)` returns the actual rate; we log mismatch.
- **Risk: drop-rate overlay misleading at startup** — first ~30 frames have small denominator. Skip rendering when `captured < 10`.
- **Risk: extra CLI flags clutter --help** — 8 new flags, but they're grouped logically. Keep TUI lean (don't prompt for these).

---

# Plan: Fix the kickup counter regression (round 9)

## Context

User reports the kickup counter stopped working after round 7. Root cause: the pose gate is too aggressive. Two failure modes:

1. **Soft case — pose can't see anyone.** `body_parts` is `{FOOT: [], KNEE: [], HEAD: []}` (no person detected). The current code calls `nearest_body_part(centroid, {}, ...)` which returns None, then `credit_last(None)` rolls back the bounce. We're penalizing bounces for the *pose model's* failure to detect a person.
2. **Hard case — proximity too tight.** Default 80px = ~7% of 1080p height. But pose keypoints are joint centers; the actual foot is 50-100px below the ankle keypoint. Ball center to ankle keypoint can easily exceed 80px on a real kickup.

Both are correctness bugs in the gate logic, not the pose model itself.

## Design

- **Soft gate:** if pose found *no* body parts at all this frame, credit the bounce anyway (default to FOOT). Only reject when pose actually saw someone but none of their keypoints were close enough to the ball.
- **Bump default proximity** from 80 to 150 to match real ankle-vs-foot offset.
- **Diagnostic log** on every bounce decision so the user can tell from the terminal which mode they're in (`kept (foot)`, `kept (no pose)`, `rejected`).
- **Pose-status line** in the settings overlay: `pose: yolo26n-pose.pt  3 kpts` so the user sees in real time whether pose is finding anyone.
- Extract the gate decision into a pure `decide_bounce_credit()` in `soccer_ball/pose.py` so it's unit-testable in isolation.

## Checklist

- [x] TDD: `decide_bounce_credit(parts, centroid, proximity_px, fallback)` — empty parts returns fallback; ball near a part returns that part; ball far from all returns None.
- [x] Implement `decide_bounce_credit` in `soccer_ball/pose.py`.
- [x] Update `RuntimeSettings.proximity_px` default 80 → 150 (`tests/test_settings.py` + `tests/test_trackbars.py`).
- [x] `detect.py`: replace the inline `nearest_body_part` call with `decide_bounce_credit`; log each bounce decision; pass total keypoint count into the settings overlay.
- [x] Update `_settings_lines` to render `pose: <name>  N kpts`.
- [x] Run pytest, commit, push.
