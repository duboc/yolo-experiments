# YOLO26 Soccer Ball Detector

Real-time soccer-ball detection from a webcam using pretrained
[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/), tuned for
Apple Silicon (M-series) MacBooks. Every parameter is tunable from a
pre-loop TUI and live cv2 trackbars, with JSON presets that persist between runs.

## How it works

YOLO26 ships pretrained on COCO. COCO has no dedicated "soccer ball" class —
it has class `32: sports ball`, which covers soccer balls, basketballs,
baseballs, tennis balls, etc. This experiment uses that class as a proxy.
If your scene contains other sports balls they will also be flagged. Fine-tune
on a soccer-ball dataset for soccer-only detection.

## Defaults

- Model: **`yolo26l.pt`** (large)
- Device: **`auto`** → MPS on M-series Macs, CUDA on Linux/Windows GPU boxes, CPU otherwise
- Source: **interactive camera picker** if `--source` is omitted
- Capture: threaded grabber, MJPG fourcc, single-frame buffer (latency-first)
- Display: green detection boxes + yellow FPS overlay (top-left)
- Tunable: TUI before loop + trackbar panel during loop

## Requirements

- Python 3.10+
- A webcam (or a video file / RTSP URL)
- Internet on first run (downloads `yolo26l.pt`, ~50 MB)
- macOS users: Terminal/iTerm needs camera permission (System Settings → Privacy & Security → Camera)

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
# Default flow: TUI → camera picker → live loop with trackbars
python detect.py
```

Pre-loop TUI walks through every launch param. Press Enter to keep the bracketed default:

```
--- Launch configuration (Enter to accept default) ---
Model [yolo26l.pt]:
Device (auto/mps/cuda/cpu) [auto]:
Source (blank = camera picker) []:
Camera width (blank = camera default) []: 1920
Camera height (blank = camera default) []: 1080
Use FP16 half-precision? [n]: y
Preset name [default]:
```

Then the camera picker, then the live window plus a `Settings` window with trackbars.

## Live trackbars

| Trackbar       | Range / Encoding                  | Notes                              |
| -------------- | --------------------------------- | ---------------------------------- |
| `conf`         | 0–100 → 0.0–1.0                   | Confidence threshold               |
| `iou`          | 0–100 → 0.0–1.0                   | NMS IoU threshold                  |
| `max_det`      | 1–300                             | Max detections per frame           |
| `imgsz`        | 10–40 → 320–1280 (×32)            | Inference image size               |
| `ball_class`   | 0–79                              | COCO class id (logged on change)   |
| `agnostic_nms` | 0/1                               | Class-agnostic NMS                 |
| `show_fps`     | 0/1                               | FPS overlay on/off                 |
| `show_label`   | 0/1                               | Box labels on/off                  |
| `show_settings`| 0/1                               | Settings overlay (top-right) on/off |
| `min_area_pct` | 0–100 → 0.0–10.0%                 | Drop boxes below this % of frame area (kickup focus) |
| `proximity_px` | 0–300 px (direct)                 | Foot/knee/head proximity radius for pose gating |
| `show_pose`    | 0/1                               | Render pose keypoint dots + proximity circles    |
| `show_debug`   | 0/1                               | Render bottom-left kickup-state debug panel      |

Drag a slider, see the effect immediately on the next frame.

## On-screen settings overlay

While the loop is running, the top-right corner shows a live readout of the
settings actually being used for inference, e.g.:

```
model:    yolo26l.pt
device:   mps   half:n
conf:     0.25  iou:0.70
imgsz:    640   max_det:300
agnostic: n
class:    32 sports ball
```

This stays in sync with the trackbars and presets. Toggle off with the
`show_settings` trackbar if it gets in the way.

## Presets

Settings are saved to `./presets/<name>.json` when you press **`s`** in the
detection window. Load a preset on startup with `--preset NAME` (default
`default`, auto-loaded if present).

```bash
python detect.py --preset indoor      # load presets/indoor.json on startup
# adjust trackbars while running, press 's' to overwrite presets/indoor.json
```

## Kickup mode

The detector is wired for soccer ball juggling out of the box:

- **Tracker** — uses Ultralytics' `model.track()` with ByteTrack and a
  `SingleBallTracker` that locks onto one ID and follows it across frames,
  so brief motion-blur drops don't lose the ball mid-bounce and a second
  ball in the scene can't hijack the counter.
- **Larger-ball focus** — `min_area_pct` drops detections smaller than 1% of
  the frame area by default, so distant balls in the background don't
  distract the counter. Drag the slider to tune.
- **Acceleration-spike gate** — a bounce only counts when the smoothed
  vertical velocity reverses *and* the velocity-change magnitude clears a
  threshold. Slow rollovers (ball drifting off a hand) no longer count.
- **Resolution-aware velocity floor** — the minimum-velocity threshold
  scales with frame height (~0.5% of height per frame), so the same
  juggling motion behaves consistently on 720p, 1080p, and 4K.
- **Pose-gated counter** — a second model (`yolo26n-pose.pt` by default)
  runs in parallel; each bounce only counts when a foot, knee, or head
  keypoint is within `--proximity-px` (default 80px) of the ball at the
  bounce frame. Pure floor bounces, throws, and balls rolling off a desk
  no longer fool the counter.
- **Per-body-part breakdown** — `KICKUPS: 12 (foot:8 knee:3 head:1)` shown
  in the settings panel; the flash text reads `KICKUPS: 12 (knee)` on the
  frame the bounce happens.
- **Pose overlay** — colored dots at each detected ankle / knee / head plus
  the proximity radius. Toggle with the `show_pose` trackbar.
- **Motion trail** — last 30 ball centroids drawn as fading orange dots so
  you can see the kickup arc.
- **Auto-reset** — counter zeroes itself after ~30 frames with no detection
  (~1s at 30 FPS), so dropping the ball starts a fresh count.

Tips for accuracy:

- Keep the ball as the largest object in the frame (close-up shots help).
- For a fast kickup loop, `--imgsz 480` cuts inference time and is plenty
  for a close-up ball.
- If pose proximity is misfiring, drag the `proximity_px` trackbar — bigger
  is more permissive, smaller is stricter.
- If your machine can't keep up with two models, `--no-pose` keeps the
  Stage 1 wins (ID tracking, acceleration gate, resolution-aware velocity)
  at full FPS but loses the floor-bounce filter.

## Debug overlay (when the count looks wrong)

If kickups stop counting and you can't tell why, flip `show_debug` (trackbar)
or just enable it via a preset. A bottom-left panel appears with the
state-machine internals:

```
state    FALLING
v        +12.3 /  5.4
a        -0.45 /  0.5
ball     yes
lock     id=3
pose     3 kpts
kickups  7  rej:2
```

- `state`: NEUTRAL → FALLING → RISING — a bounce only counts on the
  FALLING→RISING transition.
- `v`: smoothed vertical velocity / threshold (resolution-aware). For a real
  kickup this should swing from large positive to large negative.
- `a`: per-frame acceleration / `--acceleration-threshold`. The bounce only
  counts if `|a| ≥ threshold`. Lower the threshold (e.g. `--acceleration-threshold 0.2`)
  for slow / smooth kickups; raise it if you see false positives.
- `ball`: was the ball detected this frame.
- `lock`: which ByteTrack ID the SingleBallTracker is locked onto.
- `pose`: total keypoint count across all visible persons.
- `kickups`: aggregate count + rejected count (rejected = bounce detected
  but pose gate said no body part nearby).

Common diagnoses:

| Symptom | Likely cause | Fix |
|---|---|---|
| `v` stays near 0 | Ball detection drops too often | Lower `min_area_pct`, raise `conf` threshold from camera, or check lighting |
| `state` never reaches RISING | Velocity threshold too high | Lower `min_velocity_pct` (advanced) or just confirm on screen the ball is actually moving fast enough |
| `state` reaches RISING but no count | `|a|` below `--acceleration-threshold` | Drop `--acceleration-threshold` to 0.2 |
| `rej` increments while `kickups` stays | Pose gate rejecting | Raise `proximity_px` trackbar or pass `--no-pose` to disable gating |
| `pose` stays at 0 | Pose model can't see you | Move closer to the camera, improve lighting, or try `--pose-model yolo26s-pose.pt` |

## Sound

Each counted kickup plays a short sound effect — useful for kinesthetic
feedback when you're not staring at the screen. macOS uses `afplay` with
`/System/Library/Sounds/Pop.aiff` by default; Linux uses `aplay` (you supply
the file).

```bash
python detect.py                                       # default Pop.aiff on macOS
python detect.py --sound-path /System/Library/Sounds/Tink.aiff
python detect.py --sound-path ~/Music/whistle.wav      # any file afplay can read
python detect.py --no-sound                            # silent
```

The player is fire-and-forget (non-blocking `Popen`) so the inference loop
never stalls on audio. If the playback command fails once (missing binary,
bad path), the player auto-disables and the loop keeps running.

Rejected bounces (pose gate said no body part nearby) stay silent — only
credited kickups make a sound.

## Keys

| Key | Action                          |
| --- | ------------------------------- |
| `q` | Quit                            |
| `s` | Save current settings as preset |
| `r` | Reset kickup counter and trail  |

## Precedence

`builtin defaults` ← `preset` ← `CLI flags` ← `TUI` ← `live trackbars`

Each layer overlays the previous. `--no-tui` skips the TUI; `--no-trackbars`
skips the trackbar panel; combine both for fully scripted runs.

## CLI flags

### Launch config

| Flag              | Default       | Purpose                                                    |
| ----------------- | ------------- | ---------------------------------------------------------- |
| `--source`        | (picker)      | Camera index / video path / stream URL                     |
| `--model`         | `yolo26l.pt`  | Any Ultralytics-supported checkpoint                       |
| `--device`        | `auto`        | `auto` / `mps` / `cuda` / `cpu`                            |
| `--width`         | unset         | Camera width hint                                          |
| `--height`        | unset         | Camera height hint                                         |
| `--fps`           | unset         | Target capture FPS hint                                    |
| `--capture-preset`| unset         | `low` / `balanced` / `high` — bundles width+height+fps     |
| `--exposure`      | unset         | Manual exposure value (camera-specific scale)              |
| `--no-auto-exposure` | off        | Disable camera auto-exposure (best-effort)                 |
| `--focus`         | unset         | Manual focus distance (camera-specific scale)              |
| `--no-auto-focus` | off           | Disable camera auto-focus (best-effort)                    |
| `--wb-temp`       | unset         | Manual white-balance temperature in Kelvin                 |
| `--no-auto-wb`    | off           | Disable camera auto white-balance (best-effort)            |
| `--half`          | off           | Use FP16 half-precision                                    |
| `--preset`        | `default`     | Preset to load and save with `s`                           |

### Runtime settings (also live-tunable)

| Flag              | Default | Purpose                                          |
| ----------------- | ------- | ------------------------------------------------ |
| `--conf`          | `0.25`  | Confidence threshold                             |
| `--iou`           | `0.7`   | NMS IoU threshold                                |
| `--max-det`       | `300`   | Max detections per frame                         |
| `--imgsz`         | `640`   | Inference size (multiple of 32)                  |
| `--ball-class`    | `32`    | COCO class id treated as the ball                |
| `--agnostic-nms`  | off     | Class-agnostic NMS                               |
| `--no-half`       | off     | Disable FP16 (default is on for mps/cuda)        |
| `--proximity-px`  | `80`    | Foot/knee/head proximity radius (also live tunable) |
| `--pose-model`    | `yolo26n-pose.pt` | Pose checkpoint for body-part gating  |
| `--no-pose`       | off     | Disable pose-based gating (Stage 1 only)         |
| `--sound-path`    | `Pop.aiff` (macOS) | Sound played on each counted kickup    |
| `--no-sound`      | off     | Disable kickup sound effects                     |
| `--acceleration-threshold` | `0.5` | Min |Δsmoothed-velocity| to count a bounce |

### Workflow toggles

| Flag              | Purpose                                          |
| ----------------- | ------------------------------------------------ |
| `--no-tui`        | Skip the pre-loop launch TUI                     |
| `--no-trackbars`  | Skip the live trackbar panel                     |
| `--no-display`    | Skip cv2.imshow (headless smoke test)            |
| `--no-fps`        | Disable the FPS overlay at startup               |
| `--list-cameras`  | Probe cameras, print, exit                       |
| `--probe-max N`   | Max camera index to probe (default 5)            |

## Device guide

| Hardware                  | `--device` | `--half` |
| ------------------------- | ---------- | -------- |
| Apple Silicon (M1-M4)     | `mps`      | `True` (default) |
| NVIDIA GPU (any modern)   | `cuda`     | `True` (default) |
| CPU only                  | `cpu`      | forced to `False` |

`--device auto` (the default) resolves to MPS on Apple Silicon, CUDA on
NVIDIA boxes, CPU as fallback. CUDA is **NVIDIA-only**; on a Mac, asking
for `cuda` will fail.

`--half` defaults to `True`. On CPU it's automatically forced off (PyTorch
CPU FP16 inference is unsupported / much slower). Pass `--no-half` to opt
out on a GPU device for a small accuracy bump.

## Camera capture

The capture pipeline is tunable from the CLI for two real wins on fast-motion
detection:

### Presets

```bash
python detect.py --capture-preset low        # 640x480 @ 60 fps  (best for kickup)
python detect.py --capture-preset balanced   # 1280x720 @ 60 fps
python detect.py --capture-preset high       # 1920x1080 @ 30 fps
```

Higher FPS gives the kickup counter more samples per bounce; lower resolution
keeps inference fast and trades a sharper temporal signal for spatial detail.

Individual `--width`, `--height`, `--fps` flags overlay on top of any preset.

### Manual exposure / focus / WB (motion-blur fix)

Auto-exposure on most webcams ramps shutter to 1/30s in low light, smearing a
fast ball over multiple pixels per frame. Lock the camera to a short exposure
to freeze motion:

```bash
python detect.py --no-auto-exposure --exposure -7    # short shutter, may be dark
python detect.py --no-auto-focus --focus 120         # fixed focus, no hunting
python detect.py --no-auto-wb --wb-temp 4500         # fixed colors
```

These are best-effort: the loop logs which properties the camera accepted.
Built-in MacBook cameras typically ignore exposure/focus controls; external
USB cameras usually honour them.

### Drop-rate diagnostic

The FPS overlay grows when the threaded grabber has stats:

```
FPS:  18.3  (cam   60  drop 70%)
```

`cam` is the camera's measured capture rate; `drop` is the fraction of camera
frames the inference loop never saw (inference is the bottleneck). High drop
rate means: lower `--imgsz`, switch to a smaller model, or enable `--half`.

## Why a threaded frame grabber?

OpenCV's `CAP_PROP_BUFFERSIZE = 1` is silently ignored by the macOS
AVFoundation backend, which means a slow consumer (YOLO inference) backs up
behind stale camera frames and inference lags 1-2 seconds behind reality.
The threaded grabber drains the camera continuously and only ever exposes
the **latest** frame, so the visible output stays in sync with reality.

## Tests

```bash
pytest -q
```

The suite (~50 tests) covers all pure helpers — devices, cameras, capture,
detection filtering, annotation, FPS overlay, settings, preset I/O, trackbar
encoding, TUI prompts — without needing a webcam, GPU, or model weights.

## Next steps

If "any sports ball" is too loose, fine-tune `yolo26l.pt` on a soccer-ball
dataset. The Ultralytics
[training guide](https://docs.ultralytics.com/modes/train/) plus a Roboflow
Universe soccer-ball dataset gets you most of the way.
