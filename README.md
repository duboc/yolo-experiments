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

## Keys

| Key | Action                          |
| --- | ------------------------------- |
| `q` | Quit                            |
| `s` | Save current settings as preset |

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
