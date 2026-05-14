# YOLO26 Soccer Ball Detector

Real-time soccer-ball detection from a webcam using pretrained
[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/), tuned for
Apple Silicon (M-series) MacBooks.

## How it works

YOLO26 ships pretrained on COCO. COCO has no dedicated "soccer ball" class —
it has class `32: sports ball`, which covers soccer balls, basketballs,
baseballs, tennis balls, etc. This experiment uses that class as a proxy.
If your scene contains other sports balls they will also be flagged. Fine-tune
on a soccer-ball dataset for soccer-only detection (see "Next steps").

## Defaults

- Model: **`yolo26l.pt`** (large)
- Device: **`auto`** → MPS on M-series Macs, CUDA on Linux/Windows GPU boxes, CPU otherwise
- Camera: **interactive picker** if `--source` is omitted
- Capture: threaded grabber, MJPG fourcc, single-frame buffer (latency-first)
- Display: green detection boxes + a yellow FPS overlay (top-left)

## Requirements

- Python 3.10+
- A webcam (or a video file / RTSP URL)
- Internet on first run (downloads `yolo26l.pt`, ~50 MB)
- macOS users: Terminal/iTerm needs camera permission (System Settings → Privacy & Security → Camera)

## Install

```bash
pip install -r requirements.txt
```

On an M-series Mac the bundled `torch` already has MPS support; nothing extra
to install.

## Run

```bash
# First run: probes cameras and asks you to pick
python detect.py
```

Output looks like:

```
Available cameras:
  [0] index=0  1280x720 @ 30fps
  [1] index=1  1920x1080 @ 60fps
Choose camera [0-1]: 1
Loading model yolo26l.pt on device=mps
```

Press **q** in the window to quit.

### Common variants

```bash
# Skip the picker, use camera 0 directly
python detect.py --source 0

# Just list cameras and exit
python detect.py --list-cameras

# Force CPU even on a Mac (for comparison)
python detect.py --device cpu

# Bigger model, higher confidence threshold
python detect.py --model yolo26x.pt --conf 0.4

# Run on a video file
python detect.py --source path/to/match.mp4

# RTSP stream (you trust the URL)
python detect.py --source rtsp://user:pass@host/stream

# Bump camera resolution
python detect.py --width 1920 --height 1080
```

## CLI flags

| Flag             | Default        | Purpose                                                            |
| ---------------- | -------------- | ------------------------------------------------------------------ |
| `--source`       | (picker)       | Camera index / video path / stream URL. Omit to pick interactively. |
| `--model`        | `yolo26l.pt`   | Any Ultralytics-supported checkpoint.                              |
| `--device`       | `auto`         | `auto` → `mps` > `cuda` > `cpu`. Override with `mps`/`cuda`/`cpu`. |
| `--conf`         | `0.25`         | Confidence threshold for keeping detections.                       |
| `--ball-class`   | `32`           | COCO class id treated as the ball.                                 |
| `--width`        | unset          | Camera width hint (passed to OpenCV).                              |
| `--height`       | unset          | Camera height hint.                                                |
| `--probe-max`    | `5`            | Highest camera index to probe in the picker.                       |
| `--list-cameras` | off            | Probe cameras, print them, exit.                                   |
| `--no-display`   | off            | Run without opening a window (headless smoke test).                |
| `--no-fps`       | off            | Disable the FPS overlay.                                           |

## Why a threaded frame grabber?

OpenCV's `CAP_PROP_BUFFERSIZE = 1` is silently ignored by the macOS
AVFoundation backend, which means a slow consumer (YOLO inference) backs up
behind stale camera frames and inference lags 1-2 seconds behind reality.
The threaded grabber drains the camera continuously and only ever exposes the
**latest** frame, so the visible output stays in sync with what the camera
actually sees.

## Tests

```bash
pytest -q
```

The suite covers all pure helpers — device selection, camera probing/picker,
threaded grabber, FPS meter, detection filtering, and frame annotation —
without needing a webcam, GPU, or model weights.

## Next steps

If "any sports ball" is too loose, fine-tune `yolo26l.pt` on a soccer-ball
dataset. The Ultralytics
[training guide](https://docs.ultralytics.com/modes/train/) plus a Roboflow
Universe soccer-ball dataset gets you most of the way.
