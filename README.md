# YOLO26 Soccer Ball Detector

Real-time soccer-ball detection from a webcam using pretrained
[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/).

## How it works

YOLO26 ships a `yolo26n.pt` checkpoint pretrained on COCO. COCO does not have a
dedicated "soccer ball" class — it has class `32: sports ball`, which covers
soccer balls, basketballs, baseballs, tennis balls, etc. This experiment uses
that class as a proxy. If your scene contains other sports balls they will also
be flagged. To detect specifically *soccer* balls, fine-tune on a soccer-ball
dataset (see "Next steps" below).

## Requirements

- Python 3.10+
- A webcam (or a video file / RTSP URL)
- Internet on first run (to download `yolo26n.pt`)

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
# Default webcam, default model, conf threshold 0.25
python detect.py

# Second camera
python detect.py --source 1

# Local video file
python detect.py --source path/to/match.mp4

# RTSP stream (you are responsible for trusting the URL)
python detect.py --source rtsp://user:pass@host/stream

# Larger / more accurate model
python detect.py --model yolo26s.pt --conf 0.4
```

Press **q** in the window to quit.

## CLI flags

| Flag            | Default       | Purpose                                                              |
| --------------- | ------------- | -------------------------------------------------------------------- |
| `--source`      | `0`           | Camera index, video file path, or stream URL.                        |
| `--model`       | `yolo26n.pt`  | Any Ultralytics-supported checkpoint (downloaded on first use).      |
| `--conf`        | `0.25`        | Confidence threshold for keeping detections.                         |
| `--ball-class`  | `32`          | COCO class id treated as the ball.                                   |
| `--no-display`  | off           | Process frames without opening a window (headless smoke testing).    |

## Tests

```bash
pytest -q
```

The test suite covers the pure helpers (`filter_sports_ball`, `format_label`,
`annotate_frame`) and runs without a webcam, GPU, or model weights.

## Next steps

If "any sports ball" is too loose for your use case, fine-tune `yolo26n.pt`
on a soccer-ball dataset. The Ultralytics docs have a one-page
[training guide](https://docs.ultralytics.com/modes/train/), and Roboflow Universe
has several public soccer-ball datasets in YOLO format.
