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
