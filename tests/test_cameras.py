"""Tests for camera enumeration and the interactive picker."""

import pytest

from soccer_ball.cameras import (
    CameraInfo,
    format_camera_menu,
    pick_camera_interactive,
    probe_cameras,
)


class _FakeCapture:
    def __init__(self, *, opened: bool, width: float = 1280, height: float = 720, fps: float = 30):
        self._opened = opened
        self._width = width
        self._height = height
        self._fps = fps
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop: int) -> float:
        # 3 = CAP_PROP_FRAME_WIDTH, 4 = CAP_PROP_FRAME_HEIGHT, 5 = CAP_PROP_FPS
        return {3: self._width, 4: self._height, 5: self._fps}.get(prop, 0.0)

    def read(self):
        return (self._opened, object() if self._opened else None)

    def release(self) -> None:
        self.released = True


class TestProbeCameras:
    def test_returns_only_openable_indices(self):
        opened_indices = {0, 2}
        captures: list[_FakeCapture] = []

        def factory(index: int) -> _FakeCapture:
            cap = _FakeCapture(opened=index in opened_indices)
            captures.append(cap)
            return cap

        cams = probe_cameras(max_index=3, capture_factory=factory)

        assert [c.index for c in cams] == [0, 2]
        # All probed captures must be released, even the ones that didn't open.
        assert all(c.released for c in captures)

    def test_records_resolution_and_fps(self):
        def factory(index: int) -> _FakeCapture:
            return _FakeCapture(opened=True, width=1920, height=1080, fps=60)

        cams = probe_cameras(max_index=1, capture_factory=factory)

        assert cams == [CameraInfo(index=0, width=1920, height=1080, fps=60.0)]


class TestFormatCameraMenu:
    def test_renders_numbered_lines(self):
        cams = [
            CameraInfo(index=0, width=1280, height=720, fps=30.0),
            CameraInfo(index=2, width=1920, height=1080, fps=60.0),
        ]
        out = format_camera_menu(cams)

        assert "[0]" in out and "index=0" in out and "1280x720" in out and "30" in out
        assert "[1]" in out and "index=2" in out and "1920x1080" in out and "60" in out


class TestPickCameraInteractive:
    def _cams(self) -> list[CameraInfo]:
        return [
            CameraInfo(index=0, width=1280, height=720, fps=30.0),
            CameraInfo(index=2, width=1920, height=1080, fps=60.0),
        ]

    def test_returns_chosen_index(self):
        cams = self._cams()
        chosen = pick_camera_interactive(cams, input_fn=lambda _: "1")
        assert chosen == 2  # menu position 1 → camera index 2

    def test_reprompts_on_invalid_input(self):
        cams = self._cams()
        answers = iter(["foo", "99", "0"])
        chosen = pick_camera_interactive(cams, input_fn=lambda _: next(answers))
        assert chosen == 0

    def test_eof_propagates_keyboard_interrupt(self):
        cams = self._cams()

        def raise_eof(_):
            raise EOFError

        with pytest.raises(KeyboardInterrupt):
            pick_camera_interactive(cams, input_fn=raise_eof)

    def test_empty_camera_list_raises(self):
        with pytest.raises(RuntimeError):
            pick_camera_interactive([], input_fn=lambda _: "0")
