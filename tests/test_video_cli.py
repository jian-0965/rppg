import json
import cv2
import numpy as np
from rppg_liveness.cli import main
from rppg_liveness.video import FaceSampler, analyze_video


def test_missing_video(tmp_path):
    assert main(["video", str(tmp_path / "missing.mp4")]) == 2


def test_no_display_requires_duration():
    assert main(["camera", "--no-display"]) == 2


def test_invalid_threshold_before_capture():
    assert main(["camera", "--threshold", "nan"]) == 2


def test_blank_video_is_inconclusive(tmp_path):
    path = tmp_path / "blank.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (160, 120))
    assert writer.isOpened()
    for _ in range(330):
        writer.write(np.full((120, 160, 3), 120, np.uint8))
    writer.release()
    output = tmp_path / "result.json"
    assert main(["video", str(path), "--output", str(output)]) == 0
    result = json.loads(output.read_text())
    assert result["decision"] == "inconclusive"
    assert result["reason"] == "face_missing_or_multiple"
    assert result["threshold_source"] == "unvalidated_manual_or_default"


def test_three_roi_rgb_channel_order():
    class Detector:
        def detectMultiScale(self, *args, **kwargs):
            return np.array([[10, 10, 100, 100]])
    sampler = FaceSampler()
    sampler.detector = Detector()
    frame = np.full((150, 150, 3), [80, 110, 160], np.uint8)
    colors, box, regions = sampler.sample(frame)
    np.testing.assert_allclose(colors, [[160, 110, 80]] * 3)
    assert len(regions) == 3


def test_multiple_faces_rejected():
    class Detector:
        def detectMultiScale(self, *args, **kwargs):
            return np.array([[0, 0, 90, 90], [100, 0, 90, 90]])
    sampler = FaceSampler()
    sampler.detector = Detector()
    colors, _, _ = sampler.sample(np.full((150, 250, 3), 100, np.uint8))
    assert np.isnan(colors).all()


def test_camera_keeps_capturing_during_analysis(monkeypatch):
    import threading
    from rppg_liveness import video
    advanced = threading.Event()
    observations = []

    class Capture:
        count = 0
        released = False
        def isOpened(self):
            return True
        def set(self, *args):
            pass
        def read(self):
            self.count += 1
            if self.count >= 3:
                advanced.set()
            return True, np.full((120, 160, 3), 100, np.uint8)
        def release(self):
            self.released = True

    class Sampler:
        def sample(self, frame):
            return np.full((3, 3), 100.), np.array([0, 0, 100, 100]), []

    capture = Capture()
    monkeypatch.setattr(video.cv2, "VideoCapture", lambda index: capture)
    monkeypatch.setattr(video, "FaceSampler", Sampler)
    monkeypatch.setattr(video.time, "monotonic", lambda: capture.count * 0.1)

    def slow_analysis(*args, **kwargs):
        # Synchronous analysis would wait here before frame 3 could be acquired.
        observations.append(advanced.wait(timeout=2))
        return dict(status="ok", reason=None, score=0.9, bpm=72)

    monkeypatch.setattr(video, "analyze", slow_analysis)
    video.camera(seconds=1, display=False)
    assert observations and all(observations)
    assert capture.count >= 10 and capture.released
