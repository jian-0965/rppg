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
