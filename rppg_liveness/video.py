"""Local capture, face detection and three fixed proportional facial ROIs."""
from collections import deque
import time
import cv2
import numpy as np
from .core import Config, analyze, decide, rejected


class FaceSampler:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if self.detector.empty():
            raise RuntimeError("OpenCV bundled face cascade is unavailable")
        self.previous = None

    def sample(self, frame):
        height, width = frame.shape[:2]
        scale = min(1.0, 480 / width)
        gray = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                               minSize=(max(24, int(80 * scale)),) * 2)
        if len(faces) != 1:
            self.previous = None
            return np.full((3, 3), np.nan), np.full(4, np.nan), []
        raw = np.asarray(faces[0], float) / scale
        # Return raw coordinates for motion gating; smooth only sampling regions.
        box = raw if self.previous is None else 0.65 * self.previous + 0.35 * raw
        self.previous = box
        x, y, w, h = box
        regions, colors = [], []
        for left, top, right, bottom in [(0.30, 0.16, 0.70, 0.30),
                                          (0.16, 0.52, 0.38, 0.72), (0.62, 0.52, 0.84, 0.72)]:
            x0, y0 = max(0, int(x + w * left)), max(0, int(y + h * top))
            x1, y1 = min(width, int(x + w * right)), min(height, int(y + h * bottom))
            patch = frame[y0:y1, x0:x1]
            if patch.size == 0:
                return np.full((3, 3), np.nan), raw, []
            # Exclude clipped/near-black pixels; no fixed skin-color threshold.
            pixels = patch.reshape(-1, 3)
            good = np.all((pixels > 10) & (pixels < 245), axis=1)
            if good.mean() < 0.7:
                return np.full((3, 3), np.nan), raw, []
            colors.append(pixels[good].mean(axis=0)[::-1])
            regions.append((x0, y0, x1, y1))
        return np.asarray(colors), raw, regions


def analyze_video(path, config=Config()):
    """Deterministic first window, never select the best-scoring segment."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return rejected("video_open_failed")
    times, colors, boxes = [], [], []
    timestamp_fallback = False
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or not 1 <= fps <= 240:
            return rejected("invalid_video_fps")
        sampler = FaceSampler()
        origin, previous_raw = None, None
        for index in range(int(config.window_seconds * fps) + 1):
            ok, frame = capture.read()
            if not ok:
                break
            raw = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            if origin is None:
                origin = raw if np.isfinite(raw) else 0.0
            if not np.isfinite(raw) or (previous_raw is not None and raw <= previous_raw):
                timestamp_fallback = True
            previous_raw = raw
            timestamp = index / fps if timestamp_fallback else raw - origin
            color, box, _ = sampler.sample(frame)
            times.append(timestamp)
            colors.append(color)
            boxes.append(box)
            if not timestamp_fallback and timestamp >= config.window_seconds:
                break
        if timestamp_fallback:
            times = np.arange(len(times)) / fps
        result = analyze(times, colors, boxes, config)
        result["timestamp_mode"] = "fps_fallback_assumes_CFR" if timestamp_fallback else "video_timestamps"
        result["frames_read"] = len(times)
        return result
    finally:
        capture.release()


def camera(index=0, threshold=0.55, seconds=0, display=True, config=Config()):
    """Rolling window. Camera timestamps come from acquisition wall clock."""
    capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open camera {index}; check device permissions")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, 30)
    sampler, samples = FaceSampler(), deque()
    start, last_analysis = time.monotonic(), -1e9
    result = rejected("collecting_frames")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                result = rejected("camera_read_failed")
                break
            now = time.monotonic() - start
            color, box, regions = sampler.sample(frame)
            samples.append((now, color, box))
            if not np.isfinite(color).all():
                result = rejected("face_missing_or_multiple")
            while samples and now - samples[0][0] > config.window_seconds:
                samples.popleft()
            if now - last_analysis >= 2 and np.isfinite(color).all():
                result = analyze(*zip(*samples), config=config)
                last_analysis = now
                if not display:
                    print(f"{now:.1f}s {decide(result, threshold)} score={result['score']} reason={result['reason']}")
            if display:
                for x0, y0, x1, y1 in regions:
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (40, 220, 220), 1)
                lines = [f"{decide(result, threshold)} | score: {result['score'] if result['score'] is not None else '--'}",
                         f"BPM estimate: {round(result['bpm'], 1) if result['bpm'] else '--'} | {now:.1f}s",
                         result['reason'] or 'Pulse evidence only; replay may pass', 'Q / ESC: quit']
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, (12, 25 + 25 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
                cv2.imshow("rPPG Liveness Lab - experimental", frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
            if seconds > 0 and now >= seconds:
                break
    finally:
        capture.release()
        if display:
            cv2.destroyAllWindows()
    return result
