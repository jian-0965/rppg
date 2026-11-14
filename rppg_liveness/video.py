"""Local capture, face detection and three fixed proportional facial ROIs."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import time
import cv2
import numpy as np
from .core import Config, analyze, decide, rejected


def _timed_analysis(snapshot, config):
    start = time.perf_counter()
    result = analyze(*snapshot, config=config)
    result["analysis_seconds"] = time.perf_counter() - start
    return result


def _check_freshness(result, age):
    if age > 5:
        expired = rejected("analysis_too_slow", **result.get("quality", {}))
        expired["analysis_seconds"] = result.get("analysis_seconds")
        expired["diagnostics"] = {"expired_score": result.get("score"),
                                  "expired_features": result.get("features", {})}
        result = expired
    result["analysis_age_seconds"] = age
    return result


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
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or not 1 <= fps <= 240:
            return rejected("invalid_video_fps")
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        metadata_duration = (float(frame_count / fps) if np.isfinite(frame_count) and frame_count > 0 else None)
        sampler = FaceSampler()
        raw_times = []
        # Read past the nominal first window when timestamps are sparse or VFR.
        # Sampling is cheap (three color means), and the selected window remains
        # the first 15 seconds after timestamp validation below.
        max_frames = int(max(config.window_seconds * fps * 4, 300))
        for index in range(max_frames):
            ok, frame = capture.read()
            if not ok:
                break
            raw = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            raw_times.append(raw)
            color, box, _ = sampler.sample(frame)
            colors.append(color)
            boxes.append(box)
            if index > 5 and raw - raw_times[0] >= config.window_seconds:
                # A duplicate timestamp must not make us discard the rest of a
                # genuine 22-second clip on a 30/60 FPS metadata mismatch.
                observed_so_far = np.asarray(raw_times)
                if np.isfinite(observed_so_far).all() and np.all(np.diff(observed_so_far) >= 0) \
                        and (np.diff(observed_so_far) > 0).mean() >= 0.95:
                    break
        observed = np.asarray(raw_times, dtype=float)
        mode = "video_timestamps"
        if len(observed) >= 2 and np.isfinite(observed).all() and \
                np.all(np.diff(observed) >= 0) and \
                (np.diff(observed) > 0).mean() >= 0.95 and observed[-1] > observed[0]:
            # A few duplicate timestamps may be interpolated; one duplicate
            # must not trigger a permanent frame-number fallback.
            increasing = np.r_[True, np.diff(observed) > 0]
            times = np.interp(np.arange(len(observed)), np.flatnonzero(increasing), observed[increasing])
            times -= times[0]
            if not increasing.all():
                mode = "video_timestamps_interpolated"
        else:
            mode = "fps_fallback_assumes_CFR"
            times = np.arange(len(observed)) / fps
        if len(times):
            keep = np.searchsorted(times, config.window_seconds, side="right")
            times, colors, boxes = times[:keep], colors[:keep], boxes[:keep]
        result = analyze(times, colors, boxes, config)
        if mode == "fps_fallback_assumes_CFR" and result["reason"] == "too_short":
            result = rejected("unreliable_video_timestamps", **result["quality"])
        result["timestamp_mode"] = mode
        result["frames_read"] = len(times)
        result["total_frames_decoded"] = len(observed)
        result["nominal_fps"] = float(fps)
        result["metadata_frame_count"] = float(frame_count) if np.isfinite(frame_count) else None
        result["metadata_duration_seconds"] = metadata_duration
        result["last_raw_timestamp_seconds"] = float(observed[-1]) if len(observed) and np.isfinite(observed[-1]) else None
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
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rppg-analysis")
    pending, generation, result_time = None, 0, None
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
                generation += 1
                result = rejected("face_missing_or_multiple")
            while samples and now - samples[0][0] > config.window_seconds:
                samples.popleft()
            if pending is not None and pending[0].done():
                future, submitted_at, submitted_generation = pending
                computed = future.result()
                if submitted_generation == generation:
                    result = _check_freshness(computed, now - submitted_at)
                    result_time = submitted_at
                pending = None
            if result_time is not None and result["status"] == "ok":
                result = _check_freshness(result, now - result_time)
            if pending is None and now - last_analysis >= 2 and np.isfinite(color).all():
                snapshot = tuple(zip(*samples))
                pending = (executor.submit(_timed_analysis, snapshot, config), now, generation)
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
        executor.shutdown(wait=True, cancel_futures=True)
        if display:
            cv2.destroyAllWindows()
    return result
