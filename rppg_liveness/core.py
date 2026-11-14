"""Quality-gated pulse-evidence score. Weights are a heuristic, not trained."""
from dataclasses import asdict, dataclass
from itertools import combinations
import numpy as np
from .signal import pulse, spectrum


@dataclass(frozen=True)
class Config:
    window_seconds: float = 15.0
    min_seconds: float = 10.0
    min_fps: float = 15.0
    min_coverage: float = 0.90
    max_gap_seconds: float = 0.25
    max_motion: float = 0.025
    max_brightness_cv: float = 0.12

    def __post_init__(self):
        if self.min_seconds < 8 or self.window_seconds < self.min_seconds or self.min_fps < 10:
            raise ValueError("Require window >= min_seconds >= 8, min_fps >= 10")


def rejected(reason, **quality):
    return {"status": "insufficient_quality", "reason": reason, "score": None,
            "bpm": None, "quality": quality, "features": {}}


def analyze(times, colors, boxes, config=Config()):
    """Missing detections are NaN. colors: (frames, 3 ROIs, 3 RGB)."""
    t, c, b = np.asarray(times, float), np.asarray(colors, float), np.asarray(boxes, float)
    if len(t) < 3:
        return rejected("too_few_frames")
    if c.shape != (len(t), 3, 3) or b.shape != (len(t), 4):
        raise ValueError("Trace dimensions do not match")
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        return rejected("invalid_timestamps")
    duration = float(t[-1] - t[0])
    if duration < config.min_seconds:
        return rejected("too_short", duration=duration)
    valid = np.isfinite(c).all(axis=(1, 2)) & np.isfinite(b).all(axis=1)
    coverage = float(valid.mean())
    if coverage < config.min_coverage or valid.sum() < 3:
        return rejected("face_missing_or_multiple", coverage=coverage)
    tv, cv, bv = t[valid], c[valid], b[valid]
    max_gap = float(max(np.diff(tv).max(), tv[0] - t[0], t[-1] - tv[-1]))
    fs = float((len(tv) - 1) / (tv[-1] - tv[0]))
    if max_gap > config.max_gap_seconds or fs < config.min_fps:
        return rejected("low_fps_or_detection_gap", fps=fs, max_gap=max_gap, coverage=coverage)
    # Normalize motion to face size and elapsed time, expressed per nominal 30 Hz frame.
    delta = np.linalg.norm(np.diff(bv, axis=0), axis=1) / np.maximum(bv[:-1, 2], 1)
    motion = float(np.percentile(delta / np.diff(tv) / 30, 95))
    brightness = cv.mean(axis=(1, 2))
    light_cv = float(brightness.std() / max(brightness.mean(), 1e-8))
    quality = dict(duration=duration, fps=fs, coverage=coverage, max_gap=max_gap,
                   motion=motion, brightness_cv=light_cv, mean_brightness=float(brightness.mean()))
    if motion > config.max_motion:
        return rejected("excessive_motion", **quality)
    if light_cv > config.max_brightness_cv or not 35 <= brightness.mean() <= 230:
        return rejected("unstable_or_extreme_lighting", **quality)
    # Resample timestamps, including small gaps only; long gaps were rejected above.
    grid = np.linspace(tv[0], tv[-1], round((tv[-1] - tv[0]) * fs) + 1)
    fs = (len(grid) - 1) / (grid[-1] - grid[0])
    traces = np.stack([np.stack([np.interp(grid, tv, cv[:, roi, ch]) for ch in range(3)], axis=1)
                       for roi in range(3)])
    pos = [pulse(trace, fs, "pos") for trace in traces]
    chrom = [pulse(trace, fs, "chrom") for trace in traces]
    # Discard overlap-add/filter edge transients.
    trim = round(fs)
    pos, chrom = [x[trim:-trim] for x in pos], [x[trim:-trim] for x in chrom]
    specs, alt = [spectrum(x, fs) for x in pos], [spectrum(x, fs) for x in chrom]
    bpms = [s["bpm"] for s in specs]
    if any(v is None for v in bpms) or any(s["bpm"] is None for s in alt):
        return rejected("no_measurable_color_variation", **quality)
    peak = float(np.mean([s["concentration"] for s in specs]))
    agreement = float(np.exp(-np.std(bpms) / 10))
    method = float(np.exp(-np.mean([abs(a["bpm"] - z["bpm"]) for a, z in zip(specs, alt)]) / 10))
    correlation = float(np.mean([abs(np.corrcoef(a, z)[0, 1]) for a, z in combinations(pos, 2)]))
    halves = [[spectrum(x[:len(x)//2], fs)["bpm"], spectrum(x[len(x)//2:], fs)["bpm"]] for x in pos]
    stability = float(np.exp(-np.mean([abs(a - z) if a is not None and z is not None else 180
                                     for a, z in halves]) / 12))
    features = dict(spectral_concentration=peak, roi_bpm_agreement=agreement,
                    roi_correlation=correlation, method_agreement=method, temporal_stability=stability)
    # Multiplicative gate avoids awarding high scores for mutually consistent noise.
    score = peak * (0.30 * agreement + 0.25 * correlation + 0.25 * method + 0.20 * stability)
    return {"status": "ok", "reason": None, "score": float(np.clip(score, 0, 1)),
            "bpm": float(np.median(bpms)), "quality": quality, "features": features,
            "roi_bpm": bpms, "roi_snr_db": [s["snr_db"] for s in specs]}


def decide(result, threshold=0.55):
    if result["status"] != "ok":
        return "inconclusive"
    return "live_candidate" if result["score"] >= threshold else "attack_candidate"


def config_dict():
    return asdict(Config())
