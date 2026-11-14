import numpy as np
import pytest
from rppg_liveness.signal import pulse, spectrum
from rppg_liveness.core import analyze, decide


def trace(bpm=72, seconds=15, fps=30, noise=0.03):
    rng = np.random.default_rng(9)
    t = np.arange(round(seconds * fps) + 1) / fps
    wave = np.sin(2 * np.pi * bpm / 60 * t)
    base = np.array([160, 110, 80])
    colors = np.stack([base * (1 + wave[:, None] * np.array([0.002, 0.009, 0.004]))
                       + rng.normal(0, noise, (len(t), 3)) for _ in range(3)], axis=1)
    boxes = np.tile([100, 100, 200, 200], (len(t), 1))
    return t, colors, boxes


@pytest.mark.parametrize("method", ["pos", "chrom"])
@pytest.mark.parametrize("bpm", [48, 72, 105, 150])
def test_frequency_recovery(method, bpm):
    _, colors, _ = trace(bpm=bpm)
    signal = pulse(colors[:, 0], 30, method)[30:-30]
    assert abs(spectrum(signal, 30)["bpm"] - bpm) < 3


def test_coherent_pulse_and_replay_limitation():
    args = trace()
    result = analyze(*args)
    assert result["status"] == "ok"
    assert abs(result["bpm"] - 72) < 3
    assert result["score"] > 0.55
    # An identical replay retains color traces; a pulse-only baseline cannot distinguish it.
    assert analyze(*(a.copy() for a in args))["score"] == result["score"]


def test_flat_signal_never_accepted():
    t, c, b = trace()
    c[:] = [160, 110, 80]
    assert decide(analyze(t, c, b)) == "inconclusive"


def test_noise_score_below_coherent_pulse():
    t, c, b = trace()
    rng = np.random.default_rng(53)
    c = np.tile([160., 110., 80.], (len(t), 3, 1)) + rng.normal(0, 0.5, c.shape)
    assert analyze(t, c, b)["score"] < 0.55


@pytest.mark.parametrize("fault,reason", [
    ("short", "too_short"), ("missing", "face_missing_or_multiple"),
    ("gap", "low_fps_or_detection_gap"), ("order", "invalid_timestamps"),
    ("motion", "excessive_motion"), ("dark", "unstable_or_extreme_lighting")])
def test_quality_failures(fault, reason):
    t, c, b = trace()
    if fault == "short":
        t, c, b = t[:100], c[:100], b[:100]
    elif fault == "missing":
        c[:200] = np.nan
    elif fault == "gap":
        c[100:110] = np.nan
    elif fault == "order":
        t[50] = t[49]
    elif fault == "motion":
        b[::2, 0] += 100
    elif fault == "dark":
        c[:] *= 0.1
    assert analyze(t, c, b)["reason"] == reason


def test_jittered_timestamps():
    t, c, b = trace()
    t[1:-1] += np.random.default_rng(5).uniform(-0.003, 0.003, len(t)-2)
    result = analyze(t, c, b)
    assert result["status"] == "ok"
    assert abs(result["bpm"] - 72) < 3


def reference_pulse(rgb, fs, method):
    """Original scalar implementation with independent SciPy detrending."""
    from scipy.signal import butter, detrend, sosfiltfilt
    def filt(x):
        return sosfiltfilt(butter(3, [0.7, 3], fs=fs, btype="bandpass", output="sos"), detrend(x))
    width = max(24, round(1.6 * fs))
    out, counts = np.zeros(len(rgb)), np.zeros(len(rgb))
    for end in range(width, len(rgb) + 1):
        w = rgb[end-width:end]
        r, g, b = (w / np.maximum(w.mean(axis=0), 1e-8) - 1).T
        if method == "pos":
            x, y = g-b, g+b-2*r
            h = x + x.std() / max(y.std(), 1e-8) * y
        else:
            x, y = filt(3*r-2*g), filt(1.5*r+g-1.5*b)
            h = x - x.std() / max(y.std(), 1e-8) * y
        out[end-width:end] += h - h.mean()
        counts[end-width:end] += 1
    return filt(out / np.maximum(counts, 1))


@pytest.mark.parametrize("method", ["pos", "chrom"])
@pytest.mark.parametrize("fps", [15, 29.97, 60])
def test_batched_algorithm_matches_scalar_reference(method, fps):
    _, colors, _ = trace(bpm=87, seconds=10, fps=fps, noise=0.3)
    rgb = colors[:, 0]
    np.testing.assert_allclose(pulse(rgb, fps, method), reference_pulse(rgb, fps, method),
                               rtol=1e-8, atol=1e-10)
