"""Independent implementations of POS and CHROM color projections."""
import numpy as np
from functools import lru_cache
from scipy.signal import butter, sosfiltfilt, welch


@lru_cache(maxsize=32)
def _filter(fs):
    return butter(3, [0.7, 3.0], fs=fs, btype="bandpass", output="sos")


def bandpass(x, fs):
    """Linear detrend and filter along the last axis, also for batched windows."""
    x = np.asarray(x, dtype=float)
    axis = np.arange(x.shape[-1], dtype=float)
    axis -= axis.mean()
    centered = x - x.mean(axis=-1, keepdims=True)
    slope = (centered * axis).sum(axis=-1, keepdims=True) / (axis * axis).sum()
    return sosfiltfilt(_filter(float(fs)), centered - slope * axis, axis=-1)


def pulse(rgb, fs, method="pos"):
    """RGB is (time, 3), in R,G,B order; sliding 1.6 s overlap-add."""
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or not np.isfinite(rgb).all():
        raise ValueError("Expected finite RGB array of shape (time, 3)")
    if method not in {"pos", "chrom"} or not np.isfinite(fs) or fs < 10:
        raise ValueError("method must be pos/chrom and sampling rate >= 10 Hz")
    width = max(24, int(round(1.6 * fs)))
    if len(rgb) < max(width, 32):
        raise ValueError("Not enough color samples")
    # Preserve every original window and projection; batch filters instead of
    # redesigning and executing thousands of separate filters in Python.
    windows = np.moveaxis(np.lib.stride_tricks.sliding_window_view(rgb, width, axis=0), 1, 2)
    normalized = windows / np.maximum(windows.mean(axis=1, keepdims=True), 1e-8) - 1
    r, g, b = np.moveaxis(normalized, -1, 0)
    if method == "pos":
        x, y = g - b, g + b - 2 * r
        h = x + x.std(axis=-1, keepdims=True) / np.maximum(y.std(axis=-1, keepdims=True), 1e-8) * y
    else:
        x, y = bandpass(3 * r - 2 * g, fs), bandpass(1.5 * r + g - 1.5 * b, fs)
        h = x - x.std(axis=-1, keepdims=True) / np.maximum(y.std(axis=-1, keepdims=True), 1e-8) * y
    h -= h.mean(axis=-1, keepdims=True)
    out, counts = np.zeros(len(rgb)), np.zeros(len(rgb))
    for offset in range(width):
        out[offset:offset + len(h)] += h[:, offset]
        counts[offset:offset + len(h)] += 1
    return bandpass(out / np.maximum(counts, 1), fs)


def spectrum(x, fs):
    """Peak and concentration in the 42–180 BPM analysis band; not clinical SNR."""
    x = np.asarray(x)
    if np.std(x) < 1e-8:
        return {"bpm": None, "concentration": 0.0, "snr_db": -60.0}
    freq, power = welch(x, fs=fs, nperseg=min(len(x), round(8 * fs)), nfft=max(2048, len(x)))
    band = (freq >= 0.7) & (freq <= 3.0)
    peak = freq[band][np.argmax(power[band])]
    near = band & (np.abs(freq - peak) <= 0.15)
    ratio = float(np.clip(power[near].sum() / max(power[band].sum(), 1e-20), 0, 1))
    return {"bpm": float(60 * peak), "concentration": ratio,
            "snr_db": float(10 * np.log10(max(ratio, 1e-6) / max(1 - ratio, 1e-6)))}
