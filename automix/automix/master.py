"""Mastering for the finished mix: loudness measurement, a true-peak limiter and the master stage.

- ``integrated_lufs``: integrated loudness after ITU-R BS.1770-4 (K-weighting, 400 ms blocks with
  75 % overlap, absolute gate at -70 LUFS and relative gate at -10 LU).
- ``limit``: an offline look-ahead brick-wall limiter. The gain is computed from a 4x oversampled
  peak envelope, so it also catches inter-sample peaks. Because the whole file is known, the look-ahead
  costs no latency: the attenuation needed at each sample is held over the look-ahead window and then
  averaged over the same window, which reaches full attenuation exactly at the peak (never after it).
  Release is linear in dB, computed with a running maximum (no per-sample Python loop).
- ``master``: gain to a target loudness, limit to a true-peak ceiling, one correction pass for the
  loudness the limiter took away, and a final true-peak check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import lfilter, resample_poly

from .stretch import SR

TRUE_PEAK_CEILING_DB = -1.5  # dBTP; leaves room for the MP3 encoder's overshoot
DEFAULT_TARGET_LUFS = -9.0  # loud like a club mix, but well below the masters of the tracks (-4 to -8 LUFS)
OVERSAMPLE = 4
MAX_REDUCTION_DB = 12.0  # deepest limiter gain reduction master() will use to reach the target


def _stereo(audio: np.ndarray) -> np.ndarray:
    return audio if audio.ndim == 2 else audio[:, None]


def true_peak(audio: np.ndarray, oversample: int = OVERSAMPLE, chunk: int = 1 << 20) -> float:
    """Inter-sample peak (linear) estimated by 4x polyphase oversampling, as in ITU-R BS.1770.

    Works in overlapping chunks so a long mix does not need a 4x copy in memory.
    """
    env = _peak_envelope(audio, oversample, chunk)
    return float(env.max()) if len(env) else 0.0


def _peak_envelope(audio: np.ndarray, oversample: int = OVERSAMPLE, chunk: int = 1 << 20) -> np.ndarray:
    """Per input sample: the largest absolute value of the oversampled signal around it (both channels)."""
    x = _stereo(audio)
    n = len(x)
    env = np.abs(x).max(axis=1).astype(np.float64)
    pad = 64  # covers the resampling filter's support
    for i in range(0, n, chunk):
        a, b = max(0, i - pad), min(n, i + chunk + pad)
        up = np.abs(resample_poly(x[a:b], oversample, 1, axis=0)).max(axis=1)
        lo, hi = (i - a) * oversample, (min(n, i + chunk) - a) * oversample
        part = up[lo:hi]
        m = len(part) // oversample
        if m:
            # each input sample owns the oversampled points from itself up to the next sample
            blk = part[: m * oversample].reshape(m, oversample).max(axis=1)
            env[i : i + m] = np.maximum(env[i : i + m], blk)
    return env


# ------------------------------------------------------------------------------------- loudness

def _k_weighting(sr: int):
    """The two BS.1770 pre-filter stages (shelf + high-pass), as (b, a) pairs for this sample rate."""
    # Coefficients derived from the analog prototypes (as in pyloudnorm / libebur128).
    f0, g, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = np.tan(np.pi * f0 / sr)
    vh = 10 ** (g / 20)
    vb = vh ** 0.4996667741545416
    a0 = 1 + k / q + k * k
    b1 = [(vh + vb * k / q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0]
    a1 = [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    f0, q = 38.13547087602444, 0.5003270373238773
    k = np.tan(np.pi * f0 / sr)
    a0 = 1 + k / q + k * k
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, 2 * (k * k - 1) / a0, (1 - k / q + k * k) / a0]
    return (b1, a1), (b2, a2)


def integrated_lufs(audio: np.ndarray, sr: int = SR) -> float:
    """Integrated loudness in LUFS (ITU-R BS.1770-4, gated). -inf for silence."""
    x = _stereo(np.asarray(audio, dtype=np.float64))
    (b1, a1), (b2, a2) = _k_weighting(sr)
    y = lfilter(b2, a2, lfilter(b1, a1, x, axis=0), axis=0)
    block, hop = int(round(0.4 * sr)), int(round(0.1 * sr))
    if len(y) < block:
        return float("-inf")
    p = np.cumsum(np.vstack([np.zeros((1, y.shape[1])), y * y]), axis=0)
    starts = np.arange(0, len(y) - block + 1, hop)
    ms = (p[starts + block] - p[starts]) / block  # mean square per block and channel
    z = ms.sum(axis=1)  # channel weights are 1.0 for L and R
    with np.errstate(divide="ignore"):
        lk = -0.691 + 10 * np.log10(z)
    z = z[lk > -70]
    if not len(z):
        return float("-inf")
    rel = -0.691 + 10 * np.log10(z.mean()) - 10
    with np.errstate(divide="ignore"):
        z = z[-0.691 + 10 * np.log10(z) > rel]
    return float(-0.691 + 10 * np.log10(z.mean()))


# --------------------------------------------------------------------------------------- limiter

def limit(audio: np.ndarray, ceiling_db: float = TRUE_PEAK_CEILING_DB, lookahead_ms: float = 1.5,
          release_db_per_s: float = 150.0, sr: int = SR) -> tuple[np.ndarray, float]:
    """Brick-wall limit ``audio`` so its true peak stays at or under ``ceiling_db``.

    Returns (limited audio, maximum gain reduction in dB).
    """
    x = _stereo(np.asarray(audio, dtype=np.float64))
    ceiling = 10 ** (ceiling_db / 20)
    env = _peak_envelope(x)
    with np.errstate(divide="ignore"):
        att = np.maximum(0.0, 20 * np.log10(np.maximum(env, 1e-12) / ceiling))  # dB of reduction needed
    if not att.any():
        return x.copy(), 0.0
    L = max(1, int(round(lookahead_ms * 1e-3 * sr)))
    # 1) hold: every requirement spreads over +-(L-1) samples (the part before it is the look-ahead);
    held = maximum_filter1d(att, size=2 * L - 1, mode="nearest")
    # 2) release: after that it falls back linearly in dB (running max of held + r*k, minus r*k);
    r = release_db_per_s / sr
    k = np.arange(len(held), dtype=np.float64)
    released = np.maximum.accumulate(held + r * k) - r * k
    # 3) smooth: a centred L-sample average. Every point it averages lies within L/2 < L samples of the
    #    centre, so each was held at >= the centre's requirement: full depth is reached at the peak.
    smooth = uniform_filter1d(released, size=L, mode="nearest")
    gain = 10 ** (-np.maximum(smooth, att) / 20)
    y = x * gain[:, None]
    return y, float(att.max())


@dataclass
class MasterReport:
    lufs: float
    true_peak_db: float
    gain_db: float  # make-up gain applied before the limiter
    max_reduction_db: float  # deepest limiter gain reduction
    target_lufs: float | None


def master(mix: np.ndarray, target_lufs: float | None = DEFAULT_TARGET_LUFS,
           ceiling_db: float = TRUE_PEAK_CEILING_DB) -> tuple[np.ndarray, MasterReport]:
    """Level and limit the finished mix.

    With ``target_lufs=None`` the mix is only scaled so its true peak sits at the ceiling (no limiting).
    """
    x = _stereo(np.asarray(mix, dtype=np.float64))
    if not np.isfinite(x).all():
        raise ValueError("master: the mix contains NaN or infinite samples")
    peak = true_peak(x)
    if peak == 0.0:  # digital silence: nothing to level
        return x.copy(), MasterReport(float("-inf"), float("-inf"), 0.0, 0.0, target_lufs)
    if target_lufs is not None and not np.isfinite(integrated_lufs(x)):
        target_lufs = None  # too short or below the -70 LUFS gate: loudness is undefined, level by peak only
    if target_lufs is None:
        g = 10 ** (ceiling_db / 20) / (true_peak(x) or 1.0)
        y = x * g
        return y, MasterReport(integrated_lufs(y), 20 * np.log10(true_peak(y) or 1e-12), 20 * np.log10(g), 0.0, None)
    # The deepest reduction equals (true peak + gain - ceiling), so the cap is a cap on the gain.
    gain_max = ceiling_db - 20 * np.log10(true_peak(x) or 1e-12) + MAX_REDUCTION_DB
    gain_db = min(target_lufs - integrated_lufs(x), gain_max)
    y, red = limit(x * 10 ** (gain_db / 20), ceiling_db)
    # The limiter takes some loudness away, more the harder it works: correct and limit again from the
    # original, a few times, but never push the limiter past MAX_REDUCTION_DB (that would only crush it).
    for _ in range(6):
        miss = target_lufs - integrated_lufs(y)
        if abs(miss) <= 0.1:
            break
        step = min(miss, gain_max - gain_db)
        if abs(step) <= 0.05:
            break
        gain_db += step
        y, red = limit(x * 10 ** (gain_db / 20), ceiling_db)
    tp = true_peak(y)
    if tp > 10 ** (ceiling_db / 20):  # safety net: interpolation between envelope points
        y *= 10 ** (ceiling_db / 20) / tp
    return y, MasterReport(round(integrated_lufs(y), 2), round(20 * np.log10(true_peak(y) or 1e-12), 2),
                           round(gain_db, 2), round(red, 2), target_lufs)
