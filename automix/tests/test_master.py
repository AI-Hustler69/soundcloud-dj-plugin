"""Mastering: loudness measurement (BS.1770) and the true-peak limiter, on synthetic signals."""
from __future__ import annotations

import numpy as np
import pytest

from automix.master import integrated_lufs, limit, master, true_peak

SR = 44100


def _sine(freq, amp, seconds=5.0, stereo=True):
    t = np.arange(int(seconds * SR)) / SR
    s = amp * np.sin(2 * np.pi * freq * t)
    return np.stack([s, s if stereo else np.zeros_like(s)], axis=1)


def test_lufs_matches_bs1770_reference_points():
    # BS.1770: a 1 kHz sine at 0 dBFS in one channel reads -3.01 LKFS; in both channels 0 LKFS.
    assert integrated_lufs(_sine(1000, 0.1)) == pytest.approx(-20.0, abs=0.05)
    assert integrated_lufs(_sine(1000, 0.1, stereo=False)) == pytest.approx(-23.01, abs=0.05)
    # K-weighting: lows count less, highs more (values cross-checked with pyloudnorm, within 0.05 dB).
    assert integrated_lufs(_sine(100, 0.1)) == pytest.approx(-21.86, abs=0.1)
    assert integrated_lufs(_sine(10000, 0.1)) == pytest.approx(-16.69, abs=0.1)


def test_lufs_gates_silence():
    x = np.concatenate([_sine(1000, 0.1, 5), np.zeros((5 * SR, 2))])
    # silence is gated out, not averaged in (only blocks straddling the edge count, slightly lower)
    assert integrated_lufs(x) == pytest.approx(-20.0, abs=0.2)
    assert integrated_lufs(np.zeros((SR, 2))) == float("-inf")


def test_limiter_holds_the_true_peak_ceiling_including_inter_sample_peaks():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((10 * SR, 2)) * 0.3
    x[::7001] *= 5  # isolated spikes
    # fs/4 sine at 45 degrees: sample peaks 0.707, true peaks 1.0 (x 1.5)
    n = np.arange(SR)
    x[:SR, 0] += 1.5 * np.sin(2 * np.pi * n / 4 + np.pi / 4)
    y, red = limit(x, ceiling_db=-1.0)
    assert 20 * np.log10(true_peak(y)) <= -1.0 + 0.05
    assert red > 6
    # quiet passages are untouched once the release has recovered
    quiet = np.full((2 * SR, 2), 0.01)
    y2, _ = limit(np.concatenate([x, quiet]), ceiling_db=-1.0)
    assert np.allclose(y2[-SR:], 0.01, atol=1e-6)


def test_limiter_leaves_quiet_audio_alone():
    x = _sine(440, 0.1)
    y, red = limit(x, ceiling_db=-1.0)
    assert red == 0.0 and np.array_equal(y, x)


def test_master_hits_loudness_and_ceiling():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((20 * SR, 2)) * 0.05
    x[::5000] *= 6
    y, rep = master(x, target_lufs=-9.0, ceiling_db=-1.5)
    assert integrated_lufs(y) == pytest.approx(-9.0, abs=0.3)
    assert 20 * np.log10(true_peak(y)) <= -1.5 + 0.05
    assert rep.max_reduction_db > 0
    y2, rep2 = master(x, target_lufs=None, ceiling_db=-1.5)
    assert 20 * np.log10(true_peak(y2)) == pytest.approx(-1.5, abs=0.05) and rep2.max_reduction_db == 0.0


def test_master_does_not_crush_to_reach_an_impossible_target():
    from automix.master import MAX_REDUCTION_DB
    rng = np.random.default_rng(3)
    x = rng.standard_normal((10 * SR, 2)) * 0.01
    x[::4410] = 1.0  # sparse full-scale clicks over quiet noise: loud target needs huge reduction
    y, rep = master(x, target_lufs=-3.0, ceiling_db=-1.0)
    assert rep.max_reduction_db <= MAX_REDUCTION_DB + 0.5
    assert 20 * np.log10(true_peak(y)) <= -1.0 + 0.05


def test_master_handles_undefined_loudness_and_bad_samples():
    short = _sine(200, 0.05, seconds=0.3)  # shorter than one 400 ms loudness block
    y, rep = master(short, target_lufs=-9.0, ceiling_db=-1.5)
    assert rep.max_reduction_db == 0.0 and rep.target_lufs is None  # levelled by peak, not boosted 36 dB
    assert 20 * np.log10(true_peak(y)) == pytest.approx(-1.5, abs=0.05)
    silent, rep = master(np.zeros((SR, 2)))
    assert not silent.any() and rep.gain_db == 0.0
    bad = _sine(1000, 0.1, seconds=1)
    bad[100, 0] = np.nan
    with pytest.raises(ValueError):
        master(bad)
