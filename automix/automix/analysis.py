"""Track analysis: tempo, beat grid, downbeats, key, energy."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import librosa
import numpy as np

from .keys import estimate_key

SR = 44100
HOP = 512


@dataclass
class TrackAnalysis:
    path: str
    duration: float
    bpm: float
    first_downbeat: float  # seconds
    beat_period: float  # seconds
    key: str
    camelot: str
    key_confidence: float
    bars: int
    bar_energy: list[float] = field(default_factory=list)

    def downbeat(self, bar: int) -> float:
        return self.first_downbeat + bar * 4 * self.beat_period

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bar_energy"] = [round(x, 3) for x in self.bar_energy]
        return d


def fold_tempo(bpm: float, lo: float, hi: float) -> float:
    """Double/halve a tempo estimate into [lo, hi] (DnB is often detected at ~87)."""
    if bpm <= 0:
        return bpm
    while bpm < lo:
        bpm *= 2
    while bpm > hi:
        bpm /= 2
    return bpm


def _refine_grid(onset_env: np.ndarray, sr: int, bpm: float, search: float = 0.6) -> tuple[float, float]:
    """Fit a constant-tempo grid: return (bpm, phase seconds) maximising onset energy on beats."""
    fps = sr / HOP
    t_env = np.arange(len(onset_env)) / fps
    best = (-1.0, bpm, 0.0)
    for cand in np.arange(bpm - search, bpm + search + 1e-9, 0.02):
        period = 60.0 / cand
        # Phase search on a fine grid within one beat.
        phases = np.arange(0, period, 0.002)
        for ph in phases:
            idx = np.round((np.arange(ph, t_env[-1], period)) * fps).astype(int)
            idx = idx[idx < len(onset_env)]
            score = float(onset_env[idx].mean()) if len(idx) else 0.0
            if score > best[0]:
                best = (score, cand, ph)
    return best[1], best[2]


def _attack_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, 160, btype="lowpass", fs=sr, output="sos")
    env = np.abs(sosfiltfilt(sos, y))
    win = int(0.002 * sr)
    return np.convolve(env, np.ones(win) / win, mode="same")


def _fit_downbeats(y: np.ndarray, sr: int, first: float, period: float, with_score: bool = False):
    """Refine the downbeat grid to sample accuracy.

    For every bar, find the kick attack (first point where the low-band envelope
    reaches half of its local peak) within +-60 ms of the predicted downbeat, then
    fit time = intercept + slope * bar with outlier rejection.
    """
    env = _attack_envelope(y, sr)
    win = int(0.06 * sr)
    # Allow a downbeat in the first 60 ms (or slightly before 0 due to estimation error).
    while first - 4 * period > -0.06:
        first -= 4 * period
    bars: list[int] = []
    times: list[float] = []
    peaks_found: list[float] = []
    icpt, slope = first, 4 * period
    k = 0
    while True:
        # Track bar by bar: predict from the fit so far, so a slightly wrong
        # initial tempo cannot walk the search window off the kick.
        t = icpt + slope * k
        c = int(round(t * sr))
        if c + win >= len(env):
            break
        lo = max(0, c - win)
        seg = env[lo : c + win]
        peak = seg.max()
        if peak > 3 * np.median(env[max(0, c - 4 * win) : c + 4 * win]) + 1e-6:
            ip = int(np.argmax(seg))
            above = np.nonzero(seg[: ip + 1] >= 0.5 * peak)[0]
            onset = (lo + int(above[0])) / sr if len(above) else t
            bars.append(k)
            times.append(onset)
            peaks_found.append(float(peak))
            if len(bars) >= 3:
                b, tt = np.array(bars, float), np.array(times)
                slope, icpt = np.polyfit(b, tt, 1)
                resid = np.abs(tt - (icpt + slope * b))
                keep = resid < max(0.004, 3 * np.median(resid))
                if not keep.all() and keep.sum() >= 3:
                    slope, icpt = np.polyfit(b[keep], tt[keep], 1)
        k += 1
    if len(bars) < 4:
        return (first, period, 0.0) if with_score else (first, period)
    peaks = {k_: p_ for k_, p_ in zip(bars, peaks_found)}
    b, tt = np.array(bars, float), np.array(times)
    for _ in range(3):  # final outlier rejection over all bars
        slope, icpt = np.polyfit(b, tt, 1)
        resid = np.abs(tt - (icpt + slope * b))
        keep = resid < max(0.004, 3 * np.median(resid))
        if keep.all() or keep.sum() < 4:
            break
        b, tt = b[keep], tt[keep]
    slope, icpt = np.polyfit(b, tt, 1)
    # Report the first bar where a kick was actually found (not a grid point in leading silence).
    first_found = icpt + slope * b[0]
    if not with_score:
        return float(first_found), float(slope / 4)
    # Score: how many bars carry a strong, well-aligned attack (kicks on "1" score highest).
    resid = np.abs(tt - (icpt + slope * b))
    score = float(sum(peaks[int(k_)] for k_, r_ in zip(b, resid) if r_ < 0.004))
    return float(first_found), float(slope / 4), score


def analyze(path: str | Path, bpm_range: tuple[float, float] = (160.0, 190.0), bpm_hint: float | None = None) -> TrackAnalysis:
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    duration = len(y) / sr
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)

    if bpm_hint:
        bpm0 = bpm_hint
    else:
        tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=HOP, start_bpm=sum(bpm_range) / 2)
        bpm0 = fold_tempo(float(np.atleast_1d(tempo)[0]), *bpm_range)
    # Coarse grid first (fast), then a fine local refinement around it.
    bpm1, _ = _refine_grid(onset_env, sr, bpm0, search=2.0) if not bpm_hint else (bpm0, 0.0)
    bpm, phase = _refine_grid(onset_env, sr, bpm1, search=0.1)
    period = 60.0 / bpm

    fps = sr / HOP
    # Sample-accurate grid: fit each of the 4 beat phases as a candidate "1" and keep the
    # one whose bars carry the strongest aligned low-end attacks (the kick). Doing this
    # after fitting avoids picking a wrong beat when the coarse tempo is slightly off.
    best = None
    for k in range(4):
        cand = _fit_downbeats(y, sr, phase + k * period, period, with_score=True)
        if best is None or cand[2] > best[2]:
            best = cand
    first_downbeat, period, _ = best
    bpm = 60.0 / period

    # Key from harmonic content from C2 up, so the kick sub (~45 Hz) does not skew it.
    y_h = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=HOP, fmin=librosa.note_to_hz("C2"), n_octaves=6, norm=None)
    # Weight frames by harmonic energy so drum-only intros/outros barely count.
    h_rms = librosa.feature.rms(y=y_h, hop_length=HOP)[0][: chroma.shape[1]]
    w = h_rms**2
    chroma_w = (chroma[:, : len(w)] * w).sum(axis=1) / (w.sum() + 1e-12)
    key, camelot, conf = estimate_key(chroma_w)

    bars = int((duration - first_downbeat) // (4 * period))
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    bar_energy = []
    for b in range(bars):
        a = max(0, int((first_downbeat + b * 4 * period) * fps))
        e = max(0, int((first_downbeat + (b + 1) * 4 * period) * fps))
        bar_energy.append(float(rms[a:e].mean()) if e > a else 0.0)
    if bar_energy:
        m = max(bar_energy) or 1.0
        bar_energy = [x / m for x in bar_energy]

    return TrackAnalysis(
        path=str(path), duration=duration, bpm=round(bpm, 2), first_downbeat=first_downbeat,
        beat_period=period, key=key, camelot=camelot, key_confidence=round(conf, 2), bars=bars,
        bar_energy=bar_energy,
    )
