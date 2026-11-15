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


def _scan_tempo(onset_env: np.ndarray, sr: int, lo: float, hi: float, step: float = 0.05) -> float:
    """Tempo in [lo, hi] whose best-phase beat grid carries the most onset energy."""
    fps = sr / HOP
    t_end = len(onset_env) / fps
    best = (-1.0, (lo + hi) / 2)
    for cand in np.arange(lo, hi + 1e-9, step):
        period = 60.0 / cand
        n_beats = int((t_end - period) / period)
        if n_beats < 4:
            continue
        base = np.arange(n_beats) * period
        phases = np.arange(0, period, 0.004)
        idx = np.round((phases[:, None] + base[None, :]) * fps).astype(int)
        idx = np.clip(idx, 0, len(onset_env) - 1)
        score = float(onset_env[idx].mean(axis=1).max())
        if score > best[0]:
            best = (score, cand)
    return best[1]


def _attack_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    """Rise of the low-band (< 160 Hz) envelope: large where a kick starts."""
    from scipy.signal import butter, sosfiltfilt

    env = _low_envelope(y, sr)
    return np.diff(env, prepend=env[0]).clip(min=0)


def _low_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    # 400 Hz keeps the kick's initial (higher-pitched) click; a lower cutoff sees the
    # attack late because kick pitch sweeps downward.
    sos = butter(4, 400, btype="lowpass", fs=sr, output="sos")
    env = np.abs(sosfiltfilt(sos, y))
    win = int(0.002 * sr)
    return np.convolve(env, np.ones(win) / win, mode="same")


FINE_HOP = 32  # ~0.7 ms at 44.1 kHz
FLUX_LAG = 0.003  # seconds


def transient_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    """Broadband spectral flux at ~0.7 ms resolution: sharp at kick and snare transients,
    unaffected by sustained (even heavily distorted) bass."""
    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=FINE_HOP, n_fft=1024, center=True)


def _fine_phase(y: np.ndarray, sr: int, grid0: float, period: float, window: float = 0.02) -> float:
    """Shift (s) of a beat grid that maximises mean transient energy on every beat."""
    flux = transient_envelope(y, sr)
    fps = sr / FINE_HOP
    n_beats = int((len(y) / sr - grid0 - window) // period)
    if n_beats < 8:
        return 0.0
    beats = grid0 + np.arange(n_beats) * period
    offsets = np.arange(-window, window + 1e-9, 0.00025)
    pos = (beats[None, :] + offsets[:, None]) * fps
    pos = np.clip(pos, 0, len(flux) - 1.001)
    i0 = np.floor(pos).astype(int)
    frac = pos - i0
    vals = flux[i0] * (1 - frac) + flux[i0 + 1] * frac
    # Spectral flux peaks ~3 ms after a transient starts (measured on synthetic kicks with
    # known onsets); remove that constant lag so downbeats mark the attack start.
    return float(offsets[int(np.argmax(vals.mean(axis=1)))]) - FLUX_LAG


def _onset_offset(env: np.ndarray, sr: int, grid: np.ndarray) -> float:
    """Median offset (s) of the kick attack start relative to grid points.

    Per bar: peak of the envelope within -10..+25 ms of the grid point, then the first
    sample (from -10 ms) reaching half of that peak. The window is far tighter than a
    16th note, so off-beat bass stabs cannot land in it.
    """
    pre, post = int(0.010 * sr), int(0.025 * sr)
    offs, peaks = [], []
    for g in grid:
        c = int(round(g * sr))
        if c - pre < 0 or c + post >= len(env):
            continue
        seg = env[c - pre : c + post]
        ip = int(np.argmax(seg))
        pk = seg[ip]
        base = seg[: max(1, ip)].min()
        if pk <= base * 1.5 + 1e-9:
            continue
        above = np.nonzero(seg[: ip + 1] >= base + 0.5 * (pk - base))[0]
        offs.append((int(above[0]) - pre) / sr)
        peaks.append(pk)
    if len(offs) < 3:
        return 0.0
    offs, peaks = np.array(offs), np.array(peaks)
    strong = peaks >= 0.5 * np.median(peaks)
    return float(np.median(offs[strong])) if strong.sum() >= 3 else float(np.median(offs))


def _fit_downbeats(y: np.ndarray, sr: int, first: float, period: float, with_score: bool = False,
                   window: float = 0.03, tempo_search: float = 0.0, phases: int = 1):
    """Sample-accurate downbeat grid by aligning bar positions with low-band attacks.

    Searches a phase offset within +-``window`` seconds of ``first`` (and, with
    ``phases=4``, each of the four beats as the candidate "1"; with
    ``tempo_search``, tempo within +-that many BPM) for the grid whose bar
    positions carry the highest *average* kick-attack energy over the whole file.
    Averaging over every bar is what keeps loud, syncopated bass stabs (jump up,
    neuro) from posing as kicks: they do not repeat exactly on the downbeat.

    Returns (first_downbeat, period[, score]); first_downbeat is the first bar
    where a kick is actually present, not a grid point in leading silence.
    """
    att = _attack_envelope(y, sr)
    n = len(att)
    bpm0 = 60.0 / period
    tempi = [bpm0] if tempo_search <= 0 else list(np.arange(bpm0 - tempo_search, bpm0 + tempo_search + 1e-9, 0.002))
    offsets = np.arange(-window, window + 1e-9, 0.0005)
    best = (-1.0, first, period)
    for bpm in tempi:
        per = 60.0 / bpm
        bar = 4 * per
        for k in range(phases):
            base = first + k * per
            base -= np.floor((base + window) / bar) * bar  # earliest grid point near the file start
            nb = int((n / sr - base - window) // bar)
            if nb < 2:
                continue
            grid = base + np.arange(nb) * bar
            # Kick attack energy, max within +-1 ms of each grid point, averaged over all bars.
            idx = np.round((grid[None, :] + offsets[:, None]) * sr).astype(int)
            ok = (idx >= 0) & (idx < n - 45)
            vals = np.zeros(idx.shape)
            for j in range(-44, 45, 11):  # small tolerance window (+-1 ms)
                vals = np.maximum(vals, np.where(ok, att[np.clip(idx + j, 0, n - 1)], 0.0))
            scores = vals.sum(axis=1) / max(1, nb)
            i = int(np.argmax(scores))
            if scores[i] > best[0]:
                best = (float(scores[i]), float(base + offsets[i]), per)
    score, grid0, per = best
    bar = 4 * per
    # Sub-millisecond phase from broadband transients on every beat (kick and snare),
    # which stays precise even when the low end is full of distorted bass.
    grid0 += _fine_phase(y, sr, grid0, per)
    # First bar that actually carries a kick (skip leading silence / beatless intros).
    first_found = grid0
    nb = int((n / sr - grid0) // bar)
    if nb > 0:
        idx = np.clip(np.round((grid0 + np.arange(nb) * bar) * sr).astype(int), 0, n - 1)
        bar_n = int(bar * sr)
        # A bar "has a kick" when the attack at its downbeat stands well above that bar's
        # own typical attack level (so quiet drum-only intros count, silence does not).
        hits = []
        for b_, i in enumerate(idx):
            peak = att[max(0, i - 44) : i + 45].max()
            floor = np.median(att[i : min(n, i + bar_n)]) if i < n else 0.0
            if peak > 0 and peak >= 6 * floor + 1e-7:
                hits.append(b_)
        if hits:
            first_found = grid0 + hits[0] * bar
    if with_score:
        return float(first_found), float(per), float(score)
    return float(first_found), float(per)


def analyze(path: str | Path, bpm_range: tuple[float, float] = (160.0, 190.0), bpm_hint: float | None = None) -> TrackAnalysis:
    y, sr = librosa.load(str(path), sr=SR, mono=True)
    duration = len(y) / sr
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)

    if bpm_hint:
        bpm1 = bpm_hint
    else:
        # Scan the whole tempo range: librosa's own estimate can be off by a few BPM on
        # dense, distorted tracks, and the on-beat onset energy peaks very sharply at the true tempo.
        bpm1 = _scan_tempo(onset_env, sr, *bpm_range)
    bpm, phase = _refine_grid(onset_env, sr, bpm1, search=0.1)
    period = 60.0 / bpm

    fps = sr / HOP
    # Sample-accurate grid: fit each of the 4 beat phases as a candidate "1" and keep the
    # one whose bars carry the strongest aligned low-end attacks (the kick). Doing this
    # after fitting avoids picking a wrong beat when the coarse tempo is slightly off.
    # Which beat is "1", the exact phase and a final tempo touch-up, all from one
    # alignment of the bar grid with low-band attacks averaged over the whole file.
    first_downbeat, period = _fit_downbeats(y, sr, phase, period, window=0.03, tempo_search=0.04, phases=4)
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
