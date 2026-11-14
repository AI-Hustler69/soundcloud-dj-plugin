"""Plan and render a beat-matched DJ mix from analysed tracks."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, correlate, sosfiltfilt

from .analysis import TrackAnalysis, _fit_downbeats
from .keys import compatibility
from .stretch import SR, load_stereo, time_stretch

CROSSOVER_HZ = 180.0
MAX_STRETCH = 0.08  # warn above +-8 %


@dataclass
class TrackInput:
    path: str
    title: str
    analysis: TrackAnalysis
    camelot: str = ""  # from metadata (e.g. SoundCloud key_signature), overrides the estimate
    license: str = ""


@dataclass
class PlacedTrack:
    title: str
    start: float  # seconds in the mix where this track's audio begins
    first_downbeat: float  # seconds in the mix of its first downbeat
    end: float  # seconds in the mix where it is fully faded out
    bpm_original: float
    stretch_pct: float
    camelot: str
    camelot_source: str
    camelot_estimated: str
    license: str


@dataclass
class Transition:
    from_title: str
    to_title: str
    start: float  # incoming track's first downbeat lands here
    bass_swap: float
    end: float
    bars: int
    key_relation: str
    measured_offset_ms: float | None = None


@dataclass
class MixResult:
    audio: np.ndarray  # (n, 2)
    bpm: float
    tracks: list[PlacedTrack] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return len(self.audio) / SR


def _split(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = butter(4, CROSSOVER_HZ, btype="lowpass", fs=SR, output="sos")
    hi = butter(4, CROSSOVER_HZ, btype="highpass", fs=SR, output="sos")
    return sosfiltfilt(lo, audio, axis=0), sosfiltfilt(hi, audio, axis=0)


def _rms_loud(audio: np.ndarray) -> float:
    mono = audio.mean(axis=1)
    frame = SR // 2
    n = len(mono) // frame
    if n == 0:
        return float(np.sqrt(np.mean(mono**2)) + 1e-9)
    r = np.sqrt((mono[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    return float(np.mean(np.sort(r)[n // 2 :]) + 1e-9)  # loudest half


def _eq_power(x: np.ndarray, rising: bool) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.sin(x * np.pi / 2) if rising else np.cos(x * np.pi / 2)


def _ramp(n: int, a: int, b: int, rising: bool) -> np.ndarray:
    """Gain curve of length n: constant before a, equal-power ramp a..b, constant after."""
    idx = np.arange(n)
    x = (idx - a) / max(1, b - a)
    return _eq_power(x, rising)


def _measure_offset_ms(a: np.ndarray, b: np.ndarray) -> float:
    """Lag (ms) that best aligns the low-band attack envelopes of two equal-length signals."""
    lo = butter(4, 150, btype="lowpass", fs=SR, output="sos")
    ea = np.abs(sosfiltfilt(lo, a.mean(axis=1)))
    eb = np.abs(sosfiltfilt(lo, b.mean(axis=1)))
    ea = np.diff(ea, prepend=ea[0]).clip(min=0)
    eb = np.diff(eb, prepend=eb[0]).clip(min=0)
    max_lag = int(0.05 * SR)
    c = correlate(ea, eb, mode="full", method="fft")
    mid = len(eb) - 1
    window = c[mid - max_lag : mid + max_lag + 1]
    return float((np.argmax(window) - max_lag) / SR * 1000)


def order_crate(tracks: list[TrackInput]) -> list[TrackInput]:
    """Order by BPM, then greedily prefer harmonically compatible neighbours."""
    rest = sorted(tracks, key=lambda t: t.analysis.bpm)
    out = [rest.pop(0)]
    rank = {"same": 0, "adjacent": 0, "relative": 1, "energy_boost": 2, "unknown": 3, "clash": 4}
    while rest:
        cur = out[-1]
        key_of = lambda t: t.camelot or t.analysis.camelot  # noqa: E731
        rest.sort(key=lambda t: (abs(t.analysis.bpm - cur.analysis.bpm) > 3, rank[compatibility(key_of(cur), key_of(t))], t.analysis.bpm))
        out.append(rest.pop(0))
    return out


def render_mix(tracks: list[TrackInput], bpm: float | None = None, transition_bars: int = 16, phrase_bars: int = 8,
               target_rms: float = 0.18) -> MixResult:
    if not tracks:
        raise ValueError("no tracks")
    if transition_bars < 2 or transition_bars % 2:
        raise ValueError("transition_bars must be an even number >= 2")
    target = bpm or float(np.median([t.analysis.bpm for t in tracks]))
    period = 60.0 / target
    bar = 4 * period
    warnings: list[str] = []

    # 1) Stretch every track to the target tempo and re-measure its grid on the stretched audio.
    prepared = []
    for t in tracks:
        ratio = target / t.analysis.bpm
        if abs(ratio - 1) > MAX_STRETCH:
            warnings.append(f"{t.title}: tempo changed by {100 * (ratio - 1):+.1f} % (more than {MAX_STRETCH:.0%}); it may sound unnatural.")
        audio = time_stretch(load_stereo(t.path), ratio)
        audio *= target_rms / _rms_loud(audio)
        expected = t.analysis.first_downbeat / ratio
        first, per = _fit_downbeats(audio.mean(axis=1), SR, expected, period)
        # The re-fit is for sub-millisecond alignment only. It may report a later bar as
        # "first" (e.g. if the stretch softens the very first kick), so keep the original
        # bar numbering and shift by whole bars back onto the analysed downbeat.
        first -= round((first - expected) / (4 * per)) * 4 * per
        nbars = int((len(audio) / SR - first) // (4 * per))
        prepared.append((t, ratio, audio, first, nbars))

    # 2) Lay out the timeline.
    n_trans = transition_bars
    placements = []  # (start_sample, audio, first_downbeat_s, mix_out_bar, nbars)
    cursor_start = 0.0
    for i, (t, ratio, audio, first, nbars) in enumerate(prepared):
        if i == 0:
            start = 0.0
        else:
            prev_t, _, _, prev_first, prev_nbars = prepared[i - 1]
            p_start = placements[-1][0]
            out_bar = placements[-1][3]
            entry = p_start + prev_first + out_bar * bar  # mix time of prev track's mix-out downbeat
            start = entry - first
            if start < cursor_start:
                warnings.append(f"{t.title}: its intro is longer than the previous track allows; shifted later.")
        # Mix-out bar for this track: last phrase boundary leaving room for a full transition.
        last_start = nbars - n_trans
        out_bar = max(phrase_bars, (last_start // phrase_bars) * phrase_bars) if i < len(prepared) - 1 else nbars
        if i < len(prepared) - 1 and last_start < phrase_bars:
            raise ValueError(f"{t.title} is too short for a {n_trans}-bar transition")
        placements.append((start, audio, first, out_bar, nbars))
        cursor_start = start

    # 3) Render with band-split, bass-swap transitions.
    ends = []
    for i, (start, audio, first, out_bar, nbars) in enumerate(placements):
        if i < len(placements) - 1:
            ends.append(start + first + (out_bar + n_trans) * bar)
        else:
            ends.append(start + len(audio) / SR)
    total = int(np.ceil(max(ends) * SR)) + SR
    mix = np.zeros((total, 2), dtype=np.float64)

    transitions: list[Transition] = []
    placed: list[PlacedTrack] = []
    for i, (start, audio, first, out_bar, nbars) in enumerate(placements):
        t, ratio = prepared[i][0], prepared[i][1]
        s0 = int(round(start * SR))
        length = int(round(ends[i] * SR)) - s0
        seg = np.zeros((length, 2))
        avail = min(length, len(audio))
        seg[:avail] = audio[:avail]
        low, high = _split(seg)
        g_low = np.ones(length)
        g_high = np.ones(length)
        half = n_trans // 2
        # Incoming: highs rise over the first half, bass enters at the swap (1-bar ramp).
        if i > 0:
            a = int(round(first * SR))
            g_high *= _ramp(length, a, a + int(half * bar * SR), rising=True)
            g_low *= _ramp(length, a + int(half * bar * SR), a + int((half + 1) * bar * SR), rising=True)
        # Outgoing: bass leaves at the swap, highs fall over the second half.
        if i < len(placements) - 1:
            m = int(round((first + out_bar * bar) * SR))
            g_low *= _ramp(length, m + int(half * bar * SR), m + int((half + 1) * bar * SR), rising=False)
            g_high *= _ramp(length, m + int(half * bar * SR), m + int(n_trans * bar * SR), rising=False)
        track_out = low * g_low[:, None] + high * g_high[:, None]
        e = min(total, s0 + length)
        if s0 < 0:
            raise ValueError("negative placement")
        mix[s0:e] += track_out[: e - s0]

        cam = t.camelot or t.analysis.camelot
        placed.append(PlacedTrack(
            title=t.title, start=start, first_downbeat=start + first, end=ends[i], bpm_original=t.analysis.bpm,
            stretch_pct=round(100 * (ratio - 1), 2), camelot=cam,
            camelot_source="metadata" if t.camelot else f"estimated ({t.analysis.key_confidence:.0%} confidence)",
            camelot_estimated=t.analysis.camelot, license=t.license,
        ))

    for i in range(len(placements) - 1):
        a_start, a_audio, a_first, a_out, _ = placements[i]
        b_start, b_audio, b_first, _, _ = placements[i + 1]
        t_start = a_start + a_first + a_out * bar
        # Measure how well the two kicks line up over the first half of the overlap.
        n = int(n_trans // 2 * bar * SR)
        ia = int(round((t_start - a_start) * SR))
        ib = int(round((t_start - b_start) * SR))
        if ib < 0:  # incoming downbeat at (or a hair before) sample 0: skip the first few samples of both
            ia, ib = ia - ib, 0
        offset = None
        if ia + n <= len(a_audio) and ib + n <= len(b_audio) and ib >= 0:
            offset = round(_measure_offset_ms(a_audio[ia : ia + n], b_audio[ib : ib + n]), 2)
        ka = placed[i].camelot
        kb = placed[i + 1].camelot
        transitions.append(Transition(
            from_title=placed[i].title, to_title=placed[i + 1].title, start=t_start,
            bass_swap=t_start + n_trans // 2 * bar, end=t_start + n_trans * bar, bars=n_trans,
            key_relation=compatibility(ka, kb), measured_offset_ms=offset,
        ))

    # 4) Master: soft-limit and normalise to -1 dBFS peak.
    peak = np.max(np.abs(mix)) or 1.0
    if peak > 0.9:
        mix = np.tanh(mix / peak * 1.5) / np.tanh(1.5)
    mix *= 10 ** (-1 / 20) / (np.max(np.abs(mix)) or 1.0)
    return MixResult(audio=mix.astype(np.float32), bpm=round(target, 2), tracks=placed, transitions=transitions, warnings=warnings)
