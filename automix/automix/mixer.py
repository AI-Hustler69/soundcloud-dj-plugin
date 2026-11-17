"""Plan and render a beat-matched DJ mix from analysed tracks."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, correlate, sosfiltfilt

from .analysis import TrackAnalysis, _fit_downbeats
from .keys import compatibility
from .master import DEFAULT_TARGET_LUFS, TRUE_PEAK_CEILING_DB, MasterReport, master, true_peak  # noqa: F401
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
    measured_offset_ms: float | None = None  # median over 4-bar chunks of the overlap
    chunk_offsets_ms: list[float] = field(default_factory=list)
    beatless_chunks: int = 0  # parts (chunk_bars long) skipped because one side had no clear beat
    chunk_bars: int = 4  # length of the parts the alignment is measured in


@dataclass
class MixResult:
    audio: np.ndarray  # (n, 2)
    bpm: float
    tracks: list[PlacedTrack] = field(default_factory=list)
    transitions: list[Transition] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    samples: list = field(default_factory=list)  # PlacedSample
    master: MasterReport | None = None

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


BEAT_STRENGTH_MIN = 2.0  # on-beat transient energy vs. average, below this a passage counts as beatless


def _beat_strength(audio: np.ndarray, period: float) -> float:
    """How strongly transients sit on a beat grid of ``period`` (best phase), relative to average."""
    from .analysis import FINE_HOP, transient_envelope

    fl = transient_envelope(audio.mean(axis=1), SR)
    fps = SR / FINE_HOP
    n_beats = int((len(fl) / fps) // period) - 1
    if n_beats < 4 or fl.mean() <= 0:
        return 0.0
    beats = np.arange(n_beats) * period
    offs = np.arange(0, period, 0.002)
    pos = np.clip(((beats[None, :] + offs[:, None]) * fps).astype(int), 0, len(fl) - 1)
    return float(fl[pos].mean(axis=1).max() / fl.mean())


def _measure_offset_ms(a: np.ndarray, b: np.ndarray) -> float:
    """Lag (ms) that best aligns the broadband transients (kicks and snares) of two equal-length signals.

    Positive means b's beats land later than a's.
    """
    from .analysis import FINE_HOP, transient_envelope

    fa = transient_envelope(a.mean(axis=1), SR)
    fb = transient_envelope(b.mean(axis=1), SR)
    m = min(len(fa), len(fb))
    fa, fb = fa[:m] - fa[:m].mean(), fb[:m] - fb[:m].mean()
    fps = SR / FINE_HOP
    max_lag = int(0.05 * fps)
    c = correlate(fa, fb, mode="full", method="fft")
    mid = m - 1
    window = c[mid - max_lag : mid + max_lag + 1]
    return float(-(np.argmax(window) - max_lag) / fps * 1000)


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
               target_rms: float = 0.18, samples: list | None = None, clash_bars: int | None = None,
               target_lufs: float | None = DEFAULT_TARGET_LUFS) -> MixResult:
    """Beat-matched mix of ``tracks`` in the given order.

    clash_bars: a shorter blend (even, >= 2) for transitions whose keys clash, so two clashing
        basslines overlap less (DJs cut quickly between clashing keys). None = same as transition_bars.
    target_lufs: integrated loudness of the master, reached with a true-peak limiter; None = no
        limiting, only scale the true peak to the ceiling.
    """
    if not tracks:
        raise ValueError("no tracks")
    if transition_bars < 2 or transition_bars % 2:
        raise ValueError("transition_bars must be an even number >= 2")
    if clash_bars is not None and (clash_bars < 2 or clash_bars % 2):
        raise ValueError("clash_bars must be an even number >= 2")
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
        # Re-align the phase on the stretched audio (tempo is now exactly `period`): this
        # absorbs any time offset the stretcher introduces. It may report a later bar as
        # "first" if the stretch softens the very first kick, so keep the analysed bar numbering.
        first, per = _fit_downbeats(audio.mean(axis=1), SR, expected, period, window=0.02)
        first -= round((first - expected) / (4 * per)) * 4 * per
        nbars = int((len(audio) / SR - first) // (4 * per))
        prepared.append((t, ratio, audio, first, nbars))

    # 2) Lay out the timeline. trans[i] = length in bars of the blend from track i into track i + 1.
    def _key(t):
        return t.camelot or t.analysis.camelot

    trans = [clash_bars if clash_bars and compatibility(_key(a), _key(b)) == "clash" else transition_bars
             for a, b in zip(tracks, tracks[1:])]
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
        n_out = trans[i] if i < len(prepared) - 1 else 0
        last_start = nbars - n_out
        out_bar = max(phrase_bars, (last_start // phrase_bars) * phrase_bars) if i < len(prepared) - 1 else nbars
        if i < len(prepared) - 1 and last_start < phrase_bars:
            raise ValueError(f"{t.title} is too short for a {n_out}-bar transition")
        n_in = trans[i - 1] if i > 0 else 0
        if 0 < i < len(prepared) - 1 and out_bar < n_in:
            # Its mix-out would start inside the incoming blend: three tracks would overlap.
            raise ValueError(f"{t.title} is too short for a {n_in}-bar blend in and a {n_out}-bar blend out")
        placements.append((start, audio, first, out_bar, nbars))
        cursor_start = start

    # 3) Render with band-split, bass-swap transitions.
    ends = []
    for i, (start, audio, first, out_bar, nbars) in enumerate(placements):
        if i < len(placements) - 1:
            ends.append(start + first + (out_bar + trans[i]) * bar)
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
        # Incoming: highs rise over the first half, bass enters at the swap (1-bar ramp).
        if i > 0:
            half = trans[i - 1] // 2
            a = int(round(first * SR))
            g_high *= _ramp(length, a, a + int(half * bar * SR), rising=True)
            g_low *= _ramp(length, a + int(half * bar * SR), a + int((half + 1) * bar * SR), rising=True)
        # Outgoing: bass leaves at the swap, highs fall over the second half.
        if i < len(placements) - 1:
            half = trans[i] // 2
            m = int(round((first + out_bar * bar) * SR))
            g_low *= _ramp(length, m + int(half * bar * SR), m + int((half + 1) * bar * SR), rising=False)
            g_high *= _ramp(length, m + int(half * bar * SR), m + int(trans[i] * bar * SR), rising=False)
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
        n_trans = trans[i]
        a_start, a_audio, a_first, a_out, _ = placements[i]
        b_start, b_audio, b_first, _, _ = placements[i + 1]
        t_start = a_start + a_first + a_out * bar
        # Measure beat alignment over the whole overlap in 4-bar chunks; report the median,
        # so one irregular passage (e.g. an off-grid intro effect) does not dominate.
        ia0 = int(round((t_start - a_start) * SR))
        ib0 = int(round((t_start - b_start) * SR))
        if ib0 < 0:  # incoming downbeat at (or a hair before) sample 0: skip the first few samples of both
            ia0, ib0 = ia0 - ib0, 0
        chunk_bars = min(4, n_trans)  # short (clash) blends are measured too
        chunk = int(chunk_bars * bar * SR)
        chunks = []
        beatless = 0
        for c in range(n_trans // chunk_bars):
            ia, ib = ia0 + c * chunk, ib0 + c * chunk
            if ia + chunk <= len(a_audio) and ib + chunk <= len(b_audio):
                sa, sb = a_audio[ia : ia + chunk], b_audio[ib : ib + chunk]
                # Skip parts where either track has no clear beat (beatless intro, breakdown):
                # there is nothing to clash, and the measurement would only be noise.
                if _beat_strength(sa, period) < BEAT_STRENGTH_MIN or _beat_strength(sb, period) < BEAT_STRENGTH_MIN:
                    beatless += 1
                    continue
                chunks.append(round(_measure_offset_ms(sa, sb), 2))
        offset = round(float(np.median(chunks)), 2) if chunks else None
        ka = placed[i].camelot
        kb = placed[i + 1].camelot
        transitions.append(Transition(
            from_title=placed[i].title, to_title=placed[i + 1].title, start=t_start,
            bass_swap=t_start + n_trans // 2 * bar, end=t_start + n_trans * bar, bars=n_trans,
            key_relation=compatibility(ka, kb), measured_offset_ms=offset, chunk_offsets_ms=chunks,
            beatless_chunks=beatless, chunk_bars=chunk_bars,
        ))

    # 4) Samples (loops / one-shots) on top of the blended tracks, before mastering.
    placed_samples = []
    if samples:
        from .samples import render_samples

        mix, placed_samples = render_samples(mix, samples, target, placed, transitions)
        for ps in placed_samples:
            warnings.extend(f"sample {ps.title}: {n}" for n in ps.notes if "unnatural" in n or "cut" in n)

    # 5) Master: loudness target with a true-peak limiter (see master.py).
    mix, report = master(mix, target_lufs=target_lufs)
    if target_lufs is not None and report.lufs < target_lufs - 0.5:
        warnings.append(f"master: reached {report.lufs:.1f} LUFS, not {target_lufs:g}: louder would need more than "
                        f"{report.max_reduction_db:.0f} dB of limiting, which crushes the drums.")
    return MixResult(audio=mix.astype(np.float32), bpm=round(target, 2), tracks=placed, transitions=transitions,
                     warnings=warnings, samples=placed_samples, master=report)
