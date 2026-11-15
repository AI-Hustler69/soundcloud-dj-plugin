"""Samples on top of a mix: loops (tempo-synced, repeated) and one-shots (horns, vocals, impacts).

A sample is placed at an anchor on the mix timeline:
  {"track": 2, "bar": 16}              bar 16 of the 2nd track (1-based, final play order) or a track title
  {"transition": 1, "bar": -4}         4 bars before the 1st transition's incoming downbeat
  {"time": "12:30"}                    a mix time, snapped to the nearest downbeat unless "snap": false
With "align": "end" the sample *ends* at the anchor (risers into a drop).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .stretch import SR, load_stereo, time_stretch

LOOP_BARS = (1, 2, 4, 8, 16)
MAX_LOOP_STRETCH = 0.06  # auto-detect a loop only if its length is within 6 % of whole bars at the mix tempo


@dataclass
class SampleInput:
    path: str
    title: str
    at: dict
    gain_db: float = -3.0
    kind: str = "auto"  # auto | loop | oneshot
    bars: int | None = None  # loops: total length to play (repeats as needed); default one pass
    loop_bars: int | None = None  # loops: length of one pass in bars, if known
    align: str = "start"  # start | end
    snap: bool = True


@dataclass
class PlacedSample:
    title: str
    kind: str
    start: float  # mix seconds where the sample's sound begins
    end: float
    anchor: str
    loop_bars: int | None = None
    stretch_pct: float = 0.0
    gain_db: float = -3.0
    notes: list[str] = field(default_factory=list)


def parse_time(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    parts = [float(p) for p in str(value).strip().split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def _onset_index(audio: np.ndarray) -> int:
    """First sample where the sample actually starts (10 % of its peak), for one-shots with a little lead-in."""
    mono = np.abs(audio).max(axis=1)
    peak = float(mono.max())
    if peak <= 0:
        return 0
    idx = np.nonzero(mono >= 0.1 * peak)[0]
    return int(idx[0]) if len(idx) else 0


def classify(audio: np.ndarray, mix_bpm: float, kind: str = "auto", loop_bars: int | None = None) -> tuple[str, int | None, float]:
    """Return (kind, loop_bars, loop_bpm).

    A loop is a file whose length, at the mix tempo, is close to 1/2/4/8/16 bars
    (within MAX_LOOP_STRETCH). Its own tempo then follows from that length exactly,
    which is more reliable than beat tracking on a short clip.
    """
    dur = len(audio) / SR
    if kind == "oneshot" or dur <= 0:
        return "oneshot", None, 0.0
    bars_at_mix = dur * mix_bpm / 240.0
    candidates = [loop_bars] if loop_bars else list(LOOP_BARS)
    n = min(candidates, key=lambda c: abs(np.log(bars_at_mix / c)))
    dev = abs(bars_at_mix / n - 1)
    if kind == "loop" or (dur >= 1.5 and dev <= MAX_LOOP_STRETCH):
        return "loop", int(n), 240.0 * n / dur
    return "oneshot", None, 0.0


def resolve_anchor(at: dict, tracks: list, transitions: list, bar: float) -> tuple[float, str]:
    """Mix time (s) of an anchor and a human-readable description."""
    if "track" in at:
        ref = at["track"]
        if isinstance(ref, str):
            matches = [i for i, t in enumerate(tracks) if t.title == ref]
            if not matches:
                raise ValueError(f"sample anchor: no track titled {ref!r}")
            i = matches[0]
        else:
            i = int(ref) - 1
        if not 0 <= i < len(tracks):
            raise ValueError(f"sample anchor: track {ref} does not exist (mix has {len(tracks)})")
        b = float(at.get("bar", 0))
        return tracks[i].first_downbeat + b * bar, f"track {i + 1} ({tracks[i].title}) bar {b:g}"
    if "transition" in at:
        n = int(at["transition"]) - 1
        if not 0 <= n < len(transitions):
            raise ValueError(f"sample anchor: transition {n + 1} does not exist (mix has {len(transitions)})")
        b = float(at.get("bar", 0))
        return transitions[n].start + b * bar, f"transition {n + 1} {'+' if b >= 0 else ''}{b:g} bars"
    if "time" in at:
        t = parse_time(at["time"])
        return t, f"time {at['time']}"
    raise ValueError(f"sample anchor needs 'track', 'transition' or 'time': {at}")


def snap_to_downbeat(t: float, tracks: list, bar: float) -> float:
    """Nearest downbeat of the track that is playing (latest track started at or before t)."""
    playing = [tr for tr in tracks if tr.first_downbeat <= t + bar / 2]
    ref = playing[-1] if playing else tracks[0]
    k = round((t - ref.first_downbeat) / bar)
    return ref.first_downbeat + k * bar


def render_samples(mix: np.ndarray, samples: list[SampleInput], mix_bpm: float, tracks: list, transitions: list) -> tuple[np.ndarray, list[PlacedSample]]:
    """Add samples into ``mix`` (float64, (n, 2)); grows the buffer if a sample runs past the end."""
    bar = 4 * 60.0 / mix_bpm
    placed: list[PlacedSample] = []
    for s in samples:
        notes: list[str] = []
        audio = load_stereo(s.path)
        if len(audio) == 0:
            raise ValueError(f"sample {s.title}: empty file")
        kind, loop_bars, loop_bpm = classify(audio, mix_bpm, s.kind, s.loop_bars)
        t, anchor = resolve_anchor(s.at, tracks, transitions, bar)
        if "time" in s.at and s.snap:
            t2 = snap_to_downbeat(t, tracks, bar)
            if abs(t2 - t) > 1e-6:
                notes.append(f"snapped {t2 - t:+.2f} s to the nearest downbeat")
            t = t2
        stretch = 0.0
        if kind == "loop":
            ratio = mix_bpm / loop_bpm
            stretch = 100 * (ratio - 1)
            # Stretch three passes in a row and keep the middle one: its edges are surrounded by
            # the loop itself, so it repeats seamlessly, and it is exactly loop_bars long.
            want = int(round(loop_bars * bar * SR))
            three = time_stretch(np.tile(audio, (3, 1)), ratio)
            one = three[want : 2 * want]
            if len(one) < want:
                one = np.pad(one, ((0, want - len(one)), (0, 0)))
            total_bars = s.bars or loop_bars
            reps = int(np.ceil(total_bars / loop_bars))
            body = np.tile(one, (reps, 1))[: int(round(total_bars * bar * SR))]
            # Tiny fade-in (keeps the first hit's attack) and a short fade-out so the loop does not click.
            fi, fo = min(int(0.001 * SR), len(body) // 4), min(int(0.005 * SR), len(body) // 4)
            if fi > 0:
                body[:fi] *= np.linspace(0, 1, fi)[:, None]
            if fo > 0:
                body[-fo:] *= np.linspace(1, 0, fo)[:, None]
            start = t - (len(body) / SR if s.align == "end" else 0.0)
            offset = 0
        else:
            body = audio.astype(np.float64)
            onset = _onset_index(body)
            if s.align == "end":
                start = t - len(body) / SR
                offset = 0
            else:
                start = t - onset / SR  # the sound's attack, not the file start, lands on the beat
                offset = onset
        gain = 10 ** (s.gain_db / 20)
        i0 = int(round(start * SR))
        if i0 < 0:
            body = body[-i0:]
            notes.append("starts before the mix; its beginning was cut")
            i0 = 0
        i1 = i0 + len(body)
        if i1 > len(mix):
            mix = np.pad(mix, ((0, i1 - len(mix)), (0, 0)))
        mix[i0:i1] += body * gain
        placed.append(PlacedSample(
            title=s.title, kind=kind, start=(i0 + offset) / SR if kind == "oneshot" else i0 / SR,
            end=i1 / SR, anchor=anchor + (", ends there" if s.align == "end" else ""),
            loop_bars=loop_bars if kind == "loop" else None, stretch_pct=round(stretch, 2), gain_db=s.gain_db, notes=notes,
        ))
        if kind == "loop" and abs(stretch) > 8:
            notes.append(f"tempo changed by {stretch:+.1f} %, which may sound unnatural")
    return mix, placed
