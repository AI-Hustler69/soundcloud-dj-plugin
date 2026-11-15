"""Synthetic drum & bass test tracks with known tempo, key and downbeat.

Used by the test-suite and the demo, so the analyser and the mixer can be
checked against ground truth without any third-party audio.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 44100

NOTE_FREQ = {  # octave 2 roots
    "C": 65.41, "C#": 69.30, "D": 73.42, "D#": 77.78, "E": 82.41, "F": 87.31,
    "F#": 92.50, "G": 98.00, "G#": 103.83, "A": 110.00, "A#": 116.54, "B": 123.47,
}


@dataclass(frozen=True)
class SynthSpec:
    bpm: float
    root: str  # e.g. "A"
    minor: bool = True
    bars: int = 48
    lead_in: float = 0.0  # seconds of silence before the first downbeat
    seed: int = 0
    jumpup: bool = False  # loud, distorted, syncopated bass stabs that crowd the kick's band


def _env(n: int, decay: float) -> np.ndarray:
    return np.exp(-np.arange(n) / SR / decay)


def _place(out: np.ndarray, sig: np.ndarray, t: float, gain: float = 1.0) -> None:
    i = int(round(t * SR))
    if i >= len(out) or i < 0:
        return
    j = min(len(out), i + len(sig))
    out[i:j] += sig[: j - i] * gain


def synth_track(spec: SynthSpec) -> np.ndarray:
    """Return a mono float32 signal. Downbeat 0 is at ``spec.lead_in`` seconds.

    Structure: 8-bar drums-only intro, main section with bass + chords,
    8-bar drums-only outro, like a DJ-friendly club edit.
    """
    rng = np.random.default_rng(spec.seed)
    beat = 60.0 / spec.bpm
    step = beat / 4
    bar = 4 * beat
    dur = spec.lead_in + spec.bars * bar + 1.0
    n = int(dur * SR)
    out = np.zeros(n)

    kn = int(0.3 * SR)
    kt = np.arange(kn) / SR
    kick = np.sin(2 * np.pi * np.cumsum(45 + 120 * np.exp(-kt / 0.03)) / SR) * _env(kn, 0.11)
    sn = int(0.22 * SR)
    noise = rng.standard_normal(sn)
    snare = 0.5 * (noise - np.roll(noise, 1) * 0.8) * _env(sn, 0.06) + 0.5 * np.sin(2 * np.pi * 190 * np.arange(sn) / SR) * _env(sn, 0.05)
    hn = int(0.03 * SR)
    hnoise = rng.standard_normal(hn)
    hat = (hnoise - np.roll(hnoise, 1)) * _env(hn, 0.01) * 0.12

    for b in range(spec.bars):
        t0 = spec.lead_in + b * bar
        _place(out, kick, t0, 0.9)
        _place(out, kick, t0 + 10 * step, 0.75)
        _place(out, snare, t0 + 4 * step, 0.7)
        _place(out, snare, t0 + 12 * step, 0.7)
        for s in range(0, 16, 2):
            _place(out, hat, t0 + s * step, 0.8)

    # Bass + chord pad in the main section only (bars 8 .. bars-8).
    root = NOTE_FREQ[spec.root]
    third = root * (2 ** ((3 if spec.minor else 4) / 12))
    fifth = root * (2 ** (7 / 12))
    t = np.arange(n) / SR
    main_start = spec.lead_in + 8 * bar
    main_end = spec.lead_in + (spec.bars - 8) * bar
    gate = ((t >= main_start) & (t < main_end)).astype(float)
    ramp = int(0.02 * SR)
    gate = np.convolve(gate, np.ones(ramp) / ramp, mode="same")
    # Bass on the tonic (octave 2) plus a sub an octave below, as in most DnB.
    bass = np.sin(2 * np.pi * root * t) * 0.22 + np.sin(2 * np.pi * (root / 2) * t) * 0.18
    pad = (np.sin(2 * np.pi * root * 2 * t) + np.sin(2 * np.pi * third * 2 * t) + np.sin(2 * np.pi * fifth * 2 * t)) / 3 * 0.12
    out += (bass + pad) * gate

    if spec.jumpup:
        # Jump-up style: heavily saturated reese stabs on off-beat 16ths, louder than
        # the kick in the low band (including just before each downbeat).
        stab_n = int(step * 1.6 * SR)
        st = np.arange(stab_n) / SR
        f0 = root / 2
        saw = sum(np.sin(2 * np.pi * f0 * h * (1 + 0.004 * (h % 2)) * st) / h for h in range(1, 12))
        stab = np.tanh(5.0 * saw * _env(stab_n, step * 1.2)) * 0.9
        for b in range(8, spec.bars - 8):
            t0 = spec.lead_in + b * bar
            for s_ in (3, 6, 9, 11, 14, 15):
                _place(out, stab, t0 + s_ * step, 1.0)

    out /= np.max(np.abs(out)) / 0.89
    return out.astype(np.float32)
