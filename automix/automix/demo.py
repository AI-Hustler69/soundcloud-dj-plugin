"""Build a small synthetic crate so the mixer can be tried without any real tracks."""
from __future__ import annotations

from pathlib import Path

import soundfile as sf

from .synth import SR, SynthSpec, synth_track

DEMO_CRATE = [
    ("Synth One", SynthSpec(172.0, "A", True, bars=48, lead_in=0.20, seed=11), "8A"),
    ("Synth Two", SynthSpec(174.0, "E", True, bars=48, lead_in=0.55, seed=12), "9A"),
    ("Synth Three", SynthSpec(175.5, "B", True, bars=48, lead_in=0.00, seed=13), "10A"),
    ("Synth Four", SynthSpec(173.0, "D", False, bars=48, lead_in=0.31, seed=14), "10B"),
]


def build_demo(folder: str | Path) -> list[dict]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    items = []
    for title, spec, cam in DEMO_CRATE:
        path = folder / f"{title.lower().replace(' ', '_')}.wav"
        sf.write(path, synth_track(spec), SR)
        # Pass the key as metadata, the way the soundcloud-dj plugin supplies key_signature/Camelot.
        items.append({"path": str(path), "title": title, "camelot": cam, "license": "original test audio"})
    return items
