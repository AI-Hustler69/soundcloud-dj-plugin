"""Time-stretching with ffmpeg's rubberband filter (pitch preserved)."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def load_stereo(path: str | Path) -> np.ndarray:
    """Return float32 array shaped (n, 2) at 44.1 kHz, decoding anything ffmpeg can read."""
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "in.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-ac", "2", "-ar", str(SR), "-c:a", "pcm_f32le", str(wav)],
            check=True,
        )
        data, _ = sf.read(wav, dtype="float32", always_2d=True)
    return data


def time_stretch(audio: np.ndarray, tempo_ratio: float) -> np.ndarray:
    """Change tempo by ``tempo_ratio`` (>1 = faster) without changing pitch."""
    if abs(tempo_ratio - 1.0) < 1e-6:
        return audio.copy()
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "src.wav", Path(tmp) / "dst.wav"
        sf.write(src, audio, SR, subtype="FLOAT")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src),
             "-af", f"rubberband=tempo={tempo_ratio:.8f}:transients=crisp:detector=percussive:pitchq=quality",
             "-ar", str(SR), "-c:a", "pcm_f32le", str(dst)],
            check=True,
        )
        out, _ = sf.read(dst, dtype="float32", always_2d=True)
    return out
