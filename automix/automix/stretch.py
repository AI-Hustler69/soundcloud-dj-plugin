"""Time-stretching with ffmpeg's rubberband filter (pitch preserved)."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def load_stereo(path: str | Path) -> np.ndarray:
    """Return float32 array shaped (n, 2) at 44.1 kHz, decoding anything ffmpeg can read.

    Mono is duplicated to both channels at full level (ffmpeg's own upmix would drop it by 3 dB);
    more than two channels are reduced to the first two.
    """
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "in.wav"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-vn", "-ar", str(SR), "-c:a", "pcm_f32le", str(wav)],
            check=True,
        )
        data, _ = sf.read(wav, dtype="float32", always_2d=True)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return np.ascontiguousarray(data[:, :2])


def time_stretch(audio: np.ndarray, tempo_ratio: float) -> np.ndarray:
    """Change tempo by ``tempo_ratio`` (>1 = faster) without changing pitch."""
    if abs(tempo_ratio - 1.0) < 1e-6:
        return audio.copy()
    # Pad with silence: Rubber Band softens the very first transient of a file (a kick
    # on sample 0) and needs a moment to settle at the end; the padding is cut off again.
    pad = int(0.25 * SR)
    padded = np.pad(audio, ((pad, pad), (0, 0)))
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "src.wav", Path(tmp) / "dst.wav"
        sf.write(src, padded, SR, subtype="FLOAT")
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(src),
             "-af", f"rubberband=tempo={tempo_ratio:.8f}:transients=crisp:detector=percussive:pitchq=quality",
             "-ar", str(SR), "-c:a", "pcm_f32le", str(dst)],
            check=True,
        )
        out, _ = sf.read(dst, dtype="float32", always_2d=True)
    lead = int(round(pad / tempo_ratio))
    want = int(round(len(audio) / tempo_ratio))
    out = out[lead : lead + want]
    if len(out) < want:
        out = np.pad(out, ((0, want - len(out)), (0, 0)))
    return out
