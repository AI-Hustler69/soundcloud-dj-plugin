"""Tracklist, cue sheet and JSON report for a rendered mix."""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import soundfile as sf

from .mixer import MixResult
from .stretch import SR

LICENSE_NOTE = (
    "Only mix files you are allowed to use. 'Free download' usually means personal use only: "
    "before publishing, streaming or monetising this mix, check every track's license and ask the artists when in doubt."
)


def ts(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    return f"{s // 3600:d}:{s // 60 % 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def tracklist(result: MixResult) -> str:
    lines = [f"Automix at {result.bpm} BPM · {ts(result.duration)}", ""]
    for i, t in enumerate(result.tracks, 1):
        lic = f" · license: {t.license}" if t.license else ""
        lines.append(f"{i:>2}. {ts(t.first_downbeat)}  {t.title}  [{t.camelot or '?'}; {t.bpm_original} BPM, {t.stretch_pct:+.1f} %]{lic}")
    lines.append("")
    for tr in result.transitions:
        off = ""
        if tr.measured_offset_ms is None and tr.beatless_chunks:
            off = ", beat alignment not measurable (no clear beat in the blend, e.g. a beatless intro)"
        if tr.measured_offset_ms is not None and tr.beatless_chunks:
            off = f", {tr.beatless_chunks * 4} bars without a clear beat skipped"
        if tr.measured_offset_ms is not None:
            off = f", beat alignment {tr.measured_offset_ms:+.1f} ms" + off
            spread = [c for c in tr.chunk_offsets_ms if abs(c - tr.measured_offset_ms) > 5]
            if spread:
                off += f" (irregular passage: {', '.join(f'{c:+.0f}' for c in spread)} ms in some 4-bar parts)"
        lines.append(f"   {ts(tr.start)}–{ts(tr.end)}  {tr.from_title} → {tr.to_title}: {tr.bars}-bar blend, bass swap at {ts(tr.bass_swap)}, key {tr.key_relation}{off}")
    if result.warnings:
        lines += ["", "Warnings:"] + [f" - {w}" for w in result.warnings]
    lines += ["", LICENSE_NOTE]
    return "\n".join(lines) + "\n"


def cue_sheet(result: MixResult, audio_name: str) -> str:
    out = [f'FILE "{audio_name}" MP3']
    for i, t in enumerate(result.tracks, 1):
        sec = max(0.0, t.first_downbeat if i > 1 else 0.0)
        frames = int(round(sec * 75))
        out += [f"  TRACK {i:02d} AUDIO", f'    TITLE "{t.title}"',
                f"    INDEX 01 {frames // 75 // 60:02d}:{frames // 75 % 60:02d}:{frames % 75:02d}"]
    return "\n".join(out) + "\n"


def write_outputs(result: MixResult, out_base: str | Path, mp3: bool = True) -> dict[str, str]:
    base = Path(out_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    wav = base.with_suffix(".wav")
    sf.write(wav, result.audio, SR, subtype="PCM_16")
    files = {"wav": str(wav)}
    if mp3:
        mp3_path = base.with_suffix(".mp3")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-c:a", "libmp3lame", "-b:a", "320k", str(mp3_path)], check=True)
        files["mp3"] = str(mp3_path)
    txt = base.with_suffix(".tracklist.txt")
    txt.write_text(tracklist(result), encoding="utf-8")
    files["tracklist"] = str(txt)
    cue = base.with_suffix(".cue")
    cue.write_text(cue_sheet(result, Path(files.get("mp3", wav)).name), encoding="utf-8")
    files["cue"] = str(cue)
    js = base.with_suffix(".json")
    js.write_text(json.dumps({
        "bpm": result.bpm, "duration": round(result.duration, 2),
        "tracks": [asdict(t) for t in result.tracks],
        "transitions": [asdict(t) for t in result.transitions],
        "warnings": result.warnings, "license_note": LICENSE_NOTE,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    files["json"] = str(js)
    return files
