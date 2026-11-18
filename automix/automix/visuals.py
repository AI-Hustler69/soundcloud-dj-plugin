"""Audio-reactive visuals for a finished mix: a YouTube video of the whole set, or a vertical reel.

The look reacts to the drums separately, the way a VJ patches a "sender" on each instrument:
  kick  -> the centre orb swells and its rim glows
  snare -> a shockwave ring leaves the orb, plus a short flash
  hats  -> particles sparkle and speed up
  bass  -> drives the tunnel in the background
Each track of the mix gets its own colour (crossfaded during blends) and its title, taken from the
JSON report that ``automix mix`` writes next to the mix.

Everything is computed with numpy and encoded by ffmpeg; no GPU, no network, no third-party visuals.
Frames are rendered at a reduced internal size and scaled up by ffmpeg (the glow-heavy look hides it).
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMATS = {  # name: (output size, internal render size)
    "youtube": ((1920, 1080), (960, 540)),
    "reel": ((1080, 1920), (540, 960)),
    "square": ((1080, 1080), (540, 540)),
}
PALETTES = {  # hue sequence (0..1) cycled per track
    "neon": [0.55, 0.83, 0.33, 0.08, 0.72, 0.47, 0.95, 0.15],
    "ice": [0.55, 0.6, 0.5, 0.66, 0.52, 0.58],
    "fire": [0.02, 0.08, 0.95, 0.11, 0.0, 0.06],
    "toxic": [0.3, 0.22, 0.38, 0.45, 0.27],
}
ANALYSIS_SR = 22050


# ---------------------------------------------------------------------------------------- audio

def decode(path: str | Path, sr: int = ANALYSIS_SR, start: float = 0.0, duration: float | None = None) -> np.ndarray:
    """Mono float32 audio through ffmpeg (any format ffmpeg reads, including the audio of a video)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


@dataclass
class Features:
    fps: int
    kick: np.ndarray  # 0..1 per frame, fast attack, short decay
    snare: np.ndarray
    hats: np.ndarray
    bass: np.ndarray  # smoothed low-band level
    energy: np.ndarray  # smoothed overall level
    snare_hits: np.ndarray  # frame indices of snare onsets (for shockwaves)

    @property
    def n(self) -> int:
        return len(self.kick)


def _band_flux(logS: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sel = (freqs >= lo) & (freqs < hi)
    d = np.diff(logS[sel], axis=1, prepend=logS[sel][:, :1])
    return np.maximum(d, 0).mean(axis=0)


def _normalise(x: np.ndarray, fps: int, window_s: float = 8.0) -> np.ndarray:
    """Scale by a running local level so quiet and loud passages both react, then clip to 0..1."""
    from scipy.ndimage import uniform_filter1d

    local = uniform_filter1d(x, size=max(1, int(window_s * fps)), mode="nearest")
    ref = np.percentile(x, 98) if len(x) else 1.0
    return np.clip(x / (2.5 * np.maximum(local, 0.15 * ref) + 1e-9), 0, 1)


def _envelope(x: np.ndarray, fps: int, decay_s: float) -> np.ndarray:
    """Instant attack, exponential decay (a peak follower), like a VU meter with a fast release."""
    d = math.exp(-1.0 / (fps * decay_s))
    out = np.empty_like(x)
    v = 0.0
    for i, s in enumerate(x):
        v = s if s > v else v * d
        out[i] = v
    return out


def analyse(audio: np.ndarray, fps: int = 30, sr: int = ANALYSIS_SR) -> Features:
    """Per-frame drum envelopes from the mixed audio (no stems needed)."""
    import librosa
    from scipy.ndimage import uniform_filter1d

    hop = sr / fps
    n_frames = int(len(audio) / hop)
    if n_frames < 2:
        raise ValueError("audio too short for visuals")
    # Resample the STFT grid onto exact video frames: compute with an integer hop, then interpolate.
    ihop = int(round(hop))
    S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=ihop, center=True)) ** 2
    logS = np.log1p(1e3 * S / (S.max() + 1e-12))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    grid = np.arange(n_frames) * hop / ihop

    def at_frames(v: np.ndarray) -> np.ndarray:
        return np.interp(grid, np.arange(len(v)), v).astype(np.float32)

    kick = at_frames(_band_flux(logS, freqs, 35, 120))
    snare = at_frames(_band_flux(logS, freqs, 1200, 4500))
    hats = at_frames(_band_flux(logS, freqs, 7000, 14000))
    # the kick band also carries the bassline; subtract its slow part so only hits remain
    kick = np.maximum(0, kick - uniform_filter1d(kick, size=max(3, fps // 3), mode="nearest"))
    snare = np.maximum(0, snare - 0.6 * uniform_filter1d(snare, size=max(3, fps // 3), mode="nearest"))
    k, s, h = (_normalise(v, fps) for v in (kick, snare, hats))
    rms = at_frames(np.sqrt(S.mean(axis=0)))
    low = at_frames(np.sqrt(S[(freqs >= 30) & (freqs < 150)].mean(axis=0)))
    energy = uniform_filter1d(np.clip(rms / (np.percentile(rms, 95) + 1e-12), 0, 1.2), size=max(1, fps // 2))
    bass = uniform_filter1d(np.clip(low / (np.percentile(low, 95) + 1e-12), 0, 1.2), size=max(1, fps // 4))
    # snare onsets: frames where the raw snare value crosses a threshold from below, at least 1/6 s apart
    hits, last = [], -fps
    for i in range(1, len(s)):
        if s[i] >= 0.55 and s[i - 1] < 0.55 and i - last > fps // 6:
            hits.append(i)
            last = i
    return Features(fps, _envelope(k, fps, 0.12), _envelope(s, fps, 0.16), _envelope(h, fps, 0.07),
                    bass.astype(np.float32), energy.astype(np.float32), np.array(hits, dtype=int))


# ---------------------------------------------------------------------------------- the mix plan

@dataclass
class TrackSpan:
    title: str
    start: float  # mix seconds where it becomes the main track (its first downbeat)
    hue: float


@dataclass
class Plan:
    tracks: list[TrackSpan] = field(default_factory=list)
    blends: list[tuple[float, float]] = field(default_factory=list)  # (start, end) of each transition
    bpm: float | None = None
    bass_swaps: list[float] = field(default_factory=list)

    def colour_at(self, t: float) -> tuple[float, float, float]:
        """(hue_a, hue_b, mix) for time t: during a blend the colour moves from one track to the next."""
        if not self.tracks:
            return 0.55, 0.55, 0.0
        i = max(0, sum(1 for tr in self.tracks if tr.start <= t) - 1)
        for k, (a, b) in enumerate(self.blends):
            if a <= t < b and k + 1 < len(self.tracks):
                return self.tracks[k].hue, self.tracks[k + 1].hue, (t - a) / max(1e-6, b - a)
        return self.tracks[i].hue, self.tracks[i].hue, 0.0

    def title_at(self, t: float) -> tuple[str, float] | None:
        """Current title and seconds since it started, once its blend is over (or from the start)."""
        best = None
        for k, tr in enumerate(self.tracks):
            shown = tr.start if k == 0 else (self.blends[k - 1][1] if k - 1 < len(self.blends) else tr.start)
            if shown <= t:
                best = (tr.title, t - shown)
        return best


def load_plan(mix_json: str | Path | None, palette: str = "neon") -> Plan:
    hues = PALETTES.get(palette, PALETTES["neon"])
    if not mix_json:
        return Plan(tracks=[TrackSpan("", 0.0, hues[0])])
    d = json.loads(Path(mix_json).read_text(encoding="utf-8"))
    tracks = [TrackSpan(t["title"], float(t["first_downbeat"]) if i else 0.0, hues[i % len(hues)])
              for i, t in enumerate(d.get("tracks", []))]
    blends = [(float(t["start"]), float(t["end"])) for t in d.get("transitions", [])]
    swaps = [float(t["bass_swap"]) for t in d.get("transitions", [])]
    return Plan(tracks or [TrackSpan("", 0.0, hues[0])], blends, d.get("bpm"), swaps)


def pick_reel_window(plan: Plan, feats: Features, duration: float) -> float:
    """Start time of the most energetic drop: a bass swap with the loudest following part.

    The reel starts 8 bars before the swap, so the build is in the clip. Without a plan, the loudest
    window with the most kick activity wins.
    """
    fps = feats.fps
    total = feats.n / fps

    def score(a: float, b: float) -> float:
        i, j = int(a * fps), int(min(total, b) * fps)
        return float(feats.energy[i:j].mean() + 0.5 * feats.kick[i:j].mean()) if j > i else -1.0

    if plan.bass_swaps and plan.bpm:
        lead = 8 * 240.0 / plan.bpm
        cands = [s for s in plan.bass_swaps if s - lead >= 0 and s - lead + duration <= total]
        if cands:
            best = max(cands, key=lambda s: score(s, s + duration - lead))
            return max(0.0, best - lead)
    step = max(1.0, duration / 4)
    starts = np.arange(0, max(0.0, total - duration) + 1e-9, step)
    return float(max(starts, key=lambda a: score(a, a + duration))) if len(starts) else 0.0


# ------------------------------------------------------------------------------------- renderer

def _hsv(h: float, s: float, v: float) -> np.ndarray:
    i = int(h * 6) % 6
    f = h * 6 - math.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return np.array([(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i], dtype=np.float32)


def _font(size: int, bold: bool = True):
    from PIL import ImageFont

    names = (["arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"] if bold
             else ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"])
    dirs = ["", "C:/Windows/Fonts/", "/Library/Fonts/", "/System/Library/Fonts/Supplemental/",
            "/usr/share/fonts/truetype/dejavu/", "/usr/share/fonts/truetype/liberation/"]
    for n in names:
        for d in dirs:
            try:
                return ImageFont.truetype(d + n, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def _text_sprite(text: str, size: int, bold: bool = True, max_w: int | None = None) -> np.ndarray:
    """White text with a soft shadow as a float32 (h, w, 2) array: [:, :, 0] text, [:, :, 1] shadow."""
    from PIL import Image, ImageDraw, ImageFilter

    font = _font(size, bold)
    while max_w and size > 10 and font.getlength(text) > max_w:
        size = int(size * 0.92)
        font = _font(size, bold)
    l, t, r, b = font.getbbox(text)
    pad = max(4, size // 4)
    w, h = int(r - l) + 2 * pad, int(b - t) + 2 * pad
    txt = Image.new("L", (max(1, w), max(1, h)), 0)
    ImageDraw.Draw(txt).text((pad - l, pad - t), text, font=font, fill=255)
    shadow = txt.filter(ImageFilter.GaussianBlur(max(2, size // 8)))
    return np.stack([np.asarray(txt, np.float32) / 255, np.asarray(shadow, np.float32) / 255], axis=2)


RSTEPS = 2048  # resolution of the radial lookup tables (r from 0 to RMAX)
RMAX = 1.3


class Renderer:
    """Stateless frame renderer: everything that evolves (tunnel phase, particle travel, live shockwaves)
    is passed in, so frames can be rendered in any order or in parallel.

    Speed: every radially symmetric term (vignette, tunnel rings, orb, shockwaves, snare flash) is built
    as a 1-D profile over the radius each frame and looked up per pixel; the tunnel's spiral arms use
    the angle-sum identity on precomputed cos/sin tables. No transcendental function runs per pixel.
    """

    def __init__(self, w: int, h: int, seed: int = 7, header: str = ""):
        self.w, self.h = w, h
        s = min(w, h)
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        xs, ys = (x - w / 2) / s, (y - h / 2) / s
        r = np.sqrt(xs ** 2 + ys ** 2)
        th = np.arctan2(ys, xs)
        self.ridx = np.minimum((r / RMAX * (RSTEPS - 1)).astype(np.int32), RSTEPS - 1).ravel()
        b = 5.0 * th + 3.0 * np.log(r + 0.04)
        self.cosb, self.sinb = np.cos(b).ravel(), np.sin(b).ravel()
        self.rip = np.sin(12 * th).ravel()  # orb rim ripple pattern
        self.rr = np.linspace(0, RMAX, RSTEPS, dtype=np.float32)
        self.vign = np.clip(1.25 - 1.1 * self.rr, 0, 1) ** 1.5
        self.logr = np.log(self.rr + 0.04)
        rng = np.random.default_rng(seed)
        self.np = 700
        self.p_ang = rng.uniform(-math.pi, math.pi, self.np).astype(np.float32)
        self.p_rho0 = rng.uniform(0.0, 1.0, self.np).astype(np.float32)
        self.p_spd = rng.uniform(0.3, 1.0, self.np).astype(np.float32)
        self.p_tw = rng.uniform(0, 2 * math.pi, self.np).astype(np.float32)
        self.sprites: dict[tuple, np.ndarray] = {}
        self.header = _text_sprite(header, int(s * 0.045), max_w=int(w * 0.9)) if header else None

    def _sprite(self, key, text, size, bold=True):
        if key not in self.sprites:
            self.sprites[key] = _text_sprite(text, size, bold, max_w=int(self.w * 0.86))
        return self.sprites[key]

    @staticmethod
    def _blit(img: np.ndarray, spr: np.ndarray, x: int, y: int, alpha: float, colour=(1.0, 1.0, 1.0)):
        if alpha <= 0.01:
            return
        h, w = spr.shape[:2]
        H, W = img.shape[:2]
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        s = spr[y0 - y : y1 - y, x0 - x : x1 - x]
        reg = img[y0:y1, x0:x1]
        reg *= 1 - 0.65 * alpha * s[..., 1:2]  # shadow darkens
        reg += alpha * s[..., 0:1] * np.asarray(colour, np.float32)

    def frame(self, t: float, phase: float, travel: float, k: float, sn: float, hh: float, bass: float,
              en: float, hue_a: float, hue_b: float, mix: float, wave_ages: list[float],
              title: tuple[str, float] | None, progress: float) -> np.ndarray:
        ca, cb = _hsv(hue_a, 0.85, 1.0), _hsv(hue_b, 0.85, 1.0)
        c = ca * (1 - mix) + cb * mix
        c2 = _hsv((hue_a + 0.5) % 1 if mix < 0.5 else (hue_b + 0.5) % 1, 0.6, 1.0)
        rr = self.rr
        # --- radial profiles (1-D, cheap) ---
        tun_r = (0.5 + 0.5 * np.cos(9.0 * self.logr - phase)) ** 2 * self.vign * (0.22 + 0.9 * en + 0.45 * bass)
        R = 0.16 * (1.0 + 0.42 * k)
        width = 0.010 + 0.022 * k
        rim = np.exp(-((rr - R) / width) ** 2) * (0.7 + 0.8 * k)
        fill = np.clip(1.0 - rr / R, 0, 1) ** 2 * (0.25 + 0.9 * k)
        waves = np.zeros_like(rr)
        for age in wave_ages:
            rad = 0.16 + age * 0.95
            waves += np.exp(-((rr - rad) / (0.006 + 0.02 * age)) ** 2) * (1 - age / 1.1) ** 2
        flash = 0.14 * sn * self.vign
        rim_c = 0.55 * c + 0.6 * k * c2 + 0.25
        prof = (rim[:, None] * rim_c + fill[:, None] * c + waves[:, None] * (0.8 * c + 0.35 * c2 + 0.15)
                + flash[:, None] * (0.5 + 0.5 * c)).astype(np.float32)  # (RSTEPS, 3)
        # orb rim ripples with the hats: shift the lookup radius a little by angle
        idx = self.ridx
        if hh > 0.05:
            shift = (0.012 * hh * (RSTEPS - 1) / RMAX) * self.rip
            idx = np.clip(idx + shift.astype(np.int32), 0, RSTEPS - 1)
        img = np.take(prof, idx, axis=0)
        # --- tunnel: radial rings x spiral arms (angle-sum identity, no per-pixel cos) ---
        q = 0.35 * phase
        arms = 0.55 + 0.45 * (self.cosb * math.cos(q) + self.sinb * math.sin(q))
        tun = np.take(tun_r.astype(np.float32), self.ridx) * arms
        img += tun[:, None] * (0.85 * c)
        img = img.reshape(self.h, self.w, 3)
        # --- particles: a starfield flying outward; travel is the integrated speed ---
        rho = (self.p_rho0 + self.p_spd * travel) % 1.08 + 0.02
        s = min(self.w, self.h)
        px = (self.w / 2 + rho * np.cos(self.p_ang) * s * 0.9).astype(np.int32)
        py = (self.h / 2 + rho * np.sin(self.p_ang) * s * 0.9).astype(np.int32)
        ok = (px >= 1) & (px < self.w - 1) & (py >= 1) & (py < self.h - 1)
        if ok.any():
            bright = ((0.15 + 1.6 * hh) * (0.6 + 0.4 * np.sin(self.p_tw + 9 * t)) * np.clip(rho * 3, 0, 1))[ok]
            col = (0.55 + 0.45 * c2).astype(np.float32)
            px, py = px[ok], py[ok]
            for dx, dy, wgt in ((0, 0, 1.0), (1, 0, 0.5), (-1, 0, 0.5), (0, 1, 0.5), (0, -1, 0.5)):
                np.add.at(img, (py + dy, px + dx), (bright * wgt)[:, None] * col)
        # bloom: blur a quarter-size copy and add it back, for the neon glow
        if self.w % 4 == 0 and self.h % 4 == 0:
            from scipy.ndimage import uniform_filter

            small = img[1::4, 1::4]  # a sample per 4x4 block is enough for a blur this wide
            small = uniform_filter(np.maximum(small - 0.25, 0) * 0.9, size=(7, 7, 1), mode="constant")
            img.reshape(self.h // 4, 4, self.w // 4, 4, 3)[...] += small[:, None, :, None, :]
        # tone map
        img = 1.0 - np.exp(-1.9 * img)
        # text
        if self.header is not None:
            self._blit(img, self.header, (self.w - self.header.shape[1]) // 2, int(0.04 * self.h), 0.9)
        if title and title[0]:
            name, since = title
            name = name.upper()
            big = self._sprite(("big", name), name, int(s * 0.065))
            small = self._sprite(("small", name), name, int(s * 0.032), bold=False)
            fade = min(1.0, since / 0.8) * (1.0 - min(1.0, max(0.0, since - 7.0) / 1.2))
            self._blit(img, big, int(0.06 * self.w), int(0.78 * self.h), fade, 0.75 + 0.25 * c)
            self._blit(img, small, int(0.04 * self.w), int(0.915 * self.h), 0.85 * (1 - fade) + 0.15)
        y0 = self.h - max(2, self.h // 180)
        img[y0:, : int(self.w * progress)] = 0.35 + 0.65 * c
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------------------------------ top level

@dataclass
class VisualsResult:
    path: str
    start: float
    duration: float
    frames: int
    size: tuple[int, int]
    stills: list[str]


_WORKER: dict = {}


def _worker_init(w: int, h: int, header: str) -> None:
    _WORKER["r"] = Renderer(w, h, header=header)


def _worker_render(jobs: list[tuple]) -> bytes:
    r = _WORKER["r"]
    return b"".join(r.frame(*job).tobytes() for job in jobs)


def _frame_jobs(plan: Plan, feats: Features, f0: int, nfr: int, total: float) -> list[tuple]:
    """Everything that evolves over time, integrated up front, so every frame is independent."""
    fps = feats.fps
    dt = 1.0 / fps
    sl = slice(f0, f0 + nfr)
    k, sn, hh = feats.kick[sl], feats.snare[sl], feats.hats[sl]
    bass, en = feats.bass[sl], feats.energy[sl]
    # tunnel phase and particle travel: integrals of their speeds over the *whole* mix up to each frame,
    # so a reel cut from the middle looks exactly like that part of the full video
    phase_all = np.cumsum(dt * (1.2 + 5.0 * feats.bass))
    travel_all = np.cumsum(dt * (0.05 + 0.25 * feats.energy + 0.9 * feats.hats))
    hits = feats.snare_hits
    jobs = []
    for i in range(nfr):
        g = f0 + i
        t = g * dt
        recent = hits[(hits <= g) & (hits > g - int(1.1 * fps))]
        ages = [float((g - hgt) * dt) for hgt in recent[-6:]]
        ha, hb, mx = plan.colour_at(t)
        jobs.append((t, float(phase_all[g]), float(travel_all[g]), float(k[i]), float(sn[i]), float(hh[i]),
                     float(bass[i]), float(en[i]), ha, hb, mx, ages, plan.title_at(t), (t / total) if total else 0.0))
    return jobs


def render_visuals(audio_path: str, out: str, fmt: str = "youtube", mix_json: str | None = None,
                   start: float | None = None, duration: float | None = None, reel_auto: bool = False,
                   fps: int = 30, palette: str = "neon", header: str = "", stills: int = 0,
                   preset: str = "veryfast", crf: int = 20, workers: int | None = None,
                   progress=None) -> VisualsResult:
    """Render the video. ``reel_auto`` picks the most energetic drop (needs ``duration``, default 30 s)."""
    import multiprocessing as mp
    import os

    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    (ow, oh), (w, h) = FORMATS[fmt]
    plan = load_plan(mix_json, palette)
    feats = analyse(decode(audio_path), fps)
    total = feats.n / fps
    if reel_auto:
        duration = duration or 30.0
        start = pick_reel_window(plan, feats, duration)
    start = max(0.0, min(start or 0.0, total))
    duration = min(duration or total - start, total - start)
    f0, nfr = int(round(start * fps)), int(duration * fps)
    if nfr < 1:
        raise ValueError("nothing to render: start is at or past the end of the audio")
    jobs = _frame_jobs(plan, feats, f0, nfr, total)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    vf = f"scale={ow}:{oh}:flags=lanczos,format=yuv420p"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(audio_path),
           "-map", "0:v", "-map", "1:a", "-vf", vf, "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
           "-c:a", "aac", "-b:a", "320k", "-movflags", "+faststart", "-shortest", str(out)]
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    chunk = fps * 2
    batches = [jobs[a : a + chunk] for a in range(0, nfr, chunk)]
    still_at = sorted(set(np.linspace(nfr * 0.1, nfr * 0.9, stills).astype(int).tolist())) if stills else []
    still_paths: list[str] = []
    frame_bytes = w * h * 3
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        if workers > 1:
            ctx = mp.get_context("spawn")  # same behaviour on Windows, macOS and Linux
            pool = ctx.Pool(workers, initializer=_worker_init, initargs=(w, h, header))
            results = pool.imap(_worker_render, batches)
        else:
            _worker_init(w, h, header)
            pool, results = None, map(_worker_render, batches)
        done = 0
        for data in results:
            enc.stdin.write(data)
            for si in still_at:
                if done <= si < done + len(data) // frame_bytes:
                    from PIL import Image

                    fr = np.frombuffer(data, np.uint8, count=frame_bytes, offset=(si - done) * frame_bytes)
                    p = str(Path(out).with_suffix("")) + f"_still_{len(still_paths) + 1}.png"
                    Image.fromarray(fr.reshape(h, w, 3)).resize((ow, oh), Image.LANCZOS).save(p)
                    still_paths.append(p)
            done += len(data) // frame_bytes
            if progress:
                progress(done, nfr)
        if pool is not None:
            pool.close()
            pool.join()
    finally:
        enc.stdin.close()
        rc = enc.wait()
    if rc != 0:
        raise RuntimeError("ffmpeg failed while encoding the video")
    return VisualsResult(str(out), round(start, 3), round(duration, 3), nfr, (ow, oh), still_paths)


def _print_progress(i: int, n: int) -> None:
    if i == n or i % 900 < 60:
        print(f"  {100 * i / max(1, n):5.1f} %", file=sys.stderr, flush=True)
