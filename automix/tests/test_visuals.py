"""Visuals: drum detection, the mix plan, reel selection and an end-to-end render (synthetic audio only)."""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest
import soundfile as sf

from automix.visuals import (ANALYSIS_SR, Features, Plan, Renderer, TrackSpan, analyse, load_plan,
                             pick_reel_window, render_visuals)

FPS = 30
SR = ANALYSIS_SR


def _drums(seconds=8.0, kick_every=0.6, snare_at=(1.3, 3.1, 5.2), hat_every=0.15):
    t = np.arange(int(seconds * SR)) / SR
    y = np.zeros_like(t)
    kicks = np.arange(0.3, seconds - 0.3, kick_every)
    for k in kicks:
        i = int(k * SR)
        n = int(0.25 * SR)
        tt = np.arange(n) / SR
        y[i : i + n] += 0.8 * np.sin(2 * np.pi * (55 + 60 * np.exp(-tt * 40)) * tt) * np.exp(-tt * 12)
    rng = np.random.default_rng(0)
    from scipy.signal import butter, sosfilt

    hp = butter(4, 7500, "highpass", fs=SR, output="sos")
    bp = butter(4, [1500, 4000], "bandpass", fs=SR, output="sos")
    snare_at = tuple(s for s in snare_at if s < seconds - 0.2)
    for s in snare_at:
        i, n = int(s * SR), int(0.15 * SR)
        y[i : i + n] += 0.9 * sosfilt(bp, rng.standard_normal(n)) * np.exp(-np.arange(n) / SR * 25)
    hats = np.arange(0.1, seconds - 0.1, hat_every)
    for h in hats:
        i, n = int(h * SR), int(0.03 * SR)
        y[i : i + n] += 0.3 * sosfilt(hp, rng.standard_normal(n)) * np.exp(-np.arange(n) / SR * 120)
    return y.astype(np.float32), kicks, np.array(snare_at), hats


def _peaks(env, thr=0.6):
    up = np.nonzero((env[1:] >= thr) & (env[:-1] < thr))[0] + 1
    return up / FPS


def test_kicks_snares_and_hats_are_told_apart():
    y, kicks, snares, hats = _drums()
    f = analyse(y, FPS)
    assert f.n == pytest.approx(len(y) / SR * FPS, abs=1)
    kp = _peaks(f.kick)
    # every kick is found within 2 frames, and the kick envelope does not fire on snares alone
    assert all(np.min(np.abs(kp - k)) <= 2 / FPS for k in kicks[1:])
    sp = f.snare_hits / FPS
    assert all(np.min(np.abs(sp - s)) <= 2 / FPS for s in snares)
    assert len(sp) <= len(snares) + 2
    hp = _peaks(f.hats, 0.5)
    assert len(hp) >= 0.6 * len(hats)


def test_quiet_audio_gives_calm_envelopes():
    y = (0.001 * np.random.default_rng(1).standard_normal(4 * SR)).astype(np.float32)
    f = analyse(y, FPS)
    assert f.energy.mean() < 1.2 and np.isfinite(f.kick).all()


def _mix_json(tmp_path):
    d = {"bpm": 174.0,
         "tracks": [{"title": "Alpha", "first_downbeat": 0.5}, {"title": "Beta", "first_downbeat": 20.0},
                    {"title": "Gamma", "first_downbeat": 45.0}],
         "transitions": [{"start": 20.0, "end": 25.0, "bass_swap": 22.5},
                         {"start": 45.0, "end": 50.0, "bass_swap": 47.5}]}
    p = tmp_path / "mix.json"
    p.write_text(json.dumps(d))
    return p


def test_plan_colours_and_titles_follow_the_mix(tmp_path):
    plan = load_plan(_mix_json(tmp_path))
    a, b, m = plan.colour_at(10.0)
    assert a == b and m == 0.0  # inside track 1
    a, b, m = plan.colour_at(22.5)
    assert a != b and m == pytest.approx(0.5)  # halfway through the first blend
    assert plan.title_at(10.0)[0] == "Alpha"
    assert plan.title_at(24.0)[0] == "Alpha"  # the new title appears when the blend is over
    t, since = plan.title_at(26.0)
    assert t == "Beta" and since == pytest.approx(1.0)
    assert load_plan(None).title_at(5.0) == ("", 5.0)


def test_reel_window_starts_eight_bars_before_the_strongest_drop():
    plan = Plan(tracks=[TrackSpan("a", 0, 0.1)], bpm=174.0, bass_swaps=[30.0, 60.0])
    n = 100 * FPS
    energy = np.full(n, 0.3, np.float32)
    energy[60 * FPS : 80 * FPS] = 1.0  # the drop at 60 s is the big one
    z = np.zeros(n, np.float32)
    f = Features(FPS, z, z, z, z, energy, np.array([], int))
    start = pick_reel_window(plan, f, 20.0)
    assert start == pytest.approx(60.0 - 8 * 240 / 174, abs=1e-6)
    # without a plan, the loudest window wins
    s2 = pick_reel_window(Plan(), f, 20.0)
    assert 55.0 <= s2 <= 65.0


def test_frame_reacts_to_the_kick_and_is_deterministic():
    r = Renderer(160, 96)
    args = dict(t=1.0, phase=2.0, travel=0.5, sn=0.0, hh=0.2, bass=0.5, en=0.5, hue_a=0.5, hue_b=0.5, mix=0.0,
                wave_ages=[], title=None, progress=0.3)
    calm = r.frame(k=0.0, **args)
    hit = r.frame(k=1.0, **args)
    assert calm.shape == (96, 160, 3) and calm.dtype == np.uint8
    centre = (slice(30, 66), slice(62, 98))
    assert hit[centre].mean() > calm[centre].mean() + 10
    assert np.array_equal(r.frame(k=0.0, **args), calm)  # no hidden state


@pytest.mark.parametrize("workers", [1, 2])
def test_render_reel_end_to_end(tmp_path, workers):
    y, *_ = _drums(seconds=4.0)
    wav = tmp_path / "mix.wav"
    sf.write(wav, np.stack([y, y], 1), SR)
    out = tmp_path / "reel.mp4"
    res = render_visuals(str(wav), str(out), fmt="reel", header="TEST", stills=2, workers=workers,
                         start=0.5, duration=3.0)
    assert res.frames == 90 and res.size == (1080, 1920) and len(res.stills) == 2
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
                            "-show_entries", "format=duration", "-of", "json", str(out)],
                           capture_output=True, text=True, check=True)
    info = json.loads(probe.stdout)
    v = [s for s in info["streams"] if s["codec_type"] == "video"][0]
    assert (v["width"], v["height"]) == (1080, 1920)
    assert any(s["codec_type"] == "audio" for s in info["streams"])
    assert float(info["format"]["duration"]) == pytest.approx(3.0, abs=0.15)


def test_bad_arguments(tmp_path):
    y, *_ = _drums(seconds=2.0)
    wav = tmp_path / "a.wav"
    sf.write(wav, y, SR)
    with pytest.raises(ValueError):
        render_visuals(str(wav), str(tmp_path / "x.mp4"), fmt="cinema")
    with pytest.raises(ValueError):
        render_visuals(str(wav), str(tmp_path / "x.mp4"), start=99.0, workers=1)
