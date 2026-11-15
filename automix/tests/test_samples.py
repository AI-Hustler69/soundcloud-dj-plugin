"""Sample placement tests on synthetic audio (no third-party audio, no network)."""
from __future__ import annotations

import json
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from automix.__main__ import _load_crate
from automix.samples import SampleInput, classify, parse_time, render_samples, snap_to_downbeat
from automix.stretch import SR
from automix.synth import synth_loop, synth_oneshot

warnings.filterwarnings("ignore", category=UserWarning)

BPM = 174.0
BAR = 4 * 60 / BPM
TRACKS = [SimpleNamespace(title="One", first_downbeat=0.5), SimpleNamespace(title="Two", first_downbeat=0.5 + 32 * BAR)]
TRANSITIONS = [SimpleNamespace(start=0.5 + 32 * BAR)]


def _wav(tmp_path, name, audio):
    p = tmp_path / f"{name}.wav"
    sf.write(p, audio, SR)
    return str(p)


def _render(samples, seconds=120.0):
    mix = np.zeros((int(seconds * SR), 2))
    return render_samples(mix, samples, BPM, TRACKS, TRANSITIONS)


def _first_attack(audio: np.ndarray, frm: int = 0) -> int:
    mono = np.abs(audio.mean(axis=1))
    idx = np.nonzero(mono[frm:] > 0.1 * mono.max())[0]
    return frm + int(idx[0])


# ---------------------------------------------------------------- classification

def test_classify_loops_and_oneshots():
    assert classify(synth_loop(172.0, 2)[:, None], BPM) == ("loop", 2, pytest.approx(172.0, abs=0.01))
    # A half-time loop (2 bars at 87 BPM) lasts 4 bars at 174.
    kind, bars, bpm = classify(synth_loop(87.0, 2)[:, None], BPM)
    assert (kind, bars) == ("loop", 4) and bpm == pytest.approx(174.0, abs=0.05)
    assert classify(synth_oneshot(0.8)[:, None], BPM)[0] == "oneshot"
    # 2.25 bars long: too far from whole bars to auto-stretch as a loop.
    vocal = np.zeros((int(2.25 * BAR * SR), 1), dtype=np.float32)
    vocal[100] = 1.0
    assert classify(vocal, BPM)[0] == "oneshot"
    assert classify(vocal, BPM, kind="loop")[0] == "loop"


def test_parse_time():
    assert parse_time("12:30") == 750.0
    assert parse_time("1:02:03") == 3723.0
    assert parse_time(42) == 42.0


def test_snap_to_downbeat_uses_playing_track():
    t = TRACKS[1].first_downbeat + 3.3 * BAR
    assert snap_to_downbeat(t, TRACKS, BAR) == pytest.approx(TRACKS[1].first_downbeat + 3 * BAR)


# ---------------------------------------------------------------- placement

def test_oneshot_attack_lands_on_transition_bar(tmp_path):
    p = _wav(tmp_path, "horn", synth_oneshot(0.8, lead_in=0.05))
    mix, placed = _render([SampleInput(path=p, title="Horn", at={"transition": 1, "bar": -4}, gain_db=0)])
    target = TRANSITIONS[0].start - 4 * BAR
    got = _first_attack(mix) / SR
    assert got == pytest.approx(target, abs=0.001)  # the attack, not the file start, is on the beat
    assert placed[0].kind == "oneshot"
    assert placed[0].start == pytest.approx(target, abs=0.001)


def test_loop_is_synced_repeated_and_on_the_grid(tmp_path):
    p = _wav(tmp_path, "loop", synth_loop(170.0, 2))  # different tempo: must be stretched to 174
    mix, placed = _render([SampleInput(path=p, title="Loop", at={"track": "Two", "bar": 8}, bars=8, gain_db=0)])
    s = placed[0]
    assert (s.kind, s.loop_bars) == ("loop", 2)
    assert s.stretch_pct == pytest.approx(100 * (174 / 170 - 1), abs=0.05)
    start = TRACKS[1].first_downbeat + 8 * BAR
    assert s.start == pytest.approx(start, abs=1 / SR * 2)
    assert s.end - s.start == pytest.approx(8 * BAR, abs=0.002)
    # Every downbeat of the 8 bars carries the loop's accented click, within 2 ms.
    mono = np.abs(mix.mean(axis=1))
    for k in range(8):
        t = start + k * BAR
        seg = mono[int((t - 0.01) * SR) : int((t + 0.01) * SR)]
        hit = (np.nonzero(seg > 0.3 * mono.max())[0][0]) / SR - 0.01
        assert abs(hit) < 0.002, f"bar {k}: click off by {hit * 1000:.1f} ms"


def test_align_end_finishes_at_anchor(tmp_path):
    p = _wav(tmp_path, "riser", synth_loop(174.0, 4))
    _, placed = _render([SampleInput(path=p, title="Riser", at={"transition": 1}, align="end")])
    assert placed[0].end == pytest.approx(TRANSITIONS[0].start, abs=0.001)


def test_time_anchor_snaps(tmp_path):
    p = _wav(tmp_path, "hit", synth_oneshot(0.3, lead_in=0.0))
    t = TRACKS[0].first_downbeat + 5.4 * BAR
    _, placed = _render([SampleInput(path=p, title="Hit", at={"time": t})])
    assert placed[0].start == pytest.approx(TRACKS[0].first_downbeat + 5 * BAR, abs=0.001)
    assert any("snapped" in n for n in placed[0].notes)
    _, raw = _render([SampleInput(path=p, title="Hit", at={"time": t}, snap=False)])
    assert raw[0].start == pytest.approx(t, abs=0.001)


def test_gain_and_buffer_growth(tmp_path):
    p = _wav(tmp_path, "tail", synth_oneshot(2.0, lead_in=0.0))
    mix, placed = render_samples(np.zeros((int(1.0 * SR), 2)), [SampleInput(path=p, title="Tail", at={"time": 0.5}, snap=False, gain_db=-6)],
                                 BPM, TRACKS, TRANSITIONS)
    assert len(mix) >= int(2.5 * SR) - 1  # grew to fit the sample
    assert np.max(np.abs(mix)) == pytest.approx(0.8 * 10 ** (-6 / 20), rel=0.02)


@pytest.mark.parametrize("at", [{"track": 5}, {"transition": 3}, {"track": "Nope"}, {"bar": 4}])
def test_bad_anchors_raise(tmp_path, at):
    p = _wav(tmp_path, "x", synth_oneshot(0.2))
    with pytest.raises(ValueError):
        _render([SampleInput(path=p, title="X", at=at)])


# ---------------------------------------------------------------- crate format

def test_crate_object_with_samples(tmp_path):
    (tmp_path / "crate.json").write_text(json.dumps({
        "tracks": [{"path": "a.wav", "title": "A"}],
        "samples": [{"path": "horn.wav", "at": {"transition": 1}}],
    }))
    tracks, samples = _load_crate(str(tmp_path / "crate.json"))
    assert tracks[0]["path"] == str(tmp_path / "a.wav")
    assert samples[0]["path"] == str(tmp_path / "horn.wav")
    (tmp_path / "old.json").write_text(json.dumps([{"path": "a.wav"}]))
    assert _load_crate(str(tmp_path / "old.json"))[1] == []
    (tmp_path / "bad.json").write_text(json.dumps({"tracks": [{"path": "a.wav"}], "samples": [{"path": "h.wav"}]}))
    with pytest.raises(SystemExit):
        _load_crate(str(tmp_path / "bad.json"))
