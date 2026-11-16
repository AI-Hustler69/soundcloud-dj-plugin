"""Tests run on synthetic audio with known ground truth; no third-party audio, no network."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest
import soundfile as sf

from automix.analysis import analyze, fold_tempo
from automix.keys import compatibility, estimate_key, MINOR_PROFILE
from automix.mixer import TrackInput, order_crate, render_mix
from automix.report import cue_sheet, tracklist, ts, write_outputs
from automix.stretch import SR
from automix.synth import SynthSpec, synth_track

warnings.filterwarnings("ignore", category=UserWarning)


def _write(tmp_path, name, spec):
    p = tmp_path / f"{name}.wav"
    sf.write(p, synth_track(spec), SR)
    return str(p)


# ---------------------------------------------------------------- keys

def test_camelot_compatibility_rules():
    assert compatibility("8A", "8A") == "same"
    assert compatibility("8A", "9A") == "adjacent"
    assert compatibility("12A", "1A") == "adjacent"
    assert compatibility("8A", "8B") == "relative"
    assert compatibility("8A", "10A") == "energy_boost"
    assert compatibility("11B", "1B") == "energy_boost"
    assert compatibility("8A", "3B") == "clash"
    assert compatibility("", "8A") == "unknown"
    assert compatibility("13A", "8A") == "unknown"


def test_estimate_key_from_profile():
    # A pure A-minor profile must come back as A minor / 8A.
    name, code, conf = estimate_key(np.roll(MINOR_PROFILE, 9))
    assert (name, code) == ("A minor", "8A")
    assert conf > 0.5
    assert estimate_key(np.zeros(12))[1] == ""


def test_fold_tempo_into_dnb_range():
    assert fold_tempo(87.0, 160, 190) == 174.0
    assert fold_tempo(348.0, 160, 190) == 174.0
    assert fold_tempo(174.0, 160, 190) == 174.0


# ---------------------------------------------------------------- analysis

@pytest.mark.parametrize(
    "bpm,lead_in,root,minor,camelot",
    [
        (174.0, 0.37, "A", True, "8A"),
        (171.5, 1.10, "E", True, "9A"),
        (175.5, 0.00, "B", True, "10A"),  # downbeat at sample 0
        (185.0, 2.30, "F", True, "4A"),   # long leading silence
        (165.0, 0.05, "G", False, "9B"),
    ],
)
def test_analysis_tempo_and_downbeat(tmp_path, bpm, lead_in, root, minor, camelot):
    path = _write(tmp_path, "t", SynthSpec(bpm, root, minor, bars=32, lead_in=lead_in, seed=int(bpm)))
    a = analyze(path)
    assert a.bpm == pytest.approx(bpm, abs=0.05)
    assert a.first_downbeat == pytest.approx(lead_in, abs=0.003)  # within 3 ms
    assert a.bars >= 31


@pytest.mark.parametrize("bpm,lead_in", [(165.0, 0.40), (174.0, 1.00), (172.0, 0.00)])
def test_analysis_survives_distorted_jump_up_bass(tmp_path, bpm, lead_in):
    """Loud, saturated off-beat bass stabs must not pull the grid (regression: 165 read as 169.4)."""
    path = _write(tmp_path, "ju", SynthSpec(bpm, "F", True, bars=40, lead_in=lead_in, seed=int(bpm * 10), jumpup=True))
    a = analyze(path)
    assert a.bpm == pytest.approx(bpm, abs=0.05)
    assert a.first_downbeat == pytest.approx(lead_in, abs=0.004)


def test_analysis_key_estimate_majority(tmp_path):
    cases = [("A", True, "8A"), ("E", True, "9A"), ("C", False, "8B"), ("F#", True, "11A"), ("G", False, "9B"), ("A#", False, "6B")]
    hits = 0
    for i, (root, minor, code) in enumerate(cases):
        a = analyze(_write(tmp_path, f"k{i}", SynthSpec(174, root, minor, bars=32, seed=40 + i)))
        hits += a.camelot == code
    assert hits >= 5  # an estimate: metadata keys override it in the mixer


# ---------------------------------------------------------------- mixer

@pytest.fixture(scope="module")
def crate(tmp_path_factory):
    d = tmp_path_factory.mktemp("crate")
    specs = [
        ("One", SynthSpec(172.0, "A", True, bars=48, lead_in=0.20, seed=1), "8A"),
        ("Two", SynthSpec(174.0, "E", True, bars=48, lead_in=0.55, seed=2), "9A"),
        ("Three", SynthSpec(175.5, "B", True, bars=48, lead_in=0.00, seed=3), "10A"),
    ]
    tracks = []
    for title, spec, cam in specs:
        p = _write(d, title, spec)
        tracks.append(TrackInput(path=p, title=title, analysis=analyze(p), camelot=cam, license="cc-by"))
    return tracks


@pytest.fixture(scope="module")
def mix(crate):
    return render_mix(crate, bpm=None, transition_bars=16)


def test_mix_uses_median_tempo_and_keeps_order(mix):
    assert mix.bpm == pytest.approx(174.0)
    assert [t.title for t in mix.tracks] == ["One", "Two", "Three"]
    assert [t.stretch_pct for t in mix.tracks] == pytest.approx([1.16, 0.0, -0.85], abs=0.01)


def test_transitions_are_beat_aligned(mix):
    assert len(mix.transitions) == 2
    for tr in mix.transitions:
        assert tr.measured_offset_ms is not None
        assert abs(tr.measured_offset_ms) <= 3.0  # kicks land together
        assert tr.end - tr.start == pytest.approx(16 * 4 * 60 / 174.0, abs=1e-6)
        assert tr.bass_swap == pytest.approx(tr.start + 8 * 4 * 60 / 174.0, abs=1e-6)


def test_transitions_start_on_phrase_boundaries(mix):
    bar = 4 * 60 / 174.0
    for prev, tr in zip(mix.tracks, mix.transitions):
        bars_in = (tr.start - prev.first_downbeat) / bar
        assert bars_in == pytest.approx(round(bars_in), abs=1e-3)
        assert round(bars_in) % 8 == 0
        assert round(bars_in) >= 24  # plays most of the track, not cut after the intro


def test_key_relations_use_metadata(mix):
    assert [t.key_relation for t in mix.transitions] == ["adjacent", "adjacent"]
    assert all(t.camelot_source == "metadata" for t in mix.tracks)


def test_master_is_not_clipping_and_has_expected_length(mix):
    from automix.mixer import TRUE_PEAK_CEILING_DB
    peak = float(np.max(np.abs(mix.audio)))
    # Levelled by true peak, so the sample peak sits at or a little under the ceiling.
    assert 0.6 < peak <= 10 ** (TRUE_PEAK_CEILING_DB / 20) + 1e-4
    # 3 tracks of 48 bars, two 16-bar overlaps -> about 48*3 - 32 = 112 bars plus tails.
    assert 112 * 4 * 60 / 174.0 < mix.duration < 125 * 4 * 60 / 174.0


def test_bass_swap_avoids_double_bass(mix):
    """In the first half of a blend only the outgoing track's bass should play."""
    from scipy.signal import butter, sosfiltfilt

    lo = butter(4, 120, btype="lowpass", fs=SR, output="sos")
    low = sosfiltfilt(lo, mix.audio.mean(axis=1))
    tr = mix.transitions[0]
    bar = 4 * 60 / 174.0

    def rms(a, b):
        seg = low[int(a * SR) : int(b * SR)]
        return float(np.sqrt(np.mean(seg**2)))

    before = rms(tr.start - 4 * bar, tr.start)
    first_half = rms(tr.start + bar, tr.bass_swap - bar)
    # Without the swap two bass lines would add up (~+3 dB); allow a small tolerance.
    assert first_half < before * 1.25


def test_too_short_track_is_rejected(tmp_path):
    p = _write(tmp_path, "short", SynthSpec(174, "A", True, bars=16, seed=9))
    t = TrackInput(path=p, title="short", analysis=analyze(p))
    with pytest.raises(ValueError, match="too short"):
        render_mix([t, t], transition_bars=16)


def test_invalid_transition_length(crate):
    with pytest.raises(ValueError):
        render_mix(crate, transition_bars=7)


def test_auto_order_by_bpm_and_key(crate):
    shuffled = [crate[2], crate[0], crate[1]]
    assert [t.title for t in order_crate(shuffled)] == ["One", "Two", "Three"]


# ---------------------------------------------------------------- outputs

def test_outputs_and_license_note(mix, tmp_path):
    files = write_outputs(mix, tmp_path / "mix", mp3=True)
    for key in ("wav", "mp3", "tracklist", "cue", "json"):
        assert (tmp_path / files[key].split("/")[-1]).exists()
    text = tracklist(mix)
    assert "license: cc-by" in text
    assert "personal use only" in text
    data = json.loads((tmp_path / "mix.json").read_text())
    assert data["bpm"] == pytest.approx(174.0)
    cue = cue_sheet(mix, "mix.mp3")
    assert cue.count("TRACK ") == 3 and 'FILE "mix.mp3" MP3' in cue


def test_timestamp_format():
    assert ts(0) == "00:00"
    assert ts(75.4) == "01:15"
    assert ts(3725) == "1:02:05"


def test_true_peak_catches_inter_sample_overs():
    from automix.mixer import true_peak
    # A sine at fs/4 sampled at 45 degrees: every sample is 0.707, the waveform peaks at 1.0 between them.
    n = np.arange(4096)
    x = np.sin(2 * np.pi * n / 4 + np.pi / 4)
    stereo = np.stack([x, x], axis=1) * 0.9
    assert np.max(np.abs(stereo)) == pytest.approx(0.9 * 0.7071, abs=1e-3)
    assert true_peak(stereo) == pytest.approx(0.9, abs=0.02)
    # chunked and unchunked agree
    assert true_peak(stereo, chunk=1000) == pytest.approx(true_peak(stereo), abs=1e-6)


def test_master_stays_under_true_peak_ceiling(mix):
    from automix.mixer import TRUE_PEAK_CEILING_DB, true_peak
    tp_db = 20 * np.log10(true_peak(mix.audio))
    assert TRUE_PEAK_CEILING_DB - 0.1 <= tp_db <= TRUE_PEAK_CEILING_DB + 0.05
