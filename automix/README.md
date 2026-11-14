# automix (prototype)

Beat-matched DJ mixes from **local audio files you are allowed to use**, for example tracks you downloaded with the artist's own Download button after building a crate with this plugin's `dj-crate` skill. In Claude Code the `dj-automix` skill runs it for you; this page covers running it by hand.

automix **never downloads anything** and never touches streams. It works only on files you give it.

## What it does

1. **Analyses** every track: tempo, a sample-accurate beat/downbeat grid, an estimate of the musical key (Camelot), and energy per bar.
2. **Beat-matches**: time-stretches every track to one tempo (the median of the crate, or `--bpm`) with [Rubber Band](https://breakfastquay.com/rubberband/) through ffmpeg, preserving pitch, then re-measures the grid on the stretched audio.
3. **Plans** transitions on phrase boundaries (every 8 bars): the incoming track's first downbeat lands on the outgoing track's mix-out downbeat, with 16 bars of overlap by default.
4. **Blends** like a DJ with a bass swap: the incoming highs rise over the first half, the bass lines swap at the midpoint (never two bass lines at once), and the outgoing highs fade over the second half.
5. **Exports** `mix.mp3` / `mix.wav`, a tracklist with timestamps, a `.cue` sheet and a JSON report. The report includes the measured kick alignment of every transition, BPM and stretch per track, key relations and licenses.

## Install

Requirements: Python 3.11+, and ffmpeg built with `rubberband` (`ffmpeg -filters | grep rubberband`). Run the commands below from this `automix/` folder.

```bash
python3 -m venv .venv
```

```bash
.venv/bin/pip install -r requirements.txt
```

## Use

Try it without any music (generates four synthetic DnB tracks):

```bash
.venv/bin/python -m automix demo -o out/demo/mix
```

Mix your own files in play order:

```bash
.venv/bin/python -m automix mix --files a.mp3 b.mp3 c.mp3 -o out/mymix
```

Or use a crate file. `camelot` and `license` are optional and can be copied from the plugin's `get_track` output. Metadata keys override the audio estimate:

```json
[
  {"path": "tracks/one.mp3", "title": "Artist A - One", "camelot": "8A", "license": "cc-by"},
  {"path": "tracks/two.mp3", "title": "Artist B - Two", "camelot": "9A", "license": "cc-by-nc"}
]
```

```bash
.venv/bin/python -m automix mix --crate crate.json -o out/mymix --bars 16
```

Options:
- `--bpm 174` fixes the target tempo.
- `--bars 32` sets a longer blend.
- `--auto-order` sorts the crate by BPM and Camelot compatibility.
- `--bpm-range 120 135` sets the tempo range for house or techno. The default is 160–190 (drum & bass).

To inspect tracks without mixing:

```bash
.venv/bin/python -m automix analyze a.mp3 b.mp3
```

## Tests

```bash
.venv/bin/python -m pytest -q tests
```

The tests use synthetic tracks with known tempo, downbeat and key (no third-party audio, no network). They check:
- tempo within ±0.05 BPM and downbeat within 3 ms, including a downbeat at sample 0 and long leading silence;
- key estimates;
- transitions aligned within 3 ms and starting on phrase boundaries;
- the bass swap actually preventing double bass;
- no clipping, and that the outputs and the license note are written.

## Limits (prototype)

- **Constant tempo, 4/4, a kick on the downbeat** are assumed (true for most DnB, house and techno). Tracks with tempo changes, half-time intros or no kick on "1" can get a wrong grid.
- **Key detection is an estimate.** Major/minor and relative-key mix-ups happen, as with every audio-only key detector, so pass `camelot` from metadata when you have it.
- **Mix points are chosen by structure, not taste.** It blends from the last 8-bar phrase that leaves room for the transition. It does not judge energy, vocals or the "right moment" like a DJ.
- **Tempo:** every track is stretched to one tempo, and stretches beyond ±8 % are flagged.
- **No listening test was possible here.** Alignment and the bass swap are verified by measurement. Listen before you publish.

## Licensing

You are responsible for having the right to use each file. "Free download" usually means **personal use only**. Before you publish, stream or monetise a mix, check every track's license and ask the artists when in doubt. The tracklist repeats this note and lists each track's license when you provide it.
