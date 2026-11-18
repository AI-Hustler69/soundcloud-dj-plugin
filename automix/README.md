# automix (prototype)

Beat-matched DJ mixes from **local audio files you are allowed to use**, for example tracks you downloaded with the artist's own Download button after building a crate with this plugin's `dj-crate` skill. In Claude Code the `dj-automix` skill runs it for you; this page covers running it by hand.

automix **never downloads anything** and never touches streams. It works only on files you give it.

## What it does

1. **Analyses** every track: tempo, a sample-accurate beat/downbeat grid, an estimate of the musical key (Camelot), and energy per bar.
2. **Beat-matches**: time-stretches every track to one tempo (the median of the crate, or `--bpm`) with [Rubber Band](https://breakfastquay.com/rubberband/) through ffmpeg, preserving pitch, then re-measures the grid on the stretched audio.
3. **Plans** transitions on phrase boundaries (every 8 bars): the incoming track's first downbeat lands on the outgoing track's mix-out downbeat, with 16 bars of overlap by default.
4. **Blends** like a DJ with a bass swap: the incoming highs rise over the first half, the bass lines swap at the midpoint (never two bass lines at once), and the outgoing highs fade over the second half.
5. **Exports** `mix.mp3` / `mix.wav`, a tracklist with timestamps, a `.cue` sheet and a JSON report. The report includes the measured beat alignment of every transition (median over 4-bar parts), BPM and stretch per track, key relations and licenses.

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
- `--clash-bars 8` gives transitions between clashing keys a shorter blend.
- `--lufs -9` sets the master loudness (default -9 LUFS); `--no-limiter` turns the limiter off and only sets the true peak to -1.5 dBTP.

The master: every track is levelled by integrated loudness (ITU-R BS.1770) before blending, the mix is
brought to the loudness target with a look-ahead true-peak limiter (ceiling -1.5 dBTP, at most 12 dB of
gain reduction; the tracklist warns if the target needs more), and the MP3 is decoded and measured after
encoding and re-encoded quieter if it would go over -1 dBTP.

To inspect tracks without mixing:

```bash
.venv/bin/python -m automix analyze a.mp3 b.mp3
```

## Samples (loops and one-shots)

Put samples on top of the mix (horns, vocal shots, impacts, percussion loops, risers) by giving the crate an object form with a `samples` list:

```json
{
  "tracks": [
    {"path": "tracks/one.mp3", "title": "Artist A - One", "camelot": "12A"},
    {"path": "tracks/two.mp3", "title": "Artist B - Two"}
  ],
  "samples": [
    {"path": "samples/horn.wav", "at": {"transition": 1}, "gain_db": -4},
    {"path": "samples/perc_172.wav", "at": {"transition": 1, "bar": -8}, "bars": 8, "gain_db": -8},
    {"path": "samples/riser.wav", "at": {"track": 2, "bar": 32}, "align": "end"},
    {"path": "samples/vocal.wav", "at": {"time": "05:12"}}
  ]
}
```

**Where it lands (`at`):**
- `{"transition": N, "bar": B}`: the N-th transition's incoming downbeat, plus B bars. B may be negative.
- `{"track": N or "title", "bar": B}`: bar B of a track. N is 1-based in the final play order, so with `--auto-order` refer to tracks by title.
- `{"time": "mm:ss"}`: a mix time, snapped to the nearest downbeat (`"snap": false` keeps it exact).
- `"align": "end"` makes the sample *finish* at the anchor, which suits risers into a drop.

**Loop or one-shot:** this is detected automatically. Force it with `"kind": "loop"` or `"kind": "oneshot"`.
- **Loops:** a file whose length is within 6 % of 1/2/4/8/16 bars at the mix tempo (half- or double-time loops included). It is time-stretched to the mix tempo and repeated for `bars` bars (default: one pass). `loop_bars` overrides the detected loop length. Loops should start exactly on their first downbeat, as most sample-pack loops do.
- **One-shots:** everything else, played at the original tempo. The sound's attack (not the start of the file) lands on the beat, so a little silence before the hit is fine.

**Level:** `gain_db` sets it (default −3 dB). Samples are summed before the master limiter.

The tracklist lists every sample with its time, type, tempo change and anchor. The same licensing rules apply to samples: use only packs whose license allows them in a published mix.

## Video for a mix

`visuals` renders an audio-reactive MP4 of a mix: the whole set for YouTube, a vertical reel or a square clip.

```bash
.venv/bin/python -m automix visuals out/mymix.mp3 --mix-json out/mymix.json -o out/mymix_youtube.mp4 --title "DJ NAME - SESSION" --stills 4
```

```bash
.venv/bin/python -m automix visuals out/mymix.mp3 --mix-json out/mymix.json --format reel --reel-auto --duration 30 -o out/mymix_reel.mp4
```

- **Drums from the mixed audio:** kick (35–120 Hz), snare (1.2–4.5 kHz) and hats (7–14 kHz) are taken as spectral flux in their own bands, normalised to the local level so breakdowns and drops both react. No stems are needed.
- **What reacts:**
  - kick: the orb swells;
  - snare: shockwave rings and a flash;
  - hats: particles and a rim ripple;
  - bass and level: the tunnel.
- **From the mix JSON:** each track's colour (crossfaded over its blend), its title (shown once its blend is over) and the drops. `--reel-auto` starts 8 bars before the bass swap with the most energy after it.
- **Options:** `--format youtube|reel|square`, `--start`/`--duration`, `--palette neon|ice|fire|toxic`, `--fps`, `--workers` (default: CPU cores - 1), `--preset` (x264).
- **Speed:** frames are stateless (tunnel phase and particle travel are integrated up front), so they render in parallel processes. Radial elements use 1-D lookup tables, the glow is computed at quarter size, and ffmpeg scales the frames up. A 4-core machine renders about real time.

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
- **Samples are placed where you say.** It does not choose sample positions itself, and does not duck or EQ the tracks under a sample.
- **Mix points are chosen by structure, not taste.** It blends from the last 8-bar phrase that leaves room for the transition. It does not judge energy, vocals or the "right moment" like a DJ.
- **Tempo:** every track is stretched to one tempo, and stretches beyond ±8 % are flagged.
- **No listening test was possible here.** Alignment and the bass swap are verified by measurement. Listen before you publish.

## Licensing

You are responsible for having the right to use each file. "Free download" usually means **personal use only**. Before you publish, stream or monetise a mix, check every track's license and ask the artists when in doubt. The tracklist repeats this note and lists each track's license when you provide it.
