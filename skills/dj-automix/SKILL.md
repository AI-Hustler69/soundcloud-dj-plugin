---
name: dj-automix
description: Mix local audio files into one continuous, beat-matched DJ mix (time-stretch to one tempo, phrase-aligned transitions with a bass swap, optional samples such as loops, horns, vocals and risers placed on the grid, tracklist and cue sheet). Use when the user wants to auto-mix, beat-match or blend tracks they already have as files, or turn a crate built with dj-crate into an actual mix. Works only on files the user provides; never downloads anything.
---

# DJ automix (prototype)

Turn tracks the user **already has as local files** into a continuous mix with the bundled `automix` tool. Reply in the user's language.

## Hard rules

- Use **only files the user gives you** (paths they name, or files they upload). Never download, stream-rip or record audio from SoundCloud or anywhere else, and never suggest tools that do. If the user has no files yet, point them to the track pages from `dj-crate`, where they can use SoundCloud's own Download button.
- Tell the user the result was checked by measurement, not by ear, and that they should listen before publishing.
- Always pass on the license note from the tracklist. "Free download" usually means personal use only, so a published mix needs every track's license to allow it.

## 1. Set up (once)

The tool lives in `${CLAUDE_PLUGIN_ROOT}/automix`. Create its Python environment in the plugin's data folder, which survives plugin updates:

```bash
ffmpeg -hide_banner -filters | grep -q rubberband && echo "ffmpeg ok" || echo "ffmpeg without rubberband"
```

```bash
python3 -m venv "${CLAUDE_PLUGIN_DATA}/automix-venv" && "${CLAUDE_PLUGIN_DATA}/automix-venv/bin/pip" install -q -r "${CLAUDE_PLUGIN_ROOT}/automix/requirements.txt"
```

- If ffmpeg is missing or built without rubberband, tell the user how to install it (for example `brew install ffmpeg` on macOS, or a full ffmpeg build on Linux) and stop.
- Python 3.11 or newer is required.

## 2. Build the crate file

Write a crate JSON with the tracks **in play order**. When the tracks came from `dj-crate` / `get_track`, copy the Camelot key and license into it; metadata keys override the tool's audio estimate:

```json
[
  {"path": "/abs/path/one.mp3", "title": "Artist A - One", "camelot": "8A", "license": "cc-by"},
  {"path": "/abs/path/two.mp3", "title": "Artist B - Two", "camelot": "9A", "license": "cc-by-nc"}
]
```

Without metadata, list just `path` and `title`, or use `--files` (step 3) and the tool estimates the keys.

### Samples (optional)

If the user has samples (loops, horns, vocal shots, impacts, risers), use the object form of the crate: `{"tracks": [...], "samples": [...]}`. Each sample has a `path` and an anchor `at`:

- `{"transition": 1}`, or `{"transition": 1, "bar": -8}` to start 8 bars before it;
- `{"track": "Title" or 2, "bar": 32}`. Use titles when `--auto-order` may reorder the tracks;
- `{"time": "05:12"}`, snapped to the nearest downbeat.

Further fields:
- `"align": "end"` lets a riser finish on the anchor;
- `"bars": 8` makes a loop repeat for 8 bars;
- `"gain_db": -6` sets the level (default −3);
- `"kind": "loop"` or `"oneshot"` overrides auto-detection.

Loops are stretched to the mix tempo; one-shots keep their tempo, and their attack lands on the beat.

If the user does not say where samples should go, propose positions that DJs typically use and say what you chose:
- a horn or impact on a transition's incoming downbeat or on a drop;
- a percussion loop over the 8 bars before a transition;
- a riser ending on a drop.

## 3. Mix

Run it from the tool's folder so the `automix` package is importable:

```bash
cd "${CLAUDE_PLUGIN_ROOT}/automix" && "${CLAUDE_PLUGIN_DATA}/automix-venv/bin/python" -m automix mix --crate /abs/path/crate.json -o /abs/path/out/mix
```

Options:
- `--files a.mp3 b.mp3 ...` instead of `--crate`;
- `--bpm 174` for a fixed tempo (the default is the crate's median);
- `--bars 32` for longer blends (default 16; must be even);
- `--auto-order` to sort by BPM and Camelot;
- `--bpm-range 120 135` for house or techno (the default 160–190 suits drum & bass).

Write the output next to the user's files or in their project, not inside `${CLAUDE_PLUGIN_ROOT}`, which is replaced on update.

To try the tool without any music, `... -m automix demo -o /abs/path/out/demo/mix` builds four synthetic tracks and mixes them.

## 4. Report back

Read the printed tracklist (also written to `mix.tracklist.txt`) and tell the user:

- where `mix.mp3`, `mix.wav`, `mix.cue` and `mix.json` are;
- the tracklist with timestamps, each track's tempo change in %, and the key relation of every transition (`adjacent`, `relative`, `clash`, …);
- the **measured beat alignment** per transition (median over 4-bar parts). Within about ±5 ms is tight. Larger values mean the beat grid of that track is unreliable, so suggest checking it. If the tracklist names an "irregular passage", only part of the blend is off (often an intro effect that is not on the beat): mention it so the user listens to that spot;
- every warning (for example a tempo change above 8 %, or a track too short for the transition);
- every sample with its time, type (loop or one-shot), tempo change and anchor;
- the license note. It covers samples too: sample packs have their own licenses.

## Limits to mention when relevant

- The tool assumes constant tempo, 4/4 and a kick on the downbeat. Tracks with tempo changes, half-time intros or no kick on "1" can get a wrong grid.
- Keys estimated from audio can confuse major/minor or relative keys, so prefer metadata keys.
- Mix points are chosen by phrase structure, not taste: it mixes out at the last 8-bar phrase that leaves room for the blend.
