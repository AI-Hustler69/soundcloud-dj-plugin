---
name: dj-visuals
description: Make an audio-reactive video for a DJ mix - a YouTube video of the whole set (16:9), a vertical reel of the best drop (9:16) or a square clip - where the visuals react separately to kick, snare and hi-hats, each track has its own colour and its title appears when it starts. Use when the user wants a video, visualizer, reel, Instagram/TikTok clip or YouTube upload for a mix, especially one made with dj-automix. Works only on audio files the user provides.
---

# DJ visuals

Render a video for a mix the user **already has as a file**, with the `visuals` command of the bundled `automix` tool. Reply in the user's language.

## Hard rules

- Use only audio files the user gives you. Never download or rip audio.
- The video carries the music. Before the user publishes it (YouTube, Instagram, TikTok), pass on the license note: every track in the mix needs a license that allows it. "Free download" usually means personal use only.
- The look was checked on still frames, not by watching: tell the user to watch the result before posting.

## 1. Set up (once)

The tool shares its Python environment with `dj-automix`. If that skill already set up `${CLAUDE_PLUGIN_DATA}/automix-venv`, only make sure Pillow is installed (step below). Otherwise create the venv. The commands differ by OS.

**macOS and Linux:**

```bash
python3 -m venv "${CLAUDE_PLUGIN_DATA}/automix-venv" && "${CLAUDE_PLUGIN_DATA}/automix-venv/bin/python" -m pip install -q -r "${CLAUDE_PLUGIN_ROOT}/automix/requirements.txt"
```

**Windows** (Git Bash, which Claude Code uses there):

```bash
python -m venv "${CLAUDE_PLUGIN_DATA}/automix-venv" && "${CLAUDE_PLUGIN_DATA}/automix-venv/Scripts/python.exe" -m pip install -q -r "${CLAUDE_PLUGIN_ROOT}/automix/requirements.txt"
```

Below, `PY` means `${CLAUDE_PLUGIN_DATA}/automix-venv/bin/python` (`.../Scripts/python.exe` on Windows). If the venv existed before this skill, run `PY -m pip install -q -r "${CLAUDE_PLUGIN_ROOT}/automix/requirements.txt"` once to add Pillow.

ffmpeg must be on PATH. Any recent build works here, rubberband is not needed for visuals.

## 2. Render

Run from the tool's folder so the `automix` package is importable:

```bash
cd "${CLAUDE_PLUGIN_ROOT}/automix" && PY -m automix visuals /abs/path/mix.mp3 --mix-json /abs/path/mix.json -o /abs/path/out/set_youtube.mp4 --title "DJ NAME - JUMP UP SESSION" --stills 4
```

Options:
- `--mix-json`: the JSON report `automix mix` wrote next to the mix. It gives track titles, one colour per track (crossfaded during blends) and the drops for `--reel-auto`. Without it the video still works, with no titles.
- `--format youtube | reel | square`: 1920x1080, 1080x1920 or 1080x1080. The default is `youtube`.
- `--reel-auto --duration 30`: picks the most energetic drop and starts 8 bars before it. Use it for reels unless the user names a moment. A manual moment is `--start 1190 --duration 30`, in seconds.
- `--title`: the header text (DJ name, event, session name).
- `--palette neon | ice | fire | toxic`.
- `--stills 4`: also saves 4 PNG frames, so you can look at the result without playing it.
- `--fps 30` (default), `--workers N` (default: CPU cores - 1), `--preset veryfast` (x264; `medium` gives smaller files but is slower).

For a whole set, run it in the background. On a 4-core machine it renders at about real time, so a 30-minute set takes about 30 minutes; more cores make it faster. Render a 20-second test with `--stills` first so the user can approve the look and the title.

If the user has no mix JSON (a mix from other software), render without it. Offer to add titles if they give you a tracklist with timestamps: write a small JSON in the same shape, `{"bpm": 174, "tracks": [{"title": ..., "first_downbeat": seconds}], "transitions": [{"start": s, "end": s, "bass_swap": s}]}`.

## 3. Check and report

- Look at the stills (open the PNGs). Check that the title is readable and not cut off, and that the frame is neither almost black nor blown out. If it is, adjust `--title` or the palette and re-render the test.
- Tell the user where the MP4 and the stills are, the format and length, and for a reel where it starts in the mix and why (which drop).
- Pass on the license note.

## What reacts to what

| Sound | Visual |
|---|---|
| kick | the centre orb swells and its rim glows |
| snare | a shockwave ring leaves the orb, with a short flash |
| hi-hats | particles sparkle and fly faster; the orb rim ripples |
| bass and overall level | the tunnel in the background moves faster and gets brighter |

Drums are detected from the mixed audio (spectral flux in separate bands), so no stems are needed. Very dense or heavily distorted passages can blur the kick and snare apart.
