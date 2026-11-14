"""Command line: python -m automix {analyze,mix,demo} ...

Works only on local audio files you supply. It never downloads anything.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analysis import analyze
from .mixer import TrackInput, order_crate, render_mix
from .report import tracklist, write_outputs


def _load_crate(path: str) -> list[dict]:
    """Crate JSON: [{"path": "...", "title": "...", "camelot": "8A", "license": "cc-by"}, ...]

    "camelot" and "license" are optional (e.g. copied from the soundcloud-dj plugin's get_track output).
    Relative paths are resolved against the crate file's folder.
    """
    base = Path(path).parent
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise SystemExit("crate JSON must be a non-empty list")
    for it in items:
        p = Path(it["path"])
        it["path"] = str(p if p.is_absolute() else base / p)
    return items


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="automix", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="print tempo, downbeat and key for audio files")
    a.add_argument("files", nargs="+")
    a.add_argument("--bpm-range", nargs=2, type=float, default=(160, 190), metavar=("LO", "HI"))

    m = sub.add_parser("mix", help="beat-match and blend tracks into one continuous mix")
    src = m.add_mutually_exclusive_group(required=True)
    src.add_argument("--crate", help="crate JSON (order, titles, Camelot keys, licenses)")
    src.add_argument("--files", nargs="+", help="audio files in play order")
    m.add_argument("-o", "--out", default="out/mix", help="output path without extension")
    m.add_argument("--bpm", type=float, help="target tempo (default: median of the tracks)")
    m.add_argument("--bars", type=int, default=16, help="transition length in bars (even, default 16)")
    m.add_argument("--auto-order", action="store_true", help="reorder by BPM and Camelot compatibility")
    m.add_argument("--bpm-range", nargs=2, type=float, default=(160, 190), metavar=("LO", "HI"))
    m.add_argument("--no-mp3", action="store_true")

    d = sub.add_parser("demo", help="generate synthetic tracks and mix them (no third-party audio)")
    d.add_argument("-o", "--out", default="out/demo/mix")

    args = ap.parse_args(argv)

    if args.cmd == "analyze":
        for f in args.files:
            print(json.dumps({k: v for k, v in analyze(f, tuple(args.bpm_range)).to_dict().items() if k != "bar_energy"}))
        return 0

    if args.cmd == "demo":
        from .demo import build_demo

        items = build_demo(Path(args.out).parent)
        bpm_range = (160.0, 190.0)
        out, bars, bpm, auto, mp3 = args.out, 16, None, False, True
    else:
        items = _load_crate(args.crate) if args.crate else [{"path": f, "title": Path(f).stem} for f in args.files]
        bpm_range = tuple(args.bpm_range)
        out, bars, bpm, auto, mp3 = args.out, args.bars, args.bpm, args.auto_order, not args.no_mp3

    tracks = []
    for it in items:
        print(f"analysing {it.get('title') or it['path']} …", file=sys.stderr)
        tracks.append(TrackInput(path=it["path"], title=it.get("title") or Path(it["path"]).stem,
                                 analysis=analyze(it["path"], bpm_range), camelot=it.get("camelot", ""),
                                 license=it.get("license", "")))
    if auto:
        tracks = order_crate(tracks)
    print("rendering mix …", file=sys.stderr)
    result = render_mix(tracks, bpm=bpm, transition_bars=bars)
    files = write_outputs(result, out, mp3=mp3)
    print(tracklist(result))
    print(json.dumps(files, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
