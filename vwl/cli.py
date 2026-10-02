"""vwl — virtual worldlines: turn personal data exports into a life video.

  python -m vwl index  /mnt/5tb/takeouts ~/exports/instagram-*.zip   --work work
  python -m vwl plan   --work work --from 2014 --to 2019-06 --images 150 --hold 2.5 --transition 0.8
  python -m vwl render --work work --out life.mp4 --workers 12
  python -m vwl render --work work --preview-at 42 --out frame.png
  python -m vwl all    /mnt/5tb/takeouts --work work --out life.mp4
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _date(s: str | None, end: bool = False) -> float | None:
    """YYYY, YYYY-MM or YYYY-MM-DD → unix time (UTC).  With end=True, the end of that year/month/day,
    so `--to 2019` includes all of 2019."""
    if not s:
        return None
    parts = [int(x) for x in s.split("-")]
    if not 1 <= len(parts) <= 3:
        raise SystemExit(f"bad date {s!r}: use YYYY, YYYY-MM or YYYY-MM-DD")
    y, mo, d = (parts + [1, 1])[:3]
    start = datetime(y, mo, d, tzinfo=timezone.utc)
    if not end:
        return start.timestamp()
    if len(parts) == 1:
        nxt = datetime(y + 1, 1, 1, tzinfo=timezone.utc)
    elif len(parts) == 2:
        nxt = datetime(y + (mo == 12), mo % 12 + 1, 1, tzinfo=timezone.utc)
    else:
        nxt = start + timedelta(days=1)
    return nxt.timestamp() - 1


def _share(s: str) -> float:
    """'15%' or '0.15' → 0.15; '0' → 0."""
    try:
        x = float(s[:-1]) / 100 if s.endswith("%") else float(s)
        if not 0 <= x <= 1:
            raise ValueError
        return x
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a percentage like 15% (or 0 for photos only), got {s!r}")


def _count(s: str) -> int | float:
    """'150' → 150; '0.5%' → 0.005 (a fraction, resolved against the photos in range by `plan`)."""
    try:
        if s.endswith("%"):
            x = float(s[:-1]) / 100
            if not 0 < x <= 1:
                raise ValueError
            return x
        n = int(s)
        if n < 2:
            raise ValueError
        return n
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer >= 2 or a percentage in (0, 100]%, got {s!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="vwl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cpu = max(1, (os.cpu_count() or 4) - 2)

    def common(p):
        p.add_argument("--work", type=Path, default=Path("work"), help="working directory (index, plan, photos)")
        p.add_argument("--workers", type=int, default=cpu)

    def index_args(p):
        p.add_argument("paths", nargs="+", help="export zips or folders (Takeout parts, Meta/X zips, Timeline.json, GPX, photo dirs)")
        p.add_argument("--self-name", action="append", default=[], metavar="PLATFORM=NAME",
                       help="your display name in message exports, e.g. facebook='Jane Doe' (auto-detected otherwise)")

    def plan_args(p):
        p.add_argument("--from", dest="t_from", metavar="DATE",
                       help="start of the date range: YYYY, YYYY-MM or YYYY-MM-DD (default: first photo)")
        p.add_argument("--to", dest="t_to", metavar="DATE",
                       help="end of the date range, inclusive: YYYY, YYYY-MM or YYYY-MM-DD (default: last photo)")
        p.add_argument("-n", "--images", type=_count, default=120, metavar="N|X%",
                       help="number of photos and videos in the film, or a percentage of those in the date range "
                            "(e.g. 0.5%%) (default: 120)")
        p.add_argument("--hold", type=float, default=2.2, metavar="SEC",
                       help="seconds each photo is displayed (default: 2.2)")
        p.add_argument("--hold-video", type=float, default=4.0, metavar="SEC",
                       help="seconds a video clip is displayed; shorter videos play in full (default: 4.0)")
        p.add_argument("--video-share", type=_share, default=0.15, metavar="X%",
                       help="at most this share of the items may be videos; 0 = photos only (default: 15%%)")
        p.add_argument("--transition", "--trans", dest="trans", type=float, default=1.0, metavar="SEC",
                       help="seconds of morph between consecutive items (default: 1.0)")
        p.add_argument("--fps", type=int, default=30)
        p.add_argument("--size", default="1920x1080")
        p.add_argument("--title", default="")
        p.add_argument("--no-nsfw-filter", action="store_true", help="don't drop photos a local NudeNet model flags")
        p.add_argument("--person", action="append", default=[], metavar="NAME=WORK_DIR",
                       help="two-person film: give twice, e.g. --person Nikita=work --person Olga=work_olga "
                            "(each WORK_DIR indexed separately); --work is then where the joint plan goes")
        p.add_argument("--met", metavar="DATE", help="two-person film: date of the first time together "
                       "(default: detected as the first sustained co-location)")

    def render_args(p):
        p.add_argument("--out", type=Path, default=Path("life.mp4"))
        p.add_argument("--style", default="dark", choices=["dark", "light", "satellite", "topo", "osm"])
        p.add_argument("--tile-url", help="custom XYZ template, e.g. https://tiles.stadiamaps.com/tiles/alidade_smooth_dark/{z}/{x}/{y}.png?api_key=…")
        p.add_argument("--crf", type=int, default=18)
        p.add_argument("--preset", default="medium")
        p.add_argument("--morph", type=float, default=1.0, help="morph warp strength (0 = plain dissolve)")
        p.add_argument("--music", help="audio file to lay under the video")
        p.add_argument("--offline", action="store_true", help="don't download map tiles")
        p.add_argument("--preview-at", type=float, help="render just the frame at this second, as an image")

    p = sub.add_parser("index", help="scan exports → work/index")
    common(p); index_args(p)
    p = sub.add_parser("plan", help="select photos and pacing → work/plan.json")
    common(p); plan_args(p)
    p = sub.add_parser("render", help="render work/plan.json → video")
    common(p); render_args(p)
    p = sub.add_parser("all", help="index + plan + render")
    common(p); index_args(p); plan_args(p); render_args(p)
    a = ap.parse_args(argv)

    if a.cmd in ("index", "all"):
        from . import index
        names = dict(s.split("=", 1) for s in a.self_name)
        summary = index.run(a.paths, a.work, a.workers, names)
        print(json.dumps({k: v for k, v in summary.items() if k != "containers"}, indent=1, ensure_ascii=False))
    if a.cmd in ("plan", "all"):
        from . import plan
        w, h = map(int, a.size.lower().split("x"))
        if a.hold <= 0 or a.trans < 0 or a.hold_video <= 0:
            raise SystemExit("need --hold > 0, --hold-video > 0, --transition >= 0")
        if a.person:
            from . import duo
            a.work.mkdir(parents=True, exist_ok=True)
            duo.run(a.work, a.person, a.images, a.hold, a.trans, a.fps, (w, h), _date(a.t_from),
                    _date(a.t_to, end=True), a.workers, a.title, not a.no_nsfw_filter, a.hold_video,
                    a.video_share, _date(a.met))
        else:
            plan.run(a.work, a.images, a.hold, a.trans, a.fps, (w, h), _date(a.t_from), _date(a.t_to, end=True),
                     a.workers, a.title, not a.no_nsfw_filter, a.hold_video, a.video_share)
    if a.cmd in ("render", "all"):
        from . import render
        render.run(a.work, a.out, a.style, a.workers, a.crf, a.preset, a.morph, a.music,
                   a.preview_at, a.offline, a.tile_url)
