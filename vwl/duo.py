"""`vwl plan --person A=work_a --person B=work_b`: one film of two worldlines.

Each person is indexed separately (their own Takeout, Timeline, social exports); this module
combines them at plan time:

* the default date range is where both datasets overlap;
* copies of the same shot in both libraries (partner sharing, shared albums, forwarded photos)
  are kept once, preferring the owner's own upload;
* a 10-minute grid classifies every moment as together / apart / unknown from both tracks;
* photos are chosen per category (together, A alone, B alone) and laid out in spreads: a row of
  1-3 photos across the screen while together, each person's own photos on their half while apart;
* chapter cards mark the first sustained time together and reunions after long separations.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import index
from .camera import Track
from .model import Photo
from .duo_layout import DuoLayout, best_count
from .plan import Geocoder, contact_sheet, describe, export_media, fmt_date, mosaic
from .select import _largest_remainder, select

COLORS = [(255, 146, 64), (64, 200, 255)]       # left person orange, right person cyan
RECEIVED = re.compile(r"(?i)partner|shared")      # upload origins of items received from someone else
STEP = 600                                        # co-presence grid, seconds
TOGETHER_M = 500                                  # within this distance counts as together
CLOSE_M = 25_000                                  # same area: shown on one screen, though not "together"
UNKNOWN, APART, TOGETHER = 0, 1, 2


@dataclass
class Person:
    name: str
    work: Path
    color: tuple
    photos: list
    places: list
    loc: dict

    @property
    def track(self) -> Track:
        return Track(self.loc)


def load_people(specs: list[str]) -> list[Person]:
    if len(specs) != 2:
        raise SystemExit("--person must be given exactly twice (two people at most)")
    people = []
    for k, spec in enumerate(specs):
        if "=" not in spec:
            raise SystemExit(f"--person expects NAME=WORK_DIR, got {spec!r}")
        name, work = spec.split("=", 1)
        photos, _social, places, loc = index.load(Path(work))
        people.append(Person(name.strip(), Path(work), COLORS[k], photos, places, loc))
    return people


def span(p: Person) -> tuple[float, float]:
    """The person's data span, trimmed so a single misdated photo cannot stretch it."""
    t = np.concatenate([np.array([x.ts for x in p.photos], np.float64), p.loc["t"]])
    if not len(t):
        raise SystemExit(f"{p.name}: nothing indexed in {p.work} — run `vwl index … --work {p.work}` first")
    return float(np.percentile(t, 0.5)), float(np.percentile(t, 99.5))


def overlap(people: list[Person]) -> tuple[float, float]:
    spans = [span(p) for p in people]
    t0, t1 = max(s[0] for s in spans), min(s[1] for s in spans)
    if t0 >= t1:
        fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).date().isoformat()  # noqa: E731
        raise SystemExit("the two datasets don't overlap: " + "; ".join(
            f"{p.name} {fmt(a)}…{fmt(b)}" for p, (a, b) in zip(people, spans)) + " — pass --from/--to")
    return t0, t1


def merge_libraries(people: list[Person], t0: float, t1: float) -> list[tuple[Photo, int]]:
    """Both libraries in [t0, t1] as (item, owner); a shot present in both is kept once."""
    by_key: dict = {}
    for owner, p in enumerate(people):
        for x in p.photos:
            if not t0 <= x.ts <= t1:
                continue
            key = (round(x.ts), x.title.rsplit(".", 1)[0].lower(), x.kind)
            prev = by_key.get(key)
            if prev is None:
                by_key[key] = (x, owner)
            elif RECEIVED.search(prev[0].origin) and not RECEIVED.search(x.origin):
                x.favorite |= prev[0].favorite
                by_key[key] = (x, owner)          # the other copy was received: credit the camera owner
            else:
                prev[0].favorite |= x.favorite
    items = sorted(by_key.values(), key=lambda it: it[0].ts)
    n_dup = sum(len([x for x in p.photos if t0 <= x.ts <= t1]) for p in people) - len(items)
    if n_dup:
        print(f"[vwl] {n_dup} items present in both libraries kept once")
    return items


def from_world(mx, my):
    lon = np.asarray(mx) * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * np.asarray(my)))))
    return lat, lon


def haversine_m(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6_371_000 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def copresence(people: list[Person], t0: float, t1: float) -> dict:
    """State per 10-minute step: together (both known, ≤ TOGETHER_M apart), apart, or unknown."""
    t = np.arange(t0, t1, STEP, dtype=np.float64)
    pa, pb = people[0].track.where(t, strict=True), people[1].track.where(t, strict=True)
    la, lo = from_world(pa[:, 0], pa[:, 1])
    lb, lob = from_world(pb[:, 0], pb[:, 1])
    known = ~np.isnan(pa[:, 0]) & ~np.isnan(pb[:, 0])
    dist = np.full(len(t), np.nan)
    dist[known] = haversine_m(la[known], lo[known], lb[known], lob[known])
    state = np.full(len(t), UNKNOWN, np.int8)
    state[known] = np.where(dist[known] <= TOGETHER_M, TOGETHER, APART)
    return {"t": t, "state": state, "dist": dist, "lat": la, "lon": lo}


def state_at(cp: dict, ts) -> np.ndarray:
    i = np.clip(((np.asarray(ts) - cp["t"][0]) // STEP).astype(int), 0, len(cp["t"]) - 1)
    return cp["state"][i]


def daily(cp: dict):
    """Per UTC day: both-known samples and together samples."""
    day = (cp["t"] // 86400).astype(np.int64)
    d0 = day[0]
    n = day[-1] - d0 + 1
    both = np.bincount(day - d0, weights=cp["state"] != UNKNOWN, minlength=n)
    tog = np.bincount(day - d0, weights=cp["state"] == TOGETHER, minlength=n)
    return d0, both, tog


def first_together(cp: dict) -> float | None:
    """First day of the first sustained time together: ≥ 2 h together on ≥ 3 days within 30 days."""
    d0, both, tog = daily(cp)
    good = np.flatnonzero(tog >= 12)
    for k, d in enumerate(good):
        if np.sum((good >= d) & (good < d + 30)) >= 3:
            return float((d0 + d) * 86400)
    return None


def stats(cp: dict, geo: Geocoder) -> dict:
    d0, both, tog = daily(cp)
    together_days = np.flatnonzero(tog >= 6)                     # ≥ 1 h together
    gaps = np.diff(together_days) if len(together_days) > 1 else np.array([0])
    known = cp["state"] != UNKNOWN
    far = float(np.nanpercentile(cp["dist"][known], 99.9)) / 1000 if known.any() else 0.0
    # km travelled together: between daily median positions on consecutive together days
    tg = cp["state"] == TOGETHER
    day = (cp["t"] // 86400).astype(np.int64) - d0
    km = 0.0
    med = {}
    for d in together_days:
        m = tg & (day == d)
        med[d] = (np.median(cp["lat"][m]), np.median(cp["lon"][m]))
    for a, b in zip(together_days[:-1], together_days[1:]):
        if b == a + 1:
            km += float(haversine_m(*med[a], *med[b])) / 1000
    # countries together: at least 2 h together there (samples interpolated mid-flight don't count)
    idx = np.flatnonzero(tg)
    weight = 1.0
    if len(idx) > 5000:
        weight = len(idx) / 5000
        idx = idx[np.linspace(0, len(idx) - 1, 5000).astype(int)]
    per_cc: dict = {}
    for r in (geo.many([(float(cp["lat"][i]), float(cp["lon"][i])) for i in idx]) if len(idx) else []):
        if r:
            per_cc[r["cc"]] = per_cc.get(r["cc"], 0) + weight
    ccs = {cc for cc, n in per_cc.items() if n * STEP >= 2 * 3600}
    return {"days_both": int((both >= 12).sum()), "days_together": int(len(together_days)),
            "longest_apart_days": max(0, int(gaps.max()) - 1) if len(gaps) else 0, "furthest_km": round(far),
            "km_together": round(km), "countries_together": len(ccs)}


def display_modes(cp: dict, items: list[dict]) -> list[str]:
    """"together" or "apart" per item, for the screen layout: together within 500 m or in the same area
    (≤ CLOSE_M); unknown keeps the previous mode; a lone item never flips the layout on its own."""
    st = state_at(cp, [it["ts"] for it in items])
    i = np.clip(((np.array([it["ts"] for it in items]) - cp["t"][0]) // STEP).astype(int), 0, len(cp["t"]) - 1)
    d = cp["dist"][i]
    modes, prev = [], "apart"
    for s_, d_ in zip(st, d):
        if s_ == TOGETHER or (s_ == APART and d_ <= CLOSE_M):
            prev = "together"
        elif s_ == APART:
            prev = "apart"
        modes.append(prev)
    for k in range(1, len(modes) - 1):
        if modes[k - 1] == modes[k + 1] != modes[k]:
            modes[k] = modes[k - 1]
    return modes


def build_spreads(items: list[dict], cp: dict, lay: DuoLayout, hold: float) -> list[dict]:
    """Group the time-ordered items into spreads: rows of 1-3 photos across the screen while together;
    while apart, the left half shows one person's 1-2 photos and the right half the other's.

    Every spread is a contiguous slice of time (the next items in date order), so the film never goes
    back in time.  A side with no photos in its slice shows that person's latest photo again, marked
    "repeat" (the renderer dims it and leaves its old date out); that person's map follows the
    spread's own time, not the repeated photo's."""
    modes = display_modes(cp, items)
    runs = []
    for it, m in zip(items, modes):
        if runs and runs[-1][0] == m:
            runs[-1][1].append(it)
        else:
            runs.append((m, [it]))
    spreads = []

    def fits(group, x, region, kmax):
        """Would x join this row (the best count for the row with x is all of them)?"""
        asp = [y["aspect"] for y in group] + [x["aspect"]]
        return len(asp) <= kmax and best_count(asp, region, lay.gap, kmax) == len(asp)

    last = [None, None]

    def spread(mode, groups):
        flat = [x for g in groups for x in g if not x.get("repeat")]
        hold_s = max([hold] + [x["hold"] for x in flat if x["kind"] == "video"])
        t0 = min(x["ts"] for x in flat)
        ts_by = [g[0]["ts"] if g and not g[0].get("repeat") else t0 for g in groups] if mode == "apart" \
            else [flat[0]["ts"]] * 2
        sp = {"kind": "spread", "mode": mode, "ts": t0, "ts_by": ts_by,
              "year": min(flat, key=lambda x: x["ts"])["year"], "hold": round(hold_s, 3), "trans": 0.0}
        if mode == "together":
            sp["items"] = groups[0]
        else:
            sp["left"], sp["right"] = groups
        return sp

    for mode, run in runs:
        k = 0
        while k < len(run):
            if mode == "together":
                grp = [run[k]]
                k += 1
                while k < len(run) and fits(grp, run[k], lay.photos_together, 3):
                    grp.append(run[k])
                    k += 1
                for x in grp:
                    last[x["owner"]] = x
                spreads.append(spread("together", [grp]))
            else:
                sides = ([], [])
                regions = (lay.photos_left, lay.photos_right)
                while k < len(run):
                    x = run[k]
                    side = sides[x["owner"]]
                    if side and not fits(side, x, regions[x["owner"]], 2):
                        break       # this side is full: the next item opens the next spread
                    side.append(x)
                    k += 1
                for o in (0, 1):
                    if sides[o]:
                        last[o] = sides[o][-1]
                    elif last[o] is not None:      # nothing new from this person: their latest photo again
                        sides[o].append(dict(last[o], repeat=True))
                spreads.append(spread("apart", list(sides)))
    n_t = sum(s["mode"] == "together" for s in spreads)
    n_rep = sum(s["mode"] == "apart" and any(x.get("repeat") for x in s["left"] + s["right"]) for s in spreads)
    print(f"[vwl] {len(spreads)} spreads: {n_t} together, {len(spreads) - n_t} side by side"
          + (f" ({n_rep} repeating one person's latest photo)" if n_rep else ""))
    return spreads


def chapters(slots: list[dict], cp: dict, met: float | None, t0: float):
    """Chapter cards: the first sustained time together, and reunions after > 60 days apart."""
    ft = met if met is not None else first_together(cp)
    marks = []
    if ft is not None and ft > t0 + 30 * 86400:   # together from the start of the range: no "first"
        marks.append((ft, "First days together"))
    d0, _both, tog = daily(cp)
    days = np.flatnonzero(tog >= 6)
    for a, b in zip(days[:-1], days[1:]):
        if b - a > 60:
            marks.append((float((d0 + b) * 86400), f"Together again, after {b - a} days apart"))
    for when, text in marks:
        k = next((k for k, s in enumerate(slots) if s["ts"] >= when - 86400), None)
        if k is not None and "chapter" not in slots[k]:
            slots[k]["chapter"] = f"{text} · {fmt_date(when)}"


def relayout(work: Path, intro: float | None = None, outro: float | None = None) -> dict:
    """Regroup an existing two-person plan into spreads with the current rules — no re-selection, no
    re-export: seconds instead of a full `plan`.  Optionally change the intro/outro lengths."""
    path = work / "plan.json"
    plan = json.load(open(path))
    if not plan.get("persons"):
        raise SystemExit("relayout is for two-person plans (made with --person)")
    from .plan import plan_items
    items = sorted(plan_items(plan), key=lambda x: x["ts"])
    # together/apart and the totals are recomputed from both tracks, so rule changes reach old plans
    t0, t1 = plan["range"]
    people = load_people([f"{q['name']}={q['work']}" for q in plan["persons"]])
    for q in people:
        m = (q.loc["t"] >= t0) & (q.loc["t"] <= t1)
        q.loc = {k: v[m] for k, v in q.loc.items()}
    cp = copresence(people, t0, t1)
    np.savez_compressed(work / "duo.npz", t=cp["t"], state=cp["state"], dist=cp["dist"].astype(np.float32))
    plan["stats"].update(stats(cp, Geocoder()))
    params = plan.get("params", {})
    hold = params.get("hold") or min(it["hold"] for it in items if it["kind"] == "image")
    trans = params.get("trans", plan["slots"][0]["trans"] if plan["slots"] else 1.0)
    slots = build_spreads(items, cp, DuoLayout(*plan["size"]), hold)
    for k, s in enumerate(slots):
        s["trans"] = trans if k + 1 < len(slots) else 0.0
    chapters(slots, cp, params.get("met"), plan["range"][0])
    plan["slots"] = slots
    plan["params"] = dict(params, hold=hold, trans=trans)
    if intro is not None:
        plan["intro"] = intro
    if outro is not None:
        plan["outro"] = outro
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(plan, f, indent=1, ensure_ascii=False)
    tmp.replace(path)
    total = plan["intro"] + plan["outro"] + sum(s["hold"] + s["trans"] for s in slots)
    print(f"[vwl] relayout: {len(items)} items in {len(slots)} spreads → {total:.0f}s ({int(total // 60)}:{int(total % 60):02d})")
    return plan


def run(work: Path, persons: list[str], images=120, hold=2.2, trans=1.0, fps=30, size=(1920, 1080),
        t_from=None, t_to=None, workers=8, title="", nsfw_filter=True, hold_video=4.0, video_share=0.15,
        met: float | None = None, intro: float = 5.0, outro: float = 6.0) -> dict:
    people = load_people(persons)
    if t_from is None or t_to is None:
        o0, o1 = overlap(people)
        t_from = o0 if t_from is None else t_from
        t_to = o1 if t_to is None else t_to
    fmt = lambda t: datetime.fromtimestamp(t, timezone.utc).date().isoformat()  # noqa: E731
    print(f"[vwl] {people[0].name} & {people[1].name}: {fmt(t_from)} … {fmt(t_to)}")
    for p in people:
        m = (p.loc["t"] >= t_from) & (p.loc["t"] <= t_to)
        p.loc = {k: v[m] for k, v in p.loc.items()}
    library = merge_libraries(people, t_from, t_to)
    if not library:
        raise SystemExit("no photos in the date range")
    cp = copresence(people, t_from, t_to)
    st = state_at(cp, [x.ts for x, _ in library])
    cats = {"together": [x for (x, o), s in zip(library, st) if s == TOGETHER]}
    for k in (0, 1):
        cats[k] = [x for (x, o), s in zip(library, st) if s != TOGETHER and o == k]
    owner_of = {id(x): o for x, o in library}
    n = images if isinstance(images, int) else max(2, round(images * len(library)))
    w = {c: math.sqrt(len(v)) * (2.0 if c == "together" else 1.0) for c, v in cats.items() if v}
    quota = _largest_remainder(w, min(n, len(library)), {c: len(v) for c, v in cats.items() if v})
    label = {"together": "together", 0: f"{people[0].name} alone", 1: f"{people[1].name} alone"}
    print("[vwl] quotas: " + ", ".join(f"{label[c]} {quota[c]} of {len(cats[c])}" for c in quota))
    ex_file = work / "exclude.txt"
    exclude = {l.strip() for l in ex_file.read_text().splitlines() if l.strip() and not l.startswith("#")} \
        if ex_file.exists() else set()
    picked, vinfo, flagged = [], {}, []
    for c, q in quota.items():
        if q <= 0:
            continue
        got = select(cats[c], q, workers=workers, exclude=exclude, nsfw_filter=nsfw_filter,
                     video_share=video_share, hold_video=hold_video, cache_dir=work / "cache" / "videos")
        picked += got
        vinfo.update(getattr(select, "video_info", {}))
        flagged += getattr(select, "flagged", [])
    (work / "nsfw_flagged.txt").write_text("".join(u + "\n" for u in flagged))
    geo = Geocoder()
    items = []
    for k, p in enumerate(people):
        mine = sorted((x for x in picked if owner_of[id(x)] == k), key=lambda x: x.ts)
        for s in describe(mine, p.loc, p.places, geo, hold, trans, vinfo):
            s.update(owner=k, owner_name=p.name, owner_color=list(p.color))
            items.append(s)
    items.sort(key=lambda s: s["ts"])
    export_media(work, items, size, fps, workers)
    slots = build_spreads(items, cp, DuoLayout(*size), hold)
    for k, s in enumerate(slots):
        s["trans"] = trans if k + 1 < len(slots) else 0.0
    chapters(slots, cp, met, t_from)
    duo = stats(cp, geo)
    np.savez_compressed(work / "duo.npz", t=cp["t"], state=cp["state"], dist=cp["dist"].astype(np.float32))
    n_vid = sum(x.kind == "video" for x, _ in library)
    out = {"fps": fps, "size": list(size), "intro": intro, "outro": outro,
           "params": {"hold": hold, "trans": trans, "met": met},
           "title": title or f"{people[0].name} & {people[1].name}", "range": [t_from, t_to],
           "persons": [{"name": p.name, "work": str(p.work.resolve()), "color": list(p.color)} for p in people],
           "stats": {"photos": len(library) - n_vid, "videos": n_vid,
                     "span": [datetime.fromtimestamp(t_from, timezone.utc).year,
                              datetime.fromtimestamp(t_to, timezone.utc).year], **duo},
           "slots": slots}
    with open(work / "plan.json", "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    contact_sheet(work, items)
    mosaic(work, items, size, workers)
    total = out["intro"] + out["outro"] + sum(s["hold"] + s["trans"] for s in slots)
    print(f"[vwl] {duo['days_together']} days together of {duo['days_both']} with data from both; "
          f"{duo['km_together']:,} km together; {duo['countries_together']} countries together; "
          f"longest apart {duo['longest_apart_days']} days; furthest {duo['furthest_km']:,} km")
    print(f"[vwl] plan: {len(items)} items in {len(slots)} spreads → {total:.0f}s ({int(total // 60)}:{int(total % 60):02d})")
    return out
