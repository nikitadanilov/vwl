from pathlib import Path

import numpy as np
import pytest

from tests.fixtures import make_all
from vwl import index, plan, render
from vwl.model import LOC_PHOTO, LOC_RECORDS, LOC_SEMANTIC, LOC_TIMELINE


@pytest.fixture(scope="module")
def indexed(tmp_path_factory):
    root = tmp_path_factory.mktemp("exports")
    make_all(root)
    work = tmp_path_factory.mktemp("work")
    summary = index.run([str(root)], work, workers=2)
    return work, summary


def test_index(indexed):
    work, s = indexed
    kinds = sorted(s["containers"].values())
    # zips by content; each phone location export (Timeline.json, location-history.json) on its own
    assert kinds == ["facebook", "generic", "generic", "google", "instagram", "x"]
    assert sorted(Path(c).name for c, k in s["containers"].items() if k == "generic") == [
        "Timeline.json", "location-history.json"]
    photos, social, places, loc = index.load(work)
    gp = [p for p in photos if p.source == "google_photos"]
    titles = {p.title for p in gp}
    # 60 regular + truncated long name + 2 dup-named + edited + Live Photo still; album copy de-duplicated
    gp_img = [p for p in gp if p.kind == "image"]
    assert len(gp_img) == 65, sorted(titles)
    # 3 videos; the Live Photo's .MOV is folded into its still
    assert sorted(p.title for p in gp if p.kind == "video") == [
        "Screen_Recording_20190523-120000.mp4", "VID_20190520_120000.mp4", "VID_20190522_120000.mp4"]
    assert s["videos"] == 3
    assert "PXL_20190601_120000000.PORTRAIT.ORIGINAL_long_name_x.jpg" in titles
    assert sum(p.title == "IMG_dup.jpg" for p in gp) == 2
    assert all(p.ts_origin == "sidecar" for p in gp if "edited" not in p.uri)
    # the Instagram post kept its capture time, so it folds into the Google original (caption and all)
    assert {p.source for p in photos} == {"google_photos", "facebook", "x"}
    orig = [p for p in gp if p.shared]
    assert len(orig) == 1 and "Golden hour" in orig[0].description
    # 0,0 geoData means "unknown"
    assert sum(p.lat is None for p in gp) >= 8
    by = s["social_by_kind"]
    assert by["facebook/post"] == 2
    assert by["facebook/message_out"] == 3 and by["facebook/message_in"] == 1
    assert by["instagram/post"] == 1 and by["instagram/story"] == 1 and by["instagram/comment"] == 1
    assert by["x/post"] == 1
    assert s["self_names"]["facebook"] == "Nikita Danilov"
    txt = " ".join(x.text for x in social)
    assert "Париж" in txt and "Удачи" in txt and "✨" in txt  # mojibake repaired
    assert "t.co" not in txt and "RT @" not in txt
    assert any(p.name == "Musée d'Orsay" for p in places)
    assert set(np.unique(loc["src"])) >= {LOC_RECORDS, LOC_SEMANTIC, LOC_TIMELINE, LOC_PHOTO}
    assert np.all(np.diff(loc["t"]) >= 0)
    # nothing out of range, no 0,0
    assert np.all(np.abs(loc["lat"]) <= 90) and not np.any((loc["lat"] == 0) & (loc["lon"] == 0))


def test_plan_and_render(indexed, tmp_path):
    work, _ = indexed
    p = plan.run(work, images=8, hold=2.0, trans=0.5, fps=12, size=(640, 360), workers=2)
    slots = p["slots"]
    assert len(slots) == 8
    assert all(s["hold"] == 2.0 for s in slots if s["kind"] == "image")
    assert [s["trans"] for s in slots] == [0.5] * 7 + [0.0]
    assert "cards" not in slots[0]
    assert all((work / s["file"]).exists() for s in slots)
    out = tmp_path / "f.png"
    render.run(work, out, workers=1, preview_at=8.0, offline=True)
    assert out.stat().st_size > 10_000
    vid = tmp_path / "v.mp4"
    render.run(work, vid, workers=2, preset="ultrafast", offline=True)
    assert vid.stat().st_size > 50_000
    frames = render.build_timeline(p, render.load_track(work, p), render.Layout(640, 360), 256)["N"]
    assert abs(frames - 12 * (p["intro"] + p["outro"] + sum(s["hold"] + s["trans"] for s in slots))) <= len(slots) + 2


def test_date_range(indexed):
    from vwl.cli import _date
    work, _ = indexed
    t0, t1 = _date("2019-06"), _date("2019-06", end=True)
    p = plan.run(work, images=5, hold=1.0, trans=0.5, fps=12, size=(640, 360), t_from=t0, t_to=t1, workers=2)
    assert p["slots"] and all(t0 <= s["ts"] <= t1 for s in p["slots"])
    loc = render.load_track(work, p)
    assert len(loc["t"]) and loc["t"].min() >= t0 and loc["t"].max() <= t1


def test_images_count_or_percent(indexed):
    import argparse
    from vwl.cli import _count
    assert _count("150") == 150 and _count("10%") == 0.1 and _count("0.5%") == 0.005
    for bad in ("1", "0%", "150%", "abc", "-5"):
        with pytest.raises(argparse.ArgumentTypeError):
            _count(bad)
    work, _ = indexed
    photos, *_ = __import__("vwl.index", fromlist=["load"]).load(work)
    p = plan.run(work, images=0.1, hold=1.0, trans=0.5, fps=12, size=(640, 360), workers=2)
    assert len(p["slots"]) == max(2, round(0.1 * len(photos)))


def test_video_clips(indexed, tmp_path):
    work, _ = indexed
    p = plan.run(work, images=40, hold=0.3, trans=0.1, fps=12, size=(640, 360), workers=2,
                 hold_video=3.0, video_share=0.1)
    vids = [s for s in p["slots"] if s["kind"] == "video"]
    # only the 6 s clip qualifies: the 1 s one is too short, the screen recording is skipped
    assert [s["uri"].rsplit("/", 1)[-1] for s in vids] == ["VID_20190520_120000.mp4"]
    v = vids[0]
    assert v["hold"] == 3.0 and 0 <= v["clip_start"] <= 6 - 3.0
    assert (work / v["file"]).stat().st_size > 5_000 and (work / v["poster"]).exists()
    vid = tmp_path / "v.mp4"
    render.run(work, vid, workers=2, preset="ultrafast", offline=True)
    assert vid.stat().st_size > 50_000
    none = plan.run(work, images=6, hold=1.0, trans=0.5, fps=12, size=(640, 360), workers=2, video_share=0)
    assert all(s["kind"] == "image" for s in none["slots"])


def test_two_people(indexed, tmp_path):
    from tests.fixtures import olga
    from vwl import duo
    work_n, _ = indexed
    exports = tmp_path / "olga"
    exports.mkdir()
    olga(exports / "takeout-olga-001.zip")
    work_o = tmp_path / "work_olga"
    index.run([str(exports)], work_o, workers=2)
    work = tmp_path / "duo"
    work.mkdir()
    p = duo.run(work, [f"Nikita={work_n}", f"Olga={work_o}"], images=24, hold=1.0, trans=0.6, fps=12,
                size=(640, 360), workers=2, video_share=0)
    st = p["stats"]
    # together from day 35 (Paris) to day 70; both have data for ~70 days
    assert 30 <= st["days_together"] <= 37 and st["days_both"] >= 60
    assert st["countries_together"] >= 2          # France and the US
    assert st["furthest_km"] > 8000                # Moscow ↔ San Francisco
    chapters = [s.get("chapter", "") for s in p["slots"]]
    assert any(c.startswith("First days together") for c in chapters)
    # spreads: rows of 1-3 while together; each person on their own half while apart
    modes = {s["mode"] for s in p["slots"]}
    assert modes == {"together", "apart"} and all(s["kind"] == "spread" for s in p["slots"])
    for s in p["slots"]:
        if s["mode"] == "together":
            assert 1 <= len(s["items"]) <= 3
        else:
            assert all(it["owner"] == 0 for it in s["left"]) and all(it["owner"] == 1 for it in s["right"])
            assert 1 <= len(s["left"]) + len(s["right"]) <= 4
    assert all(it.get("aspect", 0) > 0 for s in p["slots"] for it in s.get("items", s.get("left", []) + s.get("right", [])))
    # the film never goes back in time, and no photo is shown twice except as a marked "latest photo"
    flat = [[it for it in s.get("items", []) + s.get("left", []) + s.get("right", []) if not it.get("repeat")]
            for s in p["slots"]]
    for a_, b_ in zip(flat[:-1], flat[1:]):
        assert min(it["ts"] for it in b_) >= max(it["ts"] for it in a_)
    files = [it["file"] for f in flat for it in f]
    assert len(files) == len(set(files))
    for s in p["slots"]:
        if s["mode"] == "apart":
            assert s["left"] and s["right"] or not any(  # a side is empty only before that person's first photo
                it["owner"] == k for it in [x for f in flat for x in f] if it["ts"] < s["ts"] for k in (0, 1)
                if not s[("left", "right")[k]])
    # the map clocks and the playhead only move forward
    assert all(a_["ts_by"][k] <= b_["ts_by"][k] for a_, b_ in zip(p["slots"][:-1], p["slots"][1:]) for k in (0, 1))
    # the partner-shared copy of Nikita's photo #40 is kept once, credited to Nikita
    people = duo.load_people([f"Nikita={work_n}", f"Olga={work_o}"])
    items = duo.merge_libraries(people, *p["range"])
    copies = [(x, o) for x, o in items if x.title.startswith("IMG_") and abs(x.ts - (people[0].photos[0].ts)) >= 0
              and x.title in {y.title for y in people[1].photos}]
    assert len(copies) == 1 and copies[0][1] == 0 and "partner" not in copies[0][0].origin.lower()
    # render: the panels merge while together
    from vwl.render import load_track
    from vwl.duo_layout import DuoLayout
    from vwl.render_duo import build_duo_timeline
    locs = [load_track(Path(q["work"]), p) for q in p["persons"]]
    z = np.load(work / "duo.npz")
    tl = build_duo_timeline(p, locs, DuoLayout(640, 360), 256, {k: z[k] for k in z.files})
    assert (tl["merge"] > 0.99).any() and (tl["merge"] < 0.01).any()
    out = tmp_path / "duo.png"
    render.run(work, out, workers=1, preview_at=8.0, offline=True)
    assert out.stat().st_size > 10_000
    vid = tmp_path / "duo.mp4"
    render.run(work, vid, workers=2, preset="ultrafast", offline=True)
    assert vid.stat().st_size > 50_000


def test_discover_input_directories(tmp_path):
    """An input directory may mix export zips, phone location exports, extracted exports and leftovers."""
    import zipfile as zf
    from vwl.archive import discover
    root = tmp_path / "exports"
    (root / "phone").mkdir(parents=True)
    (root / "Takeout" / "Google Photos").mkdir(parents=True)
    (root / "Takeout" / "Google Photos" / "a.jpg").write_bytes(b"x")
    (root / "Takeout" / "Drive").mkdir()
    with zf.ZipFile(root / "Takeout" / "Drive" / "mine.zip", "w") as z:   # user's own zip in Drive: not an export
        z.writestr("x.txt", "x")
    with zf.ZipFile(root / "takeout-001.zip", "w") as z:
        z.writestr("Takeout/Google Photos/b.jpg", "x")
    (root / "partial.zip").write_bytes(b"PK\x03\x04 still downloading")
    (root / "phone" / "Timeline.json").write_text("[]")
    (root / "ride.gpx").write_text("<gpx/>")
    (root / "VID_20200101_120000-114.mp4").write_bytes(b"x")         # leftover from extracting a zip part
    (root / "All mail-128.mbox").write_bytes(b"x")                     # ignored
    found = list(discover([str(root), str(root / "phone")]))           # overlapping inputs
    names = sorted(Path(c.path).name + ("/*" if c.files is not None else "") for c in found)
    assert names == sorted(["Timeline.json", "Takeout", "exports/*", "ride.gpx", "takeout-001.zip"])
    media = next(c for c in found if c.files is not None)
    assert media.files == ["VID_20200101_120000-114.mp4"]
