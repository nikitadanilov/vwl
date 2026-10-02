"""Synthetic exports in the documented Google / Meta / X formats, for tests and demos.

    python -m tests.fixtures /tmp/fake-exports
"""
from __future__ import annotations

import io
import json
import math
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

UTC = timezone.utc
HOME = (37.7749, -122.4194)    # San Francisco
TRIP = (48.8566, 2.3522)       # Paris
T0 = datetime(2019, 5, 1, 9, tzinfo=UTC)


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def mojibake(s: str) -> str:
    """What Meta does to non-ASCII text."""
    return s.encode("utf-8").decode("latin-1")


def picture(seed: int, w=1600, h=1200) -> bytes:
    r = np.random.default_rng(seed)
    top, bot = r.integers(40, 255, 3), r.integers(0, 200, 3)
    y = np.linspace(0, 1, h)[:, None, None]
    arr = (top * (1 - y) + bot * y).repeat(w, 1).astype(np.uint8)
    im = Image.fromarray(arr)
    d = ImageDraw.Draw(im)
    for _ in range(12):
        x0, y0 = r.integers(0, w), r.integers(0, h)
        rad = r.integers(30, 260)
        d.ellipse((x0 - rad, y0 - rad, x0 + rad, y0 + rad), fill=tuple(int(v) for v in r.integers(0, 255, 3)))
    d.text((40, 40), f"#{seed}", fill=(255, 255, 255))
    b = io.BytesIO()
    im.save(b, "JPEG", quality=85)
    return b.getvalue()


def movie(seconds: float, size="640x360") -> bytes:
    """A small H.264 test clip (moving pattern) made with the bundled ffmpeg."""
    import subprocess
    import tempfile

    from vwl.video import ffmpeg_exe
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "m.mp4"
        subprocess.run([ffmpeg_exe(), "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30",
                        "-t", str(seconds), "-pix_fmt", "yuv420p", "-c:v", "libx264", "-preset", "ultrafast",
                        str(out)], check=True)
        return out.read_bytes()


def track():
    """(dt, lat, lon): 60 days at home with wandering, a week in Paris, then home."""
    out = []
    t = T0
    r = np.random.default_rng(1)
    for day in range(70):
        base = TRIP if 30 <= day < 37 else HOME
        for k in range(24):
            ang = r.uniform(0, 2 * math.pi)
            d = r.uniform(0, 0.04)
            out.append((t + timedelta(days=day, hours=k), base[0] + d * math.sin(ang), base[1] + d * math.cos(ang)))
    return out


def google(path: Path):
    z = zipfile.ZipFile(path, "w")
    pts = track()
    z.writestr("Takeout/Location History (Timeline)/Records.json", json.dumps({"locations": [
        {"latitudeE7": int(la * 1e7), "longitudeE7": int(lo * 1e7), "accuracy": 20,
         **({"timestamp": iso(t)} if i % 2 else {"timestampMs": str(int(t.timestamp() * 1000))})}
        for i, (t, la, lo) in enumerate(pts)]}))
    z.writestr("Takeout/Location History (Timeline)/Semantic Location History/2019/2019_JUNE.json", json.dumps(
        {"timelineObjects": [{"placeVisit": {
            "location": {"latitudeE7": int(TRIP[0] * 1e7), "longitudeE7": int(TRIP[1] * 1e7), "name": "Musée d'Orsay",
                         "address": "1 Rue de la Légion d'Honneur, Paris"},
            "duration": {"startTimestamp": iso(T0 + timedelta(days=32, hours=3)),
                         "endTimestamp": iso(T0 + timedelta(days=32, hours=6))}}},
            {"activitySegment": {
                "startLocation": {"latitudeE7": int(HOME[0] * 1e7), "longitudeE7": int(HOME[1] * 1e7)},
                "endLocation": {"latitudeE7": int(TRIP[0] * 1e7), "longitudeE7": int(TRIP[1] * 1e7)},
                "duration": {"startTimestamp": iso(T0 + timedelta(days=29, hours=20)),
                             "endTimestamp": iso(T0 + timedelta(days=30, hours=7))},
                "waypointPath": {"waypoints": [{"latE7": 600000000, "lngE7": -300000000}]}}}]}))
    folder = "Takeout/Google Photos/Photos from 2019"
    variants = [".json", ".supplemental-metadata.json"]
    for i in range(60):
        dt = T0 + timedelta(days=i * 70 / 60, hours=3)
        t, la, lo = min(pts, key=lambda p: abs((p[0] - dt).total_seconds()))
        name = f"IMG_{dt:%Y%m%d_%H%M%S}.jpg"
        z.writestr(f"{folder}/{name}", picture(i))
        side = {"title": name, "description": "Sunset over the bay" if i == 5 else "",
                "photoTakenTime": {"timestamp": str(int(dt.timestamp()))},
                "geoData": {"latitude": la if i % 7 else 0.0, "longitude": lo if i % 7 else 0.0},
                "favorited": i % 11 == 0, "imageViews": str(i % 5)}
        z.writestr(f"{folder}/{name}{variants[i % 2]}", json.dumps(side))
    # a long name with truncated sidecar, a duplicate name, an edited copy, an album copy
    long = "PXL_20190601_120000000.PORTRAIT.ORIGINAL_long_name_x.jpg"
    z.writestr(f"{folder}/{long[:47]}.jpg", picture(100))
    z.writestr(f"{folder}/{(long + '.supplemental-metadata')[:46]}.json",
               json.dumps({"title": long, "photoTakenTime": {"timestamp": str(int((T0 + timedelta(days=31)).timestamp()))}}))
    z.writestr(f"{folder}/IMG_dup(1).jpg", picture(101))
    z.writestr(f"{folder}/IMG_dup.jpg(1).json", json.dumps({"title": "IMG_dup.jpg", "photoTakenTime": {"timestamp": str(int((T0 + timedelta(days=40)).timestamp()))}}))
    z.writestr(f"{folder}/IMG_dup.jpg", picture(102))
    z.writestr(f"{folder}/IMG_dup.jpg.json", json.dumps({"title": "IMG_dup.jpg", "photoTakenTime": {"timestamp": str(int((T0 + timedelta(days=41)).timestamp()))}}))
    z.writestr(f"{folder}/IMG_20190510_120000-edited.jpg", picture(0))
    first = f"IMG_{T0 + timedelta(hours=3):%Y%m%d_%H%M%S}.jpg"
    z.writestr(f"Takeout/Google Photos/Summer/{first}", picture(0))
    z.writestr(f"Takeout/Google Photos/Summer/{first}.json", json.dumps({"title": first, "photoTakenTime": {"timestamp": str(int((T0 + timedelta(hours=3)).timestamp()))}}))
    z.writestr("Takeout/Google Photos/Summer/metadata.json", json.dumps({"title": "Summer"}))
    # videos: a good 6 s clip, one too short, a screen recording, and an iPhone Live Photo pair
    for name, secs, day in (("VID_20190520_120000.mp4", 6, 19), ("VID_20190522_120000.mp4", 1, 21),
                            ("Screen_Recording_20190523-120000.mp4", 4, 22)):
        dt = T0 + timedelta(days=day, hours=3)
        z.writestr(f"{folder}/{name}", movie(secs))
        z.writestr(f"{folder}/{name}.json", json.dumps({"title": name, "favorited": True,
                   "photoTakenTime": {"timestamp": str(int(dt.timestamp()))}}))
    live = "IMG_20190524_100000"
    z.writestr(f"{folder}/{live}.jpg", picture(103))
    z.writestr(f"{folder}/{live}.jpg.json", json.dumps({"title": f"{live}.jpg", "photoTakenTime": {
        "timestamp": str(int((T0 + timedelta(days=23)).timestamp()))}}))
    z.writestr(f"{folder}/{live}.MOV", movie(2))
    z.writestr(f"{folder}/{live}.MOV.json", json.dumps({"title": f"{live}.MOV", "photoTakenTime": {
        "timestamp": str(int((T0 + timedelta(days=23)).timestamp()))}}))
    z.writestr("Takeout/Drive/notes.json", json.dumps({"photoTakenTime": "not a sidecar dir"}))
    z.close()


def phone_timeline(path: Path):
    """Android on-device Timeline export (2024+)."""
    t = T0 + timedelta(days=80)
    d = {"semanticSegments": [
        {"startTime": (t).isoformat(), "endTime": (t + timedelta(hours=2)).isoformat(),
         "visit": {"topCandidate": {"placeId": "x", "semanticType": "INFERRED_HOME",
                                    "placeLocation": {"latLng": f"{HOME[0]}°, {HOME[1]}°"}}}},
        {"startTime": (t + timedelta(hours=2)).isoformat(), "endTime": (t + timedelta(hours=4)).isoformat(),
         "timelinePath": [{"point": f"{HOME[0] + 0.01 * k}°, {HOME[1]}°", "time": (t + timedelta(hours=2, minutes=10 * k)).isoformat()} for k in range(10)]}],
        "rawSignals": [{"position": {"LatLng": f"{HOME[0]}°, {HOME[1] + 0.02}°", "accuracyMeters": 10,
                                     "timestamp": (t + timedelta(hours=5)).isoformat()}}]}
    path.write_text(json.dumps(d))


def ios_timeline(path: Path):
    t = T0 + timedelta(days=90)
    d = [{"startTime": t.isoformat(), "endTime": (t + timedelta(hours=1)).isoformat(),
          "visit": {"topCandidate": {"placeLocation": f"geo:{HOME[0]},{HOME[1]}", "semanticType": "Inferred Work"}}},
         {"startTime": (t + timedelta(hours=1)).isoformat(), "endTime": (t + timedelta(hours=2)).isoformat(),
          "activity": {"start": f"geo:{HOME[0]},{HOME[1]}", "end": f"geo:{HOME[0] + 0.1},{HOME[1]}"}},
         {"startTime": (t + timedelta(hours=2)).isoformat(), "endTime": (t + timedelta(hours=3)).isoformat(),
          "timelinePath": [{"point": f"geo:{HOME[0] + 0.1},{HOME[1] + 0.01 * k}", "durationMinutesOffsetFromStartTime": str(10 * k)} for k in range(5)]}]
    path.write_text(json.dumps(d))


def facebook(path: Path):
    z = zipfile.ZipFile(path, "w")
    root = "your_facebook_activity"
    ts = lambda days: int((T0 + timedelta(days=days)).timestamp())  # noqa: E731
    z.writestr(f"{root}/posts/media/Mobileuploads_1/fb1.jpg", picture(200))
    posts = [
        {"timestamp": ts(33), "data": [{"post": mojibake("Париж, наконец-то! Walking along the Seine all afternoon.")}],
         "attachments": [{"data": [{"place": {"name": "Pont Neuf", "coordinate": {"latitude": 48.857, "longitude": 2.341}}}]}],
         "title": "Nikita Danilov was at Pont Neuf."},
        {"timestamp": ts(12), "data": [{"post": "New job, new desk, same old coffee. Here we go."}],
         "attachments": [{"data": [{"media": {"uri": f"{root}/posts/media/Mobileuploads_1/fb1.jpg", "creation_timestamp": ts(12),
                                              "media_metadata": {"photo_metadata": {"exif_data": [{"latitude": 37.79, "longitude": -122.40}]}}}}]}]},
    ]
    z.writestr(f"{root}/posts/your_posts__check_ins__photos_and_videos_1.json", json.dumps(posts))
    z.writestr(f"{root}/comments_and_reactions/comments.json", json.dumps({"comments_v2": [
        {"timestamp": ts(20), "data": [{"comment": {"timestamp": ts(20), "comment": "Congratulations, this is wonderful news!", "author": "Nikita Danilov"}}],
         "title": "Nikita Danilov commented on Anna Smith's post."}]}))
    msgs = {"participants": [{"name": "Anna Smith"}, {"name": "Nikita Danilov"}], "title": "Anna Smith",
            "messages": [{"sender_name": "Nikita Danilov", "timestamp_ms": ts(29) * 1000, "content": "Flying to Paris tonight, will send pictures from the Louvre"},
                         {"sender_name": "Anna Smith", "timestamp_ms": ts(29) * 1000 + 60000, "content": mojibake("Удачи! Bring back some croissants please")},
                         {"sender_name": "Nikita Danilov", "timestamp_ms": ts(29) * 1000 + 120000, "content": "You sent an attachment."}]}
    z.writestr(f"{root}/messages/inbox/annasmith_123/message_1.json", json.dumps(msgs))
    msgs2 = {"participants": [{"name": "Bob Jones"}, {"name": "Nikita Danilov"}], "title": "Bob Jones",
             "messages": [{"sender_name": "Nikita Danilov", "timestamp_ms": ts(50) * 1000, "content": "Back home. Jet lag is brutal this time around."}]}
    z.writestr(f"{root}/messages/inbox/bobjones_456/message_1.json", json.dumps(msgs2))
    z.close()


def instagram(path: Path):
    z = zipfile.ZipFile(path, "w")
    root = "your_instagram_activity"
    ts = lambda days: int((T0 + timedelta(days=days)).timestamp())  # noqa: E731
    z.writestr("media/posts/201906/ig1.jpg", picture(300))
    z.writestr(f"{root}/content/posts_1.json", json.dumps([
        {"media": [{"uri": "media/posts/201906/ig1.jpg", "creation_timestamp": ts(34),
                    "title": mojibake("Golden hour at the Louvre ✨ #paris"), "media_metadata": {"photo_metadata": {"exif_data": [{"latitude": 48.8606, "longitude": 2.3376,
                    # same capture time as Google photo #30 (day 35): the post is a copy of it
                    "taken_timestamp": int((T0 + timedelta(days=30 * 70 / 60, hours=3)).timestamp())}]}}}]}]))
    z.writestr(f"{root}/content/stories.json", json.dumps({"ig_stories": [
        {"uri": "media/stories/201906/s1.mp4", "creation_timestamp": ts(35), "title": "Last night in Paris, see you soon"}]}))
    z.writestr(f"{root}/comments/post_comments_1.json", json.dumps([
        {"string_map_data": {"Comment": {"value": "This view is unreal, where exactly is this?"}, "Media Owner": {"value": "travelfriend"}, "Time": {"timestamp": ts(40)}}}]))
    z.writestr(f"{root}/messages/inbox/friend_1/message_1.json", json.dumps({
        "participants": [{"name": "friend"}, {"name": "nikita"}],
        "messages": [{"sender_name": "nikita", "timestamp_ms": ts(45) * 1000, "content": "Photos from the trip are finally up on my profile!"}]}))
    z.close()


def x_archive(path: Path):
    z = zipfile.ZipFile(path, "w")
    z.writestr("data/manifest.js", "window.__THAR_CONFIG = {}")
    z.writestr("data/account.js", 'window.YTD.account.part0 = ' + json.dumps([{"account": {"accountId": "42", "username": "nd"}}]))
    t = (T0 + timedelta(days=15)).strftime("%a %b %d %H:%M:%S +0000 %Y")
    z.writestr("data/tweets.js", 'window.YTD.tweets.part0 = ' + json.dumps([
        {"tweet": {"id_str": "1", "created_at": t, "full_text": "Shipping the thing we've worked on for a year. Proud of this team. https://t.co/abc"}},
        {"tweet": {"id_str": "2", "created_at": t, "full_text": "RT @someone: not mine"}}]))
    z.writestr("data/tweets_media/1-abc.jpg", picture(400))
    z.close()


MOSCOW = (55.7558, 37.6173)


def olga(path: Path):
    """Second person: Moscow for 35 days, then joins Nikita in Paris and travels with him (≈100 m apart)."""
    z = zipfile.ZipFile(path, "w")
    nik = track()
    r = np.random.default_rng(7)
    pts = []
    for t, la, lo in nik:
        day = (t - T0).total_seconds() / 86400
        if day < 35:
            ang, d = r.uniform(0, 2 * math.pi), r.uniform(0, 0.04)
            pts.append((t, MOSCOW[0] + d * math.sin(ang), MOSCOW[1] + d * math.cos(ang)))
        else:
            pts.append((t, la + 0.0009, lo))
    z.writestr("Takeout/Location History (Timeline)/Records.json", json.dumps({"locations": [
        {"latitudeE7": int(la * 1e7), "longitudeE7": int(lo * 1e7), "timestamp": iso(t)} for t, la, lo in pts]}))
    folder = "Takeout/Google Photos/Photos from 2019"
    for i in range(40):
        dt = T0 + timedelta(days=i * 70 / 40, hours=5)
        t, la, lo = min(pts, key=lambda p: abs((p[0] - dt).total_seconds()))
        name = f"PXL_{dt:%Y%m%d_%H%M%S}000.jpg"
        z.writestr(f"{folder}/{name}", picture(500 + i, 1200, 1600))
        z.writestr(f"{folder}/{name}.json", json.dumps({
            "title": name, "photoTakenTime": {"timestamp": str(int(dt.timestamp()))},
            "geoData": {"latitude": la, "longitude": lo},
            "googlePhotosOrigin": {"mobileUpload": {"deviceType": "IOS_PHONE"}}}))
    # a copy of Nikita's photo #40 that he shared with her via Partner Sharing
    dt = T0 + timedelta(days=40 * 70 / 60, hours=3)
    name = f"IMG_{dt:%Y%m%d_%H%M%S}.jpg"
    z.writestr(f"{folder}/{name}", picture(40))
    z.writestr(f"{folder}/{name}.json", json.dumps({
        "title": name, "photoTakenTime": {"timestamp": str(int(dt.timestamp()))},
        "googlePhotosOrigin": {"fromPartnerSharing": {}}}))
    z.close()


def make_all(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    google(out / "takeout-20190801T000000Z-001.zip")
    phone_timeline(out / "Timeline.json")
    ios_timeline(out / "location-history.json")
    facebook(out / "facebook-nikita-2025.zip")
    instagram(out / "instagram-nikita-2025.zip")
    x_archive(out / "twitter-2025.zip")
    (out / "partial-download.zip").write_bytes(b"PK\x03\x04 incomplete")
    return out


if __name__ == "__main__":
    make_all(Path(sys.argv[1] if len(sys.argv) > 1 else "fixtures"))
