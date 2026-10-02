"""Facebook and Instagram "Download your information" exports (JSON format).

Both come from Meta Accounts Center and share conventions: unix-second timestamps,
UTF-8 mojibaked into \\u00XX escapes, media referenced by a root-relative "uri".
Layouts seen in the wild (all handled by matching on file name, not full path):

  Facebook 2024+  your_facebook_activity/posts/your_posts__check_ins__photos_and_videos_1.json
                  your_facebook_activity/comments_and_reactions/comments.json
                  your_facebook_activity/messages/{inbox,e2ee_cutover,archived_threads}/*/message_N.json
  Facebook old    posts/your_posts_1.json, comments/comments.json, messages/inbox/...
  Instagram 2024+ your_instagram_activity/content/{posts_1,stories,reels}.json
                  your_instagram_activity/comments/{post_comments_1,reels_comments}.json
                  your_instagram_activity/messages/inbox/*/message_N.json
  Instagram old   content/posts_1.json, comments/post_comments.json, messages/inbox/...

A "Data logs" download (data_logs/content/N/page_M.json) holds ad/telemetry tables
only — nothing usable for the video, so it is ignored.
"""
from __future__ import annotations

import json
import re

from .archive import Member, warn
from .model import Photo, Place, Social, fix_tree, media_kind, parse_time, valid_latlon

MSG_FILE = re.compile(r"/messages/(inbox|e2ee_cutover|archived_threads|filtered_threads|message_requests)/[^/]+/message_\d+\.json$")
FB_POSTS = re.compile(r"(^|/)(your_posts[^/]*|your_uncategorized_photos|your_videos|album/\d+|group_posts_and_comments)\.json$")
FB_OTHER_POSTS = re.compile(r"(^|/)posts_on_other_pages_and_profiles\.json$")
IG_POSTS = re.compile(r"(^|/)content/(posts_\d+|stories|reels|archived_posts|igtv_videos)\.json$")
COMMENTS = re.compile(r"(^|/)comments[^/]*/[^/]*comments[^/]*\.json$|(^|/)(comments|your_comments_in_groups)\.json$")
FB_LOCATION = re.compile(r"(^|/)(location_history|your_location_history|check-ins|check_ins)[^/]*\.json$")


def detect(container_path: str, members: list[Member]) -> str | None:
    names = [m.name for m in members[:20000]]
    blob = "\n".join(names)
    if "data_logs/" in blob and "_activity/" not in blob:
        return None
    if "your_instagram_activity/" in blob or "instagram" in container_path.lower().rsplit("/", 1)[-1]:
        return "instagram"
    if "your_facebook_activity/" in blob or "facebook" in container_path.lower().rsplit("/", 1)[-1]:
        return "facebook"
    if re.search(r"(^|/)messages/inbox/", blob):
        return "instagram" if re.search(r"(^|/)content/posts_\d+\.json", blob) else "facebook"
    return None


def _load(m: Member):
    try:
        return fix_tree(json.loads(m.read()))
    except (ValueError, UnicodeDecodeError) as e:
        warn(f"cannot parse {m.uri}: {e}")
        return None


class _Resolver:
    """Map a Meta 'uri' (root-relative) to a Member, whatever wrapper dir the zip has."""

    def __init__(self, members):
        self.by_tail = {}
        for m in members:
            parts = m.name.split("/")
            for k in range(1, min(len(parts), 6) + 1):
                self.by_tail.setdefault("/".join(parts[-k:]), m)

    def __call__(self, uri: str) -> Member | None:
        if not uri or uri.startswith("http"):
            return None
        parts = uri.strip("/").split("/")
        for k in range(min(len(parts), 6), 0, -1):
            m = self.by_tail.get("/".join(parts[-k:]))
            if m:
                return m
        return None


def _exif_ll(media: dict):
    """(lat, lon, capture time) from the EXIF Meta kept; each may be None independently."""
    md = (media.get("media_metadata") or {})
    lat = lon = taken = None
    for kind in ("photo_metadata", "video_metadata"):
        for e in (md.get(kind) or {}).get("exif_data", []) or []:
            if lat is None and valid_latlon(e.get("latitude"), e.get("longitude")):
                lat, lon = e["latitude"], e["longitude"]
            taken = taken or parse_time(e.get("taken_timestamp") or e.get("date_time_original"))
    return lat, lon, taken


def _media_photo(media: dict, platform: str, resolve, caption: str, fallback_ts):
    uri = media.get("uri", "")
    kind = media_kind(uri)
    if kind is None:
        return None
    m = resolve(uri)
    if m is None:
        return None
    lat, lon, taken = _exif_ll(media)
    ts = taken or parse_time(media.get("creation_timestamp")) or fallback_ts
    if ts is None:
        return None
    origin = "exif" if taken else "post"  # "exif": the camera's capture time survived the upload
    return Photo(uri=m.uri, ts=ts, lat=lat, lon=lon, source=platform, title=m.basename, kind=kind,
                 description=caption, shared=True, ts_origin=origin)


def scan(platform: str, members: list[Member]):
    photos, social, places, raw_msgs = [], [], [], []
    resolve = _Resolver(members)
    for m in members:
        n = m.name
        if not n.endswith(".json"):
            continue
        if MSG_FILE.search(n):
            j = _load(m)
            if not isinstance(j, dict):
                continue
            parts = [p.get("name", "") for p in j.get("participants", [])]
            thread = j.get("title") or ", ".join(parts)
            for msg in j.get("messages", []):
                text = msg.get("content")
                if not text or msg.get("is_unsent"):
                    continue
                raw_msgs.append({"ts": parse_time(msg.get("timestamp_ms")), "sender": msg.get("sender_name", ""),
                                 "text": text, "thread": thread, "n": len(parts), "participants": parts})
        elif platform == "facebook" and FB_POSTS.search(n):
            j = _load(m)
            # [..], {"photos": [..]} (albums), {"other_photos_v2": [..]}, {"group_posts_v2": [..]}
            items = j if isinstance(j, list) else (j or {}).get("photos") or next(
                (v for v in (j or {}).values() if isinstance(v, list)), [])
            for p in items:
                ts = parse_time(p.get("timestamp") or p.get("creation_timestamp"))
                text = " ".join(d["post"] for d in p.get("data", []) if isinstance(d, dict) and d.get("post"))
                place, plat, plon, media, pls = "", None, None, [], []
                for att in p.get("attachments", []) or []:
                    for d in att.get("data", []):
                        if "place" in d:
                            pl = d["place"]
                            c = pl.get("coordinate") or {}
                            ok = valid_latlon(c.get("latitude"), c.get("longitude"))
                            pls.append((pl.get("name", ""), c["latitude"] if ok else None, c["longitude"] if ok else None))
                        if "media" in d:
                            media.append(d["media"])
                if pls:
                    # "X was travelling to <dest> from <origin>": attachments are [dest, origin] and the
                    # post was made at the origin
                    travel = len(pls) > 1 and "travelling to" in p.get("title", "")
                    here = pls[-1] if travel else pls[0]
                    place, plat, plon = here
                    if travel:
                        place = f"{pls[-1][0]} → {pls[0][0]}"
                if "uri" in p:  # album files list media directly
                    media.append(p)
                for med in media:
                    ph = _media_photo(med, "facebook", resolve, text or med.get("description", ""), ts)
                    if ph:
                        if ph.lat is None and plat is not None:
                            ph.lat, ph.lon = plat, plon
                        photos.append(ph)
                if ts and (text or place):
                    kind = "checkin" if place and not text else "post"
                    social.append(Social(ts, "facebook", kind, text or p.get("title", ""), place=place,
                                         lat=plat, lon=plon, media=[x.get("uri", "") for x in media]))
                if ts and place and plat is not None:
                    places.append(Place(ts, ts, plat, plon, place, "facebook"))
        elif platform == "facebook" and FB_OTHER_POSTS.search(n):
            j = _load(m)
            for p in j if isinstance(j, list) else []:
                lv = {x.get("label"): x.get("value") for x in p.get("label_values", []) if isinstance(x, dict)}
                ts = parse_time(p.get("timestamp"))
                if ts and lv.get("Message"):
                    social.append(Social(ts, "facebook", "post", lv["Message"], who=lv.get("Target") or ""))
        elif platform == "instagram" and IG_POSTS.search(n):
            j = _load(m)
            if isinstance(j, dict):  # {"ig_stories": [...]}, {"ig_reels_media": [...]}
                j = next((v for v in j.values() if isinstance(v, list)), [])
            kind = "story" if "stories" in n else "post"
            for p in j or []:
                medias = p.get("media") or [p]
                ts = parse_time(p.get("creation_timestamp") or medias[0].get("creation_timestamp"))
                caption = p.get("title") or medias[0].get("title") or ""
                for med in medias:
                    ph = _media_photo(med, "instagram", resolve, caption, ts)
                    if ph:
                        photos.append(ph)
                if ts and caption:
                    social.append(Social(ts, "instagram", kind, caption, media=[x.get("uri", "") for x in medias]))
        elif COMMENTS.search(n):
            j = _load(m)
            if isinstance(j, dict):
                j = next((v for v in j.values() if isinstance(v, list)), [])
            for c in j or []:
                if "data" in c:  # facebook comments_v2
                    for d in c.get("data", []):
                        cc = d.get("comment") or {}
                        if cc.get("comment"):
                            social.append(Social(parse_time(cc.get("timestamp") or c.get("timestamp")), platform,
                                                 "comment", cc["comment"], who=_on_whom(c.get("title", ""))))
                elif "string_map_data" in c:  # instagram
                    sm = c["string_map_data"]
                    text = (sm.get("Comment") or {}).get("value")
                    ts = parse_time((sm.get("Time") or {}).get("timestamp"))
                    if text and ts:
                        social.append(Social(ts, platform, "comment", text,
                                             who=(sm.get("Media Owner") or {}).get("value", "")))
        elif platform == "facebook" and FB_LOCATION.search(n):
            j = _load(m)
            if isinstance(j, dict):
                j = next((v for v in j.values() if isinstance(v, list)), [])
            for r in j or []:
                c = r.get("coordinate") or {}
                ts = parse_time(r.get("creation_timestamp") or r.get("timestamp"))
                if ts and valid_latlon(c.get("latitude"), c.get("longitude")):
                    places.append(Place(ts, ts, c["latitude"], c["longitude"], r.get("name", ""), "facebook"))
    social = [s for s in social if s.ts]
    return photos, social, places, raw_msgs


def _on_whom(title: str) -> str:
    m = re.search(r"(?:on|to) (.+?)'s? ", title)
    return m.group(1) if m else ""


def messages_to_social(platform: str, raw_msgs: list[dict], self_name: str | None) -> tuple[list[Social], str]:
    """Split raw messages into outgoing/incoming. The account owner is the name that
    appears in the most threads, unless given explicitly."""
    if not raw_msgs:
        return [], self_name or ""
    if not self_name:
        count = {}
        seen = set()
        for r in raw_msgs:
            key = (r["thread"], r["sender"])
            if key not in seen:
                seen.add(key)
                count[r["sender"]] = count.get(r["sender"], 0) + 1
        self_name = max(count, key=count.get)
    out = []
    for r in raw_msgs:
        if r["ts"] is None:
            continue
        mine = r["sender"] == self_name
        who = r["thread"] if r["n"] > 2 else next((p for p in r["participants"] if p != self_name), r["thread"])
        out.append(Social(r["ts"], platform, "message_out" if mine else "message_in", r["text"],
                          who=who if mine else r["sender"]))
    return out, self_name
