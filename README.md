# virtual worldlines

Turn personal data exports (Google Takeout, Facebook, Instagram, X, iCloud, GPX…) into a video of a
life as the data recorded it — or of two lives, with maps that split and merge as the worldlines do.

[![10 seconds of a two-person film: Mumbai and Goa, December 2018](docs/vwl-demo.webp)](docs/vwl-demo.mp4)

*Ten seconds excerpted from a two-person film (19–26 December 2018): the opening title over a mosaic of
every photo in the film; rows of photos while together; the screen splitting while one is in Goa and the
other in Mumbai (the side without new photos shows its latest one, dimmed and undated); the maps merging
when they meet again; video clips; the continuously moving map with its world locator; the timeline
strip (scaled to the excerpt's days); and the closing mosaic. Click for the 1080p MP4.*

* **photos** chosen from tens of thousands, morphing into each other (optical-flow warp and dissolve,
  slow Ken Burns zoom), captioned with date and place;
* a **map that moves continuously**: the dot rests at each photo's place and travels your recorded
  route to the next one (one point per day for gaps over two days), easing out and in; the camera
  follows smoothly, zooming out for trips and back in when you stay, never cutting; the trail is the
  path of the last ~5 seconds of film and fades with age, over a faint glowing trace of everywhere
  you've been;
* a **timeline strip** (shown photos above, location-data density below, with a playhead), year cards,
  and an intro/outro over the whole-life map with totals.

```
intro: whole-life map ─▶ photo₁ (hold) ─morph─▶ photo₂ … ─▶ outro: totals
                         ├ date · place (reverse-geocoded offline)
                         ├ map: trail t₁→t₂, zoom fitted to the movement
                         └ strip: where t is in your whole dataset
```

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # ffmpeg ships inside imageio-ffmpeg
```

## Use

```bash
# 1. index: safe to re-run whenever more takeout parts finish downloading
.venv/bin/python -m vwl index /mnt/5tb/takeouts/nikita-google-takeout /mnt/5tb/takeouts/facebook-*.zip --work work

# 2. plan: pick photos, write work/plan.json and work/contact_sheet.jpg
.venv/bin/python -m vwl plan --work work --from 2005 --to 2024 --images 150 --hold 2.2 --transition 1.0

# 3. look at a frame or two, then render
.venv/bin/python -m vwl render --work work --preview-at 30 --out frame.png
.venv/bin/python -m vwl render --work work --out life.mp4 --music soundtrack.mp3
```

`vwl all <paths> --work work --out life.mp4` runs all three.

### Curating

* `work/contact_sheet.jpg` shows every chosen photo, numbered. To drop one, put its file name
  (e.g. `IMG_20200228_101010.jpg`) or its `uri` from `plan.json` into `work/exclude.txt` and re-run `plan`.
* `work/plan.json` is meant to be edited by hand: change a caption, delete a slot, lengthen one hold.
  Then run `render` again. Re-running `plan` overwrites it.
* `plan` options:

  | option | meaning | default |
  |---|---|---|
  | `--from DATE`, `--to DATE` | date range, `YYYY`, `YYYY-MM` or `YYYY-MM-DD`; `--to` is inclusive (`--to 2019` = through 31 Dec 2019). Limits the photos, the map track, the timeline strip and the intro totals | all |
  | `-n/--images N` or `X%` | number of photos and videos in the film, or a percentage of those in the date range (`-n 0.3%` of 43k items ≈ 130) | 120 |
  | `--hold SEC` | how long each photo is displayed | 2.2 |
  | `--hold-video SEC` | how long a video clip is shown (shorter videos play in full) | 4.0 |
  | `--video-share X%` | at most this share of the items may be videos (`0` = photos only); a ceiling, not a quota | 15% |
  | `--transition SEC` | length of the morph between items (`0` = hard cut) | 1.0 |
  | `--intro SEC`, `--outro SEC` | opening title over the photo mosaic; closing mosaic | 5, 6 |
  | `--size`, `--fps`, `--title` | output format, intro title | 1920x1080, 30 |

  Film length = 5 s intro + Σ holds + (N − 1) × transition + 6 s outro, e.g. 150 photos at 2.2 + 1.0 s ≈ 8 min;
  `plan` prints the breakdown.
* Useful `render` flags: `--style dark|light|satellite|topo`, `--morph 0` for plain dissolves,
  `--workers N`, `--crf`, `--tile-url` for a keyed tile provider.

## Inputs

Point `index` at directories (or individual files) in any mix. A directory is searched at any depth,
and what it holds is sorted out by type:

| In an input directory | Read as |
|---|---|
| `*.zip` | one export each; Takeout part, Facebook/Instagram or X is told from the content |
| an extracted export folder (`Takeout/`, a Meta `your_*_activity/` export, an X archive) | one export each, not searched further |
| `Timeline.json`, `location-history.json`, `Records.json`, `*.gpx`, `*.tcx` | location data, one each |
| other photos and videos | one loose-media collection (a photo folder, or files left over from extracting zips) |
| anything else (mailboxes, documents, …) | ignored |

So a person's folder can simply hold their Takeout zips, the phone's `Timeline.json` and their Facebook
export side by side. Overlapping inputs (a folder and its parent) count each file once. Index each person
into their own `--work` directory.

Each export is recognised by its content:

| Export | Recognised by |
|---|---|
| Google Takeout (any number of parts, zipped or extracted) | `Takeout/…` |
| Location History: `Records.json`, `Semantic Location History`, phone exports (`Timeline.json`; Android and iPhone use different layouts, both read) | file content |
| Google Fit inside Takeout: `Fit/All data/*location.sample*.json`, `Fit/Activities/*.tcx` | file path/content |
| Facebook / Instagram "Download your information" (**JSON**) | `your_facebook_activity/`, `your_instagram_activity/`, `messages/inbox/` |
| X archive | `data/manifest.js` |
| Anything else: photo folders (iCloud export, DCIM backup), `.gpx` files | EXIF / GPX |

See **[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)** for how to request each export, which options to pick,
and what is not parsed yet.

Zips are read in place; nothing is extracted. Zips still being downloaded are skipped with a
warning. Results are cached per zip in `work/index/cache`, so re-indexing after new parts arrive only reads
the new parts. Photo↔sidecar pairing is redone globally each time, because Google puts a photo and its
JSON sidecar in *different* parts.

## How photos are chosen

1. Screenshots are dropped; received images (WhatsApp `-WA`, `FB_IMG`…) are down-weighted.
2. Photos are grouped into **events**: a new event starts after a gap of more than 4 h or a jump of more than 30 km.
3. Each **year** gets a quota ∝ √(photos that year), so the phone-camera decade doesn't drown out the
   early years, and every year with photos appears.
4. Within a year, events are ranked by √size, with bonuses for favourites and photos shared to social media,
   and a ×1.8 bonus for **trips** (more than 100 km from the most-photographed place).
5. For each chosen event a few candidates are decoded and scored on sharpness, exposure, contrast,
   colourfulness, favourite/shared/views/caption. Near-duplicates are removed by perceptual hash,
   and picks are spread in time.
6. **Nudity filter**: candidates are checked by a local NudeNet model (nothing leaves the machine). It's
   best-effort, so still look at the contact sheet. Flagged files are listed in `work/nsfw_flagged.txt`;
   `--no-nsfw-filter` turns it off.

## Videos

Videos are indexed from the same Google Photos sidecars as photos (nothing is decompressed at index
time). At plan time the best candidates are extracted from their zip to `work/cache/videos/`,
sampled at 2 frames/s and scored like photos (sharpness, exposure, colour) plus steadiness; the
steadiest, sharpest `--hold-video`-second window is chosen (avoiding the first and last second) and
transcoded to `work/clips/` at the film's frame rate, rotation applied and HDR tone-mapped to SDR.
`clip_start` in `plan.json` can be edited by hand. Clips are **muted**; during a clip the map runs in
real time, and the clip keeps playing through the morph into the next item. Screen recordings, clips
under 2 s, and iPhone Live Photo clips (folded into their still) are skipped; files over 1 GB are only
considered when favourited. Google-made compositions (collages, animations) are skipped entirely.

## Two people

One film of two worldlines, e.g. you and your partner. Index each person separately, then plan together:

```bash
vwl index /mnt/5tb/takeouts/nikita-google-takeout --work work
vwl index /mnt/5tb/takeouts/olga-google-takeout-2026.10.01 --work work_olga
vwl plan  --work work_duo --person Nikita=work --person Olga=work_olga -n 150
vwl render --work work_duo --out duo.mp4
```

* **Date range**: `--from/--to`, or by default the overlap of both datasets (trimmed at the 0.5/99.5
  percentiles, so a single misdated photo can't stretch it).
* **Shared copies**: a shot in both libraries (Partner Sharing, shared albums) is kept once, credited to
  the person whose own upload it is.
* **Together / apart / unknown** on a 10-minute grid from both tracks (together = within 500 m); unknown
  where either track has no data, which is never drawn as "apart". Across a gap in someone's data, their
  last position only counts while they evidently stayed put: if their next fix is elsewhere (a flight, a
  drive), the gap is unknown rather than "still at home", so a sparse photo-only track can't make two
  people travelling together look thousands of kilometres apart.
* **Selection** per category — together, first person alone, second person alone — √-weighted, with
  together counting double.
* **Screen layout**: the bottom third is a map band and never overlaps the photos.
  * *Together* (within 500 m, or in the same area, ≤ 25 km): a row of 1–3 consecutive photos/videos
    across the top, sized by their shapes so the row fills the area, above one wide map with both
    routes as parallel lanes.
  * *Apart*: the screen splits down the middle — the first person's 1–2 photos above their map on the
    left (orange), the second person's on the right (cyan), each map on its own person's clock. If one
    side has no new photos for a while, it keeps showing that person's latest one.
  * Every spread is a contiguous slice of time, so dates and the timeline playhead only move forward.
    When one person has no new photos in a slice, their half shows their latest photo again, dimmed
    and captioned "latest photo" without its old date, while their map follows the current time.
  * The layout switches during the transition between spreads; a single item never flips it on its own.
  * `vwl relayout --work work_duo` regroups an existing plan with the current rules in seconds (no
    re-selection or re-export); `--intro/--outro` there change the opening and closing lengths.
  * `--hold` is how long each photo is on screen, so a row of three stays up for one hold.
* **Chapter cards**: "First days together" (first sustained co-location: ≥ 2 h on ≥ 3 days within a
  month; override with `--met DATE`) and "Together again, after N days apart" (> 60 days).
* **Timeline strip**: distance between you over time (log scale): gold = together, tall = far apart, grey = no data.
* **Totals**: days together (of days with data from both), km travelled together, countries together
  (≥ 2 h there), longest time apart, furthest apart.

## Social-network exports

Facebook, Instagram and X exports are still indexed: the photos you posted there are candidates for the
video, and geotagged posts and check-ins add points to the map. A posted photo that kept its camera
capture time is matched to its original in your library: the original is marked as shared (a strong
selection boost) and gets the post's caption, and the platform's downscaled copy is dropped. Their text (posts, comments, messages) is
indexed into `work/index/social.jsonl` but no longer shown in the video, because it was usually
unrelated to the photo on screen.

## Map tiles

Place names come from the style's transparent label layer, drawn from tiles one zoom level coarser than
the map itself (1–2× the native text size) and brightened, on top of the life trace.

Tiles are fetched once before rendering into `~/.cache/vwl/tiles` and reused. The default styles use Esri's
keyless ArcGIS Online basemaps with attribution burned into the panel, which is fine for a personal,
non-commercial video. CARTO's CDN now answers keyless requests with an "API KEY REQUIRED" image, and
OpenStreetMap's own tile servers forbid bulk downloads. For those, or Stadia/MapTiler, pass a keyed template
with `--tile-url 'https://…/{z}/{x}/{y}.png?api_key=…'`. `--offline` renders with whatever is cached.

## Performance

**Prefer the zips over an extracted Takeout on exFAT/NTFS external drives.** Every file open there
is a linear directory scan, and a `Photos from 2018` folder can hold 18k files, so opening 48k
sidecars one by one takes 30+ minutes on a cold cache. The same data read from the 119 zip parts
indexes in a few minutes, because each zip's central directory lists everything at once. If you do
point `vwl` at an extracted `Takeout/` folder, it only walks Google Photos, Location History and Fit,
and never stats individual files.


About 25–40 frames/s of output across 12 worker processes at 1080p on a 16-core machine, so a 7-minute
video takes a few minutes. Rendering is split into chunks at photo boundaries, encoded in parallel,
and concatenated without re-encoding. Indexing a 100-part takeout is dominated by reading the zip directories
and sidecar JSONs, which is mostly I/O.

## Tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests
```

`tests/fixtures.py` generates synthetic exports in every supported format (including sidecar naming
quirks, Meta mojibake, a still-downloading zip) and renders a short video from them.
`python -m tests.fixtures /tmp/fake` writes them out so you can try the pipeline end to end.

## Known limitations

* Local time for captions is approximated from longitude (the date can be off by one near midnight).
* Two-person "together" uses distance only; Google's visit `placeId`s aren't indexed yet.
* The map doesn't wrap across the antimeridian (a trans-Pacific trip draws the long way round).
* WhatsApp, Telegram, Snapchat, Spotify, Uber and Swarm exports aren't parsed yet.
