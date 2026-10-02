# Getting your data out

Where each export lives, which options to pick, and what `vwl` does with it.
The menus below describe these services as of 2025–26; they move things around
often, so if a label differs, search the service's help for "download your information".

General rules:

* **Always pick JSON** (machine-readable) when a service offers HTML vs JSON.
* **Pick "all time"** and **high media quality**.
* Download links usually **expire in 4–7 days**, so start the download when you get the email.
* Put everything under one folder (e.g. `/mnt/5tb/takeouts`). `vwl index` finds the zips at any depth,
  skips zips that are still downloading, and only scans new or changed zips on re-runs.

| Source | What `vwl` uses | Parsed today |
|---|---|---|
| Google Takeout: Photos | photos, capture time, GPS, favourites, captions | ✅ |
| Google Takeout: Location History / phone Timeline export | movement track, named places | ✅ (all 4 formats) |
| Facebook | posts, check-ins (with coordinates), comments, messages, posted photos | ✅ |
| Instagram | posts, stories, comments, messages, posted photos (+EXIF GPS) | ✅ |
| X / Twitter | tweets, replies, DMs, tweet photos, geotagged tweets | ✅ |
| Apple iCloud Photos, any photo folder | photos + EXIF time/GPS | ✅ |
| Strava / Garmin / any GPX | tracks | ✅ (GPX only, not FIT) |
| Snapchat, WhatsApp, Telegram, Spotify, Uber, Swarm | — | ❌ not yet (see the end of this page) |

---

## Google Takeout (Photos, and Location History if it's still in the cloud)

1. Go to <https://takeout.google.com>, then **Deselect all**.
2. Tick **Google Photos**. The default is "All photo albums included". The `Photos from YYYY` folders
   hold everything; album folders are duplicates, which `vwl` de-duplicates.
3. Tick **Location History (Timeline)** (older name: "Location History"). Also worth adding:
   **Maps (your places)** and **Fit**.
4. **Next step**: Transfer to *Send download link via email*, Frequency *Export once*,
   File type **.zip**, File size **50 GB**. Larger parts mean fewer files; your current
   export uses 2 GB parts, which gives ~100+ zips. That works, it's just tedious.
5. You'll get an email in hours to days. Each part link can be downloaded ~5 times and expires after 7 days.

What you get: `Takeout/Google Photos/Photos from 2019/IMG_…jpg` plus a JSON "sidecar" per photo,
named `IMG_…jpg.json` (older exports) or `IMG_…jpg.supplemental-metadata.json` (late 2024+). The name
is sometimes truncated to 51 characters, and the sidecar is often **in a different zip part than the photo**.
`vwl` pairs them globally after indexing all parts. Until every part has downloaded, it falls back to
the photo's own EXIF data.

### Location History: now on your phone

From 2024 Google moved Timeline storage **onto the device**. After your account migrated, Takeout's
"Location History (Timeline)" holds little or nothing, and the full history must be exported from the phone:

* **Android**: Settings → Location → Location services → **Timeline** → **Export Timeline data**.
  This saves `Timeline.json` (`semanticSegments`, `rawSignals`).
* **iPhone**: Google Maps app → your profile picture → **Settings** → **Location & Privacy** →
  **Export Timeline data**. This saves `Timeline.json` too, but in a different layout from Android's
  (a plain list of visits and paths with `geo:` coordinates; older iOS versions named it
  `location-history.json`). `vwl` reads both. On an iPhone it may cover only the last months.

**Google Fit** (tick "Fit" in Takeout) is a good second source: its `All data/*location.sample*.json`
and GPS-tracked `Activities/*.tcx` workouts often survive after Timeline has moved to the phone.

Copy the file into your takeouts folder; `vwl` recognises it by content. Older takeouts contain
`Records.json` (every raw fix since you enabled it) and `Semantic Location History/YYYY/YYYY_MONTH.json`
(visits with place names). Both are parsed too, and all sources are merged. If you have no location
history at all, the track is built from photo GPS tags, geotagged posts and check-ins.

---

## Facebook

1. <https://accountscenter.facebook.com> → **Your information and permissions** → **Download your information**
   → **Download or transfer information**.
2. Pick your Facebook profile, then **Specific types of information** (or "All available information"). At least tick:
   *Posts*, *Photos and videos*, *Comments and reactions*, *Messages*, *Check-ins*, *Location*.
3. **Download to device**. Date range **All time**, Format **JSON**, Media quality **High**. Then **Create files**.
4. You'll get a notification when it's ready (hours to a few days). It stays downloadable for ~4 days.

> ⚠️ A download whose zip holds `data_logs/content/N/page_1.json` is a **"Data logs"** export (ad
> impressions, device telemetry, login records). It has no posts or messages; request the regular
> "profile information" export above.

**End-to-end-encrypted Messenger chats** (the default for 1:1 chats since late 2023) are *not*
in the regular download. In Messenger go to Settings → **Privacy & safety** → **End-to-end encrypted chats** →
**Message storage** → **Download secure storage data**. The format is similar. Put it next to the rest
and `vwl` will pick up `messages/e2ee_cutover/*/message_1.json` threads.

## Instagram

Same place: **Accounts Center** → Your information and permissions → Download your information →
pick the **Instagram** account → JSON, All time, High quality. Or in the app: Profile → ☰ → Accounts
Center → Your information and permissions. Tick *Content* (posts, stories, reels), *Comments*,
*Messages*, *Media*.

What you get: `your_instagram_activity/content/posts_1.json`, `…/stories.json`,
`…/comments/post_comments_1.json`, `…/messages/inbox/*/message_1.json`, and the media files
themselves (`media/posts/YYYYMM/*.jpg`). Posted photos keep EXIF GPS in `media_metadata` when available.

Meta exports write every non-ASCII character as mojibake (`Ð\u009f…` instead of `П…`); `vwl` repairs it.

## X (Twitter)

**Settings and privacy** → **Your account** → **Download an archive of your data** → re-enter password, confirm the code →
**Request archive**. It takes 24 hours to several days; you'll get an email and an in-app notification.

What you get: `data/tweets.js` (older: `tweet.js`, plus `-part1…` for big accounts), `data/direct-messages.js`,
`data/tweets_media/`, `data/account.js`. Retweets are skipped and `t.co` links stripped.

## Apple (iCloud)

1. <https://privacy.apple.com> → **Request a copy of your data**.
2. Tick **iCloud Photos** (large), optionally iCloud Drive, Notes, Mail, Contacts, Calendars. Choose the
   largest file size (splits into parts). It takes up to ~7 days.
3. You get `iCloud Photos Part 1 of N.zip` with originals (HEIC/JPG/MOV) and `Photo Details.csv`.
   `vwl` reads time and GPS straight from each photo's EXIF.

Apple **does not export location history**: "Significant Locations" stay encrypted on the device.
Photo GPS tags are the substitute. Alternatives for photos: privacy.apple.com → **Transfer a copy of your data**
→ Google Photos, which then comes out through Takeout; or on a Mac, Photos → select all → File → Export → *Export Unmodified Originals*.
For a scripted download, [`icloudpd`](https://github.com/icloud-photos-downloader/icloud_photos_downloader) works.

## Strava, Garmin, other GPS trackers

* **Strava**: Settings → My Account → *Download or Delete Your Account* → **Request your archive**. The zip
  contains `activities/*.gpx` (and `.fit.gz`; convert FIT with `gpsbabel -i garmin_fit -f x.fit -o gpx -F x.gpx`).
* **Garmin Connect**: <https://www.garmin.com/account/datamanagement/exportdata/>.
* Any `.gpx` file in the folder is added to the movement track.

---

## Not parsed yet (worth requesting now; they take days)

| Service | How | Why it's interesting |
|---|---|---|
| **Snapchat** | <https://accounts.snapchat.com> → My Data → tick JSON, date range → Submit | location history, chat history, Memories (download links) |
| **WhatsApp** | per chat: ⋮ → More → **Export chat** (with media) → `.txt` | the most personal messages for many people; no bulk export exists |
| **Telegram** | Telegram Desktop → Settings → Advanced → **Export Telegram data** → format *JSON* | full message history, media |
| **Spotify** | Account → Privacy settings → **Extended streaming history** (takes up to 30 days) | "what you were listening to", a soundtrack per year |
| **Uber** | <https://myprivacy.uber.com> → Download your data | every trip with pickup/drop-off coordinates |
| **Swarm / Foursquare** | Swarm app → Settings → Privacy → Download your data | check-ins with venue names since 2009 |
| **LinkedIn** | Settings → Data privacy → **Get a copy of your data** | job changes, which make good chapter titles |
| **Reddit** | <https://www.reddit.com/settings/data-request> | posts and comments |
| **TikTok** | Settings and privacy → Account → **Download your data** → JSON | posts, DMs |
| **Microsoft / Skype** | <https://account.microsoft.com/privacy> → Download your data; Skype export page | old chat history |
| **Amazon** | Account → *Request Your Data* | orders, a surprisingly good diary |

For services with no self-serve export, the GDPR (EU/UK, Art. 15 and 20) and the CCPA/CPRA (California)
give you the right to request your data from the company's privacy team.
