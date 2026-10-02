"""Uniform read access to export dumps, whether still zipped or already extracted.

Google Takeout arrives as dozens of 2-50 GB zip parts; Meta and X arrive as one or
a few zips.  Extracting all of that is wasteful, so every ingester works on
`Member` objects that can be read straight out of a zip.  Zips that are still
downloading (no central directory yet) are skipped with a warning and picked up
on the next `vwl index` run.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import threading
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class Member:
    container: str       # path to zip file or directory root
    name: str            # path inside container, always '/'-separated
    size_hint: int = -1  # bytes if known for free (zip directory); -1 = unknown (loose file, not stat-ed)

    @property
    def size(self) -> int:
        """Exact size; stats loose files, which is slow on exFAT/NTFS drives — prefer size_hint."""
        if self.size_hint >= 0:
            return self.size_hint
        return os.path.getsize(os.path.join(self.container, self.name))

    @property
    def basename(self) -> str:
        return self.name.rsplit("/", 1)[-1]

    @property
    def dirname(self) -> str:
        return self.name.rsplit("/", 1)[0] if "/" in self.name else ""

    @property
    def uri(self) -> str:
        return f"{self.container}::{self.name}"

    def read(self, limit: int | None = None) -> bytes:
        return read_uri(self.uri, limit)

    def open(self):
        return open_uri(self.uri)


_local = threading.local()


def _zip(path: str) -> zipfile.ZipFile:
    cache = getattr(_local, "zips", None)
    if cache is None or cache.get("_pid") != os.getpid():
        cache = _local.zips = {"_pid": os.getpid()}
    zf = cache.get(path)
    if zf is None:
        zf = cache[path] = zipfile.ZipFile(path)
    return zf


def open_uri(uri: str):
    container, name = uri.split("::", 1)
    if os.path.isdir(container):
        return open(os.path.join(container, name), "rb")
    return _zip(container).open(name)


def read_uri(uri: str, limit: int | None = None) -> bytes:
    with open_uri(uri) as f:
        return f.read() if limit is None else f.read(limit)


# In an extracted Takeout only these product folders are read; everything else (Drive, Mail,
# YouTube, …) can be 100k+ files that are slow to even list on exFAT/NTFS external drives.
TAKEOUT_KEEP = re.compile(r"(?i)^(google\s*(photos|fotos|фото|foto)|photos|location history.*|"
                          r"standortverlauf.*|historique des positions.*|история местоположений.*|fit)$")


def _walk(root: str, is_takeout: bool):
    for dirpath, dirs, files in os.walk(root):
        if is_takeout and dirpath == root:
            dirs[:] = [d for d in dirs if TAKEOUT_KEEP.match(d)]
        dirs.sort()
        yield dirpath, files


class Container:
    """A zip file or a directory tree."""

    def __init__(self, path: str | Path, files: list[str] | None = None):
        """`files`: for a directory, index only these (relative) paths — e.g. the loose photos and
        videos lying next to export zips — instead of everything under it."""
        self.path = str(Path(path).resolve())
        self.files = sorted(files) if files is not None else None
        self.is_dir = os.path.isdir(self.path)
        # a single loose export file (e.g. a Timeline.json exported from the phone)
        self.is_file = not self.is_dir and not self.path.lower().endswith(".zip")
        # an extracted Takeout, given as the Takeout/ folder itself
        self.is_takeout_dir = self.is_dir and os.path.basename(self.path) == "Takeout"

    def fingerprint(self) -> str:
        """Changes when the container changes; used to cache per-container parse results."""
        if self.is_dir:
            self.members()
            return self._fp
        st = os.stat(self.path)
        return hashlib.sha1(f"{self.path}\0{st.st_size}\0{st.st_mtime_ns}".encode()).hexdigest()[:16]

    def _list_files(self):
        """Explicit file list: fingerprint from the names and their directories' mtimes."""
        h, out = hashlib.sha1(self.path.encode()), []
        for d in sorted({os.path.dirname(f) for f in self.files}):
            h.update(f"{d}\0{os.stat(os.path.join(self.path, d)).st_mtime_ns}\n".encode())
        for f in self.files:
            h.update(f.encode("utf-8", "surrogateescape") + b"\0")
            out.append(Member(self.path, f.replace(os.sep, "/")))
        self._members, self._fp = out, h.hexdigest()[:16]

    def _list_dir(self):
        """One walk yields both the listing and the fingerprint (file names + directory mtimes:
        adding, removing or renaming a file changes its directory's mtime).  Individual files are
        never stat-ed: on exFAT every lookup is a linear directory scan, so stat-ing 100k files in
        big "Photos from YYYY" folders takes tens of minutes."""
        h, out = hashlib.sha1(self.path.encode()), []
        for root, files in _walk(self.path, self.is_takeout_dir):
            h.update(f"{root}\0{os.stat(root).st_mtime_ns}\0{len(files)}\n".encode())
            for fn in sorted(files):
                h.update(fn.encode("utf-8", "surrogateescape") + b"\0")
                if fn.lower().endswith(".zip") or fn == ".DS_Store":
                    continue  # zips are containers of their own
                rel = os.path.relpath(os.path.join(root, fn), self.path).replace(os.sep, "/")
                out.append(Member(self.path, rel))
        self._members, self._fp = out, h.hexdigest()[:16]

    def members(self) -> list[Member]:
        """Listing is done once per Container and kept, so it can be pickled to a worker."""
        if self.is_file:
            return [Member(os.path.dirname(self.path), os.path.basename(self.path), os.path.getsize(self.path))]
        if self.is_dir:
            if getattr(self, "_members", None) is None:
                self._list_files() if self.files is not None else self._list_dir()
            return self._members
        zf = _zip(self.path)
        return [Member(self.path, i.filename, i.file_size) for i in zf.infolist() if not i.is_dir()]


LOCATION_FILE = re.compile(r"(?i)^(timeline|location[-_ ]history|records)[^/]*\.json$|\.(gpx|tcx)$")
MEDIA_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff", ".gif", ".bmp",
             ".mp4", ".mov", ".m4v", ".3gp", ".avi", ".mkv", ".mts", ".webm"}


def _export_folder(d: str) -> bool:
    """An extracted export, read as one unit: Takeout/, a Meta export (your_*_activity/), an X archive."""
    if os.path.basename(d) == "Takeout":
        return True
    try:
        names = set(os.listdir(d))
    except OSError:
        return False
    return ("your_facebook_activity" in names or "your_instagram_activity" in names
            or ("data" in names and os.path.exists(os.path.join(d, "data", "manifest.js"))))


def discover(paths: list[str]) -> Iterator[Container]:
    """Expand the inputs (files or directories) into containers.  A directory is searched at any depth:

    * every complete .zip is one container (its kind — Takeout part, Facebook/Instagram, X — is
      detected from its content later);
    * an extracted export folder (Takeout/, a Meta or X export) is one container, not searched further;
    * Timeline.json / location-history.json / Records.json and .gpx / .tcx files are one container each;
    * the remaining loose photos and videos form one more container (a photo folder, or files left
      over from extracting zips);
    * anything else (mailboxes, documents, …) is ignored.

    Overlapping inputs (a directory and its parent) yield each file once.
    """
    claimed: set[str] = set()

    def first(path: str) -> bool:
        path = os.path.realpath(path)
        if path in claimed:
            return False
        claimed.add(path)
        return True

    for p in paths:
        p = os.path.expanduser(p).rstrip("/") or "/"
        if os.path.isfile(p):
            if not first(p):
                continue
            if zipfile.is_zipfile(p) or not p.lower().endswith(".zip"):
                yield Container(p)  # a zip, or a single loose file such as Timeline.json
            else:
                warn(f"skipping {p}: not a complete zip (still downloading?)")
            continue
        if not os.path.isdir(p):
            warn(f"skipping {p}: does not exist")
            continue
        if _export_folder(p):
            if first(p):
                yield Container(p)
            continue
        media = []
        for root, dirs, files in os.walk(p):
            for d in sorted(dirs):
                if _export_folder(os.path.join(root, d)) and first(os.path.join(root, d)):
                    yield Container(os.path.join(root, d))
            dirs[:] = sorted(d for d in dirs if not _export_folder(os.path.join(root, d)) and not d.startswith("."))
            for fn in sorted(files):
                full = os.path.join(root, fn)
                low = fn.lower()
                if not first(full):
                    continue
                if low.endswith(".zip"):
                    if zipfile.is_zipfile(full):
                        yield Container(full)
                    else:
                        warn(f"skipping {full}: not a complete zip (still downloading?)")
                elif LOCATION_FILE.search(fn):
                    yield Container(full)
                elif low[low.rfind("."):] in MEDIA_EXT and not fn.startswith("._"):
                    media.append(os.path.relpath(full, p))
        if media:
            yield Container(p, files=media)


def warn(msg: str) -> None:
    print(f"[vwl] {msg}", file=sys.stderr)


def _single_threaded():
    """Pool initializer: parallelism comes from the processes, so each library in a worker (OpenCV,
    onnxruntime, BLAS) gets one thread — otherwise 16 workers × 16 threads thrash the CPU."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        pass


def process_pool(workers: int):
    """Process pool that is safe after OpenCV/requests have started threads (no fork)."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                               initializer=_single_threaded)
