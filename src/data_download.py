"""
Streaming extractor for the official Yelp Open Dataset.

The official archive (Yelp-JSON.zip, ~4.3 GB) contains one entry,
"Yelp JSON/yelp_dataset.tar", which is itself a gzip-compressed tar with:
    business.json (150K rows) · checkin.json · review.json (~7M rows)
    tip.json · user.json (~2M rows, ~3.3 GB because of the `friends` lists)

Downloading and unpacking it would need ~14 GB of free disk. Instead we stream it:
    HTTP response -> ZIP raw-deflate -> gzip -> tar (stream mode) -> line by line
and keep only what the project needs:
    * business.json : all rows
    * review.json   : a systematic sample (every k-th line, see config.REVIEW_SAMPLE_STEP).
                      The file is grouped by business, so "first N lines" would only cover
                      a small subset of businesses; every k-th line covers all of them.
    * user.json     : the REAL user records (incl. the real `elite` field) of every user
                      that wrote a sampled review; the huge `friends` list is dropped.

A local copy of the archive can be used instead of the URL by setting the environment
variable YELP_ARCHIVE=/path/to/Yelp-JSON.zip (or /path/to/yelp_dataset.tar).

Authors: Uali Sembay & Arystan Aldiyev (AAI-2605)
"""

import io
import json
import os
import ssl
import struct
import tarfile
import time
import urllib.request
import zlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set

from src.config import (
    MAX_REVIEWS,
    RAW_BUSINESS_JSON,
    RAW_MANIFEST_JSON,
    RAW_REVIEW_JSON,
    RAW_USER_JSON,
    REVIEW_SAMPLE_STEP,
    YELP_ZIP_URL,
    ensure_directories,
)

LOCAL_HEADER_SIG = b"PK\x03\x04"
USER_ID_PREFIX = b'{"user_id":"'


# ---------------------------------------------------------------------------
# Low-level streaming helpers
# ---------------------------------------------------------------------------
class ZipEntryStream(io.RawIOBase):
    """Read-only stream that inflates a single raw-deflate ZIP entry on the fly."""

    def __init__(self, source, chunk_size: int = 1 << 20):
        self.source = source
        self.chunk_size = chunk_size
        self.inflater = zlib.decompressobj(-zlib.MAX_WBITS)
        self.buffer = b""
        self.eof = False
        self.compressed_bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while not self.buffer and not self.eof:
            raw = self.source.read(self.chunk_size)
            if not raw:
                self.buffer = self.inflater.flush()
                self.eof = True
                break
            self.compressed_bytes_read += len(raw)
            self.buffer = self.inflater.decompress(raw)
        n = min(len(b), len(self.buffer))
        b[:n] = self.buffer[:n]
        self.buffer = self.buffer[n:]
        return n


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _http_get(url: str, byte_range: str):
    req = urllib.request.Request(
        url,
        headers={"Range": f"bytes={byte_range}", "User-Agent": "Mozilla/5.0 (yelp-analytics-student-project)"},
    )
    return urllib.request.urlopen(req, timeout=120, context=_ssl_context())


def find_tar_entry_offset(url: str = YELP_ZIP_URL, probe_bytes: int = 1 << 20) -> int:
    """
    Locate the local-file header of the *.tar entry inside the remote ZIP by scanning
    the first `probe_bytes` bytes (the only preceding entry is a ~115 KB PDF licence).
    """
    with _http_get(url, f"0-{probe_bytes - 1}") as resp:
        head = resp.read()
    pos = head.find(LOCAL_HEADER_SIG)
    while pos != -1:
        fields = struct.unpack("<IHHHHHIIIHH", head[pos:pos + 30])
        name_len, extra_len = fields[9], fields[10]
        name = head[pos + 30:pos + 30 + name_len].decode("utf-8", "ignore")
        if name.endswith(".tar"):
            return pos
        pos = head.find(LOCAL_HEADER_SIG, pos + 4)
    raise RuntimeError("Could not locate the yelp_dataset.tar entry inside Yelp-JSON.zip")


def _open_zip_entry(raw_stream) -> ZipEntryStream:
    """Skip the 30-byte local header + name + extra field and return an inflating stream."""
    local_hdr = raw_stream.read(30)
    fields = struct.unpack("<IHHHHHIIIHH", local_hdr)
    if local_hdr[:4] != LOCAL_HEADER_SIG:
        raise RuntimeError("Unexpected ZIP structure: local file header signature not found")
    method, name_len, extra_len = fields[3], fields[9], fields[10]
    raw_stream.read(name_len + extra_len)
    if method != 8:
        raise RuntimeError(f"Unsupported ZIP compression method {method} (expected deflate=8)")
    return ZipEntryStream(raw_stream)


def _open_tar_stream(archive: Optional[str]):
    """Return (tar_file_object, closers) for either a local archive or the official URL."""
    if archive:
        path = Path(archive).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"YELP_ARCHIVE points to a missing file: {path}")
        if path.suffix == ".zip":
            fh = open(path, "rb")
            fh.seek(0)
            head = fh.read(1 << 20)
            pos = head.find(LOCAL_HEADER_SIG)
            while pos != -1:
                f = struct.unpack("<IHHHHHIIIHH", head[pos:pos + 30])
                if head[pos + 30:pos + 30 + f[9]].decode("utf-8", "ignore").endswith(".tar"):
                    break
                pos = head.find(LOCAL_HEADER_SIG, pos + 4)
            fh.seek(pos)
            inner = io.BufferedReader(_open_zip_entry(fh), buffer_size=1 << 20)
            return tarfile.open(fileobj=inner, mode="r|*"), [fh]
        fh = open(path, "rb")
        return tarfile.open(fileobj=fh, mode="r|*"), [fh]

    offset = find_tar_entry_offset()
    print(f"[STREAM] {YELP_ZIP_URL} (tar entry at byte {offset:,})")
    resp = _http_get(YELP_ZIP_URL, f"{offset}-")
    inner = io.BufferedReader(_open_zip_entry(resp), buffer_size=1 << 20)
    return tarfile.open(fileobj=inner, mode="r|*"), [resp]


def _extract_user_id(line: bytes) -> str:
    if line.startswith(USER_ID_PREFIX):
        return line[len(USER_ID_PREFIX):len(USER_ID_PREFIX) + 22].decode("ascii", "ignore")
    return json.loads(line)["user_id"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def raw_data_is_ready() -> bool:
    """True when raw files exist AND were produced by this extractor (manifest present)."""
    if not (RAW_MANIFEST_JSON.exists() and RAW_BUSINESS_JSON.exists()
            and RAW_REVIEW_JSON.exists() and RAW_USER_JSON.exists()):
        return False
    try:
        manifest = json.loads(RAW_MANIFEST_JSON.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return manifest.get("status") == "complete" and manifest.get("user_source") == "official_user_json"


def stream_extract_official_yelp_dataset(
    sample_step: int = REVIEW_SAMPLE_STEP,
    max_reviews: int = MAX_REVIEWS,
    force: bool = False,
) -> Dict[str, object]:
    """
    Stream the official Yelp archive once and write business/review/user JSONL files.
    Returns a manifest dictionary (also saved to data/raw/_extraction_manifest.json).
    """
    ensure_directories()
    if not force and raw_data_is_ready():
        manifest = json.loads(RAW_MANIFEST_JSON.read_text(encoding="utf-8"))
        manifest["cached"] = True
        return manifest

    t0 = time.perf_counter()
    archive = os.environ.get("YELP_ARCHIVE")
    tar, closers = _open_tar_stream(archive)

    counts = {
        "business_rows": 0,
        "review_lines_seen": 0,
        "review_rows_kept": 0,
        "user_lines_seen": 0,
        "user_rows_kept": 0,
    }
    review_user_ids: Set[str] = set()
    review_done = False
    tmp_user = RAW_USER_JSON.with_suffix(".json.part")

    try:
        for member in tar:
            name = member.name
            if not member.isfile():
                continue
            print(f"  -> TAR entry: {name} ({member.size / 2**20:,.1f} MB)")
            fobj = tar.extractfile(member)
            if fobj is None:
                continue

            if name.endswith("business.json"):
                with open(RAW_BUSINESS_JSON, "wb") as out:
                    for line in fobj:
                        if line.strip():
                            out.write(line if line.endswith(b"\n") else line + b"\n")
                            counts["business_rows"] += 1
                print(f"     kept {counts['business_rows']:,} businesses")

            elif name.endswith("review.json"):
                with open(RAW_REVIEW_JSON, "wb") as out:
                    for idx, line in enumerate(fobj):
                        counts["review_lines_seen"] += 1
                        if counts["review_lines_seen"] % 1_000_000 == 0:
                            print(f"     ... {counts['review_lines_seen']:,} review lines streamed")
                        if idx % sample_step != 0 or counts["review_rows_kept"] >= max_reviews:
                            continue
                        if not line.strip():
                            continue
                        try:
                            review_user_ids.add(json.loads(line)["user_id"])
                        except (json.JSONDecodeError, KeyError):
                            continue  # malformed line: skip, it would be dropped by the ETL anyway
                        out.write(line if line.endswith(b"\n") else line + b"\n")
                        counts["review_rows_kept"] += 1
                review_done = True
                print(f"     kept {counts['review_rows_kept']:,} of {counts['review_lines_seen']:,} reviews "
                      f"(every {sample_step}th line), {len(review_user_ids):,} distinct users")

            elif name.endswith("user.json"):
                with open(tmp_user, "w", encoding="utf-8") as out:
                    for line in fobj:
                        counts["user_lines_seen"] += 1
                        if not line.strip():
                            continue
                        if review_done and _extract_user_id(line) not in review_user_ids:
                            continue
                        rec = json.loads(line)
                        rec.pop("friends", None)          # multi-GB list, not used in this project
                        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        counts["user_rows_kept"] += 1
                        if counts["user_lines_seen"] % 500_000 == 0:
                            print(f"     ... {counts['user_lines_seen']:,} user lines streamed")
                tmp_user.replace(RAW_USER_JSON)
                print(f"     kept {counts['user_rows_kept']:,} of {counts['user_lines_seen']:,} real user profiles")
            else:
                # checkin.json / tip.json / licence PDF: read through and discard
                while fobj.read(1 << 22):
                    pass
    finally:
        tar.close()
        for c in closers:
            try:
                c.close()
            except Exception:
                pass

    if counts["user_rows_kept"] == 0 or counts["review_rows_kept"] == 0:
        raise RuntimeError("Extraction incomplete: review or user file was not found in the archive.")

    sizes = {p.name: round(p.stat().st_size / 2**20, 2) for p in (RAW_BUSINESS_JSON, RAW_REVIEW_JSON, RAW_USER_JSON)}
    manifest = {
        "status": "complete",
        "source": archive or YELP_ZIP_URL,
        "user_source": "official_user_json",
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
        "review_sampling": f"systematic: every {sample_step}th line of review.json (cap {max_reviews:,})",
        "sample_step": sample_step,
        **counts,
        "file_sizes_mb": sizes,
        "total_raw_mb": round(sum(sizes.values()), 2),
        "elapsed_seconds": round(time.perf_counter() - t0, 1),
        "cached": False,
    }
    RAW_MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
