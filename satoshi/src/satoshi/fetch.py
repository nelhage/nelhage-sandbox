"""Download mailing-list archives from their canonical sources."""

from __future__ import annotations

import datetime as dt
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import httpx
from tqdm import tqdm

UA = "satoshi-archive-fetcher/0.1 (+https://github.com/nelhage/nelhage-sandbox)"

CRYPTOGRAPHY_INDEX = "https://www.metzdowd.com/pipermail/cryptography/"
CYPHERPUNKS_INDEX = "https://cypherpunks.venona.com/raw/"
HASHCASH_INDEX = "https://www.freelists.org/archive/hashcash/"

# Pipermail monthly archive filename pattern, e.g. "2008-October.txt.gz"
PIPERMAIL_MONTH_RE = re.compile(r'href="(\d{4}-[A-Z][a-z]+\.txt\.gz)"')
# Cypherpunks yearly filename pattern, e.g. "cyp-1993.txt"
CYPHERPUNKS_FILE_RE = re.compile(r'href="(cyp-\d{4}\.txt)"')
# FreeLists month-year index pages, e.g. "/archive/hashcash/03-2004"
HASHCASH_MONTH_RE = re.compile(r'href="(/archive/hashcash/\d{2}-\d{4})"')
# FreeLists post URLs, e.g. "/post/hashcash/Mutt-and-hashcash,3"
HASHCASH_POST_RE = re.compile(r'href="(/post/hashcash/[^"#?]+)"')


@dataclass
class Fetched:
    list_name: str
    url: str
    path: Path
    new: bool
    nbytes: int


def _client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": UA, "Accept": "*/*"},
        follow_redirects=True,
        timeout=httpx.Timeout(30.0, connect=15.0),
    )


def _log(conn: sqlite3.Connection, list_name: str, url: str, path: Path, nbytes: int) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO fetch_log (list_name, url, path, bytes, fetched_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (list_name, url, str(path), nbytes, dt.datetime.now(dt.timezone.utc).isoformat()),
    )
    conn.commit()


def _download(
    client: httpx.Client,
    url: str,
    dest: Path,
    *,
    force: bool = False,
) -> tuple[bool, int]:
    """Download ``url`` to ``dest``; skip if the file already exists.

    Returns (was_new, bytes_written).
    """
    if dest.exists() and not force and dest.stat().st_size > 0:
        return False, dest.stat().st_size
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with client.stream("GET", url) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1 << 15):
                f.write(chunk)
    tmp.replace(dest)
    return True, dest.stat().st_size


# -------- Cryptography (metzdowd pipermail) --------


def fetch_cryptography(
    data_root: Path,
    conn: sqlite3.Connection,
    *,
    force: bool = False,
) -> list[Fetched]:
    dest_dir = data_root / "cryptography"
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Fetched] = []
    with _client() as client:
        index = client.get(CRYPTOGRAPHY_INDEX)
        index.raise_for_status()
        months = sorted(set(PIPERMAIL_MONTH_RE.findall(index.text)))
        for fname in tqdm(months, desc="cryptography", unit="mo"):
            url = CRYPTOGRAPHY_INDEX + fname
            dest = dest_dir / fname
            try:
                new, nbytes = _download(client, url, dest, force=force)
            except httpx.HTTPError as exc:
                tqdm.write(f"  fetch failed {url}: {exc}")
                continue
            _log(conn, "cryptography", url, dest, nbytes)
            out.append(Fetched("cryptography", url, dest, new, nbytes))
    return out


# -------- Cypherpunks (venona raw) --------


def fetch_cypherpunks(
    data_root: Path,
    conn: sqlite3.Connection,
    *,
    force: bool = False,
) -> list[Fetched]:
    dest_dir = data_root / "cypherpunks"
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Fetched] = []
    with _client() as client:
        index = client.get(CYPHERPUNKS_INDEX)
        index.raise_for_status()
        files = sorted(set(CYPHERPUNKS_FILE_RE.findall(index.text)))
        for fname in tqdm(files, desc="cypherpunks", unit="yr"):
            url = CYPHERPUNKS_INDEX + fname
            dest = dest_dir / fname
            try:
                new, nbytes = _download(client, url, dest, force=force)
            except httpx.HTTPError as exc:
                tqdm.write(f"  fetch failed {url}: {exc}")
                continue
            _log(conn, "cypherpunks", url, dest, nbytes)
            out.append(Fetched("cypherpunks", url, dest, new, nbytes))
    return out


# -------- HashCash (scraped from FreeLists) --------


def _fetch_text(client: httpx.Client, url: str, dest: Path, *, force: bool = False) -> tuple[bool, str]:
    """Fetch a text resource; cache on disk. Returns (was_new, text)."""
    if dest.exists() and not force and dest.stat().st_size > 0:
        return False, dest.read_text(encoding="utf-8", errors="replace")
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = client.get(url)
    resp.raise_for_status()
    text = resp.text
    dest.write_text(text, encoding="utf-8")
    return True, text


def fetch_hashcash(
    data_root: Path,
    conn: sqlite3.Connection,
    *,
    force: bool = False,
    rate_limit_s: float = 0.3,
) -> list[Fetched]:
    """Scrape the FreeLists ``hashcash`` archive into per-post HTML files.

    Structure cached under ``data_root/hashcash/``::

        index.html
        month/MM-YYYY.html
        post/<slug>.html
    """
    base = "https://www.freelists.org"
    dest_dir = data_root / "hashcash"
    (dest_dir / "month").mkdir(parents=True, exist_ok=True)
    (dest_dir / "post").mkdir(parents=True, exist_ok=True)
    out: list[Fetched] = []

    with _client() as client:
        new, index_html = _fetch_text(client, HASHCASH_INDEX, dest_dir / "index.html", force=force)
        _log(conn, "hashcash", HASHCASH_INDEX, dest_dir / "index.html", len(index_html.encode()))
        if new:
            time.sleep(rate_limit_s)

        month_paths = sorted(set(HASHCASH_MONTH_RE.findall(index_html)))
        post_urls: set[str] = set()
        for mpath in tqdm(month_paths, desc="hashcash months", unit="mo"):
            mname = mpath.rsplit("/", 1)[-1]  # e.g. "03-2004"
            murl = base + mpath
            mdest = dest_dir / "month" / f"{mname}.html"
            try:
                m_new, mhtml = _fetch_text(client, murl, mdest, force=force)
            except httpx.HTTPError as exc:
                tqdm.write(f"  fetch failed {murl}: {exc}")
                continue
            _log(conn, "hashcash", murl, mdest, len(mhtml.encode()))
            if m_new:
                time.sleep(rate_limit_s)
            for rel in HASHCASH_POST_RE.findall(mhtml):
                post_urls.add(rel)

        post_list = sorted(post_urls)
        for rel in tqdm(post_list, desc="hashcash posts", unit="msg"):
            slug = rel.rsplit("/", 1)[-1].replace(",", "_")
            purl = base + rel
            pdest = dest_dir / "post" / f"{slug}.html"
            try:
                p_new, phtml = _fetch_text(client, purl, pdest, force=force)
            except httpx.HTTPError as exc:
                tqdm.write(f"  fetch failed {purl}: {exc}")
                continue
            _log(conn, "hashcash", purl, pdest, len(phtml.encode()))
            out.append(Fetched("hashcash", purl, pdest, p_new, len(phtml.encode())))
            if p_new:
                time.sleep(rate_limit_s)
    return out


def fetch_all(
    data_root: Path,
    conn: sqlite3.Connection,
    *,
    lists: Iterable[str] = ("cryptography", "cypherpunks", "hashcash"),
    force: bool = False,
) -> None:
    lists = list(lists)
    if "cryptography" in lists:
        fetch_cryptography(data_root, conn, force=force)
    if "cypherpunks" in lists:
        fetch_cypherpunks(data_root, conn, force=force)
    if "hashcash" in lists:
        fetch_hashcash(data_root, conn, force=force)
