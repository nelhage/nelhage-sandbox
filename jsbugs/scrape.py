#!/usr/bin/env python3
"""Scrape V8 security bugs from the Chrome Releases blog.

Source
------
Chrome publishes a security-advisory-per-release on the Chrome Releases
blog (https://chromereleases.googleblog.com). Every "Stable Channel Update
for Desktop" post lists the CVEs patched in that release, in a consistent
line format::

    [$reward][bug_id] Severity CVE-YYYY-NNNN: Short description. Reported by X on YYYY-MM-DD

The blog is served by Blogger, which exposes a paginated JSON feed at
``/feeds/posts/default``. We walk that feed filtered to the
"Stable updates" category, stop once we reach posts older than the cutoff,
and extract every CVE line that mentions V8.

Outputs (in this directory):
    v8_bugs.jsonl    one JSON object per CVE
    v8_bugs.csv      same data flattened to CSV
    posts.jsonl      one JSON object per source release post we scanned
"""

from __future__ import annotations

import csv
import html
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

FEED_URL = "https://chromereleases.googleblog.com/feeds/posts/default"
CATEGORY = "Stable updates"
PAGE_SIZE = 50
USER_AGENT = "jsbugs-scraper/1.0 (+https://github.com/nelhage/nelhage-sandbox)"

OUT_DIR = Path(__file__).resolve().parent
JSONL_PATH = OUT_DIR / "v8_bugs.jsonl"
CSV_PATH = OUT_DIR / "v8_bugs.csv"
POSTS_PATH = OUT_DIR / "posts.jsonl"

# Stop once we hit posts older than this. "Last 5 years" from today.
CUTOFF = datetime.now(timezone.utc) - timedelta(days=5 * 365 + 1)

# Match the CVE line format used in Chrome security advisories.
# Example:  [$3000][486927780] High CVE-2026-5861: Use after free in V8. Reported by 5shain on 2026-02-23
CVE_LINE_RE = re.compile(
    r"""
    \[\s*(?P<reward>[^\]]*)\s*\]         # [$3000] or [N/A] or [TBD]
    \s*
    \[\s*(?P<bug>[^\]]*)\s*\]            # [486927780]
    \s*
    (?P<severity>Critical|High|Medium|Low)\s+
    (?P<cve>CVE-\d{4}-\d{4,7})\s*:\s*
    (?P<desc>.*?)
    (?:\s*\.\s*Reported\s+by\s+(?P<reporter>.*?)(?:\s+on\s+(?P<report_date>\d{4}-\d{2}-\d{2}))?)?
    \s*$
    """,
    re.VERBOSE | re.IGNORECASE,
)

V8_RE = re.compile(r"\bV8\b")
TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class Bug:
    cve: str
    severity: str
    reward: str
    bug_tracker_ids: list[str]         # some CVEs list multiple IDs
    bug_tracker_urls: list[str]
    description: str
    reporter: str
    report_date: str
    release_post_url: str
    release_post_title: str
    release_published: str
    raw_line: str


@dataclass
class Post:
    url: str
    title: str
    published: str
    v8_bug_count: int = 0
    total_cve_count: int = 0
    labels: list[str] = field(default_factory=list)


def http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"  retry {attempt + 1} after error: {exc!r} (sleep {wait}s)", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch {url}")


def iter_feed() -> Iterator[dict]:
    start = 1
    while True:
        qs = urllib.parse.urlencode(
            {
                "alt": "json",
                "max-results": PAGE_SIZE,
                "start-index": start,
                "category": CATEGORY,
            }
        )
        url = f"{FEED_URL}?{qs}"
        print(f"GET {url}", file=sys.stderr)
        data = json.loads(http_get(url))
        entries = data.get("feed", {}).get("entry", []) or []
        if not entries:
            return
        for entry in entries:
            yield entry
        if len(entries) < PAGE_SIZE:
            return
        start += PAGE_SIZE
        time.sleep(0.5)


def entry_post_url(entry: dict) -> str:
    for link in entry.get("link", []):
        if link.get("rel") == "alternate" and link.get("type") == "text/html":
            return link.get("href", "")
    return ""


def entry_labels(entry: dict) -> list[str]:
    return [c.get("term", "") for c in entry.get("category", []) if c.get("term")]


def strip_html(s: str) -> str:
    # Convert <br>/<p>/<li> to newlines so our line-based parsing works.
    s = re.sub(r"(?i)<\s*br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</\s*(p|li|div|h[1-6])\s*>", "\n", s)
    s = TAG_RE.sub("", s)
    return html.unescape(s)


def parse_post(entry: dict) -> tuple[Post, list[Bug]]:
    title = entry.get("title", {}).get("$t", "")
    published = entry.get("published", {}).get("$t", "")
    url = entry_post_url(entry)
    labels = entry_labels(entry)
    raw_html = entry.get("content", {}).get("$t", "")
    text = strip_html(raw_html)

    post = Post(url=url, title=title, published=published, labels=labels)
    bugs: list[Bug] = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line or not line.startswith("["):
            continue
        m = CVE_LINE_RE.match(line)
        if not m:
            continue
        post.total_cve_count += 1
        desc = m.group("desc") or ""
        if not V8_RE.search(desc):
            continue
        # The bug field sometimes contains several IDs, e.g. "327740539, 40072287".
        bug_ids = [b.strip() for b in re.split(r"[,\s]+", m.group("bug") or "") if b.strip().isdigit()]
        bug_urls = [f"https://issues.chromium.org/issues/{b}" for b in bug_ids]
        bugs.append(
            Bug(
                cve=m.group("cve").upper(),
                severity=m.group("severity").title(),
                reward=(m.group("reward") or "").strip(),
                bug_tracker_ids=bug_ids,
                bug_tracker_urls=bug_urls,
                description=desc.strip().rstrip("."),
                reporter=(m.group("reporter") or "").strip(),
                report_date=(m.group("report_date") or "").strip(),
                release_post_url=url,
                release_post_title=title,
                release_published=published,
                raw_line=line,
            )
        )
    post.v8_bug_count = len(bugs)
    return post, bugs


def main() -> int:
    all_bugs: list[Bug] = []
    all_posts: list[Post] = []
    cutoff_reached = False
    scanned = 0

    for entry in iter_feed():
        published_s = entry.get("published", {}).get("$t", "")
        try:
            published_dt = datetime.fromisoformat(published_s.replace("Z", "+00:00"))
        except ValueError:
            published_dt = None

        if published_dt is not None and published_dt < CUTOFF:
            cutoff_reached = True
            break

        post, bugs = parse_post(entry)
        all_posts.append(post)
        all_bugs.extend(bugs)
        scanned += 1
        if bugs:
            print(
                f"  {published_s[:10]}  {post.title[:60]:60}  +{len(bugs)} V8 ({post.total_cve_count} CVEs total)",
                file=sys.stderr,
            )

    # De-duplicate by CVE (a CVE sometimes reappears in a follow-up post).
    seen: dict[str, Bug] = {}
    for b in all_bugs:
        prev = seen.get(b.cve)
        if prev is None or b.release_published > prev.release_published:
            seen[b.cve] = b
    unique_bugs = sorted(seen.values(), key=lambda b: b.release_published, reverse=True)

    with JSONL_PATH.open("w") as f:
        for b in unique_bugs:
            f.write(json.dumps(asdict(b)) + "\n")

    with CSV_PATH.open("w", newline="") as f:
        fieldnames = [
            "cve", "severity", "reward", "bug_tracker_ids", "bug_tracker_urls",
            "description", "reporter", "report_date", "release_post_url",
            "release_post_title", "release_published", "raw_line",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for b in unique_bugs:
            row = asdict(b)
            row["bug_tracker_ids"] = ",".join(row["bug_tracker_ids"])
            row["bug_tracker_urls"] = " ".join(row["bug_tracker_urls"])
            w.writerow(row)

    with POSTS_PATH.open("w") as f:
        for p in all_posts:
            f.write(json.dumps(asdict(p)) + "\n")

    print(
        f"\nScanned {scanned} stable-update posts"
        f" ({'hit 5-year cutoff' if cutoff_reached else 'exhausted feed'}).",
        file=sys.stderr,
    )
    print(f"Collected {len(all_bugs)} V8 CVE mentions, {len(unique_bugs)} unique.", file=sys.stderr)
    print(f"Wrote: {JSONL_PATH.name}, {CSV_PATH.name}, {POSTS_PATH.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
