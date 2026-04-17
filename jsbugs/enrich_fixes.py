#!/usr/bin/env python3
"""Enrich v8_bugs.jsonl with source-code fix links via Chromium Gerrit.

The Chrome Releases advisory only gives us the issue-tracker ID for each
CVE. To get the actual fix commit, we query Gerrit:

    GET https://chromium-review.googlesource.com/changes/?q=bug:<id>

Gerrit matches the ``Bug:`` footer on every CL. We keep MERGED CLs from
the V8 and chromium/src projects (Blink-V8 bindings live there) and skip
mechanical rolls in chromium/deps/*. For each CL we record the CL number,
subject, the merged commit SHA, and a git-on-the-web URL pointing to the
commit so a reader can click through to the diff.

Outputs:
    v8_bugs_with_fixes.jsonl   original rows + "fix_commits" field
    v8_bugs_with_fixes.csv     CSV with fix commits joined as a pipe list
    fixes_unresolved.txt       bugs for which Gerrit returned nothing
"""

from __future__ import annotations

import csv
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
IN_PATH = OUT_DIR / "v8_bugs.jsonl"
JSONL_OUT = OUT_DIR / "v8_bugs_with_fixes.jsonl"
CSV_OUT = OUT_DIR / "v8_bugs_with_fixes.csv"
MISSING_OUT = OUT_DIR / "fixes_unresolved.txt"

GERRIT = "https://chromium-review.googlesource.com"
USER_AGENT = "jsbugs-scraper/1.0 (+https://github.com/nelhage/nelhage-sandbox)"

# Map Gerrit project -> gitiles base for commit URLs.
GITILES = {
    "v8/v8": "https://chromium.googlesource.com/v8/v8/+/",
    "chromium/src": "https://chromium.googlesource.com/chromium/src/+/",
}


def gerrit_get(path: str) -> list | dict:
    url = f"{GERRIT}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read().decode()
            break
        except Exception as exc:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"  retry {attempt + 1} after {exc!r} (sleep {wait}s)", file=sys.stderr)
            time.sleep(wait)
    else:
        raise RuntimeError(f"failed to fetch {url}")
    # Gerrit prepends )]}'\n to JSON responses as XSSI protection.
    if body.startswith(")]}'"):
        body = body.split("\n", 1)[1]
    return json.loads(body)


def _search(query: str) -> list[dict]:
    q = urllib.parse.quote(query)
    return gerrit_get(f"/changes/?q={q}&o=CURRENT_REVISION&o=CURRENT_COMMIT")


def _format_cls(data: list[dict], matched_via: str) -> list[dict]:
    out = []
    for cl in data:
        project = cl.get("project", "")
        if project not in GITILES:
            continue  # skip chromium/deps/*, infra repos, etc.
        sha = cl.get("current_revision", "")
        if not sha:
            continue
        subject = cl.get("subject", "")
        out.append(
            {
                "project": project,
                "branch": cl.get("branch", ""),
                "cl_number": cl.get("_number"),
                "cl_url": f"{GERRIT}/c/{project}/+/{cl.get('_number')}",
                "commit_sha": sha,
                "commit_url": f"{GITILES[project]}{sha}",
                "subject": subject,
                "submitted": cl.get("submitted", ""),
                "is_revert": subject.lower().startswith("revert"),
                "matched_via": matched_via,
            }
        )
    return out


def fixes_for_bugs(bug_ids: list[str]) -> list[dict]:
    """Look up fix CLs for one CVE, which may list multiple bug IDs.

    Strategy per ID:
      1. ``bug:<id>`` — matches the structured ``Bug:`` footer. Clean signal.
      2. If (1) yields nothing, fall back to ``message:<id>`` — finds CLs
         that mention the bug ID anywhere in the commit message. Older
         CLs sometimes predate the Bug: footer convention.
    """
    seen: dict[int, dict] = {}
    for bid in bug_ids:
        if not bid.isdigit():
            continue
        primary = _format_cls(_search(f"bug:{bid} status:merged"), matched_via=f"bug:{bid}")
        if not primary:
            primary = _format_cls(_search(f"message:{bid} status:merged"), matched_via=f"message:{bid}")
        for cl in primary:
            # dedupe across IDs
            seen.setdefault(cl["cl_number"], cl)
    out = list(seen.values())
    out.sort(key=lambda c: c["submitted"])
    return out


def main() -> int:
    rows = [json.loads(l) for l in IN_PATH.open()]
    enriched = []
    unresolved = []
    for i, row in enumerate(rows, 1):
        # Handle both the new (list) and legacy (single string) formats.
        if "bug_tracker_ids" in row:
            bugs = row["bug_tracker_ids"]
        else:
            raw = row.get("bug_tracker_id", "")
            bugs = [b.strip() for b in raw.replace(",", " ").split() if b.strip().isdigit()]
        try:
            fixes = fixes_for_bugs(bugs)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(rows)}] {row['cve']} bugs={bugs} ERROR {exc!r}", file=sys.stderr)
            fixes = []
        row["fix_commits"] = fixes
        if not fixes:
            unresolved.append(f"{row['cve']}\t{','.join(bugs)}\t{row['description']}")
        print(
            f"[{i}/{len(rows)}] {row['cve']:17} bugs={','.join(bugs):20}  {len(fixes)} fix(es)",
            file=sys.stderr,
        )
        enriched.append(row)
        time.sleep(0.3)

    with JSONL_OUT.open("w") as f:
        for r in enriched:
            f.write(json.dumps(r) + "\n")

    fieldnames = [
        "cve", "severity", "reward", "bug_tracker_ids", "bug_tracker_urls",
        "description", "reporter", "report_date", "release_published",
        "release_post_url", "fix_commit_urls", "fix_cl_urls", "fix_subjects",
    ]
    with CSV_OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in enriched:
            fixes = r.get("fix_commits", [])
            bugs = r.get("bug_tracker_ids") or [r.get("bug_tracker_id", "")]
            urls = r.get("bug_tracker_urls") or [r.get("bug_tracker_url", "")]
            w.writerow({
                "cve": r["cve"],
                "severity": r["severity"],
                "reward": r["reward"],
                "bug_tracker_ids": ",".join(bugs),
                "bug_tracker_urls": " ".join(urls),
                "description": r["description"],
                "reporter": r["reporter"],
                "report_date": r["report_date"],
                "release_published": r["release_published"],
                "release_post_url": r["release_post_url"],
                "fix_commit_urls": " | ".join(c["commit_url"] for c in fixes),
                "fix_cl_urls": " | ".join(c["cl_url"] for c in fixes),
                "fix_subjects": " | ".join(c["subject"] for c in fixes),
            })

    MISSING_OUT.write_text("\n".join(unresolved) + ("\n" if unresolved else ""))

    total = len(enriched)
    with_fix = sum(1 for r in enriched if r["fix_commits"])
    print(
        f"\nResolved fix commits for {with_fix}/{total} CVEs"
        f" ({len(unresolved)} unresolved).",
        file=sys.stderr,
    )
    print(f"Wrote: {JSONL_OUT.name}, {CSV_OUT.name}, {MISSING_OUT.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
