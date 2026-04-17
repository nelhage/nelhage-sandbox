#!/usr/bin/env python3
"""Fetch 10 random V8 fix patches for manual analysis."""
from __future__ import annotations
import base64, json, random, sys, time, urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "sample_patches"
OUT.mkdir(exist_ok=True)
GERRIT = "https://chromium-review.googlesource.com"
UA = "jsbugs/1.0"

def gerrit_text(path: str) -> str:
    req = urllib.request.Request(f"{GERRIT}{path}", headers={"User-Agent": UA})
    body = urllib.request.urlopen(req, timeout=30).read().decode()
    if body.startswith(")]}'"):
        body = body.split("\n", 1)[1]
    return body

def fetch_patch(fix: dict) -> str:
    """Fetch commit+diff from gitiles in mbox format (base64 text).

    URL pattern: https://<host>/<project>/+/<sha>^!/?format=TEXT
    Gerrit's /revisions/current/patch endpoint tends to 503 under load; the
    gitiles mirror is far more stable.
    """
    url = f"{fix['commit_url']}%5E%21/?format=TEXT"  # ^! URL-escaped
    last = None
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            body = urllib.request.urlopen(req, timeout=60).read().decode()
            return base64.b64decode(body).decode("utf-8", errors="replace")
        except Exception as e:
            last = e
            wait = 2 ** attempt
            print(f"  retry {attempt+1}: {e} (sleep {wait}s)", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"gave up on {url}: {last}")

def main():
    rows = [json.loads(l) for l in open(Path(__file__).resolve().parent / "v8_bugs_with_fixes.jsonl")]
    # Keep CVEs that have at least one non-revert fix CL in v8/v8
    candidates = []
    for r in rows:
        primary = [c for c in r["fix_commits"] if not c["is_revert"] and c["project"] == "v8/v8"]
        if primary:
            # Use the earliest non-revert v8/v8 CL as "the fix"
            r["_primary"] = primary[0]
            candidates.append(r)
    random.seed(42)
    picks = random.sample(candidates, 10)

    index = []
    for i, r in enumerate(picks, 1):
        cl = r["_primary"]["cl_number"]
        path = OUT / f"{i:02d}_{r['cve']}_cl{cl}.patch"
        print(f"[{i}/10] {r['cve']} CL {cl} -> {path.name}", file=sys.stderr)
        try:
            patch = fetch_patch(r["_primary"])
        except Exception as e:
            print(f"  failed: {e}", file=sys.stderr)
            continue
        path.write_text(patch)
        index.append({
            "n": i,
            "cve": r["cve"],
            "severity": r["severity"],
            "description": r["description"],
            "bug_ids": r["bug_tracker_ids"],
            "release_published": r["release_published"],
            "fix": r["_primary"],
            "patch_file": path.name,
            "patch_bytes": len(patch),
        })
        time.sleep(0.3)
    (OUT / "index.json").write_text(json.dumps(index, indent=2))
    print(f"\nWrote {len(index)} patches to {OUT}/", file=sys.stderr)

if __name__ == "__main__":
    main()
