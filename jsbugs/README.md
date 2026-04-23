# jsbugs

A corpus of Chrome V8 security bugs from the last 5 years, scraped from
public sources.

Generated outputs land in `data/` (gitignored). Run the scripts in order;
each one reads the previous one's output.

## Scripts

### `scrape.py`
Walks the Chrome Releases blog's Blogger JSON feed
(`chromereleases.googleblog.com/feeds/posts/default`), filtered to the
"Stable updates" category, and extracts every `[reward][bug] Severity CVE: desc`
line that mentions V8. Stops once posts are older than 5 years.

Outputs:
- `data/v8_bugs.jsonl` — one JSON object per CVE
- `data/v8_bugs.csv` — same data, flattened
- `data/posts.jsonl` — metadata for every advisory post scanned

### `enrich_fixes.py`
For each bug, queries the Chromium Gerrit REST API
(`chromium-review.googlesource.com/changes/?q=bug:<id>`) to find merged
fix CLs. Falls back to `message:<id>` for older CLs that predate the
`Bug:` footer convention. Keeps CLs in `v8/v8` and `chromium/src`; skips
mechanical rolls in `chromium/deps/*`.

Inputs: `data/v8_bugs.jsonl`
Outputs:
- `data/v8_bugs_with_fixes.jsonl` — original rows + `fix_commits[]`
  (CL URL, commit URL, SHA, subject, `is_revert`, `matched_via`)
- `data/v8_bugs_with_fixes.csv` — CSV with fix commits joined as a pipe list
- `data/fixes_unresolved.txt` — bugs with no public Gerrit CLs

### `fetch_sample.py`
Picks 10 random CVEs that have a non-revert fix in `v8/v8` and downloads
each commit as a raw patch from gitiles
(`chromium.googlesource.com/.../+/<sha>^!/?format=TEXT`). Useful for
manual bug-pattern review.

Inputs: `data/v8_bugs_with_fixes.jsonl`
Outputs:
- `data/sample_patches/NN_CVE_clXXX.patch` — 10 diffs
- `data/sample_patches/index.json` — metadata per patch

## Usage

```
python3 scrape.py          # ~1 min
python3 enrich_fixes.py    # ~2 min (one Gerrit query per bug)
python3 fetch_sample.py    # ~30 s
```

No third-party dependencies; stdlib only.
