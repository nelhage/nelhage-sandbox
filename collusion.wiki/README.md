# collusion.wiki → git

Reconstructs the [collusion.wiki](https://collusion.wiki/) wiki-farm log dump as
a git repository: one file per wiki page, one commit per revision.

## Fetch the dump

```sh
curl -sSLO https://collusion.wiki/explorer/download/full-wiki-logs.zip
unzip -q full-wiki-logs.zip -d dump/
```

The archive holds `revisions.jsonl`, `pages.jsonl`, `events.jsonl`,
`labels.jsonl`, a `manifest.json`, and `SHA256SUMS` (the files are plain JSONL,
not gzipped). It is the published write-date cut: 14,591 revisions over 4,579
pages across four wikis (`dse`, `probier`, `fractal`, `dorfwiki`), spanning
2026-05-24 to 2026-07-02.

## Build

```sh
python3 build_repo.py            # dump/ -> wiki-git/
python3 verify_repo.py           # re-check the result against the dump
```

`build_repo.py` streams the history into `git fast-import` (~20s).

## Repository layout

Each page becomes `<wiki>/<PageName>.txt`. Page names containing `/` nest as
directories, matching the wiki's own subpage convention; the `.txt` suffix on
every leaf means a page and its subpage directory can never collide. Any
character outside `[A-Za-z0-9._\[\]-]` is percent-encoded per component, and an
empty component — only `dse/StartSeite/` — becomes a bare `%`, which nothing
else can produce because a literal `%` encodes to `%25`.

Two pages differ only by case (`dse/StartSeite` and `dse/Startseite`), so the
worktree needs a case-sensitive filesystem.

## Commits

Commits are ordered by revision time, with same-second edits to one page broken
by `seq` so no page's history is replayed out of order. The subject is
`<wiki>/<page>: <change_summary>`, falling back to `create page` / `edit` for
the 972 revisions that were saved without a summary. Remaining revision
metadata follows as trailers: `Rev-Id`, `Seq`, `Label`, `IP16`, `RCS-Revision`,
`RCS-Path`, body size/lines/encoding/SHA-256, `Diff-Base`, `Hunks`, the several
clock fields (`Time-Grade`, `Winning-Clock`, `Uncertainty-Seconds`,
`Write-Date`, `Archived-At`, `Request-Time`, `Success-Time`),
`Request-Action`, and the recreation links (`Related-Event-Id`,
`Relation-Type`, `Round-Id`).

## Two things worth knowing about the data

**Bodies are latin-1-escaped bytes.** Each character of `body` in the JSON is
one byte of the original file, so `body.encode("latin-1")` recovers the exact
bytes. Only that round-trip reproduces `body_sha256`; re-encoding the string as
UTF-8 corrupts the 250 revisions whose `body_encoding` is `utf8`. The declared
`body_encoding` (14,340 ascii / 250 utf8 / 1 latin1) says how to *read* those
bytes and is recorded as a trailer rather than applied.

**A revision's `name` is the page, not the author.** The editor identity is
`label`, which is what the commits use; the 899 revisions with a blank label are
attributed to `(unlabelled)`. Emails are synthesized as
`<label>@collusion.wiki`.

## What is checked

`verify_repo.py` walks the history oldest-first and asserts, for every one of
the 14,591 revisions, that the commit touches the expected path, that the
blob's bytes hash to `body_sha256` and match `body_len`, and that the author and
committer identity and timestamp came from the metadata. It then confirms the
final tree is exactly the 4,579 pages at their last revision. 42 revisions
re-saved a page's existing text and so leave the tree unchanged; the checker
reads those bodies out of the commit instead of from a diff.

## Not represented

Only `revisions.jsonl` is replayed. The dump's 5,216 admin deletion events and
other request-log rows live in `events.jsonl`; a deleted page's file therefore
stays at its last stored revision rather than disappearing. Page-level rollups
(`pages.jsonl`) and actor rollups (`labels.jsonl`) are likewise not committed.
