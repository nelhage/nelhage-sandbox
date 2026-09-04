#!/usr/bin/env python3
"""Check the reconstructed repository against the dump.

Walks the history oldest-first and asserts, for every revision, that the commit
touches the expected page file, that the blob's bytes hash to ``body_sha256``,
and that the author identity and timestamp came from the revision metadata.
Then checks the final tree is exactly the set of pages at their last revision.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from build_repo import identity, load_revisions, page_path, parse_iso

SEP = "\x01"


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


def blob_bytes(repo: Path, refs: list[str]) -> dict[str, bytes]:
    """Read many blobs in one `git cat-file --batch` pass, keyed by request."""
    proc = subprocess.run(
        ["git", "-C", str(repo), "cat-file", "--batch"],
        input=("\n".join(refs) + "\n").encode(),
        capture_output=True,
        check=True,
    )
    out, pos, blobs = proc.stdout, 0, {}
    for ref in refs:
        eol = out.index(b"\n", pos)
        header = out[pos:eol].split()
        if len(header) != 3 or header[1] != b"blob":
            raise SystemExit(f"cat-file: unexpected header for {ref}: {out[pos:eol]!r}")
        size = int(header[2])
        pos = eol + 1
        blobs[ref] = out[pos : pos + size]
        pos += size + 1
    return blobs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", type=Path, default=Path("dump"))
    ap.add_argument("--repo", type=Path, default=Path("wiki-git"))
    args = ap.parse_args()

    revs = load_revisions(args.dump)
    raw = git(
        args.repo,
        "log",
        "--reverse",
        f"--format=C{SEP}%H{SEP}%at{SEP}%an{SEP}%ae{SEP}%cn{SEP}%ce{SEP}%ct",
        "--raw",
        "--no-abbrev",
    )

    commits, failures = [], []
    for line in raw.splitlines():
        if line.startswith(f"C{SEP}"):
            commits.append({"meta": line.split(SEP)[1:], "changes": []})
        elif line.startswith(":"):
            info, path = line.split("\t", 1)
            fields = info.split()
            commits[-1]["changes"].append((fields[3], path))  # post-image oid, path

    def fail(msg: str) -> None:
        failures.append(msg)

    if len(commits) != len(revs):
        fail(f"commit count {len(commits)} != revision count {len(revs)}")

    # A revision that re-saved a page's existing text leaves the tree unchanged
    # and so shows no --raw line; read its file out of the commit instead.
    refs, unchanged = [], 0
    for commit, rev in zip(commits, revs):
        if commit["changes"]:
            refs.append(commit["changes"][0][0])
        else:
            unchanged += 1
            refs.append(f"{commit['meta'][0]}:{page_path(rev['wiki'], rev['name'])}")
    blobs = blob_bytes(args.repo, sorted(set(refs)))

    for commit, rev, ref in zip(commits, revs, refs):
        sha, at, an, ae, cn, ce, ct = commit["meta"]
        rid = rev["rev_id"]
        want_path = page_path(rev["wiki"], rev["name"])
        if len(commit["changes"]) > 1:
            fail(f"{rid}: {len(commit['changes'])} paths changed in {sha}")
            continue
        if commit["changes"] and commit["changes"][0][1] != want_path:
            fail(f"{rid}: path {commit['changes'][0][1]!r} != {want_path!r}")
        body = blobs[ref]
        got = hashlib.sha256(body).hexdigest()
        if got != rev["body_sha256"]:
            fail(f"{rid}: blob sha256 {got} != {rev['body_sha256']}")
        if len(body) != int(rev["body_len"]):
            fail(f"{rid}: {len(body)} bytes != body_len {rev['body_len']}")
        want_name, want_email = identity(rev["label"])
        if (an, ae, cn, ce) != (want_name, want_email, want_name, want_email):
            fail(f"{rid}: identity {an} <{ae}> / {cn} <{ce}> != {want_name} <{want_email}>")
        want_time = parse_iso(rev["time"])
        if (int(at), int(ct)) != (want_time, want_time):
            fail(f"{rid}: time {at}/{ct} != {want_time}")

    # Final tree: one file per page, holding that page's last revision.
    head = {}
    for rev in revs:
        head[(rev["wiki"], rev["name"])] = rev
    tracked = set(git(args.repo, "ls-files").splitlines())
    expected = {page_path(w, n): r for (w, n), r in head.items()}
    if tracked != set(expected):
        fail(f"tree mismatch: {len(tracked - set(expected))} extra, {len(set(expected) - tracked)} missing")
    for path, rev in expected.items():
        data = (args.repo / path).read_bytes()
        if hashlib.sha256(data).hexdigest() != rev["body_sha256"]:
            fail(f"{path}: worktree content != last revision {rev['rev_id']}")

    print(f"checked {len(commits)} commits ({unchanged} re-saved identical text), {len(expected)} page files")
    for msg in failures[:20]:
        print("FAIL:", msg, file=sys.stderr)
    if failures:
        print(f"{len(failures)} failures", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
