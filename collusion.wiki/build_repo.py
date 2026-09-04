#!/usr/bin/env python3
"""Reconstruct the collusion.wiki dump as a git repository.

One file per wiki page, one commit per entry in ``revisions.jsonl``, replayed in
chronological order via ``git fast-import``.

Notes on the source data:

* Revision bodies arrive in JSON as latin-1-escaped raw bytes: each character of
  ``body`` is one byte of the original file. ``body.encode("latin-1")`` recovers
  the exact bytes, and only that round-trip reproduces ``body_sha256`` (the 250
  ``body_encoding == "utf8"`` revisions fail if you re-encode the string as
  UTF-8). ``body_encoding`` describes how those bytes should be *read*, and is
  recorded in the commit message rather than applied.
* A revision's ``name`` is the page name, not an actor. The editor identity is
  ``label``; 899 revisions have a blank label and are attributed to
  ``UNLABELLED_NAME`` below.
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
import subprocess
import sys
import time
from pathlib import Path

EMAIL_DOMAIN = "collusion.wiki"
UNLABELLED_NAME = "(unlabelled)"
UNLABELLED_SLUG = "unlabelled"

# Characters kept verbatim in a path component. Page names in the dump draw from
# alphanumerics plus "-_[]/", where "/" separates subpages.
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9._\[\]-]")


def parse_iso(ts: str) -> int:
    """Parse a trailing-Z ISO 8601 timestamp into a Unix time."""
    return calendar.timegm(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))


def encode_component(component: str) -> str:
    """Percent-encode one path component so the mapping stays injective.

    An empty component (only ``StartSeite/`` in the dump) becomes a bare ``%``,
    which no other name can produce because a literal ``%`` encodes to ``%25``.
    """
    if component == "":
        return "%"
    if component in (".", ".."):
        return component.replace(".", "%2E")
    out = []
    for ch in component:
        if _SAFE_COMPONENT.fullmatch(ch):
            out.append(ch)
        else:
            out.extend(f"%{b:02X}" for b in ch.encode("utf-8"))
    return "".join(out)


def page_path(wiki: str, name: str) -> str:
    """Repository path for a page. Subpages nest; the ``.txt`` suffix on every
    leaf keeps a page and its subpage directory from ever colliding."""
    parts = [encode_component(p) for p in name.split("/")]
    return f"{encode_component(wiki)}/{'/'.join(parts)}.txt"


def identity(label: str) -> tuple[str, str]:
    """Git author name and email for a revision label."""
    if not label:
        return UNLABELLED_NAME, f"{UNLABELLED_SLUG}@{EMAIL_DOMAIN}"
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-.") or UNLABELLED_SLUG
    return label, f"{slug}@{EMAIL_DOMAIN}"


def summarize_hunks(hunks: list[dict] | None) -> str | None:
    if not hunks:
        return None
    counts: dict[str, int] = {}
    for hunk in hunks:
        counts[hunk["op"]] = counts.get(hunk["op"], 0) + 1
    return ", ".join(f"{counts[op]} {op}" for op in sorted(counts))


def subject(rev: dict) -> str:
    """One-line commit subject: the editor's own summary when they left one."""
    page = f"{rev['wiki']}/{rev['name']}"
    summary = (rev.get("change_summary") or "").strip()
    if not summary:
        summary = "create page" if rev.get("diff_base_reason") == "page_created" else "edit"
    summary = " ".join(summary.split())
    if len(summary) > 100:
        summary = summary[:99] + "…"
    return f"{page}: {summary}"


def commit_message(rev: dict) -> str:
    """Subject line plus the revision's metadata as trailers."""
    trailers: list[tuple[str, object]] = [
        ("Rev-Id", rev["rev_id"]),
        ("Page-Id", rev["page_id"]),
        ("Wiki", rev["wiki"]),
        ("Page-Name", rev["name"]),
        ("Seq", rev["seq"]),
        ("Label", rev["label"] or UNLABELLED_NAME),
        ("IP16", rev.get("ip16")),
        ("RCS-Revision", rev.get("rcs_rev")),
        ("RCS-Path", rev.get("rcs_path")),
        ("Body-Bytes", rev.get("body_len")),
        ("Body-Lines", rev.get("lines")),
        ("Body-Encoding", rev.get("body_encoding")),
        ("Body-SHA256", rev.get("body_sha256")),
        ("Diff-Base", rev.get("diff_base")),
        ("Diff-Base-Reason", rev.get("diff_base_reason")),
        ("Hunks", summarize_hunks(rev.get("hunks"))),
        ("Time", rev.get("time")),
        ("Time-Grade", rev.get("time_grade")),
        ("Winning-Clock", rev.get("winning_clock")),
        ("Uncertainty-Seconds", rev.get("uncertainty_seconds")),
        ("Write-Date", rev.get("write_date")),
        ("Archived-At", rev.get("archived_at")),
        ("Request-Time", rev.get("request_time")),
        ("Success-Time", rev.get("success_time")),
        ("Recent-Changes-Time", rev.get("recent_changes_time")),
        ("Request-Action", rev.get("request_action")),
        ("Change-Summary", rev.get("change_summary")),
        ("Related-Event-Id", rev.get("related_event_id")),
        ("Relation-Type", rev.get("relation_type")),
        ("Round-Id", ", ".join(r for r in (rev.get("round_id") or []) if r) or None),
    ]
    lines = [subject(rev), ""]
    lines += [f"{k}: {v}" for k, v in trailers if v not in (None, "", [])]
    return "\n".join(lines) + "\n"


def quote_path(path: str) -> str:
    """fast-import path literal (always quoted, so no character needs care)."""
    escaped = path.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def load_revisions(dump: Path) -> list[dict]:
    with (dump / "revisions.jsonl").open(encoding="utf-8") as fh:
        revs = [json.loads(line) for line in fh if line.strip()]
    # Chronological, with same-second edits to one page broken by sequence
    # number so a page's own history is never replayed out of order.
    revs.sort(key=lambda r: (parse_iso(r["time"]), r["wiki"], r["name"], int(r["seq"])))
    return revs


def emit(out, revs: list[dict], branch: str) -> None:
    write = out.write

    def data(payload: bytes) -> None:
        write(b"data %d\n" % len(payload))
        write(payload)
        write(b"\n")

    write(b"feature done\n")
    for mark, rev in enumerate(revs, start=1):
        name, email = identity(rev["label"])
        when = f"{parse_iso(rev['time'])} +0000".encode()
        ident = b"%s <%s> %s" % (name.encode("utf-8"), email.encode("utf-8"), when)
        write(b"commit refs/heads/%s\n" % branch.encode())
        write(b"mark :%d\n" % mark)
        write(b"author %s\n" % ident)
        write(b"committer %s\n" % ident)
        data(commit_message(rev).encode("utf-8"))
        write(b"M 644 inline %s\n" % quote_path(page_path(rev["wiki"], rev["name"])).encode("utf-8"))
        data(rev["body"].encode("latin-1"))
    write(b"done\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dump", type=Path, default=Path("dump"), help="unpacked dump directory")
    ap.add_argument("--out", type=Path, default=Path("wiki-git"), help="repository to create")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--stream", type=Path, help="also save the fast-import stream here")
    args = ap.parse_args()

    if args.out.exists():
        print(f"{args.out} already exists; remove it first", file=sys.stderr)
        return 1

    revs = load_revisions(args.dump)
    print(f"{len(revs)} revisions across {len({(r['wiki'], r['name']) for r in revs})} pages")

    args.out.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", args.branch, str(args.out)], check=True)
    proc = subprocess.Popen(
        ["git", "-C", str(args.out), "fast-import", "--date-format=raw", "--quiet"],
        stdin=subprocess.PIPE,
    )
    stream = args.stream.open("wb") if args.stream else None
    try:
        out = proc.stdin
        if stream is not None:
            class Tee:
                def write(self, b):
                    out.write(b)
                    stream.write(b)
            emit(Tee(), revs, args.branch)
        else:
            emit(out, revs, args.branch)
        out.close()
    finally:
        if stream is not None:
            stream.close()
    if proc.wait() != 0:
        return proc.returncode

    subprocess.run(["git", "-C", str(args.out), "reset", "-q", "--hard"], check=True)
    subprocess.run(["git", "-C", str(args.out), "gc", "-q", "--aggressive", "--prune=now"], check=True)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
