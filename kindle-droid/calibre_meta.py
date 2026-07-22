#!/usr/bin/env python3
"""Push Kindle-library metadata into a Calibre library, matched by ASIN.

Series comes from the app's `Groups`/`GroupItems` tables (Amazon's own series
grouping).  Purchase dates come from the `GetContentOwnershipData` JSON that
Manage Your Content and Devices fetches for its content list -- the device DB
has no acquisition date at all (`KindleContent.DELIVERY_DATE` is only when the
emulator downloaded the file).  To capture it: open

    https://www.amazon.com/hz/mycd/digital-console/contentlist/booksAll/dateDsc/

with DevTools' Network tab open, page through the whole list, and save the
ajax response(s) as JSON.

    ./calibre_meta.py -n series
    ./calibre_meta.py series
    ./calibre_meta.py -n purchased ContentOwnershipData.json
    ./calibre_meta.py purchased ContentOwnershipData.json
"""
import argparse
import collections
import datetime
import json
import os
import sqlite3
import subprocess

LIBRARY_DB = "files-4k/kindle_library.db"
CALIBRE_LIB = os.path.expanduser("~/Calibre Library")


def calibre_asin_map():
    """asin -> [calibre book id].  Books are tagged with mobi-asin/asin ids."""
    db = os.path.join(CALIBRE_LIB, "metadata.db")
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = collections.defaultdict(list)
    for book, val in con.execute(
        "SELECT book, val FROM identifiers WHERE type IN ('mobi-asin', 'asin')"
    ):
        out[val.strip().upper()].append(book)
    return out


def calibredb(args, dry_run):
    cmd = ["calibredb", "--library-path", CALIBRE_LIB] + args
    if dry_run:
        print("  would run:", " ".join(cmd))
        return
    subprocess.run(cmd, check=True, capture_output=True)


def kindle_series():
    """asin -> (series_title, series_index)."""
    con = sqlite3.connect(f"file:{LIBRARY_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    out = {}
    for r in con.execute("""
            SELECT gi.ITEM_ID, g.GROUP_TITLE, gi.GROUP_POSITION,
                   gi.GROUP_POSITION_LABEL
              FROM GroupItems gi JOIN Groups g ON g.GROUP_ID = gi.GROUP_ID"""):
        asin = r["ITEM_ID"].split("/")[1]
        try:
            idx = float(r["GROUP_POSITION_LABEL"])
        except (TypeError, ValueError):
            idx = float(r["GROUP_POSITION"] or 0) + 1
        out[asin.upper()] = (r["GROUP_TITLE"], idx)
    return out


def cmd_series(args):
    series = kindle_series()
    books = calibre_asin_map()
    n = miss = 0
    for asin, (title, idx) in sorted(series.items()):
        ids = books.get(asin)
        if not ids:
            miss += 1
            continue
        for bid in ids:
            print(f"{asin} #{bid}: {title} [{idx:g}]")
            calibredb(["set_metadata", str(bid),
                       "--field", f"series:{title}",
                       "--field", f"series_index:{idx:g}"], args.dry_run)
            n += 1
    print(f"\n{n} books updated, {miss} series members not in Calibre")


def mycd_items(paths):
    """asin -> acquiredTime (epoch ms), merged across one or more MYCD pages."""
    out = {}
    for path in paths:
        blob = json.load(open(path))
        # Unwrap either the whole ajax envelope or a bare item list.
        if isinstance(blob, dict):
            blob = blob.get("GetContentOwnershipData", blob)
            blob = blob.get("items", blob)
        for item in blob:
            if item.get("asin") and item.get("acquiredTime"):
                out[item["asin"].upper()] = item["acquiredTime"]
    return out


def cmd_purchased(args):
    """Import 'date acquired' into a #purchased custom column."""
    books = calibre_asin_map()
    acquired = mycd_items(args.json)
    if not args.dry_run:
        # No-op (and harmless error) if the column already exists.
        subprocess.run(["calibredb", "--library-path", CALIBRE_LIB,
                        "add_custom_column", "purchased", "Date Purchased",
                        "datetime"], capture_output=True)
    n = miss = 0
    for asin, ms in sorted(acquired.items(), key=lambda kv: kv[1]):
        ids = books.get(asin)
        if not ids:
            miss += 1
            continue
        # Must be tz-aware: calibredb reads a naive timestamp as UTC, which
        # would shift every date by the local offset.
        when = datetime.datetime.fromtimestamp(
            ms / 1000, datetime.timezone.utc).astimezone().isoformat()
        for bid in ids:
            print(f"{asin} #{bid}: {when}")
            calibredb(["set_metadata", str(bid),
                       "--field", f"#purchased:{when}"], args.dry_run)
            n += 1
    print(f"\n{n} books updated, {miss} MYCD entries not in Calibre")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", "--dry-run", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("series").set_defaults(func=cmd_series)
    sp = sub.add_parser("purchased")
    sp.add_argument("json", nargs="+", help="MYCD GetContentOwnershipData JSON")
    sp.set_defaults(func=cmd_purchased)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
