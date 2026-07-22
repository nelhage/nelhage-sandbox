#!/usr/bin/env python3
"""Push Kindle-library metadata into a Calibre library, matched by ASIN.

Series comes from the app's `Groups`/`GroupItems` tables (Amazon's own series
grouping).  Purchase dates come from a CSV scraped off Manage Your Content and
Devices (see mycd_scrape.py), since the device DB has no acquisition date --
`KindleContent.DELIVERY_DATE` is only when the emulator downloaded the file.

    ./calibre_meta.py series --dry-run
    ./calibre_meta.py series
    ./calibre_meta.py purchased mycd.csv
"""
import argparse
import csv
import collections
import os
import sqlite3
import subprocess
import sys

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


def cmd_purchased(args):
    """Import 'date acquired' into a custom column (#purchased)."""
    books = calibre_asin_map()
    rows = list(csv.DictReader(open(args.csv)))
    n = miss = 0
    for row in rows:
        asin = (row.get("asin") or "").strip().upper()
        date = (row.get("acquired") or "").strip()
        ids = books.get(asin)
        if not (asin and date):
            continue
        if not ids:
            miss += 1
            continue
        for bid in ids:
            print(f"{asin} #{bid}: {date}")
            calibredb(["set_metadata", str(bid),
                       "--field", f"#purchased:{date}"], args.dry_run)
            n += 1
    print(f"\n{n} books updated, {miss} CSV rows not in Calibre")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", "--dry-run", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("series").set_defaults(func=cmd_series)
    sp = sub.add_parser("purchased")
    sp.add_argument("csv", help="CSV with asin,acquired columns")
    sp.set_defaults(func=cmd_purchased)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
