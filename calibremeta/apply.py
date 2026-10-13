"""Apply plan.json to a Calibre library. Run under calibre's Python:

    calibre-debug apply.py <library-dir> <plan.json> [--skip SECTION ...]

Calibre (the GUI) must not be running against the library.
"""

import json
import sys
from datetime import datetime, timezone

from calibre.library import db

args = sys.argv[1:]
lib, plan_path = args[0], args[1]
skip = set(args[args.index("--skip") + 1:]) if "--skip" in args else set()
plan = json.load(open(plan_path))
c = db(lib).new_api


def author_ids():
    return {v: k for k, v in c.get_id_map("authors").items()}


if "authors" not in skip:
    ids = author_ids()
    renames, sorts = {}, {}
    for op in plan["authors"]:
        aid = ids[op["old"]]
        if op["new"] != op["old"]:
            renames[aid] = op["new"]
    _, id_map = c.rename_items("authors", renames)
    # After merges, set the sort on whichever id each name ended up as.
    ids = author_ids()
    for op in plan["authors"]:
        sorts[ids[op["new"]]] = op["sort"]
    c.set_sort_for_authors(sorts)
    print(f"authors: renamed {len(renames)}, set sort on {len(sorts)}")

if "book_authors" not in skip:
    c.set_field("authors", {op["id"]: op["new"] for op in plan["book_authors"]})
    # New authors get calibre's computed sort; make sure ones we know are right.
    ids = author_ids()
    fix = {}
    for op in plan["book_authors"]:
        for name in op["new"]:
            if name == "Merriam-Webster":
                fix[ids[name]] = name
    if fix:
        c.set_sort_for_authors(fix)
    print(f"book_authors: {len(plan['book_authors'])}")

if "series_rename" not in skip:
    sids = {v: k for k, v in c.get_id_map("series").items()}
    c.rename_items("series", {sids[op["old"]]: op["new"] for op in plan["series_rename"]})
    print(f"series_rename: {len(plan['series_rename'])}")

if "series" not in skip:
    c.set_field("series", {op["id"]: op["series"] for op in plan["series"]})
    c.set_field("series_index", {op["id"]: op["index"] for op in plan["series"]})
    c.set_field("series", {op["id"]: None for op in plan["series_clear"]})
    print(f"series: set {len(plan['series'])}, cleared {len(plan['series_clear'])}")

if "titles" not in skip:
    c.set_field("title", {op["id"]: op["new"] for op in plan["titles"]})
    print(f"titles: {len(plan['titles'])}")

if "pubdates" not in skip:
    c.set_field("pubdate", {
        op["id"]: datetime.fromisoformat(op["new"]).replace(hour=12, tzinfo=timezone.utc)
        for op in plan["pubdates"]})
    print(f"pubdates: {len(plan['pubdates'])}")

if "publishers" not in skip:
    pids = {v: k for k, v in c.get_id_map("publisher").items()}
    renames = {pids[op["old"]]: op["new"] for op in plan["publishers"] if op["new"]}
    c.rename_items("publisher", renames)
    for op in plan["publishers"]:
        if op["new"] is None:
            c.remove_items("publisher", [pids[op["old"]]])
    print(f"publishers: {len(plan['publishers'])}")

if "misc" not in skip:
    for op in plan["misc"]:
        if op["op"] == "set_language":
            c.set_field("languages", {op["id"]: [op["value"]]})
        elif op["op"] == "remove_tags":
            tids = {v: k for k, v in c.get_id_map("tags").items()}
            c.remove_items("tags", [tids[t] for t in op["tags"] if t in tids])
        elif op["op"] == "delete_book":
            # Moves to calibre's trash; recoverable from the GUI.
            c.remove_books([op["id"]])
    print(f"misc: {len(plan['misc'])}")

c.close()
