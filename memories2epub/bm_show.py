"""Print the parsed back-matter entries that appear on a PDF page, for review.

  python3 bm_show.py notes 283        -> chapter, n, text of each note on p283
  python3 bm_show.py chronology 318   -> i, date, text
  python3 bm_show.py index 330        -> i, level, text | locators | see
"""
import json, pathlib, sys

kind, page = sys.argv[1], int(sys.argv[2])
d = json.loads(pathlib.Path(f"out/backmatter/{kind}.json").read_text())
if kind == "notes":
    if page == 281:
        print("INTRO:", d["intro"], "\n")
    for ch in d["chapters"]:
        for n in ch["notes"]:
            if page in n["pages"]:
                extra = ("\n    " + "\n    ".join(n["sublist"])) if n.get("sublist") else ""
                span = f" (pages {n['pages']})" if len(n["pages"]) > 1 else ""
                print(f"ch{ch['chapter']} n{n['n']}{span}: {n['text']}{extra}\n")
elif kind == "chronology":
    for i, e in enumerate(d):
        if e["page"] == page:
            print(f"i={i}  {e['date']}: {e['text']}\n")
else:
    for i, e in enumerate(d):
        if e["page"] == page:
            print(f"i={i} L{e.get('level', 0)}{' (continued)' if e.get('continued') else ''}: {e['text']} | {e.get('locators') or ''} | {e.get('see') or ''}")
