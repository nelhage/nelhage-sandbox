"""Stage 6b: Chronology (PDF 317-328) -> out/backmatter/chronology.json

From out/hocr4 (psm 4 keeps `8/44 text` on one line): a line whose first word
matches ^(\\d{1,2}/\\d{2}|\\d{4})$ and sits in the date column (x0 < textx - 60)
starts an entry; other lines continue it. Tokens left of the text column that are
not dates (blemish residue) are dropped and flagged. Entries are checked for
chronological order. Asserts 174 entries.
Usage: python3 chronology.py [pages...]
"""
import json, re, statistics, sys
import backmatter_util as bu

FIRST, LAST = 317, 328
EXPECTED = 174


def before(a, b):
    """a is chronologically before b; a bare year (month 0) only compares on the year."""
    if a[0] != b[0]:
        return a[0] < b[0]
    return a[1] < b[1] if a[1] and b[1] else False


def parse(pages):
    fl = bu.Flagger("chronology")
    entries, per_page = [], {}
    cur = None
    for p in pages:
        stats = per_page.setdefault(p, {"lines": 0, "entries": 0, "flags": 0, "warn": []})
        lines = [l for l in bu.page_lines(p, "out/hocr4") if l["text"] != "Chronology"]
        dated = [l for l in lines if bu.date_key(l["words"][0]["text"]) and len(l["words"]) > 1]
        textx = statistics.median(l["words"][1]["bbox"][0] for l in dated) if dated else 400
        for l in lines:
            stats["lines"] += 1
            words = list(l["words"])
            date = None
            while words and (words[0]["bbox"][0] < textx - 60
                             or (words[0]["bbox"][0] < textx - 15 and not re.search(r"\w", words[0]["text"]))):
                w = words.pop(0)
                if bu.date_key(w["text"]) and date is None:
                    date = w
                else:
                    fl.add(p, w["bbox"], w["text"], "token left of the text column is not a date (dropped)")
                    stats["warn"].append(f"residue {w['text']!r}@{w['bbox'][1]}")
            if not words:
                continue
            txt, hints = bu.normalize_text(" ".join(w["text"] for w in words))
            tb = bu.union_bbox([w["bbox"] for w in words])
            if date:
                cur = {"date": date["text"], "text": "", "page": p, "lines": [], "flags": [], "_lt": [],
                       "date_bbox": date["bbox"]}
                entries.append(cur); stats["entries"] += 1
                if entries[-2:-1] and before(bu.date_key(date["text"]), bu.date_key(entries[-2]["date"])):
                    cur["flags"].append("out of chronological order")
                    fl.add(p, bu.union_bbox([date["bbox"], tb]), date["text"], f"date {date['text']} is before the previous entry's {entries[-2]['date']}")
            elif cur is None:
                fl.add(p, tb, txt, "text line before any dated entry (dropped)"); continue
            elif tb[0] < textx - 30:
                fl.add(p, tb, txt, f"continuation line x0 {tb[0]} left of the text column ({textx:.0f})")
            for h in hints:
                fl.add(p, tb, txt, h); cur["flags"].append(h)
            lc = statistics.mean(w["conf"] for w in words)
            if lc < bu.CONF_THR:
                fl.add(p, tb, txt, f"mean x_wconf {lc:.0f} < {bu.CONF_THR}: {[w['text'] for w in words if w['conf'] < bu.CONF_THR]}")
                cur["flags"].append("low-confidence line")
            cur["lines"].append([p, tb]); cur["_lt"].append(txt)
        stats["flags"] = sum(1 for f in fl.flags if f["page"] == p)
    for e in entries:
        e["text"] = bu.normalize_text(bu.join_lines(e.pop("_lt")))[0]
    return entries, fl, per_page


if __name__ == "__main__":
    pages = [int(a) for a in sys.argv[1:]] or list(range(FIRST, LAST + 1))
    entries, fl, per_page = parse(pages)
    bu.OUT.mkdir(parents=True, exist_ok=True)
    (bu.OUT / "chronology.json").write_text(json.dumps(entries, indent=1, ensure_ascii=False))
    fl.write()
    print(f"chronology entries: {len(entries)} (expected {EXPECTED}); flags: {len(fl.flags)}")
    print("page  lines entries flags  warnings")
    for p, s in per_page.items():
        print(f"p{p:03d}  {s['lines']:4d} {s['entries']:6d} {s['flags']:6d}  {'; '.join(s['warn'])}")
    from collections import Counter
    print("date formats:", Counter("YYYY" if bu.date_key(e["date"])[1] == 0 else "M/YY" for e in entries))
    if pages == list(range(FIRST, LAST + 1)):
        assert len(entries) == EXPECTED, f"{len(entries)} entries != {EXPECTED}"
        print("count assertion OK")
