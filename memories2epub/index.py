"""Stage 6c: Index (PDF 329-339) -> out/backmatter/index.json

Per page: gutter found by ink projection, each column cropped from just below the
running head (or the "Index" title) and OCR'd with `--psm 4` (cached under
out/backmatter/index_ocr/). Per column, the indent level comes from x0 relative to
the column's left edge (0 / +55 sub-entry / +105 turnover); turnovers are merged
into the preceding line (also across a column/page break), hyphenation is
resolved by join_lines (word breaks dropped, range breaks like 35-/38 kept),
`See` / `See also` is split out, and the locator string is split off the end with
`,\\s*(\\d+(-\\d+)?)`. "(continued)" repeats at column tops are marked.
Usage: python3 index.py [pages...]
"""
import json, re, statistics, sys
import backmatter_util as bu

FIRST, LAST = 329, 339
OCR = bu.OUT / "index_ocr"
COLUMN_PITCH = 671   # R column edge - L column edge, 667-675 on every page
LOC = re.compile(r"^(.*?)[,.]?\s*(?:,|(?<=\))|(?<=\d\.))\s*(\d{1,3}(?:-\d{1,3})?)\.?$")
LOC_STRICT = re.compile(r"^(.*?),\s*(\d{1,3}(?:-\d{1,3})?)\.?$")
LOC_RUN = re.compile(r"^(.*?\d)\s+(\d{1,3}(?:-\d{1,3})?)\.?$")          # "187-190 260": comma lost
LOC_LOWER = re.compile(r"^(.*?\s[a-z]+\.?)\s+(\d{1,3}(?:-\d{1,3})?)\.?$")   # "development of 68"
MAX_PAGE = 330
MODEL_WORDS = {"Model", "Type", "IBM", "System/360", "Mark", "Series", "CDC", "Stretch", "Whirlwind", "Harvard", "SAGE", "AN/FSQ-7", "Ferramic"}
ABBREV = re.compile(r"(\b[A-Z]|\b(Co|Corp|Inc|Jr|Sr|Ltd|St|Mass|Calif|N\.Y|U\.S))\.$")


def strip_end(s):
    """Strip trailing separators but keep the period of an initial / abbreviation."""
    s = s.rstrip(" ,;")
    if s.endswith(".") and not ABBREV.search(s):
        s = s[:-1].rstrip(" ,;")
    return s


SEE = re.compile(r"^(.*?)(?:(?<=[.,;])|(?<=^))\s*(See(?: also)?)\s+(.+?)\.?$")


def column_lines(p, fl):
    """OCR both columns of page p; returns lines in reading order (L then R), in
    page coordinates, each with "col" and "level"."""
    from PIL import Image
    OCR.mkdir(parents=True, exist_ok=True)
    ink = bu.ink_array(p)
    top = bu.running_head_bottom(p, title=r"^Index$") + 15
    bottom = ink.shape[0] - 120
    g0, g1 = bu.find_gutter(ink, top, bottom)
    out = []
    head_x = min([l["bbox"][0] for l in bu.hocr_lines(f"out/hocr/p-{p:03d}.hocr")
                  if l["bbox"][1] < 150 or re.match(r"^Index$", l["text"])] or [None])
    edges = {}
    for col, (xa, xb) in (("L", (0, g0)), ("R", (g1, ink.shape[1]))):
        ext = bu.ink_extent(ink, top, bottom, xa + 60, xb) if col == "L" else bu.ink_extent(ink, top, bottom, xa, xb - 60)
        if ext is None:
            continue
        box = (max(0, ext[0] - 30), top, min(ink.shape[1], ext[1] + 30), bottom)
        base = OCR / f"p-{p:03d}-{col}"
        png = base.with_suffix(".png")
        if not png.exists():
            Image.open(bu.page_png(p)).crop(box).save(png)
        hocr = bu.run_tesseract(png, base, 4)
        lines = bu.hocr_lines(hocr, dx=box[0], dy=box[1])
        lines = [l for l in lines if re.search(r"[A-Za-z0-9]", l["text"])]
        if not lines:
            continue
        for l in lines:   # specks OCR'd as leading/trailing punctuation-only words
            while l["words"] and not re.search(r"\w", l["words"][0]["text"]):
                w = l["words"].pop(0)
                fl.add(p, w["bbox"], w["text"], "stray glyph before the line (dropped)")
            while len(l["words"]) > 1 and not re.search(r"[\w)]", l["words"][-1]["text"]):
                w = l["words"].pop()
                fl.add(p, w["bbox"], w["text"], "stray glyph after the line (dropped)")
            l["bbox"] = bu.union_bbox([w["bbox"] for w in l["words"]])
            l["text"] = " ".join(w["text"] for w in l["words"])
            l["conf"] = statistics.mean(w["conf"] for w in l["words"])
        # level-0 edge: lowest x0 cluster with >= 2 members; cross-checked against the
        # running head's x0 (L) or the L edge + column pitch (R), since a column can be
        # nearly all sub-entries (p334 R).
        xs = sorted(l["bbox"][0] for l in lines)
        edge = None
        for x in xs:
            cl = [v for v in xs if x <= v < x + 30]
            if len(cl) >= 2 or len(xs) < 2:
                edge = statistics.median(cl); break
        ref = head_x if col == "L" else (edges["L"] + COLUMN_PITCH if "L" in edges else None)
        if ref is not None and abs(edge - ref) > 22:
            fl.add(p, [int(ref), top, int(ref) + 60, top + 60], f"{edge:.0f}", f"column {col} edge {edge:.0f} disagrees with reference {ref:.0f}; using the reference")
            edge = ref
        edges[col] = edge
        for l in lines:
            d = l["bbox"][0] - edge
            l["level"] = 0 if d < 27 else 1 if d < 80 else 2
            l["col"] = col; l["page"] = p
            if 20 <= d < 34 or 72 <= d < 90:
                fl.add(p, l["bbox"], l["text"], f"ambiguous indent: x0 is {d:.0f} px right of the column edge (level {l['level']})")
        out.extend(lines)
    return out, (g0, g1), top


def parse(pages):
    fl = bu.Flagger("index")
    entries, per_page = [], {}
    for p in pages:
        stats = per_page.setdefault(p, {"lines": 0, "entries": 0, "flags": 0, "warn": []})
        lines, gutter, top = column_lines(p, fl)
        stats["warn"].append(f"gutter {gutter[0]}-{gutter[1]} top {top}")
        for i, l in enumerate(lines):
            stats["lines"] += 1
            txt, hints = bu.normalize_text(l["text"])
            if l["level"] == 2 or (i == 0 and l["level"] == 1 and entries and not entries[-1]["_lt"][-1].rstrip().endswith((",", ".")) and False):
                if not entries:
                    fl.add(p, l["bbox"], txt, "turnover line with nothing to attach to"); continue
                e = entries[-1]
            else:
                e = {"level": l["level"], "text": "", "locators": "", "see": None, "page": p, "lines": [], "flags": [], "_lt": []}
                entries.append(e); stats["entries"] += 1
            e["lines"].append([p, l["bbox"]]); e["_lt"].append(txt)
            for h in hints:
                fl.add(p, l["bbox"], txt, h); e["flags"].append(h)
            if l["conf"] < bu.CONF_THR:
                fl.add(p, l["bbox"], txt, f"mean x_wconf {l['conf']:.0f} < {bu.CONF_THR}: {bu.low_conf_words(l)}")
                e["flags"].append("low-confidence line")
        stats["flags"] = sum(1 for f in fl.flags if f["page"] == p)
    for k, e in enumerate(entries):
        heading = e["level"] == 0 and k + 1 < len(entries) and entries[k + 1]["level"] == 1
        s = bu.normalize_text(bu.join_lines(e.pop("_lt")))[0]
        if re.search(r"\(continued\)", s):
            s = re.sub(r"\s*\(continued\)", "", s); e["continued"] = True
        m = SEE.match(s)
        if m and not re.match(r"^See[a-z]", s[m.start(2):]):
            s, e["see"] = strip_end(m.group(1)), re.sub(r"^See als[a-z]\b", "See also", f"{m.group(2)} {m.group(3)}")
        locs, nocomma = [], []
        while True:
            m = LOC_STRICT.match(s)
            if not m:
                m = LOC.match(s) or LOC_RUN.match(s) or (LOC_LOWER.match(s) if not locs else None)
                if m and int(m.group(2).split("-")[0]) <= MAX_PAGE:
                    nocomma.insert(0, m.group(2))
                else:
                    m = None
            if not m:
                break
            s = m.group(1); locs.insert(0, m.group(2))
        s = strip_end(s)
        e["text"], e["locators"] = s, ", ".join(locs)
        if nocomma:
            e["flags"].append(f"locators {nocomma} split off without a comma")
            fl.add(e["page"], bu.union_bbox([b for _, b in e["lines"]]), s + " | " + e["locators"], f"locators {nocomma} split off without a comma; check", {"entry": s})
        for r in locs:
            if "-" in r and int(r.split("-")[0]) >= int(r.split("-")[1]):
                e["flags"].append(f"bad range {r}")
                fl.add(e["page"], bu.union_bbox([b for _, b in e["lines"]]), s, f"page range {r} is not increasing", {"entry": s})
        if not locs and not e["see"] and not heading:
            e["flags"].append("no locators and no See")
            fl.add(e["page"], bu.union_bbox([b for _, b in e["lines"]]), s, "entry has neither locators nor a See reference", {"entry": s})
        elif (m := re.search(r"(\S+)\s+(\d{1,3})$", s)) and int(m.group(2)) <= MAX_PAGE and m.group(1) not in MODEL_WORDS:
            e["flags"].append("text ends in a digit")
            fl.add(e["page"], bu.union_bbox([b for _, b in e["lines"]]), s, "entry text ends in digits; locator split may be off", {"entry": s})
        if re.search(r"[^\w\s,.;:()'’\"/&×-]", s):
            e["flags"].append("odd characters")
            fl.add(e["page"], bu.union_bbox([b for _, b in e["lines"]]), s, "unexpected characters in entry text", {"entry": s})
    return entries, fl, per_page


if __name__ == "__main__":
    pages = [int(a) for a in sys.argv[1:]] or list(range(FIRST, LAST + 1))
    entries, fl, per_page = parse(pages)
    bu.OUT.mkdir(parents=True, exist_ok=True)
    (bu.OUT / "index.json").write_text(json.dumps(entries, indent=1, ensure_ascii=False))
    fl.write()
    n0 = sum(1 for e in entries if e["level"] == 0); n1 = len(entries) - n0
    nl = sum(len(e["lines"]) for e in entries)
    print(f"index: {len(entries)} entries ({n0} main, {n1} sub) from {nl} lines; {sum(1 for e in entries if e['see'])} with See; flags: {len(fl.flags)}")
    print("page  lines entries flags  notes")
    for p, s in per_page.items():
        print(f"p{p:03d}  {s['lines']:4d} {s['entries']:6d} {s['flags']:6d}  {'; '.join(s['warn'])}")
