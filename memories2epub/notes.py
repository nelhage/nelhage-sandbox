"""Stage 6a: References and Notes (PDF 281-316) -> out/backmatter/notes.json

Parser per out/survey/backmatter.md §1.5, on out/hocr (psm 1) word boxes:
  - running head dropped, p281 title + intro paragraph kept separately;
  - "Chapter N" subheads (accepting a bare "Chapter" whose bold digit came out as
    its own one-glyph line);
  - text-column x0 per page = median x0 of >=4-word lines not starting with a number;
  - number column re-OCR'd from a crop (psm 6, digit whitelist) -> anchor y positions,
    united with the psm-1 number tokens; values assigned by sequence (prev+1, reset
    at each subhead) and cross-checked against both OCR readings;
  - entry start = anchor within 35 px of a text line; must be preceded by a
    vertical gap > 70 px (or a subhead / page top); a gap without an anchor is an
    inferred start (flagged) unless the line is a "(a)"-style list item;
  - page-top text lines with no anchor above continue the previous entry;
  - lines joined with de-hyphenation; the ch4 n8 (a)-(u) two-column list is kept
    as "sublist".
Asserts 61/39/64/86/69/59/64/48/28 = 518 notes.
Usage: python3 notes.py [pages...]
"""
import json, re, statistics, sys
import backmatter_util as bu

FIRST, LAST = 281, 316
EXPECTED = [61, 39, 64, 86, 69, 59, 64, 48, 28]
NUM = re.compile(r"^\W?(\d{1,3})[.,]?$")
SUBHEAD = re.compile(r"^Chapter\s*(\d+)?$")
LISTITEM = re.compile(r"^\((?:[a-z]|1)\)")
GAP, NEAR = 70, 35


def text_column(lines, y_from=0):
    xs = [l["bbox"][0] for l in lines if l["bbox"][1] >= y_from and len(l["words"]) >= 4
          and not NUM.match(l["words"][0]["text"])]
    return statistics.median(xs) if xs else 280


def number_anchors(p, textx, lines, subheads, y_from, fl):
    """Anchor candidates: psm-6 digit-whitelist tokens from the number-column crop
    united with psm-1 number tokens left of the text column. Each anchor:
    {"y0","y1","bbox","vals": [ocr strings]}."""
    x0 = max(0, int(textx - 230)); x1 = int(textx - 15)
    toks = bu.strip_ocr(p, (x0, max(150, y_from), x1, 2560), bu.OUT / "numstrip")
    cands = []
    for w in toks:
        b = w["bbox"]; h = b[3] - b[1]
        if any(abs((b[1] + b[3]) / 2 - (s["bbox"][1] + s["bbox"][3]) / 2) < 30 for s in subheads):
            continue                       # the subhead's own glyphs
        if not (16 <= h <= 40):
            continue
        cands.append({"y0": b[1], "y1": b[3], "bbox": b, "vals": [w["text"].strip(".,")], "src": "strip"})
    for l in lines:                       # psm-1 tokens left of the text column
        for w in l["words"]:
            if w["bbox"][0] >= textx - 60:
                break
            if NUM.match(w["text"]):
                b = w["bbox"]
                for c in cands:
                    if abs(c["y0"] - b[1]) < 20:
                        c["vals"].append(NUM.match(w["text"]).group(1)); break
                else:
                    cands.append({"y0": b[1], "y1": b[3], "bbox": b, "vals": [NUM.match(w["text"]).group(1)], "src": "psm1"})
    cands.sort(key=lambda c: c["y0"])
    return cands


def split_left(line, textx):
    """Words left of the text column (numbers / residue) vs the text words."""
    left, text = [], []
    for w in line["words"]:
        (left if w["bbox"][0] < textx - 60 and not text else text).append(w)
    return left, text


def parse(pages):
    fl = bu.Flagger("notes")
    chapters, intro_lines, title = [], [], None
    chap, cur, nxt = 0, None, 1
    per_page = {}

    def new_entry(p, line_bbox, anchor):
        nonlocal cur, nxt
        cur = {"n": nxt, "text": "", "pages": [p], "lines": [], "flags": [], "_lt": []}
        if anchor:
            vals = anchor["vals"]
            if str(nxt) not in vals:
                cur["flags"].append(f"number OCR {vals} but expected {nxt}")
                fl.add(p, bu.union_bbox([anchor["bbox"], line_bbox]), " / ".join(vals),
                       f"note number read as {vals}, sequence says {nxt}", {"chapter": chap, "n": nxt})
            cur["anchor_bbox"] = anchor["bbox"]
        chapters[-1]["notes"].append(cur)
        nxt += 1
        return cur

    for p in pages:
        stats = per_page.setdefault(p, {"lines": 0, "entries": 0, "flags": 0, "warn": []})
        lines = bu.page_lines(p)
        # -- title / intro (p281) and subheads
        y_from = 0
        if p == FIRST:
            t = [l for l in lines if re.match(r"^References and Notes$", l["text"])]
            if t:
                title = t[0]["text"]; lines.remove(t[0])
        subheads = []
        for l in list(lines):
            m = SUBHEAD.match(l["text"])
            if m:
                num = m.group(1)
                if num is None:   # bold digit came out as its own line (p311 "Chapter" / "8")
                    for d in lines:
                        if d is not l and re.match(r"^\d$", d["text"]) and abs(d["bbox"][1] - l["bbox"][1]) < 40:
                            num = d["text"]; lines.remove(d); break
                subheads.append({"bbox": l["bbox"], "num": int(num) if num else None})
                lines.remove(l)
        if p == FIRST:
            y_from = subheads[0]["bbox"][1]
            intro_lines = [l for l in lines if l["bbox"][1] < y_from]
            lines = [l for l in lines if l["bbox"][1] >= y_from]
        textx = text_column(lines)
        anchors = number_anchors(p, textx, lines, subheads, y_from, fl)
        # -- text lines
        tlines = []
        for l in lines:
            left, text = split_left(l, textx)
            for w in left:
                if not NUM.match(w["text"]) and not any(abs(a["y0"] - w["bbox"][1]) < 20 for a in anchors):
                    fl.add(p, w["bbox"], w["text"], "residue token left of the text column (dropped)")
                    stats["warn"].append(f"residue {w['text']!r}@{w['bbox'][1]}")
            if not text:
                continue
            tb = bu.union_bbox([w["bbox"] for w in text])
            if tb[0] < textx - 30:
                fl.add(p, tb, " ".join(w["text"] for w in text), f"line x0 {tb[0]} is {textx - tb[0]:.0f} px left of the text column")
            tlines.append({"bbox": tb, "words": text, "text": " ".join(w["text"] for w in text),
                           "conf": statistics.mean(w["conf"] for w in text), "left": left})
        events = sorted([("sub", s["bbox"][1], s) for s in subheads] + [("line", l["bbox"][1], l) for l in tlines], key=lambda e: e[1])
        used = set(); prev_y = None; page_top = True
        for kind, y, obj in events:
            if kind == "sub":
                chap = obj["num"] if obj["num"] else chap + 1
                if obj["num"] is None:
                    fl.add(p, obj["bbox"], "Chapter", f"subhead without a number; assumed Chapter {chap}")
                chapters.append({"chapter": chap, "notes": []})
                cur, nxt, prev_y, page_top = None, 1, None, False
                continue
            l = obj
            stats["lines"] += 1
            # anchor within NEAR px, not yet used
            near = [(abs(a["y0"] - y), i) for i, a in enumerate(anchors) if abs(a["y0"] - y) <= NEAR and i not in used]
            anchor = None
            if near:
                i = min(near)[1]; used.add(i); anchor = anchors[i]
            gap = prev_y is None or (y - prev_y) > GAP
            if anchor and not gap:
                fl.add(p, bu.union_bbox([anchor["bbox"], l["bbox"]]), l["text"], f"number anchor {anchor['vals']} but only {y - prev_y} px below the previous line")
                stats["warn"].append(f"anchor-no-gap {anchor['vals']}@{y}")
            if anchor:
                new_entry(p, l["bbox"], anchor); stats["entries"] += 1
            elif gap and not page_top and not LISTITEM.match(l["text"]) and cur is not None and not (chap == 4 and cur["n"] == 8):
                new_entry(p, l["bbox"], None); stats["entries"] += 1
                cur["flags"].append("start inferred from gap; no number found")
                fl.add(p, [int(textx - 230), l["bbox"][1], l["bbox"][2], l["bbox"][3]], l["text"],
                       f"{y - prev_y} px gap but no note number found; inferred start of note {cur['n']}", {"chapter": chap, "n": cur["n"]})
            elif gap and cur is None:
                new_entry(p, l["bbox"], None); stats["entries"] += 1
                cur["flags"].append("first line after subhead without a number")
                fl.add(p, l["bbox"], l["text"], f"first line after subhead, no number; assumed note {cur['n']}")
            elif cur is None:
                raise RuntimeError(f"p{p}: orphan line {l['text']!r}")
            if p not in cur["pages"]:
                cur["pages"].append(p)
            txt, hints = bu.normalize_text(l["text"])
            for h in hints:
                fl.add(p, l["bbox"], txt, h, {"chapter": chap, "n": cur["n"]}); cur["flags"].append(h)
            if l["conf"] < bu.CONF_THR:
                fl.add(p, l["bbox"], txt, f"mean x_wconf {l['conf']:.0f} < {bu.CONF_THR}: {bu.low_conf_words(l)}", {"chapter": chap, "n": cur["n"]})
                cur["flags"].append("low-confidence line")
            lb = list(l["bbox"])
            if anchor:
                lb[0] = min(lb[0], anchor["bbox"][0])
            cur["lines"].append([p, lb]); cur["_lt"].append(txt)
            prev_y = y; page_top = False
        for i, a in enumerate(anchors):
            if i not in used:
                fl.add(p, a["bbox"], "/".join(a["vals"]), f"number-column token {a['vals']} matches no text line (ignored)")
                stats["warn"].append(f"unused-anchor {a['vals']}@{a['y0']}")
        stats["flags"] = sum(1 for f in fl.flags if f["page"] == p)

    # -- join lines, sublist special case
    for ch in chapters:
        for e in ch["notes"]:
            lt = e.pop("_lt")
            if ch["chapter"] == 4 and e["n"] == 8:
                cit = [t for t in lt if not LISTITEM.match(t)]
                items = []
                for t in lt:
                    if LISTITEM.match(t):
                        for it in re.split(r"\s+(?=\((?:[a-z]|1)\))", t):
                            items.append(re.sub(r"^\(1\)", "(l)", it).rstrip(","))
                items.sort(key=lambda s: s[1])
                letters = "".join(s[1] for s in items)
                if letters != "abcdefghijklmnopqrstu":
                    e["flags"].append(f"sublist letters {letters!r} != a..u")
                    fl.add(e["pages"][0], e["lines"][0][1], letters, "ch4 n8 (a)-(u) sublist incomplete")
                e["sublist"] = items
                e["text"] = bu.normalize_text(bu.join_lines(cit))[0]
            else:
                e["text"] = bu.normalize_text(bu.join_lines(lt))[0]
    intro = bu.normalize_text(bu.join_lines([bu.normalize_text(l["text"])[0] for l in intro_lines]))[0]
    return {"title": title, "intro": intro, "intro_lines": [[FIRST, l["bbox"]] for l in intro_lines],
            "chapters": chapters}, fl, per_page


if __name__ == "__main__":
    pages = [int(a) for a in sys.argv[1:]] or list(range(FIRST, LAST + 1))
    doc, fl, per_page = parse(pages)
    bu.OUT.mkdir(parents=True, exist_ok=True)
    (bu.OUT / "notes.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    fl.write()
    counts = [len(c["notes"]) for c in doc["chapters"]]
    print("notes per chapter:", counts, "total", sum(counts), "(expected", EXPECTED, "=", sum(EXPECTED), ")")
    print(f"intro: {len(doc['intro'])} chars; flags: {len(fl.flags)}")
    print("page  lines entries flags  warnings")
    for p, s in per_page.items():
        print(f"p{p:03d}  {s['lines']:4d} {s['entries']:6d} {s['flags']:6d}  {'; '.join(s['warn'])}")
    spans = [(c["chapter"], e["n"], e["pages"]) for c in doc["chapters"] for e in c["notes"] if len(e["pages"]) > 1]
    print("page-spanning notes:", spans)
    if pages == list(range(FIRST, LAST + 1)):
        assert counts == EXPECTED, f"note counts {counts} != {EXPECTED}"
        assert sum(counts) == 518
        print("count assertions OK")
