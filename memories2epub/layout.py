"""Stage 4: hOCR -> typed blocks per page (out/blocks/p-NNN.json).

Handles the front-matter prose pages (11-14) and the body (17-280). Back matter
is parsed by notes.py / chronology.py / index.py.

Block types: head (dropped, kept for the record), chapter, section, subhead,
para, quote, list, figure. Each text block carries its lines and words (with
tesseract confidence) so later stages can normalise and flag at word level.

Geometry rules (300 dpi, see out/survey/body.md and structure.md):
  running head        one line with y1 < HEAD_Y
  paragraph indent    x0 > margin + 40  (indent is ~75-80 px; margin per page)
  gap                 baseline step > 1.5 x body pitch
  subhead             short bold line after a gap, no terminal period, matched
                      against the known list (SUBHEADS); unmatched candidates
                      are emitted as subhead with flag=True
  block quote / list  run of lines with pitch < 0.9 x body pitch between gaps
  figure              lines inside a figure bbox are dropped; the caption comes
                      from out/figures/figures.json

Usage: python3 layout.py [pages...] [--dump]
"""
import json, pathlib, re, statistics, sys
from lxml import html

HEAD_Y = 150   # running head y0 is 62-125 px; body first line y0 >= 189
INDENT = 40
CHAPTERS = {17: (1, "The Postwar Challenge"), 50: (2, "Searching for Memory"),
            78: (3, "A Memory from Whirlwind"), 109: (4, "Project SAGE"),
            145: (5, "Commercial Ferrite Core Memories"), 176: (6, "Project Stretch"),
            203: (7, "The Road to System/360"), 229: (8, "System/360 Memories"),
            264: (9, "Managing Technological Change")}
SECTIONS = {11: "Series Foreword", 13: "Preface"}
BODY = list(range(11, 15)) + list(range(17, 281))
SUBHEADS = """The Watson Legacy|Developments at the Moore School|UNIVAC, ERA, and Remington Rand|
New Leaders at IBM|Entering the Electronic Computer Market|A First Choice|Magnetic Cores at IBM|
Feasibility Models|Memory Considerations|New Objectives for Whirlwind|A New Memory Proposal|
Joining Forces|First Ferrite Core Main Memory|The Disputed Invention|More Inventors|
A Retrospective View|Selecting IBM|Staffing the Project|Cooperative Design Effort|
Ferrite Core Procurement|Core Testing and Wiring|XD-1 and XD-2 Memories|Core Fabrication in IBM|
Realizing the Dream|A Small 3D Array Memory Product|Indecisions on Main Memories|The Decision|
Developing Main Memories|Introducing Transistors|Ferrite Core Fabrication|Array Wiring Improvements|
Seeking Government Funds|Problems at Two Microseconds|Program Management|The Stretch Problem|
Competitive Chaos in IBM|A Plan Evolves|The SPREAD Report|Computer Control Stores|
Growing Requirements for Memory|The Forrester Patent Settlement|Management Pressures|
A Fast Control Store|Memories from Mecca|A Fast Main Memory|
Progress in Manufacturing|A Successful Technology|Cost-Performance Limits|The Manufacturing Buildup|
Not Good Enough|Innovation and Risk|Management Precepts|The IBM Team|Progress at MIT|Two Cores per Bit|
Ralph Palmer|Mike Haynes|Erich Bloch|Jay Forrester|John Gibson|Moe Every|TROS versus CCROS""".replace("\n", "").split("|")

def norm(s):
    return re.sub(r"[^a-z0-9]+", "", s.lower())
SUBHEAD_KEYS = {norm(s): s for s in SUBHEADS}

def known_subhead(t):
    """Exact or fuzzy (OCR-tolerant) match against the known subhead list."""
    key = norm(t)
    if key in SUBHEAD_KEYS:
        return SUBHEAD_KEYS[key]
    import difflib
    for k, v in SUBHEAD_KEYS.items():
        if abs(len(k) - len(key)) <= 3 and difflib.SequenceMatcher(None, k, key).ratio() >= 0.88:
            return v
    return None

def bbox_of(el):
    m = re.search(r"bbox (\d+) (\d+) (\d+) (\d+)", el.get("title", ""))
    return [int(v) for v in m.groups()]

def parse_hocr(path):
    """-> list of lines: {bbox, baseline_y, x_size, words:[{t,c,bbox}], par_id}"""
    tree = html.parse(str(path))
    lines = []
    for par_i, par in enumerate(tree.xpath('//p[@class="ocr_par"]')):
        for ln in par.xpath('.//span[@class="ocr_line" or @class="ocr_header" or @class="ocr_caption" or @class="ocr_textfloat"]'):
            title = ln.get("title", "")
            bb = bbox_of(ln)
            m = re.search(r"baseline ([-\d.]+) ([-\d.]+)", title)
            slope, off = (float(m.group(1)), float(m.group(2))) if m else (0.0, 0.0)
            m = re.search(r"x_size ([\d.]+)", title)
            xs = float(m.group(1)) if m else 40.0
            words = []
            for w in ln.xpath('.//span[@class="ocrx_word"]'):
                t = w.text_content().strip()
                if not t: continue
                c = int(re.search(r"x_wconf (\d+)", w.get("title", "")).group(1))
                words.append({"t": t, "c": c, "bbox": bbox_of(w)})
            if not words: continue
            lines.append({"bbox": bb, "base": bb[3] + off, "x_size": xs, "words": words, "par": par_i})
    lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    return merge_split_lines(lines)

def merge_split_lines(lines):
    """tesseract sometimes emits one printed line as two hOCR lines side by side (often
    putting them in different paragraphs). Merge pairs that overlap vertically by >= 70 %
    of the shorter one and are horizontally disjoint with a normal word gap between."""
    out = []
    for l in lines:
        for m in out:
            a, b = m["bbox"], l["bbox"]
            ov = min(a[3], b[3]) - max(a[1], b[1])
            if ov < 0.7 * min(a[3] - a[1], b[3] - b[1]): continue
            left, right = (m, l) if a[0] <= b[0] else (l, m)
            gap = right["bbox"][0] - left["bbox"][2]
            if gap < -4 or gap > 3 * max(m["x_size"], l["x_size"]): continue
            parts = m.get("parts", [m["bbox"]]) + [l["bbox"]]
            m["words"] = sorted(m["words"] + l["words"], key=lambda w: w["bbox"][0])
            m["bbox"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
            m["parts"] = parts
            break
        else:
            out.append(l)
    return out

def text_x0(line):
    """Left edge of the first real glyph: a fleck inside the column must not hide an indent."""
    for w in line["words"]:
        x0, y0, x1, y1 = w["bbox"]
        if x1 - x0 > 12 and y1 - y0 > 12:
            return x0
    return line["bbox"][0]

def text(line):
    return " ".join(w["t"] for w in line["words"])

def conf(line):
    return statistics.mean(w["c"] for w in line["words"])

def inside(bb, box, pad=8):
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    return box[0] - pad <= cx <= box[2] + pad and box[1] - pad <= cy <= box[3] + pad

def load_figures():
    p = pathlib.Path("out/figures/figures.json")
    if p.exists():
        figs = json.loads(p.read_text())
    else:  # fall back to the survey inventory (page fractions -> 300 dpi px)
        figs = []
        for row in pathlib.Path("out/survey/figures.tsv").read_text().splitlines()[1:]:
            f = row.split("\t")
            if not f[0].isdigit() or len(f) < 8: continue
            x0, y0, x1, y1 = [float(v) for v in f[4:8]]
            figs.append({"page": int(f[0]), "kind": f[1], "bbox": [x0 * 1609, y0 * 2578, x1 * 1609, y1 * 2578],
                         "caption_title": "", "caption_text": "", "caption_bbox": None, "file": ""})
    by_page = {}
    for f in figs:
        by_page.setdefault(f["page"], []).append(f)
    return by_page

def load_marks(n):
    p = pathlib.Path(f"out/marks/p-{n:03d}.json")
    return json.loads(p.read_text())["marks"] if p.exists() else []

def attach_marks(lines, marks):
    """Tag the word left of each superscript mark with ref/sup."""
    for m in marks:
        wb = m.get("word_bbox")
        best, bd = None, 1e9
        for ln in lines:
            for w in ln["words"]:
                if wb and w["bbox"] == wb:
                    best, bd = w, 0
                    break
                # fallback: nearest word ending just left of the mark on the same line
                mb = m["bbox"]
                if ln["bbox"][1] - 15 <= mb[3] and mb[1] <= ln["bbox"][3] + 15:
                    d = abs(w["bbox"][2] - mb[0]) + (0 if w["bbox"][2] <= mb[0] + 6 else 100)
                    if d < bd: best, bd = w, d
            if bd == 0: break
        if best is None: continue
        if m["kind"] == "ref":
            best.setdefault("refs", []).append({"n": m.get("value"), "status": m.get("status", "ok")})
        elif m.get("value") is not None:
            # non-reference superscript (M², 2¹⁰): tesseract read it as part of
            # the word, so strip it from the word text and re-add as <sup>
            v = str(m["value"])
            if best["t"].endswith(v):
                best["t"] = best["t"][:-len(v)]
                best["sup"] = v
            else:
                m2 = re.match(r"^(\w+?)" + re.escape(v) + r"(\W.*)?$", best["t"])
                if m2:
                    best["t"] = m2.group(1); best["sup"] = v
                    best["tail"] = m2.group(2) or ""
                else:
                    best["sup_unplaced"] = v

def page_metrics(lines):
    long = [l for l in lines if len(l["words"]) >= 4]
    src = long or lines
    xs = sorted(l["bbox"][0] for l in src)
    lo = xs[max(0, int(len(xs) * 0.15))]
    margin = statistics.median([x for x in xs if x <= lo + 30]) if xs else 0
    right = max(l["bbox"][2] for l in src) if src else 1500
    x_size = statistics.median(l["x_size"] for l in src) if src else 40
    steps = [b["base"] - a["base"] for a, b in zip(src, src[1:]) if 30 < b["base"] - a["base"] < 90]
    pitch = statistics.median(steps) if steps else 62
    return margin, right, x_size, pitch

def layout_page(n, figs_by_page, dump=False):
    hp = pathlib.Path(f"out/hocr2/p-{n:03d}.hocr")
    if not hp.exists(): hp = pathlib.Path(f"out/hocr/p-{n:03d}.hocr")
    lines = parse_hocr(hp)
    attach_marks(lines, load_marks(n))
    figs = figs_by_page.get(n, [])
    blocks = []
    # 1. running head
    if n not in CHAPTERS and n not in SECTIONS:
        # the running head is sometimes split into two hOCR lines (page number
        # and chapter label), so strip every line that sits in the head zone
        heads = [l for l in lines if l["bbox"][1] < HEAD_Y]
        if heads:
            blocks.append({"type": "head", "text": " ".join(text(l) for l in heads)})
            lines = [l for l in lines if l["bbox"][1] >= HEAD_Y]
    # 2. chapter / section opener: drop the numeral + title lines
    if n in CHAPTERS or n in SECTIONS:
        if n in CHAPTERS:
            k, title = CHAPTERS[n]
            blocks.append({"type": "chapter", "n": k, "title": title})
        else:
            blocks.append({"type": "section", "title": SECTIONS[n]})
        # chapter openers: numeral + display title down to y~400, body from y>=630;
        # section openers (Series Foreword, Preface): title only, body from y~550
        cut = 600 if n in CHAPTERS else 400
        lines = [l for l in lines if l["bbox"][1] > cut]
    # 3. figures: drop lines inside figure and caption boxes
    body_lines = []
    for l in lines:
        if any(inside(l["bbox"], f["bbox"]) for f in figs):
            continue
        if any(f.get("caption_bbox") and inside(l["bbox"], f["caption_bbox"]) for f in figs):
            continue
        body_lines.append(l)
    # scanner flecks OCR'd as a line of their own ("<)", ".", "*") would make the
    # next line look tight-leaded
    lines = [l for l in body_lines if not (all(not re.search(r"\w", w["t"]) for w in l["words"])
                                          and l["bbox"][3] - l["bbox"][1] < 24)]
    margin, right, x_size, pitch = page_metrics(lines) if lines else (0, 1500, 40, 62)
    # 4. classify lines
    items = []
    prev = None
    for l in lines:
        gap = (l["base"] - prev["base"]) / pitch if prev else None
        item = {"line": l, "gap": gap, "indent": text_x0(l) > margin + INDENT,
                "short": l["bbox"][2] < right - 60, "conf": conf(l)}
        items.append(item)
        prev = l
    # tight-leading runs (quotes / lists): a line whose step from the previous is < 0.9 pitch
    for i, it in enumerate(items):
        it["tight"] = it["gap"] is not None and it["gap"] < 0.9
    # 5. build blocks
    cur = None
    emitted_figs = set()
    def flush():
        nonlocal cur
        if cur and cur["lines"]:
            blocks.append(cur)
        cur = None
    for i, it in enumerate(items):
        l = it["line"]
        # figures above this line get emitted first (reading order)
        for fi, f in enumerate(figs):
            if fi not in emitted_figs and f["bbox"][3] < l["bbox"][1]:
                flush(); emitted_figs.add(fi)
                blocks.append({"type": "figure", **{k: f.get(k) for k in ("file", "kind", "caption_title", "caption_text", "bbox", "shared_caption")}})
        t = text(l)
        key = norm(t)
        nxt = items[i + 1] if i + 1 < len(items) else None
        is_gap = it["gap"] is None or it["gap"] > 1.5
        # subhead: known text, or geometric candidate
        known = known_subhead(t) if len(l["words"]) <= 8 else None
        if known:
            flush(); blocks.append({"type": "subhead", "text": known, "lines": [l]}); continue
        if (is_gap and it["gap"] is not None and it["short"] and len(l["words"]) <= 8
                and not re.search(r"[.:,;]$", t) and nxt and not nxt["indent"]
                and (nxt["gap"] or 0) < 1.5 and not it["tight"] and not (nxt and nxt["tight"])
                and t[:1].isupper() and l["bbox"][2] - l["bbox"][0] < 700):
            flush(); blocks.append({"type": "subhead", "text": t, "lines": [l], "flag": True}); continue
        # tight-leading run: quote or numbered list
        tight_run = it["tight"] or (nxt is not None and nxt["tight"] and is_gap)
        if tight_run:
            is_list = bool(re.match(r"^\d{1,2}\.\s", t)) or (cur and cur["type"] == "list" and not is_gap)
            btype = "list" if is_list else "quote"
            if cur is None or cur["type"] != btype or (is_gap and btype == "list" and re.match(r"^\d{1,2}\.\s", t)):
                flush(); cur = {"type": btype, "lines": [], "paras": [],
                                "continues": i == 0 and not it["indent"] and not any(b["type"] in ("chapter", "section") for b in blocks)}
            if is_gap or it["indent"] or re.match(r"^\d{1,2}\.\s", t):
                cur["paras"].append(len(cur["lines"]))   # paragraph start inside the block
            cur["lines"].append(l); continue
        # ordinary paragraph
        if cur is None or cur["type"] != "para" or it["indent"] or (is_gap and it["gap"] is not None):
            new = {"type": "para", "lines": [], "indent": it["indent"],
                   "continues": (i == 0 and not it["indent"] and cur is None and not any(b["type"] in ("chapter", "section") for b in blocks))}
            flush(); cur = new
        cur["lines"].append(l)
    flush()
    for fi, f in enumerate(figs):
        if fi not in emitted_figs:
            blocks.append({"type": "figure", **{k: f.get(k) for k in ("file", "kind", "caption_title", "caption_text", "bbox", "shared_caption")}})
    shared = [b for b in blocks if b["type"] == "figure" and b.get("shared_caption")]
    if shared:
        shared[-1]["last_of_group"] = True
    # the page's last paragraph may continue on the next page
    if blocks and blocks[-1]["type"] in ("para", "quote", "list"):
        last = blocks[-1]["lines"][-1]
        # a quote/list ending at the page bottom may continue with another
        # paragraph overleaf; the assembler decides by looking at the next page
        blocks[-1]["open"] = (blocks[-1]["type"] != "para" or last["bbox"][2] >= right - 60
                              or not re.search(r'[.!?"’)]$', text(last)))
    out = {"page": n, "margin": margin, "right": right, "pitch": pitch, "x_size": x_size, "blocks": blocks}
    if dump:
        print(f"===== p{n}  margin={margin:.0f} right={right} pitch={pitch:.0f}")
        for b in blocks:
            if b["type"] in ("para", "quote", "list", "subhead"):
                flag = " [FLAG]" if b.get("flag") else ""
                cont = " (cont)" if b.get("continues") else ""
                opn = " (open)" if b.get("open") else ""
                print(f"[{b['type']}{flag}{cont}{opn}]")
                for j, l in enumerate(b["lines"]):
                    mark = "¶ " if j in b.get("paras", []) else "  "
                    print("   ", mark + text(l))
            else:
                print(f"[{b['type']}] {b.get('title') or b.get('text') or b.get('caption_title')}")
    return out

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dump = "--dump" in sys.argv
    pages = [int(a) for a in args] or BODY
    figs_by_page = load_figures()
    outdir = pathlib.Path("out/blocks"); outdir.mkdir(parents=True, exist_ok=True)
    nflag = 0
    for n in pages:
        res = layout_page(n, figs_by_page, dump)
        nflag += sum(1 for b in res["blocks"] if b.get("flag"))
        (outdir / f"p-{n:03d}.json").write_text(json.dumps(res))
    print(f"laid out {len(pages)} pages; {nflag} flagged subhead candidates", file=sys.stderr)
