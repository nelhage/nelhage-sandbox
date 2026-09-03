"""Stage 5: figure crops + captions -> out/figures/.

For every row of out/survey/figures.tsv (page-fraction bboxes, accurate to ~0.02
of the page) this script

  1. refines the bbox on the 300 dpi render: the ink/tone region inside a
     search window (below the running head or the previous caption, above the
     first caption line, split from a neighbouring figure at the emptiest gap),
     unioned with the survey box, then clamped so no caption/head text is inside;
  2. finds the caption directly below the figure in out/hocr: the first
     high-confidence line at the text margin below the figure (bold title) and
     the tight-pitch paragraph that follows it, up to the next figure, the body
     text (wider pitch / gap) or the page end.  A figure with no caption of its
     own before the next figure shares the next figure's caption
     ("(above)/(below)"), and both entries carry shared_caption=true;
  3. crops, level-stretches (paper -> white, ink -> black, robust percentiles so
     photos keep their tones), downsamples photos to 200 dpi (line art stays at
     300 dpi) and writes pNNN-k.png, k = 1.. in reading order;
  4. writes figures.json, flags.json, sheet.png (thumbnails of every crop) and
     sheet_captions.png (300 dpi crops of every caption region).

Usage: python3 figures.py [pages...]     (default: every page in figures.tsv)
"""
import json, pathlib, re, statistics, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from lxml import html

PAGES = pathlib.Path("out/pages")
HOCR = pathlib.Path("out/hocr")
TSV = pathlib.Path("out/survey/figures.tsv")
OUT = pathlib.Path("out/figures")
FLAGDIR = OUT / "flagcrops"

HEAD_Y = 175          # running head top is at 62-125 px; body/caption lines start >= 189
INK = 100             # v < INK is ink (paper renders at ~140-150)
TONE_DEV = 12         # |smoothed - paper| > TONE_DEV is photo tone
MIN_CC = 40           # ignore ink components smaller than this (dust)
MARGIN = 6            # px of paper kept around the ink region
CAP_CONF = 85         # a caption line has mean x_wconf >= this
PHOTO_DPI = 200
CAPTION_PITCH = 47    # caption line pitch; body is ~62
MAX_FILE = 400_000    # bytes; photos are re-encoded with fewer levels above this

# small-caps acronyms; tesseract reads them in random case and often doubles a
# letter in the other case ("sSeEc", "XxD-1", "rRoOs", "AN/FsQ-7")
LEXICON = ["ASCC", "SSEC", "MTC", "SAGE", "UNIVAC", "ENIAC", "EDVAC", "NPL", "CCROS",
           "TROS", "BCROS", "BPS", "BOS", "SMS", "SLT", "ASCA", "EDPM", "ERA", "IBM",
           "MIT", "RCA", "XD-1", "XD-2", "ROS", "ROSDR", "IRE", "AN/FSQ-7"]
NEVER_LOWER = {"ERA", "SAGE", "BOS", "BPS"}      # real words too: need >= 2 capitals


def _lex_pattern(w):
    return "".join(f"[{c.lower()}{c}]{{1,2}}" if c.isalpha() else re.escape(c) for c in w)


LEX_RE = re.compile(r"(?<![A-Za-z])(" + "|".join(_lex_pattern(w) for w in sorted(LEXICON, key=len, reverse=True))
                    + r")(s?)(?![A-Za-z])")


def _lex_fix(m):
    t = m.group(1)
    letters = re.sub(r"[^A-Za-z]", "", t)
    # collapse doubled letters back to the lexicon spelling
    for w in LEXICON:
        wl = re.sub(r"[^A-Za-z]", "", w)
        if re.fullmatch("".join(c + "{1,2}" for c in wl), letters.upper()):
            canon = w
            break
    else:
        return m.group(0)
    ups = sum(c.isupper() for c in t)
    specific = not canon.isalpha()          # XD-1, AN/FSQ-7
    if ups < 2 and not specific and (canon in NEVER_LOWER or not (t[0].isupper() and len(letters) >= 4)):
        return m.group(0)
    return canon + m.group(2).lower()


# ----------------------------------------------------------------------------- inputs
def read_tsv():
    rows = []
    for line in TSV.read_text().splitlines()[1:]:
        f = line.split("\t")
        if not f[0].isdigit() or f[1].startswith("table"):
            continue
        rows.append({"page": int(f[0]), "tsv_kind": f[1], "tsv_title": f[2], "tsv_note": f[3],
                     "frac": [float(v) for v in f[4:8]]})
    return rows


def bbox_of(el):
    m = re.search(r"bbox (\d+) (\d+) (\d+) (\d+)", el.get("title", ""))
    return [int(v) for v in m.groups()]


def parse_hocr(path, a=None):
    """-> lines: {bbox, base, x_size, words:[{t,c,bbox}], text, conf}; `a` is the
    page array (used to spot hyphens between row fragments)."""
    tree = html.parse(str(path))
    lines = []
    for ln in tree.xpath('//span[@class="ocr_line" or @class="ocr_header" or @class="ocr_caption" or @class="ocr_textfloat"]'):
        title = ln.get("title", "")
        bb = bbox_of(ln)
        m = re.search(r"baseline ([-\d.]+) ([-\d.]+)", title)
        off = float(m.group(2)) if m else 0.0
        m = re.search(r"x_size ([\d.]+)", title)
        xs = float(m.group(1)) if m else 40.0
        words = []
        for w in ln.xpath('.//span[@class="ocrx_word"]'):
            t = w.text_content().strip()
            if not t:
                continue
            c = int(re.search(r"x_wconf (\d+)", w.get("title", "")).group(1))
            words.append({"t": t, "c": c, "bbox": bbox_of(w)})
        if not words:
            continue
        lines.append({"bbox": bb, "base": bb[3] + off, "x_size": xs, "words": words,
                      "text": " ".join(w["t"] for w in words),
                      "conf": statistics.mean(w["c"] for w in words)})
    lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    return merge_rows(lines, a)


def hyphen_in(a, y0, y1, xa, xb):
    """Is there a printed hyphen (a short horizontal bar in the middle of the
    x-height) in a[y0:y1, xa:xb]?  tesseract drops or detaches line-final
    hyphens in the small caption type."""
    if a is None or xb - xa < 6 or y1 - y0 < 8:
        return False
    reg = a[y0:y1, xa:xb] < INK
    h = y1 - y0
    rows = [y for y in range(reg.shape[0])
            if any(reg[y, i:i + 6].all() for i in range(reg.shape[1] - 5))]
    return bool(rows) and len(rows) <= 8 and all(0.25 * h <= y <= 0.8 * h for y in rows)


def merge_rows(lines, a=None):
    """tesseract sometimes splits one printed line into two side-by-side ocr_lines
    (p236: "Transformer Read" + "-Only Store (TROS)"); join fragments that share a row."""
    lines = sorted(lines, key=lambda l: l["bbox"][0])
    out = []
    for l in lines:
        for m in out:
            h = min(m["bbox"][3] - m["bbox"][1], l["bbox"][3] - l["bbox"][1])
            ov = min(m["bbox"][3], l["bbox"][3]) - max(m["bbox"][1], l["bbox"][1])
            if h <= 90 and ov > 0.6 * h and l["bbox"][0] >= m["bbox"][2] - 5 and l["bbox"][0] - m["bbox"][2] < 120:
                last, first = m["words"][-1], l["words"][0]
                gap = first["bbox"][0] - last["bbox"][2]
                if gap < 30 and (last["t"].endswith("-") or hyphen_in(a, m["bbox"][1], m["bbox"][3], last["bbox"][2] + 1, first["bbox"][0] - 1)):
                    # the row was split at a hyphen: "with U-" + "shaped", "ladder" + "shaped"
                    m["words"] = m["words"][:-1] + [{"t": last["t"].rstrip("-") + "-" + first["t"],
                                                     "c": min(last["c"], first["c"]),
                                                     "bbox": [last["bbox"][0], min(last["bbox"][1], first["bbox"][1]),
                                                              first["bbox"][2], max(last["bbox"][3], first["bbox"][3])]}] + l["words"][1:]
                else:
                    m["words"] = m["words"] + l["words"]
                m["bbox"] = [m["bbox"][0], min(m["bbox"][1], l["bbox"][1]), l["bbox"][2], max(m["bbox"][3], l["bbox"][3])]
                m["text"] = " ".join(w["t"] for w in m["words"])
                m["conf"] = statistics.mean(w["c"] for w in m["words"])
                break
        else:
            out.append(dict(l))
    out.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    return out


# ----------------------------------------------------------------------------- text
def normalise(s):
    s = re.sub(r"(?<!\w)['\"‘’“”]{1,3}(?=\w)", '"', s)      # opening quote runs
    s = re.sub(r"(?<=[\w.,;:!?)])['\"‘’“”]{2,3}(?!\w)", '"', s)  # closing runs of 2-3
    s = re.sub(r"(?<=[\w.,;:!?)])[\"“”](?!\w)", '"', s)      # single double-quote glyphs
    s = re.sub(r"(?<=\w)'(?=\w)", "’", s)                     # apostrophe inside a word
    s = re.sub(r"(?<=\d)\s*[x×]\s*(?=\d)", "×", s)           # 4x4x4 -> 4×4×4
    s = LEX_RE.sub(_lex_fix, s)
    s = re.sub(r"(?<![A-Za-z])[/\[J]BM(?= (Journal|News|Systems))", "IBM", s)   # italic IBM
    s = re.sub(r"(?<![A-Za-z])\[t(?=\s)", "It", s)                              # fleck + t
    s = re.sub(r"(\w) -(\w)", r"\1-\2", s)                  # merged row fragments: "Read -Only"
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ----------------------------------------------------------------------------- geometry
def page_margin(lines):
    xs = [l["bbox"][0] for l in lines if l["conf"] >= CAP_CONF and len(l["words"]) >= 4 and l["bbox"][1] > HEAD_Y]
    if not xs:
        xs = [l["bbox"][0] for l in lines if l["bbox"][1] > HEAD_Y] or [160]
    xs.sort()
    lo = xs[max(0, int(len(xs) * 0.15))]
    return statistics.median([x for x in xs if x <= lo + 30])


def text_like(l, margin):
    """A line that could be a caption line: confident, at the column margin, text-sized."""
    return (l["conf"] >= CAP_CONF and abs(l["bbox"][0] - margin) <= 40
            and 22 <= l["x_size"] <= 60 and l["bbox"][3] - l["bbox"][1] <= 90)


def paper_level(a):
    """Paper grey from the page frame (never the photo): median of a 60 px ring
    set 30 px in from every edge."""
    H, W = a.shape
    ring = np.concatenate([a[30:90, :].ravel(), a[H - 90:H - 30, :].ravel(),
                           a[30:H - 30, 30:90].ravel(), a[30:H - 30, W - 90:W - 30].ravel()])
    return float(np.median(ring))


def ink_masks(a, paper):
    ink = a < INK
    sm = ndimage.uniform_filter(a.astype(np.float32), 21)
    tone = np.abs(sm - paper) > TONE_DEV
    return ink, tone


def region_bbox(ink, tone, win):
    """bbox of ink/tone inside win=[x0,y0,x1,y1] (ignoring small specks), or None."""
    x0, y0, x1, y1 = [int(v) for v in win]
    sub = ink[y0:y1, x0:x1].copy()
    e = 14      # the smoothed tone mask smears text by ~10 px: keep it off the window edges
    if y1 - y0 > 2 * e and x1 - x0 > 2 * e:
        sub[e:-e, e:-e] |= tone[y0 + e:y1 - e, x0 + e:x1 - e]
    lab, n = ndimage.label(sub)
    if n == 0:
        return None
    sizes = ndimage.sum(sub, lab, range(1, n + 1))
    keep = np.isin(lab, np.nonzero(sizes >= MIN_CC)[0] + 1)
    if not keep.any():
        return None
    ys, xs = np.nonzero(keep)
    return [x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1]


def emptiest_row(ink, x0, x1, ya, yb):
    """centre of the longest run of ink-free rows in [ya, yb) between columns x0..x1."""
    prof = ink[ya:yb, x0:x1].sum(axis=1)
    best, cur, bend = 0, 0, ya
    for i, v in enumerate(prof):
        cur = cur + 1 if v <= 1 else 0
        if cur > best:
            best, bend = cur, ya + i + 1
    return (bend - best / 2) if best else (ya + yb) / 2


def emptiest_col(ink, y0, y1, xa, xb):
    prof = ink[y0:y1, xa:xb].sum(axis=0)
    best, cur, bend = 0, 0, xa
    for i, v in enumerate(prof):
        cur = cur + 1 if v <= 1 else 0
        if cur > best:
            best, bend = cur, xa + i + 1
    return (bend - best / 2) if best else (xa + xb) / 2


def y_overlap(a, b):
    return max(0, min(a[3], b[3]) - max(a[1], b[1])) / max(1, min(a[3] - a[1], b[3] - b[1]))


# ----------------------------------------------------------------------------- captions
def caption_block(lines, margin, y_from, y_to, x_from=0, x_to=10 ** 6):
    """Caption starting with the first text-like line whose top is in [y_from, y_to):
    the title line plus the following tight-pitch lines.  Returns (lines) or []."""
    cands = [l for l in lines if text_like(l, margin) and y_from <= l["bbox"][1] < y_to
             and x_from - 40 <= l["bbox"][0] <= x_to]
    if not cands:
        return []
    first = cands[0]
    block = [first]
    # the bold title (small caps, e.g. "SSEC memory components") may score < CAP_CONF:
    # walk upward over margin-aligned text-sized lines with caption pitch
    above = [l for l in lines if l["bbox"][3] <= first["bbox"][1] + 5 and l["bbox"][1] >= y_from
             and abs(l["bbox"][0] - margin) <= 60 and l["x_size"] <= 60
             and l["bbox"][3] - l["bbox"][1] <= 90 and l["conf"] >= 40]
    while above:
        prev = max(above, key=lambda l: l["base"])
        step = block[0]["base"] - prev["base"]
        if not (25 <= step <= 1.3 * CAPTION_PITCH):
            break
        block.insert(0, prev)
        above = [l for l in above if l["base"] < prev["base"] - 20]
    first = block[0]
    for l in lines:
        if l["bbox"][1] <= block[-1]["bbox"][1] or l in block:
            continue
        if l["bbox"][1] >= y_to:
            break
        step = l["base"] - block[-1]["base"]
        if step <= 0:
            continue
        if step > 1.3 * CAPTION_PITCH:           # gap or body pitch: the caption is over
            break
        if l["conf"] < 60 or abs(l["bbox"][0] - margin) > 60 or l["x_size"] > 60:
            break
        block.append(l)
    return block


SUSPECT = re.compile(r"(?<![A-Za-z])[xy][,;.]{1,2}(?=\s|$)|[^\x20-\x7e’“”‘×—–©°]|\b[A-Z][A-Z][a-z]{3,}"
                     r"|\b[A-Za-z]*[A-Z][a-z][A-Z][A-Za-z]*\b")


def ends_with_hyphen(a, line):
    """Is there a printed hyphen right after the last word (tesseract drops or
    detaches line-final hyphens in the small caption type)?"""
    x0, y0, x1, y1 = line["words"][-1]["bbox"]
    return line["text"].endswith("-") or hyphen_in(a, y0, y1, x1 + 1, x1 + 40)


def join_lines(a, lines):
    out = ""
    for i, l in enumerate(lines):
        t = l["text"].strip()
        if i < len(lines) - 1 and ends_with_hyphen(a, l):
            out += t.rstrip("-") + "-"
        else:
            out += t + " "
    return out.strip()


def caption_text(block, a):
    title = normalise(block[0]["text"])
    if title.endswith(" )") and "(" not in title:               # stray fleck read as ")"
        title = title[:-2].rstrip()
    title = re.sub(r"^[-–—.,;:'\"]+\s*(?=[A-Za-z])", "", title)   # flecks read as punctuation
    title = re.sub(r"\s+[-–—.,;:]$", "", title)
    body = normalise(join_lines(a, block[1:]))
    return title, body


def union(bbs):
    return [min(b[0] for b in bbs), min(b[1] for b in bbs), max(b[2] for b in bbs), max(b[3] for b in bbs)]


# ----------------------------------------------------------------------------- image
def stretch(a, paper):
    """paper -> white, ink -> black; percentiles so a photo keeps its tones."""
    lo = min(float(np.percentile(a, 0.5)), 40.0)
    hi = max(float(paper), float(np.percentile(a, 99.5)))
    hi = min(hi, paper + 25)      # a highlight brighter than paper is still white
    f = (a.astype(np.float32) - lo) / max(1.0, hi - lo)
    return (np.clip(f, 0, 1) * 255 + 0.5).astype(np.uint8)


def save_png(img, path):
    """8-bit grey PNG; if it is still over MAX_FILE, halve the number of grey
    levels (halftone noise compresses badly) until it fits."""
    img.save(path, optimize=True)
    levels = 128
    while path.stat().st_size > MAX_FILE and levels >= 16:
        q = np.array(img).astype(np.float32)
        step = 256 / levels
        q = np.clip(np.round(q / step) * step, 0, 255).astype(np.uint8)
        Image.fromarray(q).save(path, optimize=True)
        levels //= 2


# ----------------------------------------------------------------------------- per page
def process_page(n, rows, flags):
    im = Image.open(PAGES / f"p-{n:03d}.png").convert("L")
    a = np.array(im)
    H, W = a.shape
    paper = paper_level(a)
    ink, tone = ink_masks(a, paper)
    lines = parse_hocr(HOCR / f"p-{n:03d}.hocr", a)
    margin = page_margin(lines)
    head_bottom = max([l["bbox"][3] for l in lines if l["bbox"][1] < HEAD_Y and l["bbox"][3] < HEAD_Y + 30] + [0])
    top_limit = max(head_bottom + 30, 130)
    col_x0, col_x1 = int(W * 0.04), int(W * 0.96)

    figs = []
    for r in rows:
        x0, y0, x1, y1 = r["frac"]
        figs.append({**r, "survey": [x0 * W, y0 * H, x1 * W, y1 * H]})
    figs.sort(key=lambda f: (f["survey"][1], f["survey"][0]))

    # group side-by-side figures (strong vertical overlap) into bands
    bands = []
    for f in figs:
        if bands and y_overlap(bands[-1][-1]["survey"], f["survey"]) > 0.5:
            bands[-1].append(f)
        else:
            bands.append([f])
    for b in bands:
        b.sort(key=lambda f: f["survey"][0])

    # 1. captions per band: first text-like line below the band's survey bottom,
    #    before the next band's survey top.  None -> shared with the next band.
    for i, b in enumerate(bands):
        sv = union([f["survey"] for f in b])
        nxt = bands[i + 1] if i + 1 < len(bands) else None
        y_to = (min(f["survey"][1] for f in nxt) + 60) if nxt else H
        blk = caption_block(lines, margin, sv[3] - 60, y_to)
        bands[i] = (b, {"caption": blk, "survey": sv})

    # 2. windows and refined bboxes
    prev_bottom = top_limit
    for i, (b, info) in enumerate(bands):
        sv = info["survey"]
        cap = info["caption"]
        if cap:
            y_stop = cap[0]["bbox"][1] - 4
        elif i + 1 < len(bands):
            nsv = bands[i + 1][1]["survey"]
            y_stop = emptiest_row(ink, col_x0, col_x1, int(sv[3] - 60), int(nsv[1] + 60))
        else:
            y_stop = H - int(H * 0.04)
        info["win_y"] = (prev_bottom, y_stop)
        # horizontal windows for side-by-side figures
        xs = []
        for j, f in enumerate(b):
            xa = col_x0 if j == 0 else emptiest_col(ink, int(sv[1]), int(y_stop), int(b[j - 1]["survey"][2] - 30), int(f["survey"][0] + 30))
            xb = col_x1 if j == len(b) - 1 else emptiest_col(ink, int(sv[1]), int(y_stop), int(f["survey"][2] - 30), int(b[j + 1]["survey"][0] + 30))
            xs.append((xa, xb))
        for f, (xa, xb) in zip(b, xs):
            win = [xa, prev_bottom, xb, y_stop]
            f["win"] = win
            reg = region_bbox(ink, tone, win)
            f["ink_bbox"] = reg
            if reg is None:
                bb = [int(v) for v in f["survey"]]
                flags.append(flag(n, bb, "", f"no ink found in window for {f['tsv_title']}", f))
            else:
                # union with the survey box (it undershoots on light photos), then clamp
                s = f["survey"]
                bb = [min(reg[0], s[0]), min(reg[1], s[1]), max(reg[2], s[2]), max(reg[3], s[3])]
                bb = [max(bb[0], xa), max(bb[1], prev_bottom), min(bb[2], xb), min(bb[3], y_stop)]
                # ... but the survey box is only accurate to ~0.02 of the page, so never let it
                # push a side more than 30 px beyond the ink actually found
                bb = [max(bb[0], reg[0] - 30), max(bb[1], reg[1] - 30), min(bb[2], reg[2] + 30), min(bb[3], reg[3] + 30)]
            bb = [int(bb[0]) - MARGIN, int(bb[1]) - MARGIN, int(bb[2]) + MARGIN, int(bb[3]) + MARGIN]
            bb = [max(bb[0], 0), max(bb[1], int(prev_bottom)), min(bb[2], W), min(bb[3], int(y_stop))]
            f["bbox"] = bb
            # sanity: how far the refined box moved from the survey box
            dev = max(abs(bb[k] - f["survey"][k]) for k in range(4)) / H
            if dev > 0.06:
                fl = flag(n, bb, "", f"check crop: refined bbox differs from survey by {dev:.2f} of page ({f['tsv_title']})", f)
                fl["crop"] = f"out/figures/p{n:03d}-{{k}}.png"   # k filled in below
                flags.append(fl)
        if cap:
            prev_bottom = cap[-1]["bbox"][3] + 8
        else:
            prev_bottom = y_stop

    # 3. captions: assign, shared when a band has none of its own
    results = []
    k = 0
    for i, (b, info) in enumerate(bands):
        cap = info["caption"]
        shared = len(b) > 1
        j = i
        while not cap and j + 1 < len(bands):
            j += 1
            cap = bands[j][1]["caption"]
            shared = True
        if cap and j > i:
            bands[j][1]["shared_with_prev"] = True
        shared = shared or bool(info.get("shared_with_prev"))
        for f in b:
            k += 1
            f["k"] = k
            if cap:
                title, body = caption_text(cap, a)
                cbb = union([l["bbox"] for l in cap])
                for l in cap:
                    if l["conf"] < 80:
                        flags.append(flag(n, l["bbox"], l["text"], f"caption line conf {l['conf']:.0f} < 80", f, crop_line=(a, l)))
                    elif any(w["c"] < 65 and w["t"] in normalise(l["text"]) for w in l["words"]):
                        # low-confidence words that the normaliser did not already rewrite
                        bad = [w["t"] for w in l["words"] if w["c"] < 65 and w["t"] in normalise(l["text"])]
                        flags.append(flag(n, l["bbox"], l["text"], f"caption word conf < 65: {' '.join(bad)}", f, crop_line=(a, l)))
                    elif SUSPECT.search(normalise(l["text"])):
                        flags.append(flag(n, l["bbox"], l["text"], "suspect token in caption line (lost subscript / odd glyph): "
                                          + SUSPECT.search(normalise(l["text"])).group(0), f, crop_line=(a, l)))
            else:
                title, body, cbb = "", "", None
                flags.append(flag(n, f["bbox"], "", f"no caption found for {f['tsv_title']}", f))
            for fl in flags:
                if fl["page"] == n and fl["bbox"] == f["bbox"]:
                    fl["crop"] = fl["crop"].replace("{k}", str(k))
            kind = "photo" if f["tsv_kind"] == "photo" else "lineart"
            f["kind"] = kind
            f["caption_lines"] = cap
            results.append({"page": n, "k": k, "file": f"out/figures/p{n:03d}-{k}.png", "bbox": f["bbox"],
                            "kind": kind, "caption_title": title, "caption_text": body,
                            "caption_bbox": cbb, "shared_caption": bool(shared and cap),
                            "survey_title": f["tsv_title"]})
    # 4. crops
    for f, r in zip(sorted(figs, key=lambda f: f["k"]), results):
        x0, y0, x1, y1 = f["bbox"]
        crop = stretch(a[y0:y1, x0:x1], paper)
        img = Image.fromarray(crop)
        if f["kind"] == "photo":
            s = PHOTO_DPI / 300
            img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)
        save_png(img, OUT / f"p{n:03d}-{r['k']}.png")
        r["px"] = [img.width, img.height]
    return results, figs, a, paper


def flag(n, bbox, draft, reason, f, crop_line=None):
    FLAGDIR.mkdir(parents=True, exist_ok=True)
    name = f"p{n:03d}-{len(list(FLAGDIR.glob(f'p{n:03d}-*.png'))) + 1}.png"
    path = FLAGDIR / name
    if crop_line is not None:
        a, l = crop_line
        x0, y0, x1, y1 = l["bbox"]
        Image.fromarray(a[max(0, y0 - 10):y1 + 10, max(0, x0 - 10):x1 + 10]).save(path)
    else:
        path = None
    return {"page": n, "bbox": [int(v) for v in bbox], "crop": str(path) if path else "",
            "draft": draft, "reason": reason,
            "context": {"survey_title": f["tsv_title"], "survey_bbox": [int(v) for v in f["survey"]]}}


# ----------------------------------------------------------------------------- sheets
def font(size):
    import subprocess
    try:
        path = subprocess.run(["fc-match", "-f", "%{file}", "DejaVu Sans"], capture_output=True, text=True).stdout
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default(size=size)


def contact_sheet(results, thumb_w=300, cols=6):
    fnt, fnt2 = font(15), font(12)
    cells = []
    for r in results:
        img = Image.open(r["file"]).convert("L")
        s = thumb_w / img.width
        t = img.resize((thumb_w, max(1, round(img.height * s))), Image.LANCZOS)
        cells.append((r, t))
    rows = (len(cells) + cols - 1) // cols
    row_h = [max(t.height for _, t in cells[r * cols:(r + 1) * cols]) + 44 for r in range(rows)]
    row_y = [8 + sum(row_h[:r]) + 8 * r for r in range(rows)]
    sheet = Image.new("L", (cols * (thumb_w + 12) + 12, row_y[-1] + row_h[-1] + 8), 90)
    d = ImageDraw.Draw(sheet)
    for i, (r, t) in enumerate(cells):
        cx = 12 + (i % cols) * (thumb_w + 12)
        cy = row_y[i // cols]
        d.rectangle([cx - 2, cy - 2, cx + thumb_w + 1, cy + t.height + 1], outline=255)
        sheet.paste(t, (cx, cy))
        d.text((cx, cy + t.height + 4), f"p{r['page']} k{r['k']}  {r['kind']}  {r['px'][0]}x{r['px'][1]}"
               + ("  SHARED" if r["shared_caption"] else ""), fill=255, font=fnt)
        title = r["caption_title"] or "(no caption)"
        d.text((cx, cy + t.height + 24), title[:44], fill=230, font=fnt2)
    return sheet


def caption_sheet(results, pages_arrays, cols=2, pad=10):
    fnt = font(22)
    cells = []
    for r in results:
        a = pages_arrays[r["page"]]
        if r["caption_bbox"]:
            x0, y0, x1, y1 = r["caption_bbox"]
            crop = a[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
            crop = stretch(crop, paper_level(a))
            img = Image.fromarray(crop)
        else:
            img = Image.new("L", (600, 60), 255)
        cells.append((r, img))
    col_w = max(img.width for _, img in cells) + 20
    heights = [img.height + 36 for _, img in cells]
    per_col = (len(cells) + cols - 1) // cols
    col_heights = [sum(heights[c * per_col:(c + 1) * per_col]) for c in range(cols)]
    sheet = Image.new("L", (cols * col_w + 10, max(col_heights) + 10), 200)
    d = ImageDraw.Draw(sheet)
    for c in range(cols):
        y = 10
        for (r, img) in cells[c * per_col:(c + 1) * per_col]:
            x = 10 + c * col_w
            label = f"p{r['page']} k{r['k']}" + ("  (shared)" if r["shared_caption"] else "") + f"   [{r['survey_title']}]"
            d.text((x, y), label, fill=0, font=fnt)
            sheet.paste(img, (x, y + 28))
            d.rectangle([x - 1, y + 27, x + img.width, y + 28 + img.height], outline=0)
            y += img.height + 36
    return sheet


# ----------------------------------------------------------------------------- main
if __name__ == "__main__":
    want = [int(p) for p in sys.argv[1:]]
    rows = read_tsv()
    by_page = {}
    for r in rows:
        by_page.setdefault(r["page"], []).append(r)
    pages = [p for p in sorted(by_page) if not want or p in want]
    OUT.mkdir(parents=True, exist_ok=True)
    old_results, old_flags = [], []
    if want and (OUT / "figures.json").exists():      # page mode: keep the other pages' entries
        old_results = [r for r in json.loads((OUT / "figures.json").read_text()) if r["page"] not in want]
        old_flags = [f for f in json.loads((OUT / "flags.json").read_text()) if f["page"] not in want]
    for old in list(OUT.glob("p*.png")) + list(FLAGDIR.glob("*.png")):
        if not want or int(old.name[1:4]) in want:
            old.unlink()
    all_results, flags, arrays = [], [], {}
    for n in pages:
        res, figs, a, paper = process_page(n, by_page[n], flags)
        all_results.extend(res)
        arrays[n] = a
        for r in res:
            print(f"p{n} k{r['k']} {r['kind']:7s} bbox={r['bbox']} cap={r['caption_bbox']} "
                  f"{'SHARED ' if r['shared_caption'] else ''}| {r['caption_title'][:50]}", flush=True)
    json_out = [{k: v for k, v in r.items() if k not in ("px", "survey_title")} for r in all_results]
    json_out = sorted(old_results + json_out, key=lambda r: (r["page"], r["k"]))
    (OUT / "figures.json").write_text(json.dumps(json_out, indent=1, ensure_ascii=False))
    flags = old_flags + flags
    # sheets always cover every figure
    order = {pg: [r["tsv_title"] for r in sorted(rs, key=lambda r: (r["frac"][1], r["frac"][0]))]
             for pg, rs in by_page.items()}                  # same k order as process_page
    all_results = []
    for r in json_out:
        im = Image.open(r["file"])
        all_results.append(dict(r, px=[im.width, im.height], survey_title=order[r["page"]][r["k"] - 1]))
    for r in json_out:
        if r["page"] not in arrays:
            arrays[r["page"]] = np.array(Image.open(PAGES / f"p-{r['page']:03d}.png").convert("L"))
    seen, uniq = set(), []
    for f in flags:
        key = (f["page"], tuple(f["bbox"]), f["reason"])
        if key not in seen:
            seen.add(key); uniq.append(f)
    flags = uniq
    (OUT / "flags.json").write_text(json.dumps(flags, indent=1, ensure_ascii=False))
    contact_sheet(all_results).save(OUT / "sheet.png", optimize=True)
    caption_sheet(all_results, arrays).save(OUT / "sheet_captions.png", optimize=True)
    print(f"{len(all_results)} figures on {len(pages)} pages; {len(flags)} flags", file=sys.stderr)
