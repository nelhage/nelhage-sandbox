"""Shared helpers for notes.py / chronology.py / index.py (stage 6).

hOCR parsing, text normalisation (quote runs, small-caps lexicon), de-hyphenated
line joining, 300 dpi crops and the merged out/backmatter/flags.json.
All coordinates are 300 dpi pixels on out/pages/p-NNN.png.
"""
import json, pathlib, re, statistics, subprocess
from xml.etree import ElementTree as ET
import numpy as np
from PIL import Image

OUT = pathlib.Path("out/backmatter")
CROPS = OUT / "crops"
PAGES = pathlib.Path("out/pages")
XHTML = "{http://www.w3.org/1999/xhtml}"
CONF_THR = 80


def page_png(p):
    return PAGES / f"p-{p:03d}.png"


# ---------------------------------------------------------------- hOCR

def _bbox(el):
    m = re.search(r"bbox (\d+) (\d+) (\d+) (\d+)", el.get("title", ""))
    return [int(v) for v in m.groups()]


def _conf(el):
    m = re.search(r"x_wconf (\d+)", el.get("title", ""))
    return int(m.group(1)) if m else 0


def hocr_lines(fn, dx=0, dy=0):
    """All non-empty ocr_line spans of an hOCR file as dicts, sorted by (y0, x0).

    {"bbox": [x0,y0,x1,y1], "words": [{"text","bbox","conf"}], "text", "conf"}
    dx/dy shift the boxes back into page coordinates for crops."""
    root = ET.parse(fn).getroot()
    lines = []
    for ln in root.iter(XHTML + "span"):
        if ln.get("class") not in ("ocr_line", "ocr_header", "ocr_caption", "ocr_textfloat"):
            continue
        words = []
        for w in ln.iter(XHTML + "span"):
            if w.get("class") != "ocrx_word":
                continue
            t = "".join(w.itertext()).strip()
            if not t:
                continue
            b = _bbox(w)
            words.append({"text": t, "bbox": [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy], "conf": _conf(w)})
        if not words:
            continue
        b = _bbox(ln)
        lines.append({"bbox": [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy], "words": words,
                      "text": " ".join(w["text"] for w in words),
                      "conf": statistics.mean(w["conf"] for w in words)})
    lines.sort(key=lambda l: (l["bbox"][1], l["bbox"][0]))
    return lines


def run_tesseract(png, outbase, psm, extra=()):
    """tesseract → outbase.hocr, cached (skipped when the hocr is newer than the png)."""
    outbase = pathlib.Path(outbase)
    hocr = outbase.with_suffix(".hocr")
    png = pathlib.Path(png)
    if hocr.exists() and hocr.stat().st_mtime >= png.stat().st_mtime:
        return hocr
    subprocess.run(["tesseract", str(png), str(outbase), "-l", "eng", "--psm", str(psm), *extra, "hocr"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return hocr


def page_lines(p, hocr_dir="out/hocr", ymin=150):
    """hOCR lines of PDF page p with the running head (y0 < ymin) dropped."""
    return [l for l in hocr_lines(f"{hocr_dir}/p-{p:03d}.hocr") if l["bbox"][1] >= ymin]


def running_head_bottom(p, hocr_dir="out/hocr", title=None):
    """y1 of the running head (any line with y0 < 150) or of the section title line
    (`title` regex) on the first page; 0 when neither is found."""
    y = 0
    for l in hocr_lines(f"{hocr_dir}/p-{p:03d}.hocr"):
        if l["bbox"][1] < 150 or (title and l["bbox"][1] < 400 and re.match(title, l["text"])):
            y = max(y, l["bbox"][3])
    return y


def strip_ocr(p, box, cache_dir, psm=6, whitelist="0123456789."):
    """Re-OCR a crop of the page (e.g. the note-number column) with a character
    whitelist; returns word dicts in page coordinates. Cached under cache_dir."""
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = box
    base = cache_dir / f"p-{p:03d}-{x0}-{y0}-{x1}-{y1}-psm{psm}"
    png = base.with_suffix(".png")
    if not png.exists():
        Image.open(page_png(p)).crop((x0, y0, x1, y1)).save(png)
    hocr = run_tesseract(png, base, psm, ["-c", f"tessedit_char_whitelist={whitelist}"] if whitelist else [])
    return [w for l in hocr_lines(hocr, dx=x0, dy=y0) for w in l["words"]]


def ink_array(p):
    return np.array(Image.open(page_png(p)).convert("L")) < 128


def find_gutter(ink, y0, y1, xmin=550, xmax=1050, frac=0.004):
    """Widest ink-free vertical band between xmin and xmax over rows y0..y1
    (a column is 'free' when < frac of its rows carry ink). Returns (gx0, gx1)."""
    col = ink[y0:y1, :].sum(axis=0)
    free = col <= frac * (y1 - y0)
    best, run, start = None, 0, None
    for x in range(xmin, xmax + 1):
        if free[x]:
            if start is None:
                start = x
            run = x - start + 1
            if best is None or run > best[1] - best[0]:
                best = (start, x + 1)
        else:
            start = None
    assert best and best[1] - best[0] >= 12, f"no gutter found in [{xmin},{xmax}]"
    return best


def ink_extent(ink, y0, y1, x0, x1, min_rows=3):
    """[xa, xb) span of columns with ink in the sub-rectangle; None if blank."""
    col = ink[y0:y1, x0:x1].sum(axis=0)
    xs = np.nonzero(col >= min_rows)[0]
    if len(xs) == 0:
        return None
    return x0 + int(xs[0]), x0 + int(xs[-1]) + 1


DATE_RE = re.compile(r"^(\d{1,2})/(\d{2})$|^(\d{4})$")


def date_key(tok):
    """Chronology date token -> (year, month) for ordering; None if not a date."""
    m = DATE_RE.match(tok)
    if not m:
        return None
    if m.group(3):
        return int(m.group(3)), 0
    return 1900 + int(m.group(2)), int(m.group(1))


# ---------------------------------------------------------------- text normalisation

# Small-caps words tesseract renders in random case. "safe" ones are normalised
# even when all-lowercase (they are not English words); the others only when the
# token already carries at least one capital (ERA/era, SAGE/sage, BOS...).
SMALLCAPS_SAFE = ["ASCC", "SSEC", "MTC", "UNIVAC", "ENIAC", "EDVAC", "EDSAC", "NPL", "CCROS",
                  "TROS", "BCROS", "BPS", "SLT", "ASCA", "EDPM", "IEEE", "BINAC", "SEAC", "SWAC",
                  "ORDVAC", "ILLIAC", "JOHNNIAC", "BMEWS", "EMCC", "CDC", "CPC", "NDRC", "OSRD",
                  "ONR", "IBM", "MIT", "CTR", "RCA", "MITRE"]
SMALLCAPS_GUARDED = ["SAGE", "ERA", "BOS", "SMS", "IAS", "RAND", "AEC", "MAC"]
# outright misreads of small-cap words seen in the survey
MISREADS = {"epvac": "EDVAC", "jeee": "IEEE", "1eee": "IEEE", "sacge": "SAGE"}
_SC_SAFE = {w.lower(): w for w in SMALLCAPS_SAFE}
_SC_GUARD = {w.lower(): w for w in SMALLCAPS_GUARDED}
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _fix_token(m):
    t = m.group(0)
    lo = t.lower()
    if lo in MISREADS:
        return MISREADS[lo]
    if lo in _SC_SAFE:
        return _SC_SAFE[lo]
    if lo in _SC_GUARD and t != lo and t != t.capitalize():
        return _SC_GUARD[lo]
    return t


QUOTE_RUN = re.compile(r"(?:[\"“”]|['‘’`]{2}|['‘’`](?=[\"“”]))(?:[\"“”'‘’`])*")


def normalize_text(s):
    """Global normalisations that need no eyes. Returns (text, hints).

    hints: things a reviewer must look at (not applied): `1.` that is probably `I.`
    (author initial), `Mark 1` etc."""
    hints = []
    s = QUOTE_RUN.sub('"', s)
    s = re.sub(r"(?<=\w)'(?=\w)", "’", s)            # apostrophes inside words
    s = re.sub(r"(?<=[sS])'(?=\s)", "’", s)          # plural possessives
    s = re.sub(r"(?<=[^\ssS])'(?=[\s,.;:)]|$)", '"', s)  # lone closing quote misread as '
    s = re.sub(r"(?:(?<=^)|(?<=[\s(]))'(?=[A-Z])", '"', s)  # lone opening quote
    s = _TOKEN.sub(_fix_token, s)
    s = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "×", s)          # 256 x 256 -> 256×256
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r'\s+([,.;:)])', r"\1", s)
    s = re.sub(r'"\s+(?=[,.;:])', '"', s)
    if re.search(r"(?<![\d/])\b1\.\s+[A-Z]\b", s):
        hints.append("initial `1.` is probably `I.`")
    if re.search(r"\b(Mark|War|Volume|Part)\s+(1|11|111)\b", s):
        hints.append("`1`/`11` after Mark/War is probably roman `I`/`II`")
    if re.search(r"\b[0-9]+[A-Za-z]+[0-9]*\b|\b[A-Za-z]+[0-9]+[A-Za-z]+\b", s) and not re.search(r"\b(RJ|AN/FSQ|G520|MC-?|IBM|System/|[A-Z]{1,4}\d+|\d+[A-Z]{1,3}\b|\d+s\b|\d+(st|nd|rd|th|d)\b)", s):
        hints.append("mixed digit/letter token")
    return s, hints


def join_lines(texts):
    """Join OCR lines; a hyphen at a line end is dropped when the next line starts
    lowercase (word break) and kept otherwise (35-/38, IBM-/MIT, Read-/Only)."""
    out = ""
    for t in texts:
        t = t.strip()
        if not t:
            continue
        if not out:
            out = t
        elif out.endswith("-") and not out.endswith(" -"):
            head = re.search(r"(\w+)-$", out)
            nxt = re.match(r"\w+", t)
            if t[:1].islower():
                out = out[:-1] + t
            elif head and nxt and (head.group(1) + nxt.group(0)).lower() in _SC_SAFE:
                out = out[:-1] + t                       # ED-/VAC -> EDVAC
            else:
                out = out + t
        else:
            out = out + " " + t
    return out


# ---------------------------------------------------------------- crops / flags

def crop(page, bbox, name, pad=10):
    """Save a 300 dpi crop of out/pages/p-NNN.png; returns the path (relative to cwd)."""
    CROPS.mkdir(parents=True, exist_ok=True)
    path = CROPS / name
    im = Image.open(page_png(page))
    x0, y0, x1, y1 = bbox
    box = (max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad))
    im.crop(box).save(path)
    return str(path)


def union_bbox(bboxes):
    return [min(b[0] for b in bboxes), min(b[1] for b in bboxes),
            max(b[2] for b in bboxes), max(b[3] for b in bboxes)]


class Flagger:
    """Collects escalation flags for one stage and writes flags_<stage>.json plus the
    merged flags.json (the union of all flags_*.json present)."""

    def __init__(self, stage):
        self.stage = stage
        self.flags = []
        self.n = 0

    def add(self, page, bbox, draft, reason, context=None):
        self.n += 1
        c = crop(page, bbox, f"{self.stage}-p{page:03d}-{self.n:03d}.png")
        self.flags.append({"page": page, "bbox": list(bbox), "crop": c, "draft": draft,
                           "reason": reason, "context": context or {}})
        return self.flags[-1]

    def write(self):
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"flags_{self.stage}.json").write_text(json.dumps(self.flags, indent=1, ensure_ascii=False))
        merged = []
        for f in sorted(OUT.glob("flags_*.json")):
            merged.extend(json.loads(f.read_text()))
        merged.sort(key=lambda f: (f["page"], f["bbox"][1]))
        (OUT / "flags.json").write_text(json.dumps(merged, indent=1, ensure_ascii=False))
        return merged

    def summary(self):
        from collections import Counter
        return Counter(f["page"] for f in self.flags)


def low_conf_words(line, thr=CONF_THR):
    return [w["text"] for w in line["words"] if w["conf"] < thr]
