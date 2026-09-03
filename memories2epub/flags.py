"""Stage 8a: choose the words that need human/vision review, and build the review
material for sub-agents.

Selection rule for a body word (see out/survey/body.md §3-4):
  * tesseract x_wconf < CONF            (2-4 % of words, ~55 % precision)
  * suspect pattern                     (superscript residue, glued digits, odd case,
                                          I/1 confusion, stray marks)
  * disagrees with the PDF's own text layer (Internet Archive tesseract) after
    normalisation and is below CONF_IA   (engine disagreement flags nearly every real
                                          error; harmless glyph variants are folded first)

Outputs
  out/flags/p-NNN.json   flagged words with reasons and the draft line text
  out/flags/p-NNN.png    one strip per flagged line, flagged words boxed
  out/flags/summary.tsv  per-page counts
Usage: python3 flags.py [pages...] [--stats]
"""
import collections, json, pathlib, re, sys
from PIL import Image, ImageDraw
from normalize import norm_word

CONF = 80
CONF_IA = 92
OUT = pathlib.Path("out")

SUSPECT = [
    (re.compile(r"[.,;:?!\"'’][\'’®°!*¢©>”“]+$"), "superscript residue"),
    (re.compile(r"[.,;:?!\"'’]\d{1,2}$"), "digits glued after punctuation"),
    (re.compile(r"^[A-Za-z]*[a-z][A-Z]"), "odd case"),
    (re.compile(r"\d[Il]|[Il]\d"), "I/1 confusion"),
    (re.compile(r"^[^\w\"'(\[$—–-]"), "leading stray mark"),
    (re.compile(r"[^\w\"'.,;:?!)\]%’—–-]$"), "trailing stray mark"),
    (re.compile(r"^\W+$"), "punctuation only"),
]

def strip(t):
    return re.sub(r"[^\w]+", "", norm_word(t)).lower()

def ia_words(page):
    p = OUT / "text" / f"p{page:03d}.txt"
    if not p.exists(): return collections.Counter()
    return collections.Counter(strip(w) for w in p.read_text().split() if strip(w))

def flag_page(page, stats):
    d = json.loads((OUT / "blocks" / f"p-{page:03d}.json").read_text())
    ia = ia_words(page)
    flags = []
    n_words = 0
    for b in d["blocks"]:
        if b["type"] not in ("para", "quote", "list", "subhead"): continue
        for l in b["lines"]:
            line_text = " ".join(norm_word(w["t"]) for w in l["words"])
            for w in l["words"]:
                n_words += 1
                t = norm_word(w["t"])
                reasons = []
                if w["c"] < CONF: reasons.append(f"conf {w['c']}")
                for rx, why in SUSPECT:
                    if rx.search(t) and not (why == "digits glued after punctuation" and w.get("refs")):
                        reasons.append(why)
                core = strip(t)
                if core and ia[core] == 0 and w["c"] < CONF_IA:
                    reasons.append("IA disagrees")
                    stats["ia"] += 1
                if reasons:
                    for r in reasons: stats[r.split()[0]] += 1
                    flags.append({"page": page, "bbox": w["bbox"], "line_bbox": l["bbox"], "word": w["t"],
                                  "norm": t, "conf": w["c"], "reasons": reasons, "line": line_text,
                                  "refs": w.get("refs", [])})
    stats["words"] += n_words
    return flags

def make_sheet(page, flags):
    if not flags: return
    im = Image.open(OUT / "pages" / f"p-{page:03d}.png").convert("RGB")
    by_line = collections.OrderedDict()
    for f in flags:
        by_line.setdefault(tuple(f["line_bbox"]), []).append(f)
    strips = []
    for lb, fs in by_line.items():
        x0, y0, x1, y1 = lb
        pad = 14
        crop = im.crop((max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad))
        dr = ImageDraw.Draw(crop)
        for f in fs:
            bx = f["bbox"]
            dr.rectangle((bx[0] - x0 + pad - 3, bx[1] - y0 + pad - 3, bx[2] - x0 + pad + 3, bx[3] - y0 + pad + 3),
                         outline=(220, 0, 0), width=3)
        strips.append(crop)
    W = max(s.width for s in strips); H = sum(s.height + 8 for s in strips)
    sheet = Image.new("RGB", (W, H), (255, 255, 255)); y = 0
    for s in strips:
        sheet.paste(s, (0, y)); y += s.height + 8
    sheet.save(OUT / "flags" / f"p-{page:03d}.png")

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    pages = [int(a) for a in args] or sorted(int(p.stem.split("-")[1]) for p in (OUT / "blocks").glob("p-*.json"))
    (OUT / "flags").mkdir(exist_ok=True)
    stats = collections.Counter(); rows = []; total = 0
    for n in pages:
        fl = flag_page(n, stats)
        total += len(fl)
        rows.append(f"{n}\t{len(fl)}")
        (OUT / "flags" / f"p-{n:03d}.json").write_text(json.dumps(fl, indent=1))
        if "--stats" not in sys.argv:
            make_sheet(n, fl)
    (OUT / "flags" / "summary.tsv").write_text("\n".join(rows) + "\n")
    print(f"pages {len(pages)}  words {stats['words']}  flagged {total} ({total/len(pages):.1f}/page)")
    for k, v in stats.most_common():
        if k != "words": print(f"  {k}: {v}")
