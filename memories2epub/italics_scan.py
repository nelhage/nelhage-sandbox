"""Slant heuristic for italic words: shear the word's ink until the column projection is
sharpest; italics peak at shear ~ -0.25, upright at 0. Writes out/italics/candidates.json
(runs of adjacent candidate words per line) and crops for review."""
import json, glob, pathlib, sys, numpy as np
from PIL import Image
OUT = pathlib.Path("out"); (OUT / "italics" / "crops").mkdir(parents=True, exist_ok=True)
TH = -0.15
SHEARS = np.linspace(-0.45, 0.45, 19)
def slant(im, bbox):
    x0, y0, x1, y1 = bbox
    a = np.asarray(im.crop((x0 - 2, y0, x1 + 2, y1)).convert("L")) < 128
    h, w = a.shape
    if h < 15 or w < 15 or a.sum() < 30: return None
    ys = np.arange(h)[:, None]; best = None
    for sh in SHEARS:
        cols = np.arange(w)[None, :] + (sh * (h - ys)).astype(int)
        acc = np.zeros(w + 40); xs = np.clip(cols + 20, 0, w + 39)
        np.add.at(acc, xs[a], 1)
        sc = (acc ** 2).sum()
        if best is None or sc > best[0]: best = (sc, sh)
    return best[1]
def main():
    pages = [int(a) for a in sys.argv[1:]] or sorted(int(p.stem[2:]) for p in (OUT / "blocks").glob("p-*.json"))
    already = set()
    for f in glob.glob("out/corrections/*.json"):
        for c in json.load(open(f)):
            if c.get("italic") and "bbox" in c: already.add((c["page"], tuple(c["bbox"])))
    cands = []; n = 0
    for p in pages:
        im = Image.open(f"out/pages/p-{p:03d}.png")
        d = json.load(open(f"out/blocks/p-{p:03d}.json"))
        for b in d["blocks"]:
            if b["type"] not in ("para", "quote", "list"): continue
            for l in b["lines"]:
                run = []
                for w in l["words"]:
                    n += 1
                    s = slant(im, w["bbox"]) if len(w["t"].strip(".,;:\"'()")) >= 3 else None
                    hit = s is not None and s <= TH and (p, tuple(w["bbox"])) not in already
                    if hit: run.append(w)
                    elif run:
                        cands.append({"page": p, "line_bbox": l["bbox"], "words": [{"t": x["t"], "bbox": x["bbox"]} for x in run],
                                      "line": " ".join(x["t"] for x in l["words"])}); run = []
                if run:
                    cands.append({"page": p, "line_bbox": l["bbox"], "words": [{"t": x["t"], "bbox": x["bbox"]} for x in run],
                                  "line": " ".join(x["t"] for x in l["words"])})
    for i, c in enumerate(cands):
        x0, y0, x1, y1 = c["line_bbox"]
        im = Image.open(f"out/pages/p-{c['page']:03d}.png").crop((x0 - 10, y0 - 10, x1 + 10, y1 + 10))
        c["crop"] = f"out/italics/crops/{i:04d}.png"; im.save(c["crop"])
    json.dump(cands, open(OUT / "italics" / "candidates.json", "w"), indent=1)
    print(f"{n} words scanned, {len(cands)} candidate runs ({sum(len(c['words']) for c in cands)} words)")
if __name__ == "__main__":
    main()
