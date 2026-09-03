"""Stage 1: page images for OCR.

Reads the 300 dpi grayscale renders in out/png300 (made with
`pdftoppm -r 300 -gray -png book.pdf out/png300/p`) and writes out/pages/p-NNN.png
with the scanner blemish removed.

The scan has a fixed blemish (a ~12x15 px caret-shaped blob) at about
x=145-200, y=1660-1715 (300 dpi) on every even (verso) page; it corrupts the
first word of a body line or the number/date column of the back matter. Its
shape is constant, so we template-match it inside a small search box and erase
only the template's own pixels: that leaves glyphs it touches intact (minus a
few pixels) and never removes letters or digits that merely happen to be small.
"""
import pathlib, sys
import numpy as np
from PIL import Image
from scipy import ndimage

SRC = pathlib.Path("out/png300")
DST = pathlib.Path("out/pages"); DST.mkdir(parents=True, exist_ok=True)
BOX = (125, 1635, 225, 1740)   # search box x0, y0, x1, y1
DARK = 110
MATCH = 0.85                   # fraction of template ink that must be dark

def make_template():
    """The blob as it appears isolated in the margin of PDF page 120."""
    a = np.array(Image.open(SRC / "p-120.png").convert("L"))
    sub = a[1690:1722, 140:168] < DARK       # the blob sits alone in this box
    lab, n = ndimage.label(sub)
    sizes = ndimage.sum(sub, lab, range(1, n + 1))
    i = int(np.argmax(sizes)) + 1
    sl = ndimage.find_objects(lab)[i - 1]
    t = lab[sl] == i
    assert 8 <= t.shape[1] <= 18 and 9 <= t.shape[0] <= 20, t.shape
    return t

def clean(a, tmpl):
    x0, y0, x1, y1 = BOX
    sub = a[y0:y1, x0:x1]
    dark = sub < DARK
    th, tw = tmpl.shape
    n_ink = tmpl.sum()
    tmpl = np.pad(tmpl, 2)[2:-2, 2:-2] if False else tmpl
    best, pos = 0.0, None
    for dy in range(0, sub.shape[0] - th):
        for dx in range(0, sub.shape[1] - tw):
            s = (dark[dy:dy+th, dx:dx+tw] & tmpl).sum() / n_ink
            if s > best:
                best, pos = s, (dy, dx)
    if best < MATCH:
        return 0
    dy, dx = pos
    paper = int(np.median(sub[~dark])) if (~dark).any() else 200
    # erase the blob plus a 2 px ring: the JPX halo around it is otherwise
    # still dark enough for tesseract to read as a stray glyph
    foot = ndimage.binary_dilation(tmpl, iterations=2)
    y0f, x0f = max(0, dy - 2), max(0, dx - 2)
    region = sub[y0f:y0f + foot.shape[0], x0f:x0f + foot.shape[1]]
    region[foot[:region.shape[0], :region.shape[1]]] = paper
    return 1

if __name__ == "__main__":
    pages = [int(p) for p in sys.argv[1:]] or None
    tmpl = make_template()
    hits = []
    for src in sorted(SRC.glob("p-*.png")):
        n = int(src.stem.split("-")[1])
        if pages and n not in pages: continue
        im = Image.open(src).convert("L")
        a = np.array(im)
        if n % 2 == 0 and clean(a, tmpl):
            hits.append(n)
        Image.fromarray(a).save(DST / src.name)
    print("blob erased on", len(hits), "pages; even pages without a match:",
          [n for n in range(2, 345, 2) if n not in hits and (not pages or n in pages)])
