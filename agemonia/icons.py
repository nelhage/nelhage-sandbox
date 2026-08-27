"""Locate the inline icons in the rulebook's body text.

The PDF is format 1.3, so InDesign's transparency flattener rasterised every
soft-shadowed element into image fragments. Two consequences shape this script:

* One icon often arrives as several adjacent tiles (a 12x12 die as a 4x12 plus
  an 8x12), so touching fragments are merged into one logical occurrence.
* Each inline icon was downsampled to roughly 15x15 pixels, so the art itself
  is unrecoverable. We only record *where* an icon sits; naming it is left to
  the vision pass, which reads the crisp legend boxes instead.

Output: out/icons/occurrences.json, which extract.py splices into the text
layer as [[ICON]] markers.
"""

import io
import json
import pathlib

import pymupdf
from PIL import Image

import layout

PDF = pathlib.Path("pdfs/Rulebook.pdf")
OUT = pathlib.Path("out/icons")

MAX_PT = 34.0
MIN_PT = 3.5
# Fragments of one icon sit flush together; 1.5pt of slack merges them without
# swallowing a genuinely separate neighbour.
MERGE_SLACK = 1.5


def body_lines(page, body):
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for ln in b.get("lines", []):
            txt = "".join(s["text"] for s in ln["spans"])
            if txt.strip() and body[0] - 4 <= ln["bbox"][0] <= body[1]:
                out.append((pymupdf.Rect(ln["bbox"]), txt))
    return out


def inline_line(rect, lines):
    """The body text line this image rides on, if any."""
    best = None
    for lb, txt in lines:
        overlap = min(rect.y1, lb.y1) - max(rect.y0, lb.y0)
        if overlap < 0.55 * rect.height:
            continue
        if lb.x0 - 30 <= rect.x0 and rect.x1 <= lb.x1 + 30:
            if best is None or overlap > best[0]:
                best = (overlap, lb, txt)
    return (best[1], best[2]) if best else None


def sits_in_gap(rect, spans, thresh=0.5):
    """True if the image occupies a gap in the text rather than lying under it.

    A real inline icon interrupts the line: the span before it ends, the span
    after it begins. Flattening artifacts instead sit *within* a span's box, so
    anything mostly covered by text is rejected.
    """
    covered = 0.0
    for s in spans:
        sb = s["bbox"]
        if sb[3] < rect.y0 or sb[1] > rect.y1:
            continue
        covered += max(0.0, min(rect.x1, sb[2]) - max(rect.x0, sb[0]))
    return covered < thresh * rect.width


def merge_fragments(rects):
    parent = list(range(len(rects)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rects)):
        gi = pymupdf.Rect(rects[i]) + (-MERGE_SLACK, -MERGE_SLACK, MERGE_SLACK, MERGE_SLACK)
        for j in range(i + 1, len(rects)):
            if gi.intersects(rects[j]):
                parent[find(i)] = find(j)

    groups = {}
    for i, r in enumerate(rects):
        groups.setdefault(find(i), []).append(r)
    out = []
    for members in groups.values():
        u = pymupdf.Rect(members[0])
        for m in members[1:]:
            u |= m
        out.append(u)
    return out


def has_ink(page, rect):
    """Reject pure-background tiles, which would become phantom icons."""
    pix = page.get_pixmap(dpi=150, clip=rect + (-0.3, -0.3, 0.3, 0.3))
    im = Image.open(io.BytesIO(pix.tobytes("png"))).convert("L")
    px = im.getdata()
    return (max(px) - min(px)) >= 30


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PDF)

    occurrences = []
    for pno, page in enumerate(doc):
        body = layout.body_region(page)
        lines = body_lines(page, body)
        if not lines:
            continue
        spans = layout.text_spans(page, size_range=(0.0, 99.0))
        cands = []
        for info in page.get_images(full=True):
            for r in page.get_image_rects(info[0]):
                if not (MIN_PT <= r.width <= MAX_PT and MIN_PT <= r.height <= MAX_PT):
                    continue
                if r.x0 < body[0] - 4 or r.x1 > body[1] + 4:
                    continue
                if inline_line(r, lines):
                    cands.append(r)
        for u in merge_fragments(cands):
            if u.width > MAX_PT or u.height > MAX_PT:
                continue
            hit = inline_line(u, lines)
            if not hit or not sits_in_gap(u, spans) or not has_ink(page, u):
                continue
            occurrences.append(
                {
                    "page": pno + 1,
                    "rect": [round(v, 2) for v in u],
                    "context": hit[1].strip()[:120],
                }
            )

    (OUT / "occurrences.json").write_text(json.dumps(occurrences, indent=1))
    per_page = {}
    for o in occurrences:
        per_page[o["page"]] = per_page.get(o["page"], 0) + 1
    print(f"{len(occurrences)} inline icon occurrences across {len(per_page)} pages")
    print("busiest:", sorted(per_page.items(), key=lambda kv: -kv[1])[:10])


if __name__ == "__main__":
    main()
