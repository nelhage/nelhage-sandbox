"""Page geometry for the Agemonia rulebook.

The rulebook is laid out as spreads: the body column sits on one side and a
sidebar of examples on the *outer* edge, so the sidebar swaps sides from page
to page (right on printed p6, left on printed p7). Nothing may assume a fixed
split -- both the icon finder and the text extractor derive the regions from
where real text actually lands.

The sidebar is invisible here on purpose: the PDF 1.3 transparency flattener
rasterised it, so it contributes no text spans. That is exactly what makes
"where is the text?" a reliable way to find the body.
"""

import pymupdf

PAGE_W = 595.28
# Body text runs 10-13pt; anything larger is a page title that may span a
# gutter, anything much smaller is fine print inside a rasterised graphic.
BODY_SIZE = (7.0, 13.5)
# A real sidebar column is ~160pt wide. The decorative page edge is ~65pt and
# must not be mistaken for one (page 43's outer border otherwise qualifies).
MIN_SIDE_W = 90.0
GUTTER_MIN = 10.0


def _contained(inner, outer, tol=0.6):
    return (
        inner[0] >= outer[0] - tol
        and inner[1] >= outer[1] - tol
        and inner[2] <= outer[2] + tol
        and inner[3] <= outer[3] + tol
    )


def text_spans(page, size_range=BODY_SIZE):
    """Body spans, with the flattener's duplicate glyphs removed.

    Flattening left a second copy of each line's first glyph as its own span,
    sitting exactly on top of the full line ("...blue box followed by" plus a
    lone "A"). Any span wholly inside a strictly wider one is that artifact.
    """
    out = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                if s["text"].strip() and size_range[0] <= s["size"] <= size_range[1]:
                    out.append(s)

    out.sort(key=lambda s: -(s["bbox"][2] - s["bbox"][0]))
    kept = []
    for s in out:
        if any(_contained(s["bbox"], k["bbox"]) for k in kept):
            continue
        kept.append(s)
    kept.sort(key=lambda s: (round(s["bbox"][3], 0), s["bbox"][0]))
    return kept


def line_coverage(page):
    """For each 1pt column of the page, how many text lines span it.

    Taking extremes (min x0, max x1) over spans is not safe: a diagram label
    can reach far into the sidebar's half of the page and swallow it (page 8's
    "Class & Profession cards" callout does exactly this). Counting *lines*
    instead makes the body a broad plateau and the sidebar a near-zero floor.
    """
    cov = [0] * int(PAGE_W)
    lines = {}
    for s in text_spans(page):
        lines.setdefault(round(s["bbox"][3], 0), []).append(s)
    for spans in lines.values():
        a = int(min(s["bbox"][0] for s in spans))
        b = int(max(s["bbox"][2] for s in spans))
        for i in range(max(0, a), min(len(cov), b + 1)):
            cov[i] += 1
    return cov


def _plateaus(cov, frac=0.20):
    """Contiguous x-ranges carrying a real share of the page's text lines."""
    peak = max(cov) if cov else 0
    if not peak:
        return []
    thresh = frac * peak
    out, run = [], None
    for i, v in enumerate(list(cov) + [0]):
        if v >= thresh:
            if run is None:
                run = i
        elif run is not None:
            if i - run >= 8:  # ignore stray specks
                out.append((run, i))
            run = None
    return out


def body_region(page):
    """Horizontal extent of the body column(s)."""
    regions = _plateaus(line_coverage(page))
    if not regions:
        spans = text_spans(page)
        if not spans:
            return (0.0, PAGE_W)
        return (
            max(0.0, min(s["bbox"][0] for s in spans) - 14),
            min(PAGE_W, max(s["bbox"][2] for s in spans) + 14),
        )
    return (max(0.0, regions[0][0] - 14), min(PAGE_W, regions[-1][1] + 14))


def side_region(page, body=None):
    """The rasterised sidebar band, or None if this page has no sidebar."""
    bx0, bx1 = body if body else body_region(page)
    left_w, right_w = bx0, PAGE_W - bx1
    if max(left_w, right_w) < MIN_SIDE_W:
        return None
    band = (0.0, bx0) if left_w >= right_w else (bx1, PAGE_W)
    # A sidebar holds artwork; an empty outer margin does not.
    for info in page.get_images(full=True):
        for r in page.get_image_rects(info[0]):
            if r.x0 >= band[0] - 4 and r.x1 <= band[1] + 4 and r.width * r.height > 400:
                return band
    return None


def columns(page, body=None):
    """Split the body into reading columns by locating the gutter.

    Some pages set the body as two columns; merging those by baseline would
    interleave them ("How to Get Started The Stock and the Archive").  Page
    titles are excluded from the coverage map because they span the gutter.
    """
    bx0, bx1 = body if body else body_region(page)
    lo, hi = int(bx0), int(bx1) + 1
    covered = bytearray(max(1, hi - lo))
    for s in text_spans(page):
        if s["bbox"][2] < bx0 or s["bbox"][0] > bx1:
            continue
        a = max(lo, int(s["bbox"][0])) - lo
        b = min(hi, int(s["bbox"][2]) + 1) - lo
        for i in range(max(0, a), max(0, b)):
            covered[i] = 1

    inner_lo, inner_hi = bx0 + 0.22 * (bx1 - bx0), bx0 + 0.78 * (bx1 - bx0)
    best, run = None, 0
    for i in range(len(covered) + 1):
        if i < len(covered) and not covered[i]:
            run += 1
            continue
        if run >= GUTTER_MIN:
            mid = lo + (i - run + i) / 2
            if inner_lo < mid < inner_hi and (best is None or run > best[0]):
                best = (run, mid)
        run = 0

    if best is None:
        return [(bx0, bx1)]
    return [(bx0, best[1]), (best[1], bx1)]


def describe(page):
    body = body_region(page)
    side = side_region(page, body)
    return {
        "body": [round(v, 1) for v in body],
        "side": [round(v, 1) for v in side] if side else None,
        "columns": [[round(v, 1) for v in c] for c in columns(page, body)],
    }


if __name__ == "__main__":
    doc = pymupdf.open("pdfs/Rulebook.pdf")
    for pno in range(doc.page_count):
        d = describe(doc[pno])
        print(
            f"pdf {pno + 1:>2}  body={d['body']}  side={d['side']}  cols={len(d['columns'])}"
        )
