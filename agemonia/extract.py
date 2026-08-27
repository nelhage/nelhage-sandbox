"""Per-page extraction: text layer with icon markers, plus high-DPI renders.

Two facts about this PDF drive the design (see icons.py and layout.py):

* The body column's text is real text, so it can be transcribed exactly. But
  icons sit in that flow as images and the text layer just concatenates around
  them -- "rolling a number of equal to your level" -- so a [[ICON]] marker is
  spliced in at the right character offset.
* The sidebar (and some shadowed headings) were rasterised by the PDF 1.3
  transparency flattener and have no text layer at all. Those can only be read
  from an image, so every page is also rendered in bands sized to stay crisp
  after a vision model downscales them to ~1568px.
"""

import argparse
import json
import pathlib

import pymupdf

import layout

PDF = pathlib.Path("pdfs/Rulebook.pdf")
OUT = pathlib.Path("out")

TARGET_PX = 1560
BANDS = 3
FOLIO_Y = 800.0


def band_dpi(width_pt, height_pt):
    return max(72, min(600, int(TARGET_PX / (max(width_pt, height_pt) / 72.0))))


def render_page(doc, pno, dest):
    """Full-page overview plus body/sidebar bands at vision-legible DPI."""
    page = doc[pno]
    H = page.rect.y1
    body = layout.body_region(page)
    side = layout.side_region(page, body)
    made = []

    # Clear stale renders: if a page's column count or sidebar changed, its old
    # band files would otherwise linger and mislead a transcriber.
    for old in dest.glob(f"p{pno + 1:02d}_*.png"):
        old.unlink()

    page.get_pixmap(dpi=band_dpi(page.rect.x1, H)).save(dest / f"p{pno + 1:02d}_full.png")
    made.append(f"p{pno + 1:02d}_full.png")

    # A full-width two-column page rendered whole lands near 200 DPI, which is
    # marginal for 11pt text; rendering each column alone roughly doubles it.
    cols = layout.columns(page, body)
    if len(cols) > 1:
        regions = [(f"col{i + 1}_", c) for i, c in enumerate(cols)]
    else:
        regions = [("body", body)]
    if side:
        regions.append(("side", side))

    for tag, (x0, x1) in regions:
        step = H / BANDS
        for i in range(BANDS):
            # Overlap bands slightly so nothing is cut mid-line.
            y0 = max(0, i * step - 14)
            y1 = min(H, (i + 1) * step + 14)
            name = f"p{pno + 1:02d}_{tag}{i + 1}.png"
            page.get_pixmap(dpi=band_dpi(x1 - x0, y1 - y0), clip=pymupdf.Rect(x0, y0, x1, y1)).save(
                dest / name
            )
            made.append(name)
    return made


def style_tag(span):
    """Compact font descriptor: the agent uses it to pick heading levels."""
    return f"{span['font'].split('+')[-1]}@{span['size']:.1f}"


def line_records(page, icons):
    """Body lines, in reading order, with icon markers spliced in.

    PyMuPDF breaks a visual line into several `line` records wherever an inline
    icon interrupts it, so spans are regrouped by baseline first -- otherwise
    one icon lands on both halves and gets transcribed twice.
    """
    body = layout.body_region(page)
    cols = layout.columns(page, body)
    spans_all = [
        s
        for s in layout.text_spans(page, size_range=(0.0, 99.0))
        if body[0] - 6 <= s["bbox"][0] <= body[1]
    ]

    out, used = [], set()
    for ci, (cx0, cx1) in enumerate(cols):
        col = [s for s in spans_all if cx0 - 4 <= s["bbox"][0] < cx1]
        if not col:
            continue
        if len(cols) > 1:
            out.append({"y": 0.0, "x": cx0, "text": f"=== COLUMN {ci + 1} ===", "icons": 0})

        by_baseline = {}
        for s in col:
            by_baseline.setdefault(round(s["bbox"][3], 0), []).append(s)

        for base in sorted(by_baseline):
            spans = sorted(by_baseline[base], key=lambda s: s["bbox"][0])
            y0 = min(s["bbox"][1] for s in spans)
            y1 = max(s["bbox"][3] for s in spans)

            # Each icon belongs to exactly one line: the one it sits on.
            mine = []
            for i, ic in enumerate(icons):
                if i in used:
                    continue
                cy = (ic["rect"][1] + ic["rect"][3]) / 2
                if y0 - 2 <= cy <= y1 + 2 and cx0 - 4 <= ic["rect"][0] < cx1:
                    mine.append(ic)
                    used.add(i)

            events = [("span", s["bbox"][0], s) for s in spans]
            events += [("icon", ic["rect"][0], ic) for ic in mine]
            events.sort(key=lambda e: e[1])

            pieces, cur = [], None
            for kind, _, obj in events:
                if kind == "icon":
                    pieces.append("[[ICON]]")
                else:
                    st = style_tag(obj)
                    if st != cur:
                        pieces.append(f"<{st}>")
                        cur = st
                    pieces.append(obj["text"])
            out.append(
                {
                    "y": round(y0, 1),
                    "x": round(spans[0]["bbox"][0], 1),
                    "text": "".join(pieces),
                    "icons": len(mine),
                }
            )
    return out


def folio(page):
    """The page number as printed at the foot of the page."""
    best = None
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                t = s["text"].strip()
                if t.isdigit() and len(t) <= 3 and s["bbox"][1] > FOLIO_Y:
                    if best is None or s["bbox"][1] > best[0]:
                        best = (s["bbox"][1], int(t))
    return best[1] if best else None


HEADER = """# PDF page {pno} of {total}
# Printed page number (folio): {printed}
# Body column x-range: {body}   Sidebar: {side}
# Inline icon markers found: {nicons}
#
# Below is the BODY column's text layer, one line per printed line, in reading
# order.  <Font@size> tags mark style changes.  [[ICON]] marks an inline icon
# image; its raster is ~15px and unreadable, so infer it from context and the
# glossary.  [[ICON]] is a LOWER BOUND -- vector-drawn icons are not marked, so
# trust the page renders over these markers.
#
# '=== COLUMN n ===' marks a two-column body: read column 1 fully, then 2.
#
# The SIDEBAR has NO text layer (the PDF's transparency flattener rasterised
# it).  Read it from the p{pno:02d}_side*.png renders.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pages",
        help="comma-separated PDF page numbers to regenerate; default all. "
        "Existing index entries for other pages are preserved.",
    )
    args = ap.parse_args()
    only = {int(p) for p in args.pages.split(",")} if args.pages else None

    render_dir, text_dir = OUT / "render", OUT / "text"
    render_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    occ = json.loads((OUT / "icons" / "occurrences.json").read_text())
    by_page = {}
    for o in occ:
        by_page.setdefault(o["page"], []).append(o)

    prior = {}
    if only and (OUT / "index.json").exists():
        prior = {e["pdf_page"]: e for e in json.loads((OUT / "index.json").read_text())}

    doc = pymupdf.open(PDF)
    index = []
    for pno in range(doc.page_count):
        if only and (pno + 1) not in only:
            if pno + 1 in prior:
                index.append(prior[pno + 1])
            continue
        page = doc[pno]
        recs = line_records(page, by_page.get(pno + 1, []))
        printed = folio(page)
        imgs = render_page(doc, pno, render_dir)
        geo = layout.describe(page)

        head = HEADER.format(
            pno=pno + 1,
            total=doc.page_count,
            printed=printed if printed is not None else "NONE (unnumbered page)",
            body=geo["body"],
            side=geo["side"] or "none",
            nicons=sum(r["icons"] for r in recs),
        )
        body_txt = "\n".join(f"{r['y']:>6} x{r['x']:<6} {r['text']}" for r in recs)
        (text_dir / f"p{pno + 1:02d}.txt").write_text(head + "\n" + body_txt + "\n")

        index.append(
            {
                "pdf_page": pno + 1,
                "printed": printed,
                "icons": sum(r["icons"] for r in recs),
                "sidebar": geo["side"] is not None,
                "columns": len(geo["columns"]),
                "lines": len(recs),
                "images": imgs,
            }
        )

    (OUT / "index.json").write_text(json.dumps(index, indent=1))
    missing = [e["pdf_page"] for e in index if e["printed"] is None]
    print(f"{len(index)} pages rendered; no folio on PDF pages {missing}")
    print(f"total icon markers: {sum(e['icons'] for e in index)}")
    print(f"two-column pages: {[e['pdf_page'] for e in index if e['columns'] > 1]}")


if __name__ == "__main__":
    main()
