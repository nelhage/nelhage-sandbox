"""Dump the PDF's existing (Internet Archive) OCR text layer, one file per page.

out/text/p{NNN}.txt  -- plain text in reading order (PyMuPDF sort=True)
out/text/index.tsv   -- page, chars, n_lines, first line (for quick surveys)
"""
import pathlib, sys
import pymupdf

doc = pymupdf.open("book.pdf")
out = pathlib.Path("out/text"); out.mkdir(parents=True, exist_ok=True)
rows = []
for i, page in enumerate(doc):
    n = i + 1
    txt = page.get_text("text", sort=True)
    (out / f"p{n:03d}.txt").write_text(txt)
    lines = [l for l in txt.splitlines() if l.strip()]
    rows.append(f"{n}\t{len(txt)}\t{len(lines)}\t{lines[0] if lines else ''}")
(out / "index.tsv").write_text("\n".join(rows) + "\n")
print("done", len(rows))
