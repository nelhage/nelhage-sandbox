# Survey brief (shared context for all survey agents)

Project dir: /home/nelhage/code/sandbox/memories2epub (all commands run from here; the
dev shell is auto-loaded via direnv, so tesseract 5.5 (eng), pdftoppm/pdfimages,
imagemagick, pandoc and python3 with pymupdf+pillow are on PATH).

The book: "Memories That Shaped an Industry" (Emerson W. Pugh, MIT Press 1984), an
Internet Archive scan, 344 PDF pages. book.pdf is a symlink to it.
Printed page number = PDF page - 16. Chapter starts (PDF pages): 17, 50, 78, 109,
145, 176, 203, 229, 264. "References and Notes" (endnotes) 281-316, Chronology
317-328, Index 329-339. Front matter 1-16, back cover etc. 340-344.

Already available:
- out/render/p-NNN.png : 150 dpi grayscale render of every page (view these with the
  Read tool to see the page; render specific pages at higher dpi yourself with
  `pdftoppm -r 300 -gray -f N -l N -png book.pdf out/thumb/x` if you need detail).
- out/text/pNNN.txt : the PDF's existing OCR text layer (Internet Archive's
  tesseract, reading-order sorted). Useful for grepping, but has errors.
- out/text/index.tsv : page, chars, lines, first line.
- Run fresh OCR yourself with e.g.
  `tesseract out/thumb/x-021.png out/thumb/x021 -l eng --psm 1 hocr txt`
  (--psm 1 warns about osd, harmless; try --psm 3/4/6 too). hOCR gives per-word
  bboxes and x_wconf confidences.

Rules: only write under out/ (out/survey/ for reports, out/thumb/ for scratch).
Don't install anything; if you'd need a tool, say so in the report. Do NOT try
to transcribe pages; the goal is to characterise the document and the OCR so a
pipeline can be designed. Be concrete: cite PDF page numbers for every claim.
Write your report to the file named in your task, then return a ~15-line summary.
