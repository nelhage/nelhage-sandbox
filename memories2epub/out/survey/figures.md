# Figures, photographs, diagrams and tables

All page numbers are PDF pages (printed page = PDF page - 16). Page is 5.36 x 8.59 in
(386 x 619 pt). Deliverable list: `out/survey/figures.tsv` (68 figure entries on 44 pages,
plus one row standing in for the 12-page chronology). Bboxes are page fractions
(x0 y0 x1 y1), accurate to ~0.02.

## 1. Inventory and how it was found

**Result: 44 figure pages carrying 67 figures** (46 photographs, 15 line-art diagrams,
3 patent drawings, 1 notebook facsimile, 1 graph). There are no numbered figures and no
true tables. Every figure is in the body chapters (pp. 17-280); front matter, endnotes,
chronology and index contain none.

Figure pages: 25 32 37 40 47 58 62 66 67 69 71 76 81 86 88 91 96 99 126 127 130 132 136
138 141 157 163 165 172 174 185 190 193 196 198 217 236 239 243 248 254 256 261 271.

Pure line-art pages (nothing in the photographic layer): 58 66 67 86 99 130 165 196 217
236 243 248. Mixed photo + line art on one page: 32, 157, 193, 261.

Three independent detectors were used, then every candidate was confirmed by eye on
contact sheets (`out/thumb/sheet0-4.png`, made from out/render at 300x480 px per page):

1. **Caption-text grep** of out/text: `(Photograph courtesy of ...)`, `(From the IBM
   Journal ...)`, `(Illustration from ...)`. Hits: 25 32 37 40 58 66 67 76 81 86 88 91 96
   132 136 163 165 174 196 217 236 239 243 271. Only ~half the figures carry a credit, so
   this alone misses ~20 pages (e.g. 47, 62, 69, 71, 126, 127, 138, 141). "Figure N"/
   "Table N" grep finds only p248, where "FIG. 4"/"FIG. 8" are labels inside a reproduced
   patent drawing, not book captions.
2. **MRC background-layer size** (`pdfimages -list book.pdf`, out/thumb/imglist.txt): the
   643x1030 JPX background is 1.17-1.26 KB on every text-only page and 1.47-6.5 KB on every
   page with a halftone photo. Threshold 1.4 KB gives exactly the 32 photo pages (min p157
   at 1466 B; next non-photo is p99 at 1362 B, which is the patent drawing's gray). It
   cannot see line art (p58, p130, p165, ... have 1.2 KB backgrounds).
3. **Ink-run detector on the 150 dpi renders** (`out/thumb/detect.py`, output
   out/thumb/detect.txt): binarise at v<100 (render background is ~150, not white!),
   drop 6% margins, and measure the longest run of consecutive rows with >1.5% ink.
   Text pages: median 21 rows, max 46 (line pitch ~31 px). Figure pages: min 68 (p196), all
   others >= 123. With threshold 55 this flags all 44 figure pages plus only two false
   positives, p229 (the large "8" chapter numeral) and p293 (a dense endnote block). Long
   horizontal dark runs (>=70 px) are a secondary line-art cue: 0 on every text page,
   8-40 on typeset diagrams, hundreds on photos.

Recommendation for the pipeline: use detector 3 (or 2 OR 3) to pick pages, then locate
figures within the page as described in section 3.

Text-only pages that look suspicious on line count alone (chapter ends etc.) and were
checked and rejected: 49 108 144 175 202 228 263 280 229 293.

## 2. Caption convention

Every figure has a caption **directly below it**, left-aligned to the text column, in the
same column width as the body text. Structure (see p25, p91, p165, p236):

    <Title line in bold roman, sentence case>            e.g. "Ceramic Memory I" (p91),
                                                          "Block diagram of the Type 738 memory" (p165)
    <One paragraph of roman caption text, 1-12 lines>
    <optional credit in parentheses at the end of the paragraph>
        "(Photograph courtesy of the MITRE Corporation Archives.)"      p88, p91, p96
        "(From the IBM Journal of Research and Development, April 1957, p. 104.)"  p165
        "(Illustration from R. G. Counihan's internal IBM report of September 9, 1951.)" p66

- **No figure numbers** anywhere ("Figure", "Fig." never appear as captions). Figures are
  referenced in the text only informally.
- Caption text is set ~2 pt smaller than body: line pitch 23 px at 150 dpi (~11 pt) vs 31 px
  (~15 pt) for body text (measured on p66/p67 vs p108). Title is bold at the same small size.
- When two figures share a page they usually get **two separate title+caption blocks** (p88,
  p126, p141, p165, p172, p271), but sometimes **one caption covers both**, using
  "(above)"/"(below)" or "(above, left)"/"(below, right)" (p32, p40, p62, p71, p81, p236,
  p248, p256). Multi-part photos are labelled "(a)"/"(b)" in prose (p239).
- Title line quirks the OCR must survive: small caps in titles/captions ("SSEC", "SAGE",
  "XD-1", "UNIVAC", "ROS") come out as mixed case in IA's OCR ("ssEC", "saGE", "uNIvAC",
  "Ros"); italic journal names are OCR'd with "JBM"/"[BM" for "IBM" (p58, p163, p165).
- Full-page figure pages have only the running head + figure(s) + caption(s); many
  captions end with the figure page and the body text resumes on the next page, so the
  caption never interleaves with body prose. Exception pattern: none found.

## 3. Image layers and the best achievable resolution

`pdfimages -list` shows each page as: image 0 = 643x1030 RGB JPX at 120 ppi
(background), image 1 = 1930x3093 RGB JPX at 360 ppi (foreground colour, 4-5 KB, i.e.
a near-flat dark fill), image 2 = 1930x3093 1-bit JBIG2 smask at 360 ppi. Extracted for
p25 and p60 to out/thumb/img{25,60}-00{0,1,2}.png; montage in out/thumb/layers25.png.

What each layer carries on a photo page (p25, p91):
- **Background (120 ppi)** carries the continuous-tone photo, but heavily blurred and
  JPX-crushed (2.5-6.5 KB per page). Fine detail is *absent*: on p25 the printing on the
  punched card does not exist in the background at all (it is a flat grey rectangle,
  see out/thumb/cmp_p25.png, left panel).
- **JBIG2 mask (360 ppi)** carries all text AND a thresholded copy of every dark/high-
  frequency part of the photo: halftone dots, edges, the card printing, wire outlines.
  It is 1-bit, so tones are posterised, but this is where the *sharpness* lives.
- **Foreground** is just the ink colour painted through the mask.

Consequence: **cropping the background layer alone is not acceptable** (blur, missing
detail, p25 card unreadable). The photo must be taken from the composited render, where
the mask supplies edges at 360 ppi on top of the 120 ppi tones. Compare
out/thumb/cmp_p91.png: left = bg 120 ppi, middle = pdftoppm 150 dpi, right = 300 dpi.

Best achievable: effective tonal resolution is 120 ppi (a full-column photo 4.3 in wide
is ~515 px of real tone), with binary edge detail at 360 ppi. Recommendation:

    pdftoppm -r 300 -gray -f N -l N -png book.pdf x     # or -r 360 to hit the mask grid
    crop to bbox, then downscale to ~200 dpi (≈860 px for a full-column figure) with a
    Lanczos/area filter and stretch levels (paper ≈150 -> white, ink ≈0-40 -> black).

The downscale softens the 1-bit posterisation while keeping the mask's edge detail; going
above ~200 dpi in the EPUB only enlarges dither. For **line art** (p58, p130, p165, p217,
p236, p243, p248, patents p99/p248, notebook p86) everything is in the mask, so render at
300-360 dpi and keep it at that resolution (or binarise) - these come out crisp.

Text bleed: the background layer contains no text at all (dark-pixel fraction outside the
photo boxes <= 0.1% on all 33 photo pages), and the mask's text lies in bands that never
touch the figure area, so a correct bbox crop from the composite has no bleed. Running
heads are at y≈0.04 and all figures start at y>=0.07; captions start >=0.02 below the
figure's lowest ink. The only hazard is *labels inside* figures (p32 block diagram,
p165, p196, p217) which are part of the figure and must not be OCR'd into the text.

Locating the bbox automatically: for photos, threshold the background layer (v<125) and
take the connected component bbox (this is what produced the photo boxes in the TSV);
it undershoots on light-background photos (p69, p88, p96, p132, p185 - the bg box is
0.05-0.1 smaller than the visible photo), so union it with the ink-run region from the
render, and stop the box at the first high-confidence hOCR caption line below it. For
line art use the ink-run region from the render directly. Nested/split components (p25,
p47, p88 produced sub-boxes) need merging by overlap. Two figures on one page are
separated by the intervening caption block (gap >= 0.03 of page height).

## 4. Tesseract on figure pages

Ran `tesseract out/thumb/r300-{025,130}.png ... --psm 1 hocr txt` (300 dpi renders);
output in out/thumb/ocr025.* and out/thumb/ocr130.*.

- **Photos (p25)**: tesseract emits short garbage lines inside the photo region
  ("l?;xxyl'hllmlll'", "GGE 6656", "(ERE FRX ]", "9985324"). hOCR line confidences for
  those are 0-9 (x_wconf), while every real text/caption line on the page is 87-91. The
  SSEC photo produced no words at all. Easy to detect either way: mean x_wconf < 60, or
  bbox inside the figure region. IA's own text layer (out/text/p025.txt lines 2-7) has
  the same kind of junk ("OooO0 Fae", "$9 SMMBS 595M 133:"), so the existing layer also
  needs this filter.
- **Hand-lettered diagrams (p130)**: much worse - 27 lines of plausible-looking pseudo-
  words from the block diagram and timing chart ("Rogram Se/ectian", "Timing and Contre/
  Secthon", "Set MBR", "Sample on red"). Confidences are mixed: mean 48.8 per line, but a
  few label lines score 79-90 ("Time, psec. 0 1 2 3 ...", "Set MBR", "Start"). Confidence
  alone would let ~5 of 27 lines through; **bbox overlap with the figure region is the
  reliable filter**, confidence is a useful second signal. Caption lines on the same page
  score 89.8-92.0, so a threshold of ~85 on mean line confidence separates caption from
  figure junk on both pages tested.
- Typeset diagrams (p165, p196, p217) will OCR their labels *correctly* (IA's layer on
  p165 reads "WAVEFORM GENERATOR", "MATRIX SWITCH", ...), so high confidence is not proof
  of body text; those must be dropped by bbox.
- Layout: tesseract --psm 1 keeps caption paragraphs intact and in order on both pages;
  it did not merge caption text with the figure junk.

## 5. Tables

There are **no ruled or multi-column data tables** in the book. The only tabular layout is
the **Chronology, pp. 317-328**: two columns, a narrow date column ("2/46", "12/47",
"1911") at x≈0.10-0.16 and an event paragraph at x≈0.27-0.90, one blank line between
entries, entries 1-3 lines long, no rules. The Index (329-339) is a plain two-column
list and the endnotes (281-316) are a numbered list (note the sub-lettered (g)...(u)
two-column run on p293 that fooled the ink detector).

Tesseract on the chronology (out/thumb/ocr318*.txt, 300 dpi):
- `--psm 1` **breaks the association**: it emits all 15 dates as one block, then all the
  event paragraphs as another block (out/thumb/ocr318.txt lines 1-30 vs 30+).
- `--psm 4` and `--psm 6` keep each date on the same line as the first line of its event
  ("2/46 The Electronic Numerical Integrator and Computer / (ENIAC) is dedicated ..."),
  with blank lines between entries - directly parseable into (date, text) pairs. IA's
  existing text layer (out/text/p318.txt) also preserves the pairing via leading spaces.
  Use psm 4/6 (or IA's layer) for pp. 317-328 and emit a definition list or a two-column
  HTML table in the EPUB. Minor OCR errors seen: "Chonology" (the printed running head
  is actually misspelled that way on p318 in both OCRs - check the scan), "Labortories".

## Scratch files
out/thumb/flag.py (abandoned first attempt), out/thumb/detect.py + detect.txt (working
detector), imglist.txt, sheet0-4.png, grid0-2.png, layers25.png, layers60.png,
cmp_p25.png, cmp_p91.png, bg/pNN-000.png (background layers of the 33 photo pages),
r300-{025,091,130,318}.png, ocr*.{hocr,txt,tsv}.
