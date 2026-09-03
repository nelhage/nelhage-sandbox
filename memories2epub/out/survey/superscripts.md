# Superscript endnote references — survey report

Prototype: `out/survey/superscript_proto.py` (detector + digit OCR + mask-and-re-OCR),
`out/survey/superscript_seq.py` (whole-book numbering-prior analysis),
ground truth: `out/survey/superscript_gt.json`. Scratch images in `out/thumb/ss/`.

## 0. Two facts about the scan that change the problem

1. **The PDF is an MRC scan: the text is a 1-bit JBIG2 mask at 360 ppi** (1930x3093 px per
   page) composited over a 120-ppi JPX background (`pdfimages -list -f 20 -l 21 book.pdf`).
   Rendering at 400/450 dpi therefore adds nothing: it only up-samples the 360-ppi bilevel
   mask (verified: the 450-dpi render of PDF 20 shows exactly the same broken glyph as the
   extracted mask `out/thumb/ss/raw-20-002.png`). 300 dpi is a mild down-sample of the
   mask and is perfectly adequate; `pdfimages -png` gives the lossless mask if ever wanted.
2. **Some superscripts are physically missing from the mask.** PDF 20 line 32
   ("more complicated calculations.⁸") has only a 3x5-px speck where the 8 should be (see
   `out/thumb/ss/zoomM-020-L32.png`); PDF 50 line 23 ("logic.¹") likewise. These are
   scan/thresholding losses, unrecoverable from the image; only the numbering sequence (and
   the endnote text) can supply the value. Whole-book count: 3 of ~808 references (0.4 %).

Body superscripts are ~0.95-1.0 x-height tall (≈21 px at 300 dpi, i.e. the same height as
a lower-case x), their bottom sits 0.45-0.55 x-heights above the baseline and their top
0.4-0.5 x-heights above the x-height line. Apostrophes/quotes are 0.55-0.75 x-height tall,
4-7 px wide and sit with their bottom 0.85-1.0 x-heights up — separable by geometry alone.

## 1. Ground truth (PDF pages 20, 21, 22, 60, 130, 210)

Read from 450-dpi line strips (`out/thumb/ss/strip-NNN-k.png`, labelled with the
tesseract line index L used below). 22 references, 21 legible.

| PDF | line | preceding text | ref | tesseract 5.5 @300 dpi (--psm 1) reads |
|---|---|---|---|---|
| 20 | L07 | leader of IBM. | 4 | `IBM.*` |
| 20 | L15 | United States. | 6 | `States.®` |
| 20 | L24 | Educational Research. | 7 | `Research.”` |
| 20 | L32 | complicated calculations. | 8 (glyph missing in scan) | `calculations.'` |
| 21 | L06 | planetary motion. | 8 | `motion.8` |
| 21 | L11 | electronic computers. | 8 (re-cite) | `computers.®` |
| 21 | L25 | out of education?" | 7 (re-cite) | `education?"”` |
| 21 | L29 | mathematical operations. | 9 | `operations.®` |
| 21 | L33 | estimate of $100,000. | 10 | `$100,000.1°` |
| 21 | L35 | August 7, 1944. | 9 (re-cite) | `1944.9` |
| 21 | L37 | from its construction. | 11 | `construction.!!` |
| 22 | L06 | 23 significant figures." | 12 | `figures.''12` |
| 22 | L12 | about five tons. | 12 (re-cite) | `tons.!?` |
| 22 | L16 | Charles Babbage. | 13 | `Babbage.13` |
| 22 | L26 | their contributions. | 14 | `contributions.!*` |
| 22 | L33 | Aiken might conceive. | 10 (re-cite) | `conceive.!9` |
| 22 | L36 | as its director. | 8 (re-cite) | `director.!` |
| 60 | L23 | the i-th digit-level." | 13 | `digit-level." 13` |
| 60 | L34 | was to be selected. | 13 (re-cite) | `selected.!3` |
| 130 | — | (figure page: two hand-lettered diagrams + captions) | none | — |
| 210 | L07 | for the 8000 series." | 14 | `series.'14` |
| 210 | L14 | bitterness, and rage. | 12 | `rage.!>` |
| 210 | L28 | 1401 and 1410 computers. | 12 (re-cite) | `computers.!12` |

Non-reference superscripts on these pages (must not be turned into notes): the
mathematical `M²` three times on PDF 60 (L11, L12, L22); PDF 21 L01 `hour."` is a closing
quote, not a reference. Tesseract's raw text gets the number right in only 7/21 cases.

**Numbering is NOT monotone.** Notes are numbered per chapter in order of *first*
citation, but re-citations are common and appear out of order: 7 of the 22 references
above are re-cites (p21: 8, 7, 9 after 10; p22: 10, 8 after 14; p60: 13; p210: 12), and
over the whole book (section 4) about a third of all marks are re-citations. The usable
prior is therefore: value ∈ {1 .. max_so_far+1} within the chapter, and the first
citations of 1..N (N = number of notes in the chapter, countable from References and
Notes) occur in increasing order with no gaps. Also seen: two references on one word,
`distributed.²⁵,²⁶` (PDF 28 L28), and references after a colon (`units:¹⁹`, PDF 24 L31)
and after closing quotes.

## 2. Detector (superscript_proto.py)

Per page: render at 300 dpi (pymupdf) → `tesseract --psm 1 hocr` (already needed for the
body text) → for every text line use the hOCR `bbox`, `baseline`, `x_size`,
`x_ascenders`, `x_descenders` (x-height = x_size − asc − desc ≈ 21-22 px) → connected
components (pure-python run-length union-find, no numpy needed; ~1-2 s/page) inside the
line box extended upward by 0.5·x_size → a component is a superscript candidate if
bottom is 0.25-0.75 x-heights above the baseline, top ≥ 1.2 x-heights above the baseline,
height 0.75-1.25 x-heights and width ≥ 0.25 x-height → adjacent candidates (gap ≤ 0.6
x-height) form one mark → the mark is a **reference** if the nearest component to its
left (ignoring specks above the line) is a period/comma/colon-dot (height < 0.5 x-h,
narrow, sits on or near the baseline) or a quote stroke (narrow, bottom ≥ 0.6 x-h up,
optionally two strokes and a period behind it), or if it directly follows another
reference mark (chained `25,26`). Marks after a full-height glyph (M²) are reported as
`other`. Lines are skipped unless their x_size is within ±25 % of the page's body x_size
and either the line or its hOCR paragraph has mean word confidence ≥ 50 (this removes the
hand lettering inside figures such as PDF 130 while keeping one-word paragraph-final lines
like `division.¹⁵` whose own confidence is low precisely because of the superscript).

Digit reading: crop the mark with 3 px padding, binarise, upscale 4x (LANCZOS), pad with
white, `tesseract --psm 7 -c tessedit_char_whitelist=0123456789`.

**Mask-and-re-OCR:** paint the detected marks white and run the page OCR again; the body
word then comes out clean and the reference is re-inserted at the word's right edge.

## 3. Results on the 6 pages (300 dpi)

| | value |
|---|---|
| reference marks detected | 21 of 21 legible (recall 0.955 incl. the missing-glyph case; 1.0 on legible) |
| false positives | 0 (0 on the figure page 130; the three `M²` are correctly reported as `other`) |
| digit OCR, psm 7 crop | **21/21** correct |
| digit OCR, psm 8 crop (tesseract's "single word" mode) | 7/21 — drops the leading 1 of 12/13/14, 8→3, 6→2, 9→2 |
| per-component psm 10 | 20/21 (6→0) |
| nearest-template on 16x24 normalised glyphs, leave-one-out, 33 digits | 32/33 (one 6→0; the book has only one superscript 6 in the sample) |
| body word after mask-and-re-OCR | 21/21 clean (`IBM.`, `States.`, `education?"`, `$100,000.`, `figures."`, `series."` …) with word confidence 86-92 (38 and 52 for the two quote cases) |
| native 360-ppi JBIG2 mask instead of 300-dpi render | same detections; no benefit (and psm-8 digit OCR was slightly worse on the crisp mask) |

The psm-8 vs psm-7 difference is the single most important detail: tesseract's single-word
mode treats the tiny crop badly and its confidences (0-60) are uninformative, while psm 7
returns the right digits with confidence 90-97 on clean glyphs.

What still fails on these pages:
* PDF 20 L32 `calculations.⁸` — glyph absent from the scan (speck only). The detector has
  nothing to find; tesseract emits `calculations.'`. The numbering prior does *not* fire
  here (8 is still max+1 when it is next cited at PDF 21 L06), so the only automatic
  signal is the stray `'` after terminal punctuation in the OCR text; that text-pattern
  check is therefore part of the recommendation.
* Nothing else on the six pages.

## 4. Whole-book run (PDF 17-280)

Detector run over all body pages (`out/thumb/ss/runall.py`, ~4 s/page incl. tesseract;
detections in `out/thumb/ss/det-300dpi-NNN.json`), then `superscript_seq.py`: psm-7 digit
OCR, a nearest-template second reader bootstrapped from the book itself (1276 digit glyphs
from marks with psm-7 confidence >= 90), consensus, and the numbering prior. Full output:
`out/thumb/ss/seq-final.txt`.

| ch | PDF pages | marks | max seen / notes in References | first cites | re-cites | gaps | flagged |
|---|---|---|---|---|---|---|---|
| 1 | 17-49 | 105 | 61 / 61 | 59 | 45 | 1 | 4 |
| 2 | 50-77 | 66 | 39 / 39 | 37 | 28 | 1 | 5 |
| 3 | 78-108 | 92 | 64 / 64 | 58 | 31 | 3 | 7 |
| 4 | 109-144 | 123 | 86 / 86 | 86 | 37 | 0 | 6 |
| 5 | 145-175 | 111 | 69 / 69 | 69 | 41 | 0 | 4 |
| 6 | 176-202 | 92 | 59 / 59 | 57 | 34 | 1 | 7 |
| 7 | 203-228 | 88 | 64 / 64 | 60 | 26 | 2 | 6 |
| 8 | 229-263 | 91 | 48 / 48 | 46 | 44 | 1 | 2 |
| 9 | 264-280 | 37 | 28 / 28 | 28 | 9 | 0 | 1 |
| all | 264 pages | **805** | 518 / 518 | 500 | 295 (37 %) | 9 | **42 marks on 36 pages** |

* **Coverage:** in every chapter the highest number found equals the number of notes in
  References and Notes, and 509 of the 518 note numbers are found as first citations in
  increasing order; the other 9 are the "gaps" (see below). 216 of the 264 body pages
  carry at least one mark, 3.05 per page on average; the busiest are PDF 128 (9), 227 (8),
  21 and 104 (7).
* **Digit reading:** psm-7 tesseract alone is wrong on ~4 % of marks, almost all
  **5 -> 3** (`55`->`33`/`35`, `57`->`37`, `52`->`32`, `5`->`3`) and a few `x5`->`x35`
  (55 hand-labelled hard cases in `superscript_seq.py`: psm-7 37/55 right). The template
  reader gets 54/55 of those; with both readers agreeing or the prior choosing between
  them, 54/55 (the remaining one is a chained `71,72` label error on my side). Over the book
  the two readers disagree on 30 marks (3.7 %), and every disagreement was a psm-7 error.
* **Both readers wrong, agreeing:** 2 of 805 — PDF 43 L22 `year.⁵⁵` read 35 (conf 93) and
  PDF 90 L31 `performance.²⁵` read 23 (conf 90). Both produce a plausible "re-cite" and
  are caught only by the chapter-level gap check (55 and 25 never get a first citation).
* **Gaps (numbering prior fires):** 9. Two are those agreed misreads (ch1 55, ch3 25); for
  the other seven (ch2 26, ch3 42 and 46, ch6 30, ch7 33 and 48, ch8 32) I looked at every
  line between the neighbouring marks (`out/thumb/ss/gap-g*.png`): there is no superscript
  there, and the IA text layer has no trace either, so the book itself cites those notes
  out of order or not at all (46 and 32 do appear later as re-cites). They are cheap to
  escalate and the answer will be "nothing to fix".
* **False positives:** 1 unflagged-by-geometry (PDF 157 L08, hand lettering "Amplifiers"
  in a figure that slipped through the paragraph-confidence filter) — it is flagged anyway
  (conf 7, odd geometry, violates the prior). On the printed body text I found no false
  positive; `M²`, `2¹⁰` (PDF 223 L11), `Fe₂O₃`, °C, apostrophes and quote marks are all
  rejected.
* **Missed by geometry:** the scan-dropped glyphs. Three found: PDF 20 L32
  `calculations.⁸`, 50 L23 `logic.¹`, 55 L19 `Allegheny-Ludlum.¹⁰?` (all reduced to a
  1-3 px speck; `out/thumb/ss/dropped.png`). A tight text-layer pattern — terminal
  punctuation followed by one stray `' ’ ® ° ! * ¢` with no detection at that word — has
  5 hits in the whole book, 3 of them real. The looser pattern in `superscript_seq.py`
  (127 hits) is dominated by close-quotes and decimals and is not worth using.
* **Low confidence:** 33 marks have psm-7 confidence < 70 although 31 of them read
  correctly once the template reader agrees; with agreement between the two readers as
  the acceptance rule instead of the confidence, the flagged set shrinks to ~13 marks
  (9 gaps + 2 odd + 2 dropped-glyph text hits) plus the 3 text-only hits.

## 5. Recommended production approach

1. **Geometry first, on the 300-dpi render, using the hOCR the body OCR already produces.**
   The component rule above is essentially perfect on printed body text (0 FP on 6 pages,
   and the whole-book run flags only a handful of doubtful marks). Restrict it to lines of
   text blocks (paragraph confidence ≥ 50, x_size within ±25 % of body); never run it on
   figure regions. Keep `M²`-style marks (preceded by a full-height glyph) out of the
   reference list but keep them in the text as `<sup>` (tesseract already reads them as
   `M2`).
2. **Mask the marks and re-OCR the page** (or at least the affected lines) so the body
   words are clean; then splice `<a epub:type="noteref">n</a>` after the word whose bbox
   ends at the mark's left edge. This is far more robust than trying to strip tesseract's
   junk suffixes (`.®`, `.!?`, `.2¢`, `?"”`) from the raw text.
3. **Read the digits twice**: `tesseract --psm 7` on a 4x binarised crop (21/21 on the six
   pages, but ~4 % wrong over the book, nearly all 5→3), **and** a nearest-template
   classifier built from the book's own superscript glyphs (bootstrap templates from
   high-confidence psm-7 marks; 54/55 on the hard cases). Accept when both agree; when
   they disagree let the numbering prior choose (that resolved all 30 disagreements in the
   book); escalate if neither fits.
4. **Apply the per-chapter numbering prior as a checker, not as a corrector**: with
   max_so_far m, a reading v is fine if v ≤ m+1; v = m+2..m+3 means a reference between was
   missed or unreadable (flag the span); v > m+3 or v > N_chapter is a misread. Because ~1/3
   of marks are re-cites, the prior cannot pick the value on its own, but combined with
   the reader it is a strong consistency check. Also verify at chapter end that every
   1..N was cited at least once (N from References and Notes, which the pipeline should
   parse anyway) and that the endnote's content plausibly matches the sentence for any
   mark that was decided by the prior alone.
5. **Escalate to a vision sub-agent** when: the geometric detector fires but digit OCR is
   empty or < 70 conf or the two readers disagree; a component count does not match the
   digit count; the numbering check flags a gap/violation; or a text-layer word ends in
   terminal punctuation plus one stray `' ’ ® ° ! * ¢` (`[.,;:?!][\'’®°!*¢]$`) with no
   detection nearby (this catches the scan-dropped glyphs; 5 hits book-wide, 3 real). Send the sub-agent the 450-dpi crop of the line plus the
   allowed value set {1..m+1} and the candidate endnote texts. Expected volume from the
   whole-book run: ~15-45 marks depending on how conservative the rule is: 42 marks on 36 pages with the
   confidence rule (`CONF_ESC=70`), ~15 on ~14 pages if two agreeing readers are accepted
   regardless of confidence (9 gaps, 1 figure false positive, 2 agreed misreads found via
   gaps, 3 dropped-glyph text hits). Either way it is a page-level batch of at most a few
   dozen crops, not hundreds. Every escalation must be given the allowed set
   {1..max_so_far+1} and the References entries, because the vision model will face the
   same 3/5 ambiguity at 300 dpi; the numbering context resolves it.
