# Survey: ordinary body pages

Sample (PDF pages): 22, 23, 37, 58, 71, 95, 120, 137, 158, 170, 195, 222, 240,
258, 275. Three of these (37, 58, 71) turned out to be full-page figure+caption
pages; the other twelve are full text pages. Chapter opener p50 and list page
p207 were glanced at for the pitfalls section only.

Scratch outputs (all under out/thumb/): `x-NNN.png` (300 dpi renders, 1609x2578 px),
`p1_NNN.{hocr,txt}` (tesseract --psm 1), `p3_NNN.{hocr,txt}` (--psm 3),
`hocr.py` (tiny hOCR dumper: `python3 hocr.py lines|low FILE [thr]`),
`crops/` (detail crops used below). All pixel coordinates below are 300 dpi.

## 1. Typography inventory

Body type is a Times-like serif, justified with *very* loose word spacing and
**no hyphenation at all**: zero line-end hyphens in psm1 output on all 15 pages,
and zero lines ending in `[a-z]-` in the IA text layer over pp. 17-280. All
hyphens in the text are therefore lexical (card-feed, flip-flop, four-times,
half-select) and must be kept. Body line pitch is 61-63 px (0.205 in), text
measure ~1300 px, left margin 132-190 px depending on page (see section 5).

| Feature | Occurs? | Examples |
|---|---|---|
| Paragraph first-line indent | yes, universal | 75-80 px (~0.25 in): p22 y=189 x0=220 vs margin 145; p137 x0=250 vs 173; p258 x0=256 vs 181; p275 x0=211 vs 132. No extra vertical space between paragraphs (pitch stays 62 px, p22 y=1136->1191). |
| Non-indented paragraphs | yes | The first paragraph after a subheading (p170 y=655 "The first IBM core planes", p258 y=2277 "Large quantities"), the first paragraph of a chapter (p50 "The search for memory"), and every paragraph inside a block quote (p137 y=2143, 2452) are flush left. |
| Block quotations | yes (1 in sample) | p137 y=2099-2493: a quoted progress report. Same left margin (x0=168 vs body 170-175), same width, same x_size (39-40 px) but tighter leading, pitch 53 px vs 62, and a ~2-line blank gap before/after (1982->2099 = 117 px; 2300->2408 = 108 px). It has its own mini headings "Ferroxcube Cores —", "RCA Cores —" (roman, flush left, own line, followed by an em dash). Not indented, not visibly smaller: the only geometric cue is line pitch + surrounding blank lines. Introduced by a sentence ending in a colon plus endnote: "three potential vendors:58". |
| Italics | yes, sparse (~0-2 per page) | emphasis: p22 "proud of *his* automatic calculating machine"; titles: p58 caption "*IBM News*", p95 "*Quarterly Progress Report*"; run-in paragraph heads: p275 "*Commitment* — Hank DiMarco says...", "*Competition* — External competition...". Tesseract 5 emits no `<em>` and no `x_font` (checked with `-c hocr_font_info=1`: only `x_fsize 9/10`, useless). Italics are invisible to the OCR except via errors (see 3d). |
| Small caps | yes, frequent (1-6 per page) | Machine/project acronyms are set in true small caps: ASCC (p22 x6, p23), SSEC (p23 x6), MTC (p95 x5, p120 x3), SAGE (p120, p137 x2), UNIVAC (p158 x2, p275), NPL (p222 x3), CCROS/TROS (p240 x11), BPS/BOS (p258), BCROS (p275), SMS/SLT (p207). By contrast IBM, MIT, RCA, NSA, ERA, STL, CuMn, H-200 are full caps (p22, p137, p195, p275). Proper nouns in headers are also full caps ("Project SAGE" p137 header). |
| Section subheadings | yes | Bold, flush left, sentence case, own line, ~2 line pitches of space above (p170 y=468->601 = 133 px; p258 2087->2217 = 130 px) and normal pitch below (54-60 px). p170 "Array Wiring Improvements", p258 "A Successful Technology", p50 "A First Choice". Tesseract gives them their own `ocr_par`. Also an un-headed extra blank line on p95 y=346->462 (116 px gap, next paragraph indented) — a section break with no heading. |
| Figures + captions | yes | p37 (photo), p58 (line drawing with in-figure labels "1/2 Current", "Selected Core"), p71 (two photos). Caption = bold title line ("An early magnetic drum", "Coincident current selection", "The first functional ferrite core memory") then caption text in smaller type (x_size 36 vs 39-40, pitch 47 px vs 62), flush left, no indent, sometimes hanging past the body margin (p37 x0=118-123 vs body 145). Text inside figures gets OCR'd as garbage lines: p71 "108" (psm1) / "ALPHABETICAL" (IA) from the typed label on the photo; p58 "Selected Core", "1/2", "Current". |
| Endnote references | yes, 2-6 per page | Superscript digits after terminal punctuation, e.g. p22 `figures."12`, `tons.12`, `Babbage.13`, `contributions.14`, `conceive.10`, `director.8`; multiple refs joined by comma: p120 `1954.27,33`, p222 `development.30,31`; after a colon: p137 `vendors:58`. The superscripts are small, raised, roughly x-height. |
| Footnotes at page bottom | **never** | All notes are endnotes (pp. 281-316). |
| Tables | **never** in sample | |
| Bulleted lists | **never** in sample | |
| Numbered lists | not in sample, but exist | p207 (printed 191): "1. Improve the SMS technology...", "2. Finish...", "3. Initiate..." — number flush left, no hanging indent, items separated by a blank line, set with block-quote-style tighter leading. Also p230. |
| Equations | **never** in sample | Only inline dimension products "80×12", "64×64×17", "32×32×17" (p71, p95, p120, p240) and fractions "1/8", "1/16", "0.3" set as plain text. |
| Drop caps | **never** | p50 chapter opener: big chapter number + sans title, then plain flush-left paragraph, no drop cap. |
| Quotation marks | | Double quotes are straight typewriter-style `"` (two ticks, see crops/montage1.png row 1), apostrophes are curly `’` ("IBM’s"). Ellipses are spaced ". . ." (p95, p137, p222). Dashes are true em dashes without spaces ("year—an", p23). |

## 2. Running header pattern

One line, ~120 px above the first body line (header y0 62-125, body starts y0
189-247), spanning the full text measure (x0 ≈ margin, x1 ≈ right margin).

* Verso (even PDF page = even printed page): page number flush left, "Chapter N"
  flush right. p22 `6 ... Chapter 1`, p58 `42 ... Chapter 2`, p120 `104 ... Chapter 4`,
  p158 `142 ... Chapter 5`, p222 `206 ... Chapter 7`, p240 `224 ... Chapter 8`.
* Recto (odd): chapter title flush left, page number flush right. p23
  `The Postwar Challenge ... 7`, p95 `A Memory From Whirlwind ... 79`, p137
  `Project SAGE ... 121`, p195 `Project Stretch ... 179`, p275
  `Managing Technological Change ... 259`.
* Header type is slightly smaller than body (x_size 36-38 vs 39-41).
* Fresh tesseract at 300 dpi reads all 15 headers correctly as one line
  ("6 Chapter 1", "The Postwar Challenge 7"). The IA layer is unreliable
  here: it concatenates the two parts without a space or garbles the number —
  p22 "Chapter 16", p58 "Chapter 242", p222 "Chapter 7206", p240 "Chapter 8224",
  p23 "The Postwar Challenge i", p37 "... al", p95 "... 719", p137 "Project SAGE 2211".
  Recommendation: detect the header as the first line with y0 < ~140 and strip it
  by geometry rather than by text; the printed page number = PDF page - 16 anyway.

## 3. OCR error catalogue (tesseract 5.5, 300 dpi, psm 1)

Overall: 2-9 real word errors per ~370-word page (roughly 1-2% of words), plus
casing loss on every small-caps word and glyph-variant noise on every quote
mark. Nearly all of the damage is concentrated in a few classes:

**(a) Superscript endnote numbers — the dominant error class.** 43 words with
a trailing superscript in the 12 text pages; 27 of them (63%) are wrong,
including the quote-glyph errors, ~22 wrong counting digits only. Digits are
replaced by look-alike punctuation: 1→`!`, 0→`°`/`©`/`9`/`>`, 2→`?`, 4→`*`,
5→`S`/`>`, 6→`®`/`¢`, 7→`”`/`’`, 8→`®`, and the comma between two refs becomes a
period. Examples (truth → psm1 / IA):
`tons.12` → `tons.!?` / `tons.!?` (p22); `contributions.14` → `contributions.!*` (both, p22);
`director.8` → `director.!` / `director.` (p22; the IA layer simply drops it);
`1930s.10` → `1930s.1°©` / `1930s.!°` (p23); `division.15` → `division. !>` / `division.!5` (p23);
`1954.27,33` → `1954.27.33` / `1954.27.33,` (p120); `wiring.27` → `wiring.2”` / `wiring.*’7` (p120);
`vendors:58` → `vendors:38` / `vendors:58` (p137); `capacity.35` → `capacity.33` / `capacity.35` (p158);
`blackmailed."55` → `blackmailed."5S` / `blackmailed.''55` (p170); `Sr.60` → `Sr.60` / `Sr.®°` (p170);
`structure.61` → `structure.¢!` / `structure.®!` (p170); `lines.45` → `lines.*>` (both, p195);
`divisions.28` → `divisions.?28` / `divisions.?®` (p222); `development.30,31` → `development.30.3!` / `.3°.3!` (p222);
`unit.21` → `unit.2!` (p240); `each.18` → `each.!8` (p240); `accomplished.46` → `accomplished.*¢` / `.*®` (p258);
`contacts.47` → `contacts.4?` / `contacts.47` (p258); `commitment.20` → `commitment.2?` / `.2°` (p275);
`points.21` → `points.2!` / `points.?!` (p275).
Correct in both: `check.34`, `(MTC).33`, `tubes.32`, `tablets.74`, `Harvest.42`,
`supported.38`, `machines.34`, `Bureau.22`, `line.23`, `bit.3` (p258; IA drops it).
Pattern for a pipeline: any token matching `[.:"’]+[0-9!?*°®¢©>S”’]{1,5}$` at the
end of a sentence is a note-ref candidate; map the look-alikes back to digits and
range-check against the chapter's note count. Expect to hand-check most of them.

**(b) Small caps / all caps.** Full-caps words are reliable (IBM conf 86-91,
MIT, RCA, NSA, STL, UNIVACs conf 91 all correct). Small caps come out in random
case, and sometimes with letter substitutions: ASCC → `Ascc`, `ascc`, `AscCc,` (p22),
`Ascc` (p23); SSEC → `SSEC`, `sSEc` (p23; IA: `ssEC`, `SsEC`); MTC → `MTC`, `mTC,` (p95;
IA: `mrc`, `mtc`, `MTc`); SAGE ok in psm1, `sAGE` in IA (p120); NPL → `NeL),` (conf 4,
p222); TROS → `TrRos` (p240; IA `TRos`); CCROS → `ccros` (p240); BOS → `BOs` (p258);
UNIVACs → `uNIvACs` (IA, p275). Fix: build a lexicon of the small-caps acronyms
(ASCC, SSEC, MTC, SAGE, UNIVAC, NPL, CCROS, TROS, BCROS, BPS, BOS, SMS, SLT, ERA
...) and case-insensitively normalise, then tag as `<span class=sc>`.

**(c) Hyphenation.** None at line ends (see section 1), so no de-hyphenation
step is needed. Real hyphens survive fine ("flip-flop", "molybdenum-Permalloy",
"three-hole-core"). The only hyphen error is a fleck: p120 "be segmented" →
`be-segmented` (psm1, conf 18) / `be.segmented` (IA).

**(d) Italics.** Italic text is mostly read correctly ("his", "Quarterly
Progress Report", "Commitment", "Competition"), but italic capital I fails:
p58 "*IBM News*" → `/BM` (psm1, conf 35) / `JBM` (IA). No italic markup is
available from tesseract; italics would have to be recovered from a known
list (titles, the run-in heads on p275) or by eye.

**(e) Digits and dates.** Very good: 3304, 72, 24, 23, 51, 530, 1945, 1024,
17,408, 0.12, 1/16, 2841, 4032, 73,728, 200,000, $675 all correct (conf 88-96).
Failures: "1s and 0s" → `1s and Os` (both engines, conf 81 — plausible-looking, so
hard to catch); "Harvard Mark I" → `Mark 1,` (psm1, conf 77; IA right); "M-4I" →
`M-41` (psm1, conf 71; IA right); roman "I" vs "1" is the recurring ambiguity.
Multiplication signs: printed "64×64", "80×12", "12×60" come out as `x` with
random spacing: `64 x64`, `64 x 64`, `80x 12`, `12x 60`, `96 x60`, `32x32x17`
(conf 50-66 when split). Normalise `\d+\s*x\s*\d+` → `×`.

**(f) First/last lines and header.** psm1 reads the header and the last body
line correctly on every page; no truncation at either edge. The IA layer's
header is bad (section 2) but its body first/last lines are fine.

**(g) Scan flecks and the fixed caret mark.** Every verso page has a small
caret-shaped mark "^" in the left margin at the same place: about x=147-159,
y=1697-1713 (p120, p158; 12x15 px, i.e. ~0.5 in from the left edge, 5.67 in
from the top), just left of line ~25. Confirmed on p22, 120, 158, 170, 222,
240, 258 and on ten more even pages 24-46 (crops/versomarks.png) and the
chapter opener p50. When the text margin is close to it, it is read as part of
the first word: p22 `nproject` (conf 44), p120 a separate `>` line emitted *first*
in reading order (conf 94!), p222 "for" → `tor` (conf 59), p240 "line" → `Ifne.23`
(conf 19), p258 "cost" survived. Recommendation: mask a ~40x40 px box around
(150,1705) on even pages before OCR. Random dots also occur: p22 "as well as·" →
`ass` (conf 0); p23 stray `.` line at y=1202; p95 `"` before "was called";
p137 `‘General`; p158 "reliability of" → `reliability Yof` (there is a pen mark
on the page).

**(h) Quote marks.** The straight `"` is rendered by tesseract as `"`, `''`,
`"'`, `'"'`, `""`, `"”` or `‘…’` more or less at random (p158: `'"'but`, `tools.""`,
`''banks`; p23 `'at least 250 times as fast'`). Normalise any run of
`['"‘’“”]{1,3}` adjacent to a word boundary to `"` (keep a single `’` inside a word).

**(i) Other single-word errors (psm1):** p120 "six" → `Six` (76), "plane" → `planc` (71);
p195 "all" → `ail` (53); p23 "connecting a" is correct in psm1 but `connectinga` in IA;
p137 IA inserts `_` for a wide gap ("has _ been"); IA `Minnesota. )`.

### psm 1 vs psm 3 vs IA layer

* **psm 1 and psm 3 are byte-identical** on all 15 pages (text and hOCR).
  Disagreement between them is useless as a signal on this book; pick either.
* **IA layer vs fresh psm1:** comparable body quality, with different mistakes.
  IA is worse on headers, on spacing (`connectinga`, `has _ been`), drops some
  refs (`director.`, `bit.`), and garbles superscripts a bit more (`®°`, `%`); psm1
  has the `Mark 1`, `M-41`, `Six`, `planc`, `ail`, `tor`, `Ifne`, `NeL` errors.
  Across the 12 text pages there were ~60 differing lines; almost every one
  contains a real error in at least one engine. Cases where both agree and both
  are wrong are rare: `Os` (p95), `lines.*>` (p195), `Ifne.23` (p240),
  `contributions.!*`/`tons.!?` (p22), plus all small-caps casing. So
  **engine disagreement is a good low-confidence flag** (high recall for
  superscripts, flecks and quote glyphs; misses ~3-5 shared errors per 15 pages),
  but note that ~1/3 of the disagreements are harmless glyph variants
  (`''` vs `"`, `‘` vs `'`, `64 x64` vs `64x64`) that a normaliser should fold first.

## 4. hOCR x_wconf as a signal

Word counts: 338-390 words/page; words with conf < 80: 6-15 per page (1.7-4.2%);
conf < 90: 38-73 per page (11-20%, too many to review).

p22, all 15 words with conf < 80 (wrong marked *):
`"accomplished` 52 (ok), `figures.''12` 54 (*quotes), `about` 76, `five` 76,
`tons.!?` 67 *, `Babbage.13` 71, `1,` 77 * (Mark I), `AscCc,` 47 *, `nproject` 44 *,
`contributions.!*` 21 *, `the` 78, `ass` 0 *, `conceive.!9` 64 *, `The` 73, `director.!` 63 *.
→ 9 wrong / 15 flagged; every real error on the page is flagged except the
small-caps casing of `Ascc` (conf 80-89) which is a class of its own.

p222, 13 words < 80: `development.2’”` 51 *, `programmers` 43, `and` 57, `forty-four` 78,
`college` 72, `IBM` 74, `divisions.?28` 35 *, `requirements.2®` 41 *, `tor` 59 *,
`development.30.3!` 61 *, `NeL),` 4 *, `"I` 73, `NPL.` 23. → 6 wrong / 13 flagged,
all six real errors caught (the wide-gap line y=440 drags four correct words down).

p240, 14 words < 80: `TROS.` 78, `unit.2!` 77 *, `TrRos` 70 *, `products.??` 74 *,
`ccros` 73 (casing), `Ifne.23` 19 *, `Each` 74, `12x` 50, `60` 50, `96` 66, `x60` 66,
`each.!8` 44 *, `predicted.` 66, `Air` 47. → 5-6 wrong / 14.

Errors that slip through at 80: `points.2!` 88 (p275), `Os` 81 (p95), `M-41` 71
(caught), `Mark 1` 77 (caught), `bit.3` 91 (correct), `decision."33` 80 (quote
only). Among the 43 superscript-bearing words, 34 have conf < 80; the only
substantively wrong one above 80 is `points.2!`.

Verdict: confidence is a useful signal here. **Threshold 80** gives ~2-4% of
words to review with roughly 50-60% precision and catches nearly every real
error except (i) small-caps casing (conf 73-91, handle by lexicon), (ii)
plausible substitutions like `Os`, `Six`, `points.2!` (81-88). Going to 90 triples
the review load for very few extra catches. A better selector than a flat
threshold: conf < 80 **or** token matches the superscript / multiplication /
acronym patterns **or** engines disagree.

## 5. Paragraph boundaries from geometry

Measured from `ocr_line` bboxes (psm1):

* Left margin per page (mode of line x0): p22 145-160, p23 ~161, p95 134-145,
  p137 170-175, p170 160-167, p222 ~187, p258 173-191, p275 132-135. It drifts
  by up to 15-18 px *within* a page from top to bottom (skew: p22 145→160,
  p258 173→191), so compute the margin per page (or per page-third), not globally.
* Indented first lines sit at margin + 75-80 px: p22 220/227/232, p95 220/221/215/212,
  p137 250/250, p170 240/243, p258 250/256/260, p275 211-213. Gap between the
  two clusters is ~60 px wide — a threshold of margin+40 px separates them cleanly
  on every page. Nothing else in the sample starts at that offset.
* Last lines of paragraphs are short: x1 = 519, 1272, 346, 965, 666, 462, 1279,
  827, 883, 1120 vs full lines at 1443-1497. But "short" can be nearly full
  (p22 `contributions.14` line x1=1272, p137 `memories.` x1=1353 vs 1473), so use
  x1 < full-width − ~60 px, and never rely on it alone: a paragraph can end on
  a full line, in which case only the next line's indent tells.
* Combined rule: new paragraph iff line.x0 > margin+40, **or** previous line was
  short (x1 < right−60) *and* a vertical gap > 1.5 pitches or a heading intervenes.
  Tesseract's own `ocr_par` segmentation on these pages actually matched this on
  all 12 text pages (it split at indents and at the p137 quote / p170, p258
  headings) — it can be used as a first pass.

Pitfalls:
1. **Non-indented paragraph starts**: after a bold subheading (p170 y=655, p258
   y=2277, p50 "By 1948 IBM engineers"), at chapter start (p50), and inside block
   quotes (p137). Detect headings by the ~130 px gap above + short bold line
   (bold is not reported by tesseract; use gap + line length + no terminal period).
2. **Block quotes** are not indented and are the same x_size; distinguish by line
   pitch 53 vs 62 px and 108-117 px blank gaps on both sides (p137 y=1982→2099,
   2300→2408). Their internal "Ferroxcube Cores —" lines look like headings.
   Numbered lists (p207) use the same tighter leading with a blank line between items.
3. **Page-top continuation vs new paragraph**: first body line indent works
   (p22, p95, p258, p275 continue; p137 "John Gibson, undertook" continues at
   margin; p240, p258 start new paragraphs at margin+77). No drop caps anywhere.
4. **Un-headed extra space** (p95 y=346→462) — treat as a section break, not a
   heading.
5. The verso caret mark can create a bogus 1-word line/paragraph (p120's `>`
   at y=1697 was emitted as the *first* paragraph of the page) and shift a
   line's x0 left by ~10-25 px (p22 y=1682 x0=145 vs neighbours 155): mask it.
6. Captions (p37, p58, p71) start left of the body margin (x0 118-123 vs 145)
   with pitch 47 px; treat any page whose text block starts below y≈2000 as
   figure+caption, and drop OCR lines lying inside the figure bbox.
7. Figure-label garbage (p58 "1/2 Current", p71 "108") appears as short lines
   in the top part of the page.
