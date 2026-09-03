# Back matter survey: References and Notes, Chronology, Index

Pages are PDF pages (printed = PDF - 16). All fresh OCR is tesseract 5.5 on 300-dpi
grayscale renders (`out/thumb/bm-NNN.png`); outputs are in `out/thumb/`:

- `bmfull_NNN.{hocr,txt}`  psm 1, pages 281-339
- `bm4full_NNN.{hocr,txt}` psm 4, pages 281-328
- `bm1_/bm3_/bm4_/bm6_NNN` psm 1/3/4/6 on the sample pages 281-283, 300, 316-318, 328-330, 339
- `numstrip_NNN[w].hocr`   OCR of just the note-number column (see 1.4)
- `col_NNN_{L,R}.{png,hocr,txt}` index pages split into columns and OCR'd with psm 4
- `refparse.py` the prototype entry parser used for the counts below.

A scanner blemish sits at the same image position on every page: a ~12x15 px blob at
x≈175-200, y≈1660-1680 (300 dpi; ≈ x 88-100, y 830-840 at 150 dpi). It lands in the
number/date column of verso pages and produces junk tokens: p292 `e 4.`, p300 `44.`
(ok), p306/p318 `L}`, p310 `$6.` for `56.`, p318 `8 a three-dimensional`, p322
`a machines.`, p324 `A/57` for `4/57`, p326 `. approves`, p288 `"3,`. Mask small
connected components in that box before OCR (it is small; the real numbers there are
28 px tall) or drop 1-2 char tokens with bbox inside it.

---

## 1. References and Notes (PDF 281-316, printed 265-300)

### 1.1 Layout

- p281: section title "References and Notes", then an 11-line **introductory
  paragraph** at the full text margin (x0≈122 @300dpi) explaining sources and the
  `(*)`/`(**)` markers (MITRE archives / MIT MC-140 collection). This paragraph must be
  kept: the markers appear in dozens of notes (e.g. p300 notes 41, 42, 46).
- Then bold subheads **"Chapter 1" … "Chapter 9"** (p281 y=1107, p285 y=1242, p288
  y=1180, p292 y=1211, p297 y=1756, p302 y=794, p306 y=1592, p311 y=733, p314 y=2152).
  No other subdivision.
- Under each subhead, numbered entries restarting at 1. Number `N.` in the same roman
  font as the text (not bold, not a different face), in its own column: number x0 ≈
  115-155 on rectos, ≈ 150-215 on versos; text column x0 ≈ 265-290 (recto) / 315-350
  (verso). Hanging indent: every continuation line starts exactly at the text column
  (±5 px). Line pitch ≈ 47-50 px; the gap between entries ≈ 86-107 px (i.e. ~1 blank
  line). Entries are justified.
- Entries can start anywhere on the page and the section is set continuously (a chapter
  subhead can fall mid-page, e.g. p292 "Chapter 4" after ch. 3 note 64).
- p316 (last page) has only 5 entries (ch. 9 notes 24-28), rest blank.

### 1.2 Counts (verified by three independent passes; see 1.4)

| Chapter | notes | PDF pages |
|---|---|---|
| 1 | 61 | 281-285 |
| 2 | 39 | 285-288 |
| 3 | 64 | 288-292 |
| 4 | 86 | 292-297 |
| 5 | 69 | 297-302 |
| 6 | 59 | 302-306 |
| 7 | 64 | 306-311 |
| 8 | 48 | 311-314 |
| 9 | 28 | 314-316 |
| **total** | **518** | |

Each chapter's numbers are a gap-free 1..N sequence. These are the targets the body
superscripts must map onto 1:1. (The IA text layer carries body superscripts only
sporadically as digits glued to words, e.g. ch.1 pages show 0-58 with most values
missing, so the body-side numbering has to come from fresh OCR, not `out/text`.)

### 1.3 Entry properties

- **Spanning pages**: yes, 9 entries: ch2 n6 (285→286), ch3 n8 (288→289), ch4 n8
  (292→293), ch5 n18 (298→299), ch5 n33 (299→300), ch6 n50 (305→306), ch7 n5
  (306→307), ch7 n15 (307→308), ch8 n16 (312→313). A page-top line at the text column
  x0 with no number to its left is a continuation.
- **Multiple paragraphs inside one entry**: effectively no. Only one entry has internal
  structure: ch4 n8 (p292 bottom - p293 top) is a citation followed by a blank line and
  a **two-column list of dated items (a)…(u)** (p292 `(a) December 3, 1952   (d) March 9,
  1953` etc.; the list continues at the top of p293 with (g)…(n) / (o)…(u)). Every other
  "large gap" inside an entry that the parser flagged turned out to be a note number
  that tesseract dropped (see 1.4). So the rule "vertical gap > ~70 px ⇒ new entry"
  holds with that single exception, which should be special-cased (render as a
  sub-list inside the note).
- **Length**: of 518 entries, 185 are one line, 175 two lines, 75 three, 29 four, 12
  five, 12 six, and ~12 longer; the longest (ch4 n8) is 25 lines.
- **Italics** (from the renders; tesseract does not recover them): used for book
  titles (p281 n1 *The Lengthening Shadow*, n2 *IBM Yesterday and Today*, n5 *Herman
  Hollerith, Forgotten Giant of Information Processing*), journal/magazine names
  (p281 n6 *Business Machines*, p283 n30 *Computer*, n31 *RCA Review*, *Proceedings of
  the Institute of Radio Engineers*, n36 *Annals of the History of Computing*, p300 n45
  *IBM Journal of Research and Development*), and titled brochures (p282 n18 *IBM
  Selective Sequence Electronic Calculator*). Article titles are in straight double
  quotes, not italic. Patents, interviews, letters ("X to Y, date"), reports and
  testimony are roman. Roughly 20-25% of entries contain an italic span.
  Recovery options: tesseract's legacy engine (`--oem 0 -c hocr_font_info=1`, tested on
  p281) emits no `<em>` and bogus `x_font` names, so it is useless; a per-word slant
  measurement on the bbox crops would work but is a project; the cheap route is an
  LLM/vision cleanup pass over each entry with its crop (the entries need correction
  anyway) that also marks `*title*`. Recommend the LLM pass.
- **Pure references vs discursive notes**: most are bare citations, but a substantial
  minority (est. 10-15%) add commentary sentences or are pure commentary: p281 n7
  ("…Benjamin D. Wood was the head of…"), p293 ch4 n17 ("Others contributing to the
  solution…"), p299 ch5 n18 (continuation "…The 770, including the memory, was so
  reliable…"), p299 n26 ("The three engineers were Edward Bauer, William Lawrence, and
  Robert Ward."), p299-300 n33 (tape-recorded meeting; "As revealed in note 18 of this
  chapter…" — note the intra-chapter cross-reference), p311 ch8 n7 (Elfant, 7 lines),
  p315 ch9 ("I have personal knowledge of these vendor programs."). Short cross-refs
  ("Belden and Belden: pp. 209-210.", "Nancy Stern, p. 77.") are common (p283 has 8).
  Because of the discursive notes, the notes must be rendered as real notes, not
  reduced to a bibliography.

### 1.4 How the numbers OCR, and why the IA layer mangles them

The IA layer renders isolated numbers as words: p281 `ihe`, `Dey`, `Ep`, `Sy`, `pp` for
1, 2, 3, 5, 7; p282 `Li.`, `FZ.`, `1S.`, `iP`, `24.` for 11, 12, 15, 17, 21. The
numbers are ordinary roman digits in the text font (see p281/p283 renders) — nothing
special about the glyphs. The cause is context: each number is a one-token "line"
with 130+ px of white space to the text, so the old tesseract segmented it as a
separate block and its dictionary/language model preferred word-like outputs for a
lone 2-3 glyph token at low resolution. Fresh tesseract 5.5 at 300 dpi reads them
correctly *when it emits them* (p281-283 all correct), but has two other problems:

1. **Layout dissociation.** psm 1/3 put all the numbers of a page in a separate block
   before/after the text (p282, p283, p289, p299, p307, p311 …). psm 4 keeps
   `N. text` on one line on versos (p282, p318) but still splits on many rectos
   (p283, p289, p299, p301, p307, p311). So the `.txt` output is unusable for
   structure; the hOCR bboxes are needed regardless of psm.
2. **Dropped numbers.** On several rectos tesseract silently omits some numbers:
   psm 1 drops `8.` `9.` on p282; psm 4 drops `9.` on p289, `4.` (blemish) and `9.` on
   p292/293, `55.` on p301, `6.`-`9.` on p307, `2.`-`9.` on p311. A parser that trusts
   only detected number tokens undercounts (my first pass gave 501, not 518).
3. **Misreads in the number column** even when detected: `27.`→`217.` (p287, p303,
   p316, three times!), `22,`→`222,`, `7.`→`s`/`U`, `12.`→`117`, `22.`→`200`,
   `55.`→`5s.`. The whitelist run (`-c tessedit_char_whitelist=0123456789.`) on a crop
   of just the number column found every number (each chapter yields ≥ N tokens whose
   *positions* are complete) but still misreads ~5% of the values. Values must
   therefore be validated as "previous + 1", never trusted.

### 1.5 Proposed parser

Work from hOCR word boxes, per page, at 300 dpi:

1. Drop the running head (y < 150) and mask/ignore the blemish box.
2. Detect `Chapter N` subheads (regex `^Chapter\s*\d*$` on a line; on p311 the bold
   `8` came out as a separate one-char line, so accept "Chapter" alone and increment).
   The p281 intro paragraph is everything between the title and the first subhead.
3. Text-column x0 = median x0 of lines with ≥4 words after the first subhead (per
   page: rectos ≈ 270-290, versos ≈ 315-350). Number column = tokens with x0 <
   textx − 60 matching `^\W?\d{1,3}[.,]?$`.
4. OCR the number column a second time on its own crop
   (`x ∈ [textx−230, textx−15]`, psm 6, digit whitelist) — this finds every number's
   y position; use the y positions as entry anchors and assign values by sequence
   (expected = prev+1, reset at each subhead), logging any token whose OCR value
   disagrees. Cross-check: an entry start must also be preceded by a vertical gap
   > 70 px (or be the first line after a subhead); a gap > 70 px with no number
   anchor within 35 px is a warning (it is either a dropped number or the ch4 n8 list).
5. Attach every text line (x0 ≥ textx − 30) to the most recent anchor whose y ≤ line
   y + 35; a page's first text line with no anchor above it continues the previous
   page's last entry.
6. Dehyphenate line ends, join lines, keep the ch4 n8 (a)-(u) list as a nested list.
7. Assert per-chapter counts = 61/39/64/86/69/59/64/48/28.

### 1.6 Rendering in the EPUB — recommendation

Options:

- **pandoc footnotes `[^c3n12]`.** Pros: trivial markup, pandoc emits EPUB3
  `<aside epub:type="footnote">` with `noteref` links so Apple Books/Kobo/KOReader
  show pop-ups. Cons: pandoc numbers notes *consecutively through the whole book*
  (1…518) — print numbering restarts per chapter and the text refers to "note 18 of
  this chapter" (p300), so displayed numbers would diverge from the printed ones;
  pandoc also places each chapter's notes at the end of that chapter file, which is
  fine but loses the per-chapter "Chapter N" grouping of the printed section.
- **Linked endnotes chapter.** Keep "References and Notes" as its own chapter with the
  nine "Chapter N" subheads, each entry as `<li id="c3n12">` (ordered lists restarting
  per chapter), and body superscripts as `<a epub:type="noteref" href="notes.xhtml#c3n12"><sup>12</sup></a>`,
  with a `↩` backlink from each note. Pandoc passes raw HTML spans/links through to
  EPUB, so this needs no post-processing. Pop-up display works in readers that honour
  `epub:type="noteref"` regardless of where the target lives (Apple Books, KOReader do;
  ADE does not, but then the link just jumps).

**Recommend the linked endnotes chapter** with `epub:type="noteref"`: it preserves the
printed numbering (so "note 18 of this chapter" stays true), keeps the intro paragraph
about `(*)`/`(**)`, and matches the book's structure. If pop-ups in every reader are
the priority, additionally wrap each entry in `<aside epub:type="footnote">` in the
notes file — pop-up-capable readers use it, others show the page.

### 1.7 Error count, sample page p283 (printed 267; 18 entries, 33 lines, ~1,500 chars)

Fresh tesseract (psm 1 and psm 4 identical on text): **2 character errors**, both
straight-quote doubling (`'"The Birth`, `Systems,'"`), plus the structural problem
that all 18 numbers come out as a separate block. IA layer on p283: also clean on
text but 4 of 18 numbers garbled. On p282 the fresh run has 3 errors (`JEEE` for
*IEEE*, `WU.S.` for `U.S.`, `''` for `"`) in 21 entries. Recurrent systematic issues
seen across pages: `"` → `''`/`'"` (very common), small-caps SAGE → `sAGE` (p281),
`1.`→`I.` in initials (p286 `1. P. Eckert`), and the number-column issues above.
Expect ≈1-3 errors per page in the text plus number fixes.

---

## 2. Chronology (PDF 317-328, printed 301-312)

### 2.1 Layout

Title "Chronology" on p317 (running head on later pages is the book's own typo
"Chonology", p318-328). Two-column list: a **date column** (x0 ≈ 130-140 recto /
≈ 175-210 verso) and a text column (x0 ≈ 390 / ≈ 420; ~250 px indent), hanging
indent, ~1 blank line between entries, no year headings, no rules. 174 entries
(13-17 per page: p317 13, p318 15, …, p328 14). Dates come in two formats only:
`YYYY` (9 entries: 1911, 1914, 1924, 1942, 1952, 1956, 1968, 1970, 1979) and `M/YY`
(165 entries, e.g. `8/44`, `12/47`), chronologically ordered with repeats (p317 has
`8/44` twice, `1/46` three times). Entries are 1-6 lines (18 single-line, 88 two-line,
55 three-line, 12 four-line, 4 longer); none spans a page and none has an internal
paragraph break. Small caps are used for machine names (ASCC, SSEC, EDVAC, ENIAC,
SAGE, ASCA) — see 2.2.

### 2.2 How it OCRs

- IA layer (p317): text good, dates good, but every entry's date and text are on the
  same line only by luck of spacing; small caps come out mixed (`EDvAc`, `Ascc`).
- Fresh psm 1/3 split the date column into a separate block (p317, p318) — same
  dissociation as the notes. **psm 4 keeps `8/44 text…` on one line on every one of
  the 12 pages**, and the hOCR date tokens sit in a tight column, so parsing is easy:
  first token of a line matching `^(\d{1,2}/\d{2}|\d{4})$` at x0 < textx − 60 starts an
  entry; everything else at the text column continues it. My parser found 177 starts,
  of which exactly 3 were blemish artefacts (p318, p322, p326 at y≈1660) — 174 real.
- p318 (psm 4), 15 entries, ~1,900 chars: **4 errors** — `ssec` (small caps SSEC),
  `"kludge'"` (extra apostrophe), `8 a three-dimensional` (blemish token inserted),
  `Pépian` for Papian; psm 1 on the same page had 3 (no Pépian, `L}` blemish line).
  p317: 6 errors, all small-caps/quote related (`ascc`, `Epvac` ×2, `AscA`, `''First`,
  `Mark 1` for `Mark I`). So: small caps are the dominant error class here; a fixed
  list (ASCC, SSEC, EDVAC, ENIAC, SAGE, ASCA, EDPM, …) can be normalised by regex.
  "Labortories" (p318) and "Chonology" are the book's own typos.

### 2.3 Representation

Recommend a **pandoc definition list** (`8/44` as the term, text as the definition):
it becomes `<dl><dt>8/44</dt><dd>…</dd>` which reflows, keeps the date visually
distinct, and can be styled with a small CSS hanging indent
(`dt{float:left;width:4em;font-weight:bold} dd{margin-left:4.5em}`). A table works
too but is worse on narrow screens and in reading systems with poor table support;
plain paragraphs (`**8/44** text`) are the fallback if `<dl>` renders badly somewhere.
Keep the dates as printed (`8/44`), optionally expand to "Aug 1944" — the two
formats are unambiguous.

---

## 3. Index (PDF 329-339, printed 313-323)

### 3.1 Layout

Title "Index" on p329; running heads "314 … Index" thereafter. **Two columns** of
≈ 600 px each with a white gutter 37-42 px wide whose position varies with the
page (x ≈ 744-792 on rectos, 772-847 on versos — it must be found per page, not
fixed). No letter headings; letter groups are separated by a blank line (p329
A→B, p330 C→D and D→E: gaps of 138-140 px vs 47 px line pitch). Three indent levels
(300 dpi, relative to the column's left edge): 0 = main entry, +50-60 px = sub-entry
(p329 "Air defense system" → "coordination between IBM and / MIT on, 93-128"),
+100-110 px = turnover line of either. Totals across the 11 pages: 905 lines, **429
main entries**, 221 sub-entry lines, 255 turnover lines, 62 lines with *See*/*See also*
(italic), 357 lines with page ranges, 81 lines ending in a hyphen (word hyphenation
"Commis-/sion" and range breaks "35-/38", "218-/225", "282-/283" both occur).
Entries continue across columns/pages 7 times (e.g. p331 L top "Mecca task force work
on, 230-", p335 R top "Whirlwind computer develop-"). Last page p339 is a half page
ending with "XD-2 memory, 95, 112-117".

### 3.2 OCR

- IA layer interleaves the two columns on the same text line (p330: "Cape Cod System,
  96                 Crawford, David J., 136, 139, 140,"); the characters are largely
  right but recovering columns from the whitespace is fragile — not worth it.
- Fresh **psm 1** on the full page (p329, p330, p339) gets the columns as blocks and the
  text mostly right, but justified entries with wide word spacing get words torn out
  into separate blocks: p330 `Cryogenic computer\n209-210, 216` with `technology,`
  emitted 5 lines later, and `See` / `Co.` orphaned at the end; psm 4 on the full page
  interleaves columns (unusable).
- **Splitting by ink projection then psm 4 per column** (`col_NNN_{L,R}`) is clean:
  every line of p330 in order, sub-entry/turnover x-offsets preserved in the hOCR,
  no torn words. Errors on p330 (49 lines, ~1,700 chars): **3** — `IJr.` for `Jr.`,
  `(EMCCQC)` for `(EMCC)`, `alse` for `also`; p329: 2 (`Ampezx`, `Schonberg` for
  Schönberg); p339: 3 (`Weizsicker`, `alsc`, `World War 11`). The column crops must
  start below the running head using its bbox rather than a fixed fraction — my 8%
  crop clipped the first line on p334 R (came out as `Cillviliv=illaglivoiulll`).
  Expect ≈2-4 errors per page, concentrated in names and *See also*.
- Parse: per column, x0 level (0/+55/+105) → entry / sub-entry / continuation;
  dehyphenate; split trailing locators with `,\s*(\d+(-\d+)?)`; *See*/*See also* by
  regex; a level-1/2 line at the top of a column continues the previous column.

### 3.3 Include it?

Page numbers are meaningless in reflow unless printed-page anchors exist. EPUB3 can
carry them: insert `<span epub:type="pagebreak" id="pg123" title="123" role="doc-pagebreak"/>`
at each printed page start in the body (raw HTML passes through pandoc) and add a
`<nav epub:type="page-list">` to nav.xhtml. Pandoc does not generate the page-list
nav itself, so it means post-processing the EPUB (unzip, edit nav.xhtml, rezip) or a
Lua filter — a small, mechanical job. We know every page boundary exactly from the
per-page OCR, so placing the spans (at line boundaries, even mid-paragraph) is
cheap, and it is independently useful (readers show "page 123" in the go-to dialog,
and the notes' own `pp.` references to the book itself, e.g. index "Buffer memory,
54-61", become checkable).

**Recommendation: include the index only if page-list anchors are added, and then
link every locator** (`<a href="ch3.xhtml#pg93">93</a>`, ranges linking to their first
page). That turns a 429-entry, ~1,300-locator index into a working navigation tool
for a modest amount of work (anchors + one regex pass). Without anchors the index is
dead text and should be dropped (a *See also*-only concept list is not worth
keeping). The two-column OCR is reliable enough (3 errors/page) that OCR cost is not
the deciding factor.

---

## 4. Error counts per sample page (fresh tesseract, 300 dpi)

| Section | page | what | char/word errors | structural issues |
|---|---|---|---|---|
| References | p283 | 18 entries, 33 lines | 2 (both `"`→`'"`) | all 18 numbers emitted as a separate block (psm 1 and 4) |
| References | p282 | 21 entries | 3 (`JEEE`, `WU.S.`, `''`) | psm 1 dropped numbers 8, 9 |
| Chronology | p318 | 15 entries | 4 with psm 4 (`ssec`, `kludge'"`, blemish `8`, `Pépian`); 3 with psm 1 | dates kept inline only with psm 4 |
| Index | p330 | 49 lines / 2 columns | 3 with column split (`IJr.`, `EMCCQC`, `alse`) | psm 1 full-page tears out 3 words; psm 4 full-page interleaves columns |

Systematic classes to normalise globally: straight double quotes rendered as `''`/`'"`;
small caps → lowercase/mixed (SAGE, SSEC, EDVAC, ENIAC, ASCC, ASCA, EDPM); `27`→`217`
in the number column; `I.`/`1.` confusion in author initials; the fixed-position
blemish; diacritics (Schönberg, Weizsäcker) dropped or garbled.
