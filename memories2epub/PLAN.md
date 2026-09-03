# memories2epub — plan of attack

Source: `book.pdf` (Internet Archive scan of Pugh, *Memories That Shaped an
Industry*, MIT Press 1984; 344 PDF pages, printed page = PDF − 16).
Survey reports with all the measurements are in `out/survey/*.md`.

## What the survey established

**The scan.** MRC-compressed: a 1-bit JBIG2 text mask at 360 ppi over a 120 ppi
JPX background. Rendering above 300 dpi buys nothing. Text is ~11 pt Times-like,
justified with loose spacing, **no line-end hyphenation anywhere** in the body.
The PDF's own text layer (IA tesseract) is a usable *second engine* but not a
usable source: it garbles headers, drops most superscripts, interleaves index columns.

**Structure.** Front matter 1–16 (keep: series page 6, title 7, copyright 8,
Series Foreword 11–12, Preface 13–14; drop the rest and regenerate Contents).
Nine chapters at PDF 17, 50, 78, 109, 145, 176, 203, 229, 264; References and
Notes 281–316; Chronology 317–328; Index 329–339; 340–344 junk. Chapter openers:
big sans numeral + sans title, no running head. One level of subheads (bold serif
at body size, ~2 line pitches of space above, following paragraph unindented;
~55 total, candidate list in `structure.md`). Running head is one line at
y < ~140 px (300 dpi): verso `N … Chapter K`, recto `Title … N`. Paragraph =
first-line indent of 75–80 px over a per-page margin (margin drifts 132–190 px
between pages and ~15 px within a page from skew). Block quotes: same margin
and size, tighter pitch (53 vs 62 px) with ~2-line gaps around. Numbered lists
exist (p207, p230). No tables, no footnotes, no drop caps.

**Figures.** 67 figures on 44 pages (list + bboxes in `out/survey/figures.tsv`),
unnumbered, each with a caption directly below: bold title line + smaller-type
paragraph, optional "(Photograph courtesy of …)". Tones live at 120 ppi in the
background, edges in the 360 ppi mask, so crops must come from the composited
render. Figure-page detector: longest vertical ink run > 55 rows at 150 dpi.

**Endnotes.** 518 notes, per chapter 61/39/64/86/69/59/64/48/28, gap-free.
Body superscripts: 805 marks on 216 pages, 37 % re-citations (so the numbering
is *not* monotone; the prior is `v ≤ max_so_far + 1` and "1..N are first-cited in
order"). Tesseract mangles the superscripts (1→!, 0→°, 7→”, 5→3 …) but a
geometric detector on hOCR line boxes + connected components finds 21/21 on the
test pages with 0 false positives, and over the whole book every chapter's max
equals its note count. Prototype: `out/survey/superscript_proto.py`.

**OCR quality.** 2–9 real errors per ~370-word page. Classes: superscripts (dominant),
small caps → random case (`AscCc`, `mTC`), quote glyph noise (`''`, `'"`), `×`→`x`,
italic capital I → `/`/`J`, I/1 confusion, a **fixed scanner blemish** at
~(150–200, 1660–1710) px @300 dpi that corrupts a word on every verso and the
number/date column in the back matter. `x_wconf < 80` flags 2–4 % of words with
~55 % precision and catches nearly every real error except small-caps casing;
psm 1 and psm 3 are byte-identical (useless as a pair) but **IA layer vs fresh
tesseract disagreement flags almost every real error** once glyph variants are
normalised. Tesseract cannot see italics at all.

## Pipeline

Everything is a plain script under this directory; `out/` is disposable.
Stages 1–7 are fully mechanical (no agents) and produce a complete draft EPUB
with unresolved items left as visible markers. Agents are only used in stage 8.

1. **`render.py`** — 300 dpi grayscale PNGs (done: `out/png300`); mask small
   connected components inside the blemish box on every page (don't blank the
   box: on some pages the text margin overlaps it).
2. **`ocr.py`** — tesseract 5.5 `--psm 1` hOCR for every page → `out/hocr`.
   Back matter: `--psm 4` for chronology (keeps date+text on one line);
   index pages split at the gutter by ink projection, `--psm 4` per column.
   Notes pages: number column re-OCR'd separately with a digit whitelist.
3. **`superscripts.py`** — geometric detection → paint the marks white →
   re-OCR the 216 affected pages so body words come out clean → read each mark
   twice (psm 7 on a 4× crop + nearest-template classifier bootstrapped from the
   book's own high-confidence glyphs) → apply the per-chapter prior as a checker
   → emit `out/marks/pNNN.json` (word, value, status ∈ ok/flag). Also the
   text-pattern check for the 3 scan-dropped glyphs. Expected flags: ~15–45.
4. **`layout.py`** — per page, hOCR → typed blocks: `head` (drop), `figure`
   (drop OCR lines inside figure bbox), `caption` (title + text), `chapter`
   opener, `subhead` (gap ≥ 1.8 pitch + ≤ 8 words + no terminal period +
   next line unindented; must match the candidate list or is flagged),
   `para` (indent rule, per-page margin), `quote` (pitch rule), `list`,
   with a page-top continuation flag. Output `out/blocks/pNNN.json`.
5. **`figures.py`** — crop from the 300 dpi render using `figures.tsv` bboxes
   refined by the ink region and stopped at the first caption line; level-stretch
   (paper 150→white); photos downscaled to ~200 dpi, line art kept at 300 dpi;
   `out/figures/pNNN-k.png`. Contact sheet of all crops for one visual check.
6. **`notes.py` / `chronology.py` / `index.py`** — parsers per `backmatter.md`
   §1.5, §2.2, §3.2. Hard asserts: 518 notes with the per-chapter counts,
   174 chronology entries, ~429 index entries. Index locators become links to
   printed-page anchors.
7. **`normalize.py` + `assemble.py`** — text fixes that need no eyes: quote
   runs → `"`, `\d+\s*x\s*\d+` → `×`, small-caps lexicon (ASCC, SSEC, MTC, SAGE,
   UNIVAC, ENIAC, EDVAC, NPL, CCROS, TROS, BCROS, BPS, BOS, SMS, SLT, ASCA, …)
   → caps, blemish tokens dropped, `. . .` kept. Then build Markdown per
   chapter with raw-HTML noterefs
   (`<a epub:type="noteref" href="notes.xhtml#c3n12"><sup>12</sup></a>`),
   `<span epub:type="pagebreak" id="pg93" title="93"/>` at every printed page
   start, figures as `![caption](…)` with the bold caption title. Then
   `pandoc --to epub3 --toc-depth=2` with a small CSS, metadata from the
   copyright page (ISBN 0-262-16094-3), title-page render as cover, and a
   post-pass that adds the `page-list` nav. `epubcheck` at the end.
   **Draft v0 exists at the end of this stage**, with every unresolved item as
   an inline `⟦?…⟧` marker so it is greppable.
8. **Escalation (sub-agents).** Selection rule for a body word: `x_wconf < 80`
   OR matches a suspect pattern (superscript residue, `x` between digits,
   lexicon near-miss, stray `*`/`^` at line start) OR disagrees with the IA
   layer after normalisation. Agents never see whole pages: they get a
   montage of the flagged *line strips* at 300 dpi, the draft text of each
   line, and return corrections as JSON. Batches:
   - superscript flags: 2–3 agents, each given the allowed set {1..max+1} and
     the candidate note texts;
   - body flagged lines: ~15 pages per agent → ~18 agents (expect ~5 residual
     flags/page after normalisation and IA-agreement filtering);
   - References and Notes: 36 pages, ~6 agents — they also mark italic titles
     (`*…*`), which tesseract cannot recover; body italics (0–2/page) are
     picked up by the same flagged-line review plus a slant-angle heuristic on
     word crops, to be tried first and kept only if it validates on the known
     italic samples from the survey;
   - chronology + index: 3 agents; figure crops + captions: 1–2 agents;
   - final QA: 3–4 agents each read one chapter of the built EPUB text
     against the renders at a coarse level and report anything structural.
   ≈ 35–40 agents total, each small.
9. **Verification** (`verify.py`): superscript invariants per chapter; note,
   chronology, index counts; every subhead matched to the candidate list;
   per-page vocabulary coverage of the final text vs. the IA layer (flags any
   page that lost a paragraph); no `⟦?` markers left; epubcheck clean.

## Decisions to confirm before "go"

1. **Endnotes**: linked "References and Notes" chapter with per-chapter
   numbering and `epub:type="noteref"` (recommended; keeps printed numbering
   and "note 18 of this chapter" true) rather than pandoc `[^n]` footnotes,
   which renumber 1..518 globally.
2. **Index**: include it, with printed-page anchors and hyperlinked locators
   (recommended; mechanical, and it makes the index usable). Alternative: drop it.
3. **Italics**: try the slant heuristic + agent review for notes/body flags
   (recommended). Alternative: italics only in the notes via agents; body
   italics ignored.
4. **Cover**: the front board is a featureless cloth rectangle; use the title
   page render as the cover image.
5. **Front matter kept**: series page, title, copyright (library stamp and
   call number stripped), Series Foreword, Preface. Chapter titles follow the
   printed Contents ("Searching for Memory", "A Memory from Whirlwind").
