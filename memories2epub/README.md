# memories2epub

Turns an Internet Archive scan of Emerson Pugh's *Memories That Shaped an
Industry* (MIT Press, 1984; 344 PDF pages) into an EPUB with a working table of
contents, section headings, linked endnotes, figures, chronology and a
page-anchored index.

```sh
make ocr          # 300 dpi renders, blob removal, tesseract hOCR   (~5 min)
make marks        # superscript endnote references                (~15 min)
make figures backmatter
make epub verify  # out/book.epub
```

Everything under `out/` is regenerable except `out/corrections/` (the
sub-agent review results) and `out/survey/`, which are committed. `PLAN.md` is the plan that was agreed
before building; `CONTRACT.md` fixes the data formats between stages;
`out/survey/*.md` are the survey reports the plan was based on.

## Approach

The bulk of the work is plain tesseract. The design question was where a
vanilla OCR pass fails on *this* book, and the survey of sample pages answered
that precisely enough to handle each failure with a targeted heuristic, leaving
only a few hundred words for vision review:

| Problem | Handling |
| --- | --- |
| Superscript endnote numbers become punctuation (`hour.''`, `$100,000.1°`) | Geometric detection of the raised glyph on hOCR line geometry, mask it, re-OCR the page for a clean body word, read the digits twice (tesseract on an upscaled crop + a template classifier bootstrapped from the book's own glyphs), check against the per-chapter numbering prior (`superscripts.py`) |
| A scanner blemish at a fixed position on every verso corrupts one word | Template-matched and erased before OCR (`render.py`) |
| Running heads, figure lettering, library stamps | Dropped by position (`layout.py`, `figures.py`) |
| Paragraphs, block quotes, numbered lists, subheads | First-line indent over a per-page margin, line pitch, vertical gaps; subheads confirmed against a list from the survey (`layout.py`) |
| Full-page figures interrupt paragraphs | The figure is deferred until the paragraph closes (`assemble.py`) |
| Small caps come out in random case (`AscCc`, `mTC`) | Acronym lexicon (`normalize.py`) |
| Endnote numbers in the References section are dropped or misread | Number column re-OCR'd with a digit whitelist, values assigned by sequence, counts asserted (`notes.py`) |
| Italics are invisible to tesseract | Shear-projection slant score per word (`italics_scan.py`) flags ~120 runs book-wide; a vision reviewer confirms them and extends the span |
| Block quotes, numbered lists, a schedule table | Tight leading + indent/gap paragraphing inside the block; a run of `task ... Month d, yyyy` lines becomes a table (`assemble.py`) |

Escalation to vision sub-agents is driven by `flags.py`: a word goes to review
if tesseract's confidence is below 80, if it matches a suspect pattern, or if
it disagrees with the PDF's own (older tesseract) text layer after normalising
harmless glyph variants. Reviewers see montages of the flagged line strips, not
whole pages, and return `out/corrections/*.json`:

| file | shape |
| --- | --- |
| `body-*.json`, `qa-*.json` | `{page, bbox, text \| italic \| ref \| refs}` for one hOCR word, or `{page, line_bbox, text}` for a whole line in Markdown with `^N` endnote markers |
| `figures*.json` | `{page, k, caption_title, caption_text}` or `{page, k, md}` |
| `notes*.json` | `{chapter, n, md}` |
| `chronology*.json`, `index*.json` | `{i, ...replacement fields}` |

Later files override earlier ones (sorted by name). In the final run the
1,734 flagged body words yielded about 330 corrections; a read-through QA of
the built EPUB by four further reviewers found roughly 0.5 residual defects per
thousand words, mostly dropped italics, and a handful of layout bugs that were
fixed in code rather than by corrections.

## EPUB structure

* Chapters split at level-1 headings; subheads are level 2; `--toc-depth=2`.
* Endnotes are a linked "References and Notes" chapter keeping the printed
  per-chapter numbering. Body references are `<a epub:type="noteref">` links;
  each note is an `<aside>`-style div with `epub:type="footnote"` so pop-up
  capable readers show it in place, with a backlink to the first citation.
* Every printed page start is an `epub:type="pagebreak"` anchor and the EPUB
  carries a `page-list` nav, so the index's page numbers are live links.
* Figures are cropped from the composited 300 dpi render (the scan is MRC:
  tones live in a 120 ppi background, edges in a 360 ppi mask), photos
  downsampled to ~200 dpi, line art kept at 300 dpi.

## Verification

`verify.py` checks: per-chapter superscript maxima equal the note counts and
every note number was first-cited in order; 518 notes in 61/39/64/86/69/59/64/48/28;
174 chronology entries; ~429 index entries; every known subhead appears once;
every referenced note exists; per-page vocabulary coverage against the IA text
layer; no unresolved `⟦?…⟧` markers; epubcheck clean.
