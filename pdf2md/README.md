# pdf2md

Converts a LaTeX-produced report PDF into reflowable Markdown (and EPUB) for
reading on an e-reader.

```sh
make          # md + epub into out/
make md       # markdown only
```

Put the source PDF in `pdfs/`; output lands in `out/` (both are gitignored).

## Approach

The input is a pdfTeX document with clean, regular geometry, so this is a
*layout-driven* extractor rather than an ML/OCR one — every structural decision
comes from a measurable property of the page, which makes the result exact and
debuggable rather than approximate:

| Structure | Signal |
| --- | --- |
| Headings | Bold line whose text matches an entry in the PDF outline; nesting level comes from the outline too |
| Paragraphs | First-line indent (x0 = 123pt); continuation lines sit at the margin (108pt) |
| Bibliography | Same rule inverted — entries use a hanging indent, so a line *at* the margin starts a new entry |
| Lists | A `•` / `1.` / `a.` marker indented past 128pt; deeper indents nest |
| Emphasis | Font family: `NimbusRomNo9L-Medi` → bold, `-ReguItal` → italic, `SFTT` → monospace |
| Footnotes | A short rule at the left margin splits body from notes; refs are ~7pt superscripts, definitions start with a ~6pt number |
| Figures/tables | Image blocks, rendered to PNG at 200 dpi and linked in reading order |
| Dropped | The printed table of contents (detected by dot leaders) and page-number footers |

Footnotes become Pandoc-style `[^n]` references with definitions collected at
the end of the file, so an e-reader renders them as popups or endnotes.

### Details worth knowing

* **Spaces.** Justified TeX output positions many words as individual spans,
  sometimes with a real space glyph and sometimes with pure positioning. Both
  cases are handled: whitespace spans are preserved, and a geometric gap test
  fills in the rest. Dropping either produces `runtogetherwords`.
* **Lines.** PyMuPDF splits one visual line into several `line` records when
  inter-word gaps are wide (common in the bibliography), so lines are regrouped
  from raw spans by shared baseline.
* **Hyphenation.** A line-ending hyphen may be a TeX break (`introspec-tion`) or
  part of the word (`well-being`). The converter first looks for evidence
  elsewhere in the same document — if `wellbeing` or `well-being` appears
  unbroken, that settles it — then falls back to a compound-prefix list, then to
  merging. The run prints a breakdown of which rule fired how often.
* **Floats.** Figures and their captions are lifted out of the paragraph they
  interrupt, so a paragraph that flows around a figure stays whole.

## Verifying a conversion

The script reports block/footnote/figure counts, the hyphenation breakdown, and
any outline heading it failed to find in the text. Beyond that, the check that
actually caught bugs here was a shingle diff: every 6-word sequence in the PDF
should appear somewhere in the Markdown. On this document the only residual
differences are footnote-boundary artifacts (notes move to the end) and places
where the Markdown is *better* than the raw text — rejoined hyphenated words and
un-wrapped URLs.

## Adapting to another PDF

The page-geometry constants at the top of `pdf2md.py` (margins, indent, footer
band) and the font-family names in `style_of` are specific to this document's
LaTeX class. Everything else generalizes. Dump a page's lines with their `x0`
and font info to find the new values:

```sh
uv run python -c "
import pymupdf, pdf2md
d = pymupdf.open('pdfs/YOUR.pdf')
for ln in pdf2md.page_lines(d[10]):
    print(round(ln.x0), round(ln.y), [(s.style, s.text[:60]) for s in ln.segs])
"
```
