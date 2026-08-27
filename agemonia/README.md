# agemonia

Converts the Agemonia *Rules Reference Book* PDF into a single reflowable
Markdown file, `Agemonia-Rulebook.md`, with the game's iconography written out
in words and every heading tagged with the page number printed in the book.

```sh
uv run python icons.py      # locate inline icons -> out/icons/occurrences.json
uv run python extract.py    # text layer + page renders -> out/text, out/render
#   ... vision sub-agents transcribe pages into out/md/pNN.md ...
uv run python tagheadings.py
uv run python assemble.py   # concatenate + verify -> Agemonia-Rulebook.md
```

`extract.py --pages 8,20,32` regenerates only those pages, preserving the rest
of `out/index.json`.

## Why this isn't a text-extraction job

`pdftotext` does badly here, and the reason is structural rather than cosmetic.
The file is **PDF 1.3**, which has no transparency model, so InDesign's
flattener rasterised every soft-shadowed element. That single fact produces all
four of the real problems:

| Problem | Consequence |
| --- | --- |
| The whole right-hand **sidebar** was rasterised | It has *no text layer at all* — worked examples and card callouts are invisible to any text tool |
| Icons became **15×15px JPEGs** | Inline icons are unreadable at any render resolution; they can only be inferred from context |
| Flattening **split and duplicated glyphs** | One icon arrives as several tiles; each line's first glyph is emitted twice, giving `"...followed by" + "A"` |
| Display type is **letter-spaced** | Headings reach the text layer as `Ab ility Checks`, `It ems`, `Meta morphose` |

So the conversion is a hybrid: the body column's text is recovered *exactly*
from the text layer, while the sidebar, diagram labels and all iconography are
read from high-resolution renders by vision sub-agents.

## How the pieces fit

**`layout.py` — page geometry.** The book is laid out as spreads, so the
sidebar alternates edges (left on odd PDF pages, right on even), and some pages
are two-column. Nothing may assume a fixed split. Regions are derived from
*where text actually lands* — specifically from a per-column count of how many
text lines cover it, which makes the body a broad plateau and the sidebar a
near-zero floor.

> Taking simple extremes (min x0, max x1) is not safe, and this cost a rework:
> on page 8 a diagram callout, "Class & Profession cards", reaches into the
> sidebar's half of the page and swallowed it, so seven pages were transcribed
> believing they had no sidebar.

**`icons.py` — where the icons are.** The text layer silently concatenates
around an inline icon (`"rolling a number of equal to your level"`), so icon
positions must be recovered separately and spliced back in as `[[ICON]]`
markers. Touching image fragments are merged into one logical icon, and a
candidate is rejected unless it sits in a *gap* between text spans — that test
is what separates a real icon from a flattening artifact lying under a glyph.
The markers are a lower bound: vector-drawn icons aren't images and aren't
marked, so the renders remain authoritative.

**`extract.py` — the two views of each page.** Emits the body text with
`[[ICON]]` markers and font tags, plus renders in bands sized so the long edge
lands near 1568px, the resolution a vision model downscales to. Two-column
pages are rendered per column, which roughly doubles their effective DPI.

**`ICONS.md` / `TRANSCRIBE.md` — the shared contract.** A fleet of agents will
only agree with each other if the naming and structure rules live in one place.
`ICONS.md` is a glossary built from the book's own legend pages (the back cover
is a labelled legend for nearly every symbol); `TRANSCRIBE.md` fixes heading
levels, sidebar handling, and the `[Icon Name]` notation.

**`assemble.py` — verification.** Concatenates the pages and checks the result
against the PDF two ways:

* **Vocabulary coverage** (primary): which words the page prints are absent
  from the Markdown. Order-insensitive, so it measures omission directly.
* **Sequence coverage** (secondary): 40-character runs surviving in order.

Sequence coverage sits near 73%, which sounds alarming and isn't: where the
book sets a table, the PDF's baseline order often emits a row's label *after*
its description while the Markdown table puts it first. Nothing is lost, but
every run crossing that boundary breaks. Vocabulary coverage is the number that
means something.

## Result

Vocabulary coverage is **99.99%** — one word, `Raid`, is "missing", and it
turns out to be hidden text in the PDF (a leftover object sitting behind a
graphic, never printed), so the true figure is complete.

Icons resolve to ~94 distinct glossary names across ~1100 insertions. Where an
icon genuinely could not be identified — mostly item-card rarity stars and
craft-material tokens whose source art is below legible resolution — it is
flagged inline as `[?icon: description]` rather than guessed at or dropped.
