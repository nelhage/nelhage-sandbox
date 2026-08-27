# Transcription spec — Agemonia Rules Reference Book → Markdown

You are transcribing assigned pages of the Agemonia rulebook into Markdown.
Working directory: `/home/nelhage/code/sandbox/agemonia`.

## Read these first

* `ICONS.md` — the canonical icon glossary. **Always use these exact names.**
* For each assigned PDF page `NN` (2-digit, zero-padded):
  * `out/text/pNN.txt` — the body column's **exact text layer**, plus a header
    describing that page's geometry.
  * `out/render/pNN_full.png` — whole-page overview (use it for layout: what is
    a heading, a box, a sidebar, a table, an illustration).
  * `out/render/pNN_body1..3.png` — body column in 3 vertical bands, crisp.
    On two-column pages these are `pNN_col1_1..3` and `pNN_col2_1..3` instead.
  * `out/render/pNN_side1..3.png` — the sidebar column, if the page has one.

## The two sources, and which one wins

* **Body text is real text.** `out/text/pNN.txt` is character-exact. Copy from
  it verbatim. Do **not** retype body prose from the image — you will introduce
  errors. Do not "fix" the book's spelling, grammar, or inconsistent wording.
  * *One exception:* headings are letter-spaced display type, and the text
    layer renders that as a spurious internal space — `Ab ility Checks`,
    `It ems`, `Ga ining Item Cards`, `Loc ations and Services`. The page
    actually prints `Ability Checks`, `Items`, and so on. Close those up. Check
    the render if you are unsure whether a space is real.
* **The sidebar has no text layer at all** (the PDF's transparency flattener
  rasterised it). It exists *only* in `pNN_side*.png`. Transcribe it by reading
  those images carefully. The same is true of a few drop-shadowed headings and
  of text baked into diagrams and card illustrations.
* If the text layer and the image disagree about *what text exists*, the image
  is the truth about what is printed; the text layer is the truth about
  spelling. Text present in the image but absent from the text layer must still
  be transcribed.

## Icons

`[[ICON]]` in the text layer marks an inline icon image. Replace each with the
right glossary name in square brackets, e.g. `roll a number of [Action Die]`.

* `[[ICON]]` markers are a **lower bound** — icons drawn as vector art are not
  marked. Read the render and transcribe **every** icon you see, marked or not.
* Inline icons in body text are 15×15px and look like coloured blobs. Do not
  guess from those pixels — identify them from the sentence and from `ICONS.md`.
* Use the glossary's spelling exactly, including capitalisation (`[Action Die]`,
  not `[Action die]`). Read `ICONS.md`'s "Notes on usage" — it fixes how ability
  checks, die faces, numbers-attached-to-icons, and initiative symbols are
  written.
* If an icon is genuinely not in the glossary and you cannot identify it, write
  `[?icon: short description]` so it can be found later. Do not silently drop it.
  (This supersedes the `<!-- ? -->` convention mentioned in `ICONS.md`.)

## Output

Write **one file per page**: `out/md/pNN.md`. Nothing else.

Begin each file with an HTML comment, then the content:

```markdown
<!-- PDF page 10 | printed page 6 -->

### Ability Checks (p6)

An ability check may be either active or reactive. ...
```

### Page numbers in headings

**Every heading gets the printed page number in parentheses**, e.g.
`### Ability Checks (p6)`. Use the folio from the text-layer header — *not* the
PDF page number. For the unnumbered front/back matter use `(cover)`,
`(components)`, `(contents)`, or `(back cover)` in place of `(pN)`.

If a section continues from the previous page with no new heading, do not invent
one; just continue the prose.

### Heading levels (fixed — keep these consistent across pages)

| Level | What it is | Font in the text layer |
| --- | --- | --- |
| `#` | Book title (cover only) | — |
| `##` | Part title: Components, Contents, Campaign Rules, Scenario Rules, Additional Rules | `Varna-Regular@15.8` |
| `###` | Section heading, set in a green bar | `Varna-Regular@12.4` |
| `####` | Subsection heading | `Varna-Bold@11.4` |
| `#####` | Minor run-in heading | `Landa-Bd@10.9` |

### Other conventions

* **Bulleted lists.** The book's `»` bullets become `-`.
* **Inset boxes** (green-bordered rules boxes, "Note:" boxes) become
  blockquotes, with the box's title bolded on the first line:
  ```markdown
  > **Action dice**
  >
  > There are various symbols on an Action die:
  >
  > - [Success] 1 success
  ```
* **Sidebar callouts** become blockquotes beginning with a bold label, placed in
  the body at the point the sidebar's leader line points to. If the anchor is
  unclear, place it after the nearest paragraph vertically. Mark them:
  ```markdown
  > **Sidebar — Example:** *You get a modifier of +2 [Success] to this active
  > agility check if you have Lock picking. ...*
  ```
  Keep the book's own italics for example text.
* **Tables.** Where the original is a genuine table or a labelled grid (the
  Components spread, class-skill lists, item lists), use a Markdown table.
* **Illustrations, diagrams, card images.** Do not describe the art at length,
  but **do transcribe any text or icons printed inside it**, and add a short
  italic caption noting what it depicts, e.g.
  `*Diagram: an Enemy card, with callouts labelled Enemy Movement, Enemy Attack, Horns Special Ability, Action Symbol.*`
* **Two-column pages.** Read column 1 top-to-bottom, then column 2. The text
  layer marks this with `=== COLUMN 1 ===` / `=== COLUMN 2 ===`.
* **Cross-references.** Keep them as printed ("see p. 45"); do not convert to links.
* **Emphasis.** Preserve bold and italic as the book uses them.

## Rules

1. **Preserve all text.** Every word printed on the page must appear in your
   output, in reading order. This is the single most important requirement.
2. Do not summarise, paraphrase, reorder, or editorialise.
3. Do not add commentary, footnotes, or explanations of your own.
4. Do not add a trailing summary section to the file.

## When done

Report, briefly: which pages you wrote, any icons you could not identify, and
anything on the page you were unsure about.
