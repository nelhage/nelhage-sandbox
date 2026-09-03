# Stage contract

Shared conventions for every script in this directory. Read PLAN.md first for the
overall pipeline and `out/survey/*.md` for the measurements behind it.

## Conventions

- Page numbers are **PDF pages**, 1-based, zero-padded to 3 digits in file names
  (`p-021`). Printed page = PDF page − 16. Chapter k starts at PDF
  17, 50, 78, 109, 145, 176, 203, 229, 264 (k = 1..9); References and Notes
  281–316, Chronology 317–328, Index 329–339.
- All coordinates are **pixels at 300 dpi** on `out/pages/p-NNN.png`
  (same geometry as `out/png300`), as `[x0, y0, x1, y1]`.
- Every script is idempotent, runs from this directory, takes optional page
  numbers as arguments (default: all), and writes only under `out/`.
- Text is UTF-8 with straight double quotes `"`, curly apostrophe `’`, `×` for
  dimension products, `—` for em dashes. Unknown/illegible content is written as
  `⟦?hint⟧` so it can be grepped later. Do **not** guess italics; leave plain.
- Escalation items (things a vision agent must look at) go in a per-stage
  `flags.json`: a list of `{"page": N, "bbox": [...], "crop": "out/.../x.png",
  "draft": "text as OCR'd", "reason": "...", "context": {...}}`. Crops are
  300 dpi line strips with ~10 px padding, PNG.

## Inputs (already produced)

- `out/pages/p-NNN.png` — 300 dpi grayscale, scanner blob removed (`render.py`).
- `out/hocr/p-NNN.hocr` (+ `.txt`) — tesseract 5.5 `--psm 1` (`ocr.py`).
- `out/hocr4/p-NNN.hocr` — `--psm 4` for the Chronology pages only.
- `out/survey/figures.tsv` — figure inventory with page-fraction bboxes.

## Stage outputs

### superscripts.py → `out/marks/`
- `out/pages2/p-NNN.png`: page image with detected superscript marks painted paper-colour
  (only pages that have marks).
- `out/hocr2/p-NNN.hocr`: `--psm 1` hOCR of `out/pages2` for those pages. Consumers
  use `out/hocr2` when present, else `out/hocr`.
- `out/marks/p-NNN.json`:
  ```json
  {"page": 21, "chapter": 1,
   "marks": [{"bbox": [..], "kind": "ref"|"other", "value": 10,
              "readings": {"psm7": "10", "template": "10"}, "conf": 93,
              "status": "ok"|"flag", "reason": "",
              "line_bbox": [..], "word_bbox": [..], "word_text": "$100,000."}]}
  ```
  `word_bbox`/`word_text` is the word in `out/hocr2` immediately left of the mark
  (the one the reference attaches to). Chained marks (`25,26`) are two entries
  with the same word. `kind: other` = non-reference superscript (M², 2¹⁰);
  keep its text so the assembler can write `<sup>`.
- `out/marks/summary.json`: per chapter `{max, notes_expected, first_cites_in_order,
  gaps, n_marks, n_flags}`; `out/marks/flags.json` per the escalation format, each
  flag also carrying `"allowed": [1..max_so_far+1]`.

### notes.py / chronology.py / index.py → `out/backmatter/`
- `notes.json`: `{"intro": "...", "chapters": [{"chapter": 1, "notes":
  [{"n": 1, "text": "...", "pages": [281], "lines": [[page, bbox], ...],
  "flags": []}]}]}`. Text is de-hyphenated and joined; the ch4 n8 (a)–(u) list is
  kept as `"sublist": [...]`. Asserts counts 61/39/64/86/69/59/64/48/28.
- `chronology.json`: `[{"date": "8/44", "text": "...", "page": 317}]` (174 entries).
- `index.json`: `[{"level": 0|1, "text": "Air defense system", "locators":
  "93-128, 140", "see": "See also …" or null, "page": 329}]` in reading order,
  turnover lines already merged, hyphenation resolved. Locators stay a string
  (the assembler links them).
- `flags.json` for all three (low-confidence lines, number-sequence anomalies).

### figures.py → `out/figures/`
- `pNNN-k.png` crops (k = 1.. top to bottom), level-stretched, photos ≈200 dpi,
  line art 300 dpi.
- `figures.json`: `[{"page": 25, "k": 1, "file": "out/figures/p025-1.png",
  "bbox": [..], "kind": "photo"|"lineart", "caption_title": "IBM punched card",
  "caption_text": "...", "caption_bbox": [..], "shared_caption": false}]`.
  When one caption covers two figures, both entries carry the same caption and
  `shared_caption: true`. Caption text comes from `out/hocr` lines below the figure.
- `sheet.png`: contact sheet of every crop with its page number, for review.

### layout.py → `out/blocks/p-NNN.json` (written by the main session)
Typed blocks per page: head, chapter, subhead, para, quote, list, figure,
caption, with `continues` flags. Consumes hocr2/marks/figures.json.
