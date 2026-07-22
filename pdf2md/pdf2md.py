#!/usr/bin/env python3
"""Convert a pdfTeX-produced report PDF into reflowable Markdown.

The extractor is layout-driven rather than ML-driven: the source PDF is a
LaTeX document with a clean, regular geometry, so structure can be recovered
exactly from font, size and x-offset:

  * headings          bold spans whose text matches the PDF outline (TOC)
  * paragraphs        first-line indent at x0 = BODY_INDENT; the bibliography
                      uses the inverse (hanging indent) convention
  * lists             a bullet/number/letter marker at a deeper indent
  * emphasis          font family (Medi = bold, Ital = italic, SFTT = mono)
  * footnotes         a short horizontal rule splits body from notes; markers
                      are ~7pt superscript digits
  * figures           image blocks, rendered to PNG and linked

Usage: pdf2md.py INPUT.pdf OUTDIR
"""

from __future__ import annotations

import argparse
import collections
import os
import re
import sys
from dataclasses import dataclass, field

import pymupdf

# --- page geometry (points) -------------------------------------------------
BODY_LEFT = 108.0  # left text margin
BODY_INDENT = 123.0  # first line of a paragraph
LIST_MIN_X = 128.0  # anything indented past this is list-ish
FOOTER_Y = 735.0  # page-number band
RULE_MAX_W = 260.0  # footnote separator is a short rule at the left margin

SUPER_MAX_SIZE = 7.6  # footnote markers in the body
NOTE_NUM_MAX_SIZE = 6.6  # footnote numbers in the note block itself

# Words where a line-ending hyphen is part of the word rather than a
# TeX hyphenation break, used only when the document offers no evidence.
COMPOUND_PREFIXES = {
    "non",
    "self",
    "well",
    "quasi",
    "semi",
    "pre",
    "post",
    "sub",
    "meta",
    "multi",
    "anti",
    "co",
    "cross",
    "inter",
    "intra",
    "re",
    "un",
    "long",
    "short",
    "high",
    "low",
    "human",
    "value",
    "goal",
    "state",
    "task",
    "open",
    "closed",
    "fine",
    "large",
    "real",
}

LIST_MARKER = re.compile(r"^(•|\d{1,2}\.|[a-z]\.)\s*")
FIGURE_CAPTION = re.compile(r"^(Figure|Table)\s+\d+:")
EXTRA_HEADINGS = re.compile(r"^(References|Abstract|Acknowledge?ments?|Appendix\b.*)$")


# --- extracted primitives ---------------------------------------------------
@dataclass
class Seg:
    """A run of text sharing one style."""

    text: str
    style: str  # plain | bold | italic | bolditalic | mono | super | note-num
    x0: float = 0.0
    x1: float = 0.0


@dataclass
class Line:
    segs: list[Seg]
    x0: float
    x1: float
    y: float
    size: float = 0.0

    def text(self) -> str:
        return "".join(s.text for s in self.segs)


@dataclass
class Block:
    """One output element."""

    kind: str  # para | heading | list | figure | caption | note | raw
    segs: list[Seg] = field(default_factory=list)
    level: int = 0
    marker: str = ""
    indent: int = 0
    note_id: str = ""
    path: str = ""


def style_of(span: dict) -> str:
    font, size = span["font"], span["size"]
    if font.startswith("SFTT"):
        return "mono"
    if size <= NOTE_NUM_MAX_SIZE and span["text"].strip():
        return "note-num"
    if size <= SUPER_MAX_SIZE and span["text"].strip():
        return "super"
    bold = "Medi" in font or "Bold" in font
    italic = "Ital" in font
    if bold and italic:
        return "bolditalic"
    if bold:
        return "bold"
    if italic:
        return "italic"
    return "plain"


def page_lines(page: pymupdf.Page) -> list[Line]:
    """Group all spans on a page into visual lines by shared baseline.

    PyMuPDF splits a single justified line into several `line` records when
    inter-word gaps are large (common in the bibliography), so we regroup from
    raw spans rather than trusting its line segmentation.
    """
    spans = [
        s
        for b in page.get_text("dict")["blocks"]
        if b["type"] == 0
        for ln in b["lines"]
        for s in ln["spans"]
        if s["text"]
    ]
    spans.sort(key=lambda s: (s["origin"][1], s["origin"][0]))

    lines: list[Line] = []
    cur: list[dict] = []
    for s in spans:
        if cur and s["origin"][1] - max(c["origin"][1] for c in cur) > 4.5:
            lines.append(_make_line(cur))
            cur = []
        cur.append(s)
    if cur:
        lines.append(_make_line(cur))
    return lines


def _make_line(spans: list[dict]) -> Line:
    """Merge spans into styled runs, restoring inter-word spaces.

    Justified TeX output positions many words as separate spans; some gaps are
    real space glyphs and some are pure positioning, so we fall back to a
    geometric test when no space character is present.
    """
    spans = sorted(spans, key=lambda s: s["origin"][0])
    segs: list[Seg] = []
    prev_x1 = None
    for s in spans:
        text = s["text"]
        # A whitespace-only span carries no style of its own; keep the run open.
        st = segs[-1].style if (not text.strip() and segs) else style_of(s)
        if (
            prev_x1 is not None
            and text[:1] not in (" ", "")
            and s["bbox"][0] - prev_x1 > 0.18 * s["size"]
            and segs
            and not segs[-1].text.endswith(" ")
        ):
            text = " " + text
        # A wide gap means a separate column (the author grid); keep it as its
        # own run so column structure survives. `coalesce` rejoins them later.
        column_break = prev_x1 is not None and s["bbox"][0] - prev_x1 > 12.0
        prev_x1 = s["bbox"][2]
        if segs and segs[-1].style == st and not column_break:
            segs[-1].text += text
            segs[-1].x1 = s["bbox"][2]
        else:
            segs.append(Seg(text, st, s["bbox"][0], s["bbox"][2]))
    if not segs:
        segs = [Seg("", "plain")]
    inked = [s for s in spans if s["text"].strip()] or spans
    return Line(
        segs=segs,
        x0=min(s["bbox"][0] for s in inked),
        x1=max(s["bbox"][2] for s in inked),
        y=max(s["origin"][1] for s in spans),
        size=max(s["size"] for s in spans),
    )


def footnote_rule_y(page: pymupdf.Page) -> float:
    """y of the short rule separating body text from footnotes, else +inf."""
    best = float("inf")
    for d in page.get_drawings():
        r = d["rect"]
        if (
            r.height <= 2.0
            and 40 < r.width <= RULE_MAX_W
            and abs(r.x0 - BODY_LEFT) < 4
            and r.y0 > 200
        ):
            best = min(best, r.y0)
    return best


# --- text rendering ---------------------------------------------------------
ESCAPE = re.compile(r"([\\*_`\[\]<>])")


def escape(text: str) -> str:
    return ESCAPE.sub(r"\\\1", text)


WRAP = {"bold": "**", "italic": "*", "bolditalic": "***", "mono": "`"}


def coalesce(segs: list[Seg]) -> list[Seg]:
    """Merge neighbouring runs of the same style (e.g. a URL split by a line)."""
    out: list[Seg] = []
    for seg in segs:
        if out and out[-1].style == seg.style:
            out[-1] = Seg(out[-1].text + seg.text, seg.style)
        else:
            out.append(Seg(seg.text, seg.style))
    return out


def render(segs: list[Seg], plain: bool = False) -> str:
    """Turn styled segments into Markdown, keeping emphasis markers tight.

    `plain` drops emphasis, for contexts (headings, captions) that are already
    styled by their own Markdown construct.
    """
    out: list[str] = []
    for seg in coalesce(segs):
        if plain and seg.style not in {"super", "note-num"}:
            out.append(escape(seg.text))
            continue
        text = seg.text
        if seg.style == "super":
            # Whitespace around a marker belongs to the sentence, not the ref.
            lead = " " if text[:1].isspace() else ""
            trail = " " if text[-1:].isspace() else ""
            out.append(f"{lead}{footnote_ref(text)}{trail}")
            continue
        if seg.style == "mono":
            body = text.strip()
            if body:
                lead = " " if text[:1].isspace() else ""
                trail = " " if text[-1:].isspace() else ""
                out.append(f"{lead}`{body}`{trail}")
            continue
        if seg.style in WRAP:
            body = text.strip()
            if not body:
                out.append(text)
                continue
            mark = WRAP[seg.style]
            lead = " " if text[:1].isspace() else ""
            trail = " " if text[-1:].isspace() else ""
            out.append(f"{lead}{mark}{escape(body)}{mark}{trail}")
            continue
        out.append(escape(text))
    return "".join(out)


def footnote_ref(text: str) -> str:
    """Render a superscript marker as a Pandoc footnote reference."""
    label = note_label(text.strip())
    return f"[^{label}]" if label else escape(text)


def note_label(raw: str) -> str:
    raw = raw.strip()
    if raw.isdigit():
        return raw
    if raw in {"†", "‡", "*"}:
        return {"†": "dagger", "‡": "ddagger", "*": "star"}[raw]
    return ""


# --- hyphenation ------------------------------------------------------------
WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")


class Dehyphenator:
    """Decide whether a line-ending hyphen is a TeX break or a real hyphen.

    Preference order: evidence from elsewhere in the same document, then a
    small list of compound-forming prefixes, then join (TeX only breaks at
    legal hyphenation points, so joining is the common case).
    """

    def __init__(self, vocab: set[str]):
        self.vocab = vocab
        self.stats = collections.Counter()

    def join(self, left: str, right: str) -> str:
        m = re.search(r"([A-Za-z][A-Za-z'’]*)-$", left)
        if not m or not right[:1].isalpha():
            # Numbers, DOIs, URLs: the break is never a word break.
            self.stats["joined (non-word)"] += 1
            return left + right
        head, tail_m = m.group(1), WORD.match(right)
        tail = tail_m.group(0) if tail_m else right
        if (head + tail).lower() in self.vocab:
            self.stats["merged (vocab)"] += 1
            return left[:-1] + right
        if f"{head}-{tail}".lower() in self.vocab:
            self.stats["kept hyphen (vocab)"] += 1
            return left + right
        if head.lower() in COMPOUND_PREFIXES:
            self.stats["kept hyphen (prefix)"] += 1
            return left + right
        if right[:1].isupper():
            # Proper nouns are more often genuinely hyphenated than broken.
            self.stats["kept hyphen (proper noun)"] += 1
            return left + right
        self.stats["merged (default)"] += 1
        return left[:-1] + right


def build_vocab(doc: pymupdf.Document) -> set[str]:
    vocab: set[str] = set()
    for page in doc:
        for line in page_lines(page):
            for word in WORD.findall(line.text()):
                vocab.add(word.rstrip("-").lower())
    return vocab


# --- document conversion ----------------------------------------------------
class Converter:
    def __init__(self, doc: pymupdf.Document, outdir: str, image_dpi: int = 200):
        self.doc = doc
        self.outdir = outdir
        self.image_dpi = image_dpi
        self.toc = {norm(t[1]): t[0] for t in doc.get_toc()}
        self.dehyph = Dehyphenator(build_vocab(doc))
        self.blocks: list[Block] = []
        self.notes: dict[str, Block] = {}
        self.note_order: list[str] = []
        self.last_note: Block | None = None
        self.in_biblio = False
        self.figure_no = 0
        self.seen_headings: set[str] = set()
        self.title = ""
        self.authors: list[str] = []

    # -- block accumulation --------------------------------------------------
    def append_text(self, target: Block, line: Line) -> None:
        """Append a wrapped continuation line to an existing block."""
        if not line.segs:
            return
        text = line.text().lstrip()
        prev_seg = target.segs[-1] if target.segs else None
        prev = prev_seg.text if prev_seg else ""
        if prev.rstrip().endswith("-"):
            joined = self.dehyph.join(prev.rstrip(), text)
            # Keep only the left part here; the line's own segs supply `text`.
            prev_seg.text = joined[: len(joined) - len(text)]
            line = trim_leading(line)
        elif (
            prev_seg is not None
            and prev_seg.style == "mono"
            and line.segs[0].style == "mono"
            and not prev.endswith(" ")
        ):
            pass  # a URL wrapped mid-token: rejoin with no space
        elif prev and not prev.endswith(" "):
            prev_seg.text = prev + " "
        target.segs.extend(line.segs)

    def start(self, block: Block) -> Block:
        self.blocks.append(block)
        return block

    # -- main loop -----------------------------------------------------------
    def run(self) -> None:
        self.front_matter()
        current: Block | None = None
        for pno in range(1, self.doc.page_count):
            page = self.doc[pno]
            if is_toc_page(page):
                continue
            current = self.convert_page(page, pno, current)

    def convert_page(self, page, pno: int, current: Block | None) -> Block | None:
        rule = footnote_rule_y(page)
        lines = [ln for ln in page_lines(page) if not is_page_number(ln)]
        figures = self.figure_rects(page)

        for line in lines:
            while figures and figures[0].y1 <= line.y:
                # Keep `current` open: a figure floats out of a paragraph, so
                # text resuming below it continues the same paragraph.
                self.emit_figure(page, figures.pop(0))
            if line.y >= rule:
                self.note_line(line)
                continue
            current = self.body_line(line, current)
        for rect in figures:
            self.emit_figure(page, rect)
        return current

    def body_line(self, line: Line, current: Block | None) -> Block | None:
        text = line.text().strip()
        if not text:
            return current

        level, key = self.heading_level(line, text)
        if level:
            self.start(Block(kind="heading", segs=line.segs, level=level))
            self.seen_headings.add(key)
            if key == "references":
                self.in_biblio = True
            return None

        if FIGURE_CAPTION.match(text):
            # Like the figure itself, a caption floats out of the surrounding
            # paragraph; text below it resumes that paragraph.
            self.start(Block(kind="caption", segs=line.segs))
            return current

        marker = LIST_MARKER.match(text)
        if marker and line.x0 >= LIST_MIN_X:
            indent = 1 if line.x0 >= 148 else 0
            block = self.start(
                Block(
                    kind="list",
                    marker=marker.group(1),
                    indent=indent,
                    segs=trim_leading(line, len(marker.group(0))).segs,
                )
            )
            return block

        starts_new = (
            abs(line.x0 - BODY_LEFT) < 3 if self.in_biblio else line.x0 > BODY_LEFT + 6
        )
        if current is None or (starts_new and line.x0 < LIST_MIN_X):
            return self.start(Block(kind="para", segs=list(line.segs)))
        self.append_text(current, line)
        return current

    def note_line(self, line: Line) -> None:
        """Footnote text: a leading tiny-superscript number starts a new note."""
        if line.segs and line.segs[0].style == "note-num":
            label = note_label(line.segs[0].text)
            if label:
                block = Block(
                    kind="note",
                    note_id=label,
                    segs=trim_leading(line, 0, drop_first=True).segs,
                )
                self.notes[label] = block
                self.note_order.append(label)
                self.last_note = block
                return
        if self.last_note is not None:
            self.append_text(self.last_note, line)

    def heading_level(self, line: Line, text: str) -> tuple[int, str]:
        """Return (Markdown level, normalized title) for a heading line."""
        if not line.segs or FIGURE_CAPTION.match(text):
            return 0, ""
        inked = [s for s in line.segs if s.text.strip()]
        if not all(s.style in {"bold", "bolditalic", "super"} for s in inked):
            return 0, ""
        if not any(s.style in {"bold", "bolditalic"} for s in inked):
            return 0, ""
        # A heading may carry a footnote marker; match the title without it.
        key = norm("".join(s.text for s in inked if s.style != "super"))
        if key in self.toc:
            return self.toc[key] + 1, key
        if EXTRA_HEADINGS.match(key.title()):
            return 2, key
        return 0, ""

    # -- figures -------------------------------------------------------------
    def figure_rects(self, page) -> list[pymupdf.Rect]:
        rects = [
            pymupdf.Rect(b["bbox"])
            for b in page.get_text("dict")["blocks"]
            if b["type"] == 1
        ]
        # Skip logos and decorative marks; real figures span the text column.
        rects = [r for r in rects if r.width >= 100 and r.height >= 60]
        return sorted(rects, key=lambda r: r.y1)

    def emit_figure(self, page, rect: pymupdf.Rect) -> None:
        self.figure_no += 1
        name = f"figure-{self.figure_no}.png"
        os.makedirs(os.path.join(self.outdir, "images"), exist_ok=True)
        pix = page.get_pixmap(clip=rect, dpi=self.image_dpi)
        pix.save(os.path.join(self.outdir, "images", name))
        self.start(Block(kind="figure", path=f"images/{name}"))

    # -- front matter --------------------------------------------------------
    def front_matter(self) -> None:
        page = self.doc[0]
        lines = page_lines(page)
        title = next((ln for ln in lines if ln.size > 14), None)
        self.title = title.text().strip() if title else ""

        rule = footnote_rule_y(page)
        body = [ln for ln in lines if ln is not title and ln.y < rule]
        abstract_at = next(
            (i for i, ln in enumerate(body) if norm(ln.text()) == "abstract"), len(body)
        )

        # Author grid: bold rows hold names laid out in columns, and the row
        # directly beneath holds the matching affiliations.
        authors: list[str] = []
        rows = body[:abstract_at]
        for i, row in enumerate(rows):
            if not any(s.style == "bold" for s in row.segs if s.text.strip()):
                continue
            names = columns(row)
            affils = columns(rows[i + 1]) if i + 1 < len(rows) else []
            for j, name in enumerate(names):
                affil = affils[j] if j < len(affils) else ""
                authors.append(f"{name} — {affil}" if affil else name)
        self.authors = authors
        if authors:
            body_text = "\n".join(
                "- " + escape(a).replace("†", "[^dagger]") for a in authors
            )
            self.start(Block(kind="raw", segs=[Seg(body_text, "plain")]))

        current: Block | None = None
        for line in body[abstract_at:]:
            text = line.text().strip()
            if norm(text) == "abstract":
                self.start(Block(kind="heading", segs=line.segs, level=2))
                current = None
                continue
            if current is None:
                current = self.start(Block(kind="para", segs=list(line.segs)))
            else:
                self.append_text(current, line)

        for line in lines:
            if line.y >= rule:
                self.note_line(line)

    # -- output --------------------------------------------------------------
    def frontmatter(self) -> str:
        """YAML metadata, so pandoc can build a proper EPUB title page."""
        names = [re.sub(r"\s*—.*", "", a).replace("†", "") for a in self.authors]
        date = re.match(
            r"D:(\d{4})(\d{2})(\d{2})", self.doc.metadata.get("creationDate") or ""
        )
        lines = ["---", f'title: "{self.title}"']
        if names:
            lines.append("author:")
            lines += [f'  - "{n}"' for n in names]
        if date:
            lines.append(f"date: {'-'.join(date.groups())}")
        lines.append("---")
        return "\n".join(lines)

    def markdown(self) -> str:
        out: list[str] = [self.frontmatter()]
        for b in self.blocks:
            if b.kind == "heading":
                out.append(
                    "#" * min(b.level, 6) + " " + render(b.segs, plain=True).strip()
                )
            elif b.kind == "caption":
                out.append("*" + render(b.segs, plain=True).strip() + "*")
            elif b.kind == "figure":
                out.append(f"![]({b.path})")
            elif b.kind == "list":
                marker = "-" if b.marker == "•" else b.marker
                pad = "    " * b.indent
                out.append(f"{pad}{marker} " + render(b.segs).strip())
            elif b.kind == "raw":
                out.append(b.segs[0].text)
            else:
                out.append(render(b.segs).strip())

        for label in self.note_order:
            body = render(self.notes[label].segs).strip()
            out.append(f"[^{label}]: {body}")
        return "\n\n".join(out) + "\n"

    def report(self) -> None:
        missing = [t for t in self.toc if t not in self.seen_headings]
        print(
            f"blocks: {len(self.blocks)}  footnotes: {len(self.note_order)}  "
            f"figures: {self.figure_no}",
            file=sys.stderr,
        )
        print(f"hyphenation: {dict(self.dehyph.stats)}", file=sys.stderr)
        if missing:
            print(
                f"WARNING: {len(missing)} TOC headings not matched in text:",
                file=sys.stderr,
            )
            for m in missing:
                print(f"  - {m}", file=sys.stderr)


# --- helpers ----------------------------------------------------------------
def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def columns(line: Line, gap: float = 20.0) -> list[str]:
    """Split a visually multi-column line (the author grid) into its cells."""
    cells: list[str] = []
    prev_x1 = None
    for seg in line.segs:
        if not seg.text.strip():
            continue
        if prev_x1 is None or seg.x0 - prev_x1 > gap:
            cells.append(seg.text.strip())
        else:
            cells[-1] += seg.text
        prev_x1 = seg.x1
    return [re.sub(r"\s+", " ", c).strip() for c in cells if c.strip()]


def is_page_number(line: Line) -> bool:
    return line.y > FOOTER_Y and line.text().strip().isdigit()


def is_toc_page(page: pymupdf.Page) -> bool:
    return page.get_text().count(". . . .") >= 3


def trim_leading(line: Line, n: int = 0, drop_first: bool = False) -> Line:
    """Return a copy of `line` with `n` leading characters (or the first seg) cut."""
    segs = [Seg(s.text, s.style) for s in line.segs]
    if drop_first and segs:
        segs = segs[1:]
    while n and segs:
        take = min(n, len(segs[0].text))
        segs[0].text = segs[0].text[take:]
        n -= take
        if not segs[0].text:
            segs.pop(0)
    if segs:
        segs[0].text = segs[0].text.lstrip()
    return Line(segs=segs, x0=line.x0, x1=line.x1, y=line.y, size=line.size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf")
    ap.add_argument("outdir")
    ap.add_argument("--dpi", type=int, default=200, help="figure render DPI")
    args = ap.parse_args()

    doc = pymupdf.open(args.pdf)
    conv = Converter(doc, args.outdir, args.dpi)
    conv.run()

    os.makedirs(args.outdir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.pdf))[0]
    path = os.path.join(args.outdir, stem + ".md")
    with open(path, "w") as f:
        f.write(conv.markdown())
    conv.report()
    print(path)


if __name__ == "__main__":
    main()
