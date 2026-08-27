"""Assemble per-page Markdown into one book, and verify nothing was lost.

The check that actually catches dropped or reworded text is a shingle diff:
every run of the PDF's body text should survive somewhere in the Markdown.
Icon transcriptions are stripped before comparing, because the PDF's text layer
has nothing where an icon sits -- `[Action die]` in the Markdown corresponds to
a gap in the source.

Shingles are taken over *characters with whitespace removed*, not words,
because the two sides legitimately disagree about spacing: letter-spaced
headings reach the text layer as "Ab ility Checks" while the page prints
"Ability Checks", and line-broken words rejoin differently.

Shingles alone understate fidelity, though: where the book sets a table, the
PDF's baseline order can emit a row's label *after* its description, while the
Markdown table puts it first. Nothing is lost, but every shingle across that
boundary breaks. So the primary measure is the order-insensitive one -- which
words from the page are absent from the Markdown entirely -- and the shingle
score is kept as a secondary signal about ordering and phrasing.

Only the body column can be checked this way: the sidebar was rasterised and
has no text layer, so its transcription is unverifiable here by construction.
"""

import argparse
import pathlib
import re
import unicodedata

import pymupdf

import layout

PDF = pathlib.Path("pdfs/Rulebook.pdf")
MD_DIR = pathlib.Path("out/md")
BOOK = pathlib.Path("Agemonia-Rulebook.md")
SHINGLE = 40  # characters
STRIDE = 3

ICON_TOKEN = re.compile(r"\[[^\[\]\n]{1,40}\]")
KEEP = re.compile(r"[a-z0-9]+")
MIN_WORD = 3  # 1-2 char fragments are mostly extraction noise


def words(text):
    text = unicodedata.normalize("NFKD", text).replace("’", "'").replace("‘", "'")
    return KEEP.findall(text.lower())


def normalize(text):
    """Lowercase alphanumerics only -- spacing and punctuation carry no signal."""
    return "".join(words(text))


def pdf_body_text(doc):
    """Normalised body text per page, in reading order."""
    pages = {}
    for pno, page in enumerate(doc):
        body = layout.body_region(page)
        chunks = []
        for cx0, cx1 in layout.columns(page, body):
            spans = [
                s
                for s in layout.text_spans(page, size_range=(0.0, 99.0))
                if cx0 - 4 <= s["bbox"][0] < cx1 and body[0] - 6 <= s["bbox"][0] <= body[1]
            ]
            spans.sort(key=lambda s: (round(s["bbox"][3], 0), s["bbox"][0]))
            chunks.append(" ".join(s["text"] for s in spans))
        pages[pno + 1] = " ".join(chunks)
    return pages


def shingles(text, n=SHINGLE, stride=1):
    return {text[i : i + n] for i in range(0, max(0, len(text) - n + 1), stride)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    parts = sorted(MD_DIR.glob("p*.md"))
    if not parts:
        raise SystemExit(f"no per-page Markdown in {MD_DIR}")

    if not args.verify_only:
        # A comment, not content: the file should read as the book itself.
        preamble = (
            "<!--\n"
            "Agemonia Rules Reference Book, transcribed from pdfs/Rulebook.pdf.\n"
            "Headings carry the page number as printed in the book, e.g. '(p6)'.\n"
            "Icons are written inline in square brackets, e.g. [Action Die];\n"
            "ICONS.md is the glossary of those names.\n"
            "-->\n"
        )
        body = "\n\n".join(p.read_text().strip() for p in parts)
        BOOK.write_text(preamble + "\n" + body + "\n")
        print(f"wrote {BOOK} ({len(body):,} chars from {len(parts)} pages)")

    md_text = ICON_TOKEN.sub(" ", BOOK.read_text())
    md_vocab = set(words(md_text))
    md_shingles = shingles(normalize(md_text))

    doc = pymupdf.open(PDF)
    pages = pdf_body_text(doc)

    # Primary check: words the page prints that appear nowhere in the Markdown.
    tot_w = drop_w = 0
    dropped = []
    for pno, text in sorted(pages.items()):
        ws = [w for w in words(text) if len(w) >= MIN_WORD]
        if not ws:
            continue
        absent = {w for w in ws if w not in md_vocab}
        # The book letter-spaces its display type, which reaches the text layer
        # as split words ("Ab ility Checks", "Meta morphose"). Transcribers close
        # those up, so a fragment that is a substring of some word in the
        # Markdown is not a dropped word.
        gone = sorted(w for w in absent if not any(w in m for m in md_vocab))
        tot_w += len(set(ws))
        drop_w += len(gone)
        if gone:
            dropped.append((len(gone), pno, gone))

    print(f"\nvocabulary coverage: {tot_w - drop_w:,}/{tot_w:,} distinct words = "
          f"{100 * (tot_w - drop_w) / tot_w:.2f}%")
    dropped.sort(reverse=True)
    if dropped:
        print("\npages with words absent from the Markdown (likely dropped content):")
        for n, pno, gone in dropped[:20]:
            print(f"  pdf p{pno:<3} printed p{pno - 4:<3} {n:>3} missing: {', '.join(gone[:14])}")
    else:
        print("  every word printed in the body text appears in the Markdown.")

    # Secondary check: ordering/phrasing drift.
    total = missing = 0
    for text in pages.values():
        want = shingles(normalize(text), stride=STRIDE)
        total += len(want)
        missing += len(want - md_shingles)
    pct = 100 * (total - missing) / total if total else 100.0
    print(f"\nsequence coverage (secondary): {pct:.2f}% of {SHINGLE}-char runs "
          f"survive in order")


if __name__ == "__main__":
    main()
