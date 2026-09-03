"""Stage 7: blocks + back matter + figures -> out/book.md -> out/book.epub.

Reads  out/blocks/p-NNN.json      (layout.py)
       out/marks/p-NNN.json       (superscripts.py; via the words' refs/sup tags)
       out/figures/figures.json   (figures.py)
       out/backmatter/{notes,chronology,index}.json
       frontmatter.md, epub.css, out/corrections/*.json (optional, from review)
Writes out/book.md, out/cover.png, out/book.epub, out/assemble.log

Markup choices (all verified to survive pandoc -> EPUB3):
  endnote ref   [<sup>12</sup>](#c3n12){.noteref epub:type="noteref"}
  note body     ::: {#c3n12 .note epub:type="footnote"} ... :::
  page anchor   []{#pg93 .pb epub:type="pagebreak" title="93" role="doc-pagebreak"}
  figure        ![**Title** caption](file)   (implicit_figures -> <figure>)
Unresolved items are written as ⟦?...⟧ so they can be grepped.

Usage: python3 assemble.py [--no-epub]
"""
import json, pathlib, re, subprocess, sys, zipfile
from normalize import norm_word, norm_text

OUT = pathlib.Path("out")
CHAPTER_START = {17: 1, 50: 2, 78: 3, 109: 4, 145: 5, 176: 6, 203: 7, 229: 8, 264: 9}
ROMAN = {11: "vii", 12: "viii", 13: "ix", 14: "x"}
log = []

# ---------------------------------------------------------------- corrections
JOURNALS = ["IBM Journal of Research and Development", "IBM Journal of Reserch and Development",
            "Proceedings of the Institute of Radio Engineers", "Journal of Applied Physics"]
def ital_titles(s):
    """Journal titles are italic in the printed captions; tesseract cannot see italics."""
    for j in JOURNALS:
        s = s.replace(j, f"*{j}*")
    return s

def load_figure_corrections():
    """out/corrections/figures*.json: [{page, k, caption_title?, caption_text?, md?}]"""
    d = {}
    for f in sorted((OUT / "corrections").glob("figures*.json")) if (OUT / "corrections").exists() else []:
        for c in json.loads(f.read_text()):
            d.setdefault((c["page"], c["k"]), {}).update({k: v for k, v in c.items() if k not in ("page", "k")})
    return d
FIG_CORR = load_figure_corrections()

def load_corrections():
    """Review results: word replacements, italic spans, ref values, line replacements."""
    by_page = {}
    files = sorted((OUT / "corrections").glob("body*.json")) + sorted((OUT / "corrections").glob("qa-*.json"))
    for f in files if (OUT / "corrections").exists() else []:
        for c in json.loads(f.read_text()):
            by_page.setdefault(c["page"], []).append(c)
    return by_page

def apply_corrections(page, blocks, corrs):
    for c in corrs:
        hit = False
        for b in blocks:
            for l in b.get("lines", []):
                if "line_bbox" in c and (l["bbox"] == c["line_bbox"] or c["line_bbox"] in l.get("parts", [])):
                    # whole-line replacement in Markdown; `^12` after a word marks
                    # an endnote reference (`^12,13` for a chained pair)
                    new_words = []
                    for tok in c["text"].split():
                        m = re.match(r"^(.*?)\^(\d+(?:,\d+)*)$", tok)
                        w = {"t": m.group(1) if m else tok, "c": 100, "bbox": l["bbox"], "raw": True}
                        if m:
                            w["refs"] = [{"n": int(x), "status": "ok"} for x in m.group(2).split(",")]
                        new_words.append(w)
                    l["words"] = new_words
                    hit = True
                for w in l["words"]:
                    if "bbox" in c and w["bbox"] == c["bbox"]:
                        if "text" in c:
                            w["t"] = c["text"]; w["c"] = 100
                            w.pop("sup_unplaced", None)          # reviewer wrote the word as printed
                            if "ref" not in c and "refs" not in c:   # reviewer saw no superscript there
                                w["refs"] = [r for r in w.get("refs", []) if r.get("n")]
                        if c.get("italic"): w["italic"] = True
                        if "ref" in c:
                            w["refs"] = [{"n": c["ref"], "status": "ok"}] if c["ref"] else []
                        if "refs" in c:
                            w["refs"] = [{"n": r, "status": "ok"} for r in c["refs"]]
                        hit = True
        if not hit:
            log.append(f"p{page}: correction did not match anything: {c}")

# ---------------------------------------------------------------- text
ESC = re.compile(r"([\\`*_{}\[\]<>#^~$@|])")
def esc(s):
    return ESC.sub(r"\\\1", s)

def chapter_of(page):
    k = 0
    for start, n in CHAPTER_START.items():
        if page >= start: k = n
    return k

CHEM = re.compile(r"^\(?(?=[^\d]*\d)(?:[A-Z][a-z]?\d*){2,}\)?[.,;:]?$")

def render_word(w, page):
    t = w["t"] if w.get("raw") else norm_word(w["t"])
    s = t if w.get("raw") else esc(t)
    if CHEM.match(t):                       # Fe2O3 -> Fe~2~O~3~
        s = re.sub(r"(?<=[A-Za-z])(\d+)", r"~\1~", s)
    if w.get("italic"):
        s = f"*{s}*"
    if w.get("sup"):
        s += f"<sup>{esc(w['sup'])}</sup>" + esc(w.get("tail", ""))
    if w.get("sup_unplaced"):
        s += f"⟦?sup{w['sup_unplaced']}⟧"
    refs = w.get("refs", [])
    if refs:
        k = chapter_of(page)
        parts = []
        for i, r in enumerate(refs):
            n = r.get("n")
            if n is None or r.get("status") == "flag":
                parts.append(f"⟦?ref{n or ''}⟧")
                continue
            ident = f"c{k}n{n}"
            first = ident not in seen_refs
            seen_refs.add(ident)
            attrs = f"{{{'#r-' + ident + ' ' if first else ''}.noteref epub:type=\"noteref\"}}"
            if i: parts.append("<sup>,</sup>")
            parts.append(f"[<sup>{n}</sup>](#{ident}){attrs}")
        s += "".join(parts)
    return s

seen_refs = set()

page_text = {}   # page -> plain words emitted (for verify.py's coverage check)

DATED = re.compile(r"^(.*\S)\s+((?:January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4})$")
def norm_line_text(l):
    return " ".join(norm_word(w["t"]) for w in l["words"])

def render_lines(lines, page, anchor=None):
    """Join the words of a run of lines. `anchor` is inserted before the first word."""
    out = []
    for l in lines:
        ws = []
        for i, w in enumerate(l["words"]):
            r = render_word(w, page)
            # closing paren / comma after a superscript is a separate hOCR word: no space
            if i and w["t"] in (")", ",", ".", ";", ":") and l["words"][i - 1].get("refs") and ws:
                ws[-1] += r
            else:
                ws.append(r)
        out.append(" ".join(ws))
        page_text.setdefault(page, []).extend(norm_word(w["t"]) for w in l["words"])
    s = norm_text(" ".join(out))
    s = re.sub(r"(?<=\S)\* \*(?=\S)", " ", s)         # *a* *b* -> *a b*
    s = re.sub(r"^(\d+)\.", r"\1\\.", s)          # don't let "1. ..." start a list
    s = re.sub(r"^([-+])", r"\\\1", s)
    if anchor:
        s = anchor + s
    return s

def page_anchor(page):
    if page in ROMAN:
        n = ROMAN[page]
    elif page >= 17:
        n = str(page - 16)
    else:
        return ""
    return f'[]{{#pg{n} .pb epub:type="pagebreak" title="{n}" role="doc-pagebreak"}}'

# ---------------------------------------------------------------- body
def emit_body(md):
    pages = sorted(int(p.stem.split("-")[1]) for p in (OUT / "blocks").glob("p-*.json"))
    corrs = load_corrections()
    open_block = None         # a para/quote/list that may continue on the next page
    pending = []              # markdown lines of the open block
    deferred = []             # figures waiting for the open block to close
    def emit_figure(b, page, anchor):
        fc = FIG_CORR.get((page, int((b.get("file") or "-0")[:-4].rsplit("-", 1)[1])), {})
        title = fc.get("caption_title", b.get("caption_title") or "")
        cap = fc.get("caption_text", b.get("caption_text") or "")
        if "md" in fc:                      # reviewer supplied the full caption in Markdown
            caption = fc["md"]
        else:
            title = norm_text(" ".join(norm_word(w) for w in title.split()))
            cap = norm_text(" ".join(norm_word(w) for w in cap.split()))
            caption = (f"**{esc(title)}** " if title else "") + ital_titles(esc(cap))
        page_text.setdefault(page, []).extend((title + " " + cap).split())
        if not b.get("file"):
            log.append(f"p{page}: figure without file")
            md.append(anchor + f"⟦?figure p{page}⟧\n"); return
        if anchor: md.append(anchor + "\n")
        if b.get("shared_caption") and not b.get("last_of_group"):
            caption = ""                     # one caption under a pair: keep it on the last image
        md.append(f"![{caption}]({b['file']})\n")
    def close():
        nonlocal open_block, pending, deferred
        if open_block is not None:
            md.extend(pending); md.append("")
        open_block, pending = None, []
        for f in deferred:
            emit_figure(*f)
        deferred = []
    def render_block(b, page, anchor):
        t = b["type"]
        if t == "para":
            return [render_lines(b["lines"], page, anchor)]
        if t == "quote" and len(b["lines"]) >= 3 and all(DATED.search(norm_line_text(l)) for l in b["lines"]):
            # a two-column schedule (task ... date) set as a block: render as a table
            rows = []
            for l in b["lines"]:
                m = DATED.search(render_lines([l], page)); rows.append((m.group(1), m.group(2)))
            out = ["| | |", "|---|---|"] + [f"| {a} | {d} |" for a, d in rows]
            return [(anchor or "") + "\n".join(out)]
        if t in ("quote", "list"):
            starts = sorted(set(b.get("paras", [])) | {0})
            paras = []
            for i, s in enumerate(starts):
                e = starts[i + 1] if i + 1 < len(starts) else len(b["lines"])
                paras.append(render_lines(b["lines"][s:e], page, anchor if i == 0 else None))
            if t == "quote":
                out = []
                for p in paras:               # one blockquote, several paragraphs
                    if out: out.append(">")
                    out.append("> " + p)
                return out
            items = []
            for p in paras:
                m = re.match(r"^(\d{1,2})\\?\.\s*(.*)$", p, re.S)
                if items: items.append("")
                items.append(f"{m.group(1)}. {m.group(2)}" if m else "    " + p)
            return items
        raise ValueError(t)

    for page in pages:
        d = json.loads((OUT / "blocks" / f"p-{page:03d}.json").read_text())
        blocks = d["blocks"]
        apply_corrections(page, blocks, corrs.get(page, []))
        anchor = page_anchor(page)
        for i, b in enumerate(blocks):
            t = b["type"]
            if t == "head":
                continue
            if t == "chapter":
                close()
                md.append(f"# <span class=\"chapno\">{b['n']}</span> {b['title']}\n"); md.append(anchor + "\n"); anchor = ""
                continue
            if t == "section":
                close()
                md.append(f"# {b['title']}\n"); md.append(anchor + "\n"); anchor = ""
                continue
            if t == "subhead":
                close()
                if anchor: md.append(anchor + "\n"); anchor = ""
                if b.get("flag"):
                    log.append(f"p{page}: unverified subhead candidate: {b['text']}")
                    md.append(f"## ⟦?subhead⟧ {esc(b['text'])}\n")
                else:
                    md.append(f"## {esc(b['text'])}\n")
                continue
            if t == "figure":
                # a figure that interrupts an open paragraph (full-page figure
                # between two text pages) is emitted after the paragraph ends
                if open_block is not None:
                    deferred.append((b, page, ""))
                    continue
                emit_figure(b, page, anchor); anchor = ""
                continue
            # text blocks
            if b.get("continues") and open_block is not None and open_block["type"] == "list" and t == "quote":
                # a list item's second paragraph overleaf: tight leading, no number
                new = ["    " + x[2:] for x in render_block(b, page, anchor) if x != ">"]; anchor = ""
                pending.append(""); pending.extend(new)
                if not b.get("open"): close()
                continue
            if b.get("continues") and open_block is not None and open_block["type"] == t:
                # continuation of the previous page's last block
                new = render_block(b, page, anchor); anchor = ""
                if t == "para":
                    pending[-1] = pending[-1] + " " + new[0]
                else:
                    # quote/list: does the first paragraph continue the previous
                    # one, or start a new one? Decide from the sentence boundary.
                    first = new[0].lstrip("> ") if t == "quote" else new[0]
                    ends = re.search(r'[.!?"’)]\s*(\[.*?\]\(#c\d+n\d+\)\{[^}]*\})*$', pending[-1])
                    starts_new = first[:1].isupper() or re.match(r"^\d{1,2}\. ", first)
                    if ends and starts_new:
                        pending.extend(new)
                    else:
                        pending[-1] = pending[-1] + " " + first
                        pending.extend(new[1:])
                if not b.get("open"):
                    close()
                else:
                    open_block = b
                continue
            if b.get("continues") and open_block is None:
                log.append(f"p{page}: block marked continues but nothing open; started new paragraph")
            close()
            rendered = render_block(b, page, anchor); anchor = ""
            if b.get("open"):
                open_block, pending = b, rendered
            else:
                md.extend(rendered); md.append("")
        if anchor:   # figure-only page inside a paragraph: the break sits in the text flow
            if open_block is not None and pending:
                pending[-1] += anchor
            else:
                md.append(anchor + "\n")
    close()

# ---------------------------------------------------------------- back matter
def bm_corrections(kind):
    """Review results for back matter: out/corrections/<kind>*.json.
    notes: [{"chapter": k, "n": n, "md": "text with *italics*"}]
    chronology / index: [{"i": index in the JSON list, ...replacement fields}]"""
    res = {}
    for f in sorted((OUT / "corrections").glob(f"{kind}*.json")) if (OUT / "corrections").exists() else []:
        for c in json.loads(f.read_text()):
            key = (c["chapter"], c["n"]) if kind == "notes" else c["i"]
            res[key] = c
    return res

def apply_bm(entries, corr):
    """Apply {i: {...fields}} corrections; an "append" field inserts a new entry after i."""
    out = []
    for i, e in enumerate(entries):
        c = corr.get(i, {})
        e.update({k: v for k, v in c.items() if k not in ("i", "append")})
        out.append(e)
        if c.get("append"):
            a = dict(c["append"]); a.setdefault("page", e.get("page")); out.append(a)
    return out

def emit_notes(md):
    p = OUT / "backmatter" / "notes.json"
    if not p.exists():
        md.append("# References and Notes\n\n⟦?notes missing⟧\n"); log.append("notes.json missing"); return
    d = json.loads(p.read_text())
    corr = bm_corrections("notes")
    for ch in d["chapters"]:
        for note in ch["notes"]:
            c = corr.get((ch["chapter"], note["n"]))
            if c: note["md"] = c["md"]
    md.append("# References and Notes\n")
    md.append(page_anchor(281) + "\n")
    md.append(esc(norm_text(d.get("intro", ""))) + "\n")
    last_page = 281
    for ch in d["chapters"]:
        k = ch["chapter"]
        md.append(f"## Chapter {k}\n")
        for note in ch["notes"]:
            n = note["n"]
            ident = f"c{k}n{n}"
            txt = note.get("md") or esc(norm_text(note["text"]))
            back = f" [↩](#r-{ident}){{.backlink}}" if f"c{k}n{n}" in seen_refs else ""
            new_pages = [pg for pg in note.get("pages", []) if pg > last_page]
            last_page = max([last_page] + new_pages)
            anchors = "".join(page_anchor(pg) for pg in new_pages)
            md.append(f'::: {{#{ident} .note epub:type="footnote"}}')
            line = f"[{n}.]{{.noteno}} {anchors}{txt}"
            if note.get("sublist"):
                line += "\\\n" + "\\\n".join(esc(item) for item in note["sublist"])   # hard line breaks
            md.append(line + back)
            md.append(":::\n")

def emit_chronology(md):
    p = OUT / "backmatter" / "chronology.json"
    md.append("# Chronology\n")
    md.append(page_anchor(317) + "\n")
    if not p.exists():
        md.append("⟦?chronology missing⟧\n"); log.append("chronology.json missing"); return
    last_page = 317
    corr = bm_corrections("chronology")
    entries = apply_bm(json.loads(p.read_text()), corr)
    for e in entries:
        anchor = ""
        if e.get("page") and e["page"] != last_page:
            anchor = page_anchor(e["page"]); last_page = e["page"]
        md.append(f"{esc(e['date'])}\n:   {anchor}{esc(norm_text(e['text']))}\n")

LOC = re.compile(r"(\d+)(?:[-–](\d+))?")
def link_locators(s):
    def rep(m):
        a = m.group(1)
        return f"[{m.group(0)}](#pg{a})"
    return LOC.sub(rep, s)

def emit_index(md):
    p = OUT / "backmatter" / "index.json"
    md.append("# Index\n")
    md.append(page_anchor(329) + "\n")
    if not p.exists():
        md.append("⟦?index missing⟧\n"); log.append("index.json missing"); return
    md.append("::: {.index}")
    last_page = 329
    pending_anchor = ""
    corr = bm_corrections("index")
    entries = apply_bm(json.loads(p.read_text()), corr)
    for e in entries:
        anchor = ""
        if e.get("page") and e["page"] != last_page:
            anchor = page_anchor(e["page"]); last_page = e["page"]
        if e.get("continued"):      # "(continued)" repeat of a main entry at a column top
            if anchor: pending_anchor = anchor
            continue
        anchor = pending_anchor + anchor; pending_anchor = ""
        text = esc(norm_text(e["text"]))
        loc = link_locators(esc(e.get("locators") or ""))
        body = text + (", " + loc if loc else "")
        if e.get("see"):            # "CDC. *See* Control Data Corp."
            m = re.match(r"^(See(?: also)?)\s*(.*)$", e["see"])
            body += ". " + (f"*{m.group(1)}* {esc(m.group(2))}" if m else f"*{esc(e['see'])}*")
        md.append(f"::: {{.ix{e.get('level', 0)}}}\n{anchor}{body}\n:::")
    md.append(":::\n")

# ---------------------------------------------------------------- build
def build_cover():
    if not (OUT / "cover.png").exists():
        subprocess.run(["pdftoppm", "-r", "200", "-gray", "-f", "7", "-l", "7", "-png", "-singlefile",
                        "book.pdf", str(OUT / "cover")], check=True)

def fix_alt(x):
    """Pandoc copies the whole caption into alt; use the bold title instead, since the
    figcaption already carries the text."""
    def repl(m):
        img, cap = m.group(1), m.group(2)
        t = re.search(r"<strong>(.*?)</strong>", cap)
        alt = re.sub(r"<[^>]+>", "", t.group(1)) if t else "figure"
        img = re.sub(r'alt="[^"]*"', f'alt="{alt}"', img)
        return img + cap
    return re.sub(r'(<img [^>]*>)(\s*<figcaption[\s\S]*?</figcaption>)', repl, x)

def add_page_list(epub):
    """Pandoc does not emit an EPUB3 page-list nav; add one from the pg anchors."""
    z = zipfile.ZipFile(epub)
    names = z.namelist()
    entries = []
    for n in names:
        if n.startswith("EPUB/text/") and n.endswith(".xhtml"):
            x = z.read(n).decode()
            for m in re.finditer(r'id="pg([^"]+)"', x):
                entries.append((m.group(1), f"text/{pathlib.Path(n).name}#pg{m.group(1)}"))
    nav = z.read("EPUB/nav.xhtml").decode()
    def key(e):
        v = e[0]
        return (0, {"vii": 7, "viii": 8, "ix": 9, "x": 10}[v]) if not v.isdigit() else (1, int(v))
    lis = "\n".join(f'<li><a href="{h}">{t}</a></li>' for t, h in sorted(entries, key=key))
    pl = f'\n<nav epub:type="page-list" id="page-list" hidden="hidden"><h1>Pages</h1><ol>{lis}</ol></nav>\n'
    nav = nav.replace("</body>", pl + "</body>")
    tmp = epub.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w") as zo:
        zo.writestr(zipfile.ZipInfo("mimetype"), z.read("mimetype"), compress_type=zipfile.ZIP_STORED)
        for n in names:
            if n == "mimetype": continue
            data = nav.encode() if n == "EPUB/nav.xhtml" else z.read(n)
            if n.startswith("EPUB/text/") and n.endswith(".xhtml"):
                data = fix_alt(data.decode()).encode()
            zo.writestr(n, data, compress_type=zipfile.ZIP_DEFLATED)
    z.close(); tmp.replace(epub)
    return len(entries)

def main():
    md = []
    md.append(pathlib.Path("frontmatter.md").read_text())
    emit_body(md)
    emit_notes(md)
    emit_chronology(md)
    emit_index(md)
    text = "\n".join(md)
    (OUT / "book.md").write_text(text)
    (OUT / "pagetext").mkdir(exist_ok=True)
    for pg, ws in page_text.items():
        (OUT / "pagetext" / f"p-{pg:03d}.txt").write_text(" ".join(ws))
    n_unres = text.count("⟦?")
    (OUT / "assemble.log").write_text("\n".join(log) + f"\n\nunresolved markers: {n_unres}\n")
    print(f"book.md: {len(text)} chars, {n_unres} unresolved ⟦?⟧ markers, {len(log)} log lines")
    if "--no-epub" in sys.argv:
        return
    build_cover()
    subprocess.run(["pandoc", str(OUT / "book.md"), "-o", str(OUT / "book.epub"), "--to", "epub3",
                    "--from", "markdown+smart+raw_html+bracketed_spans+fenced_divs+link_attributes+definition_lists+implicit_figures-tex_math_dollars",
                    "--toc", "--toc-depth=2", "--css", "epub.css", "--epub-cover-image", str(OUT / "cover.png"),
                    "--metadata-file", "metadata.yaml"], check=True)
    n = add_page_list(OUT / "book.epub")
    print(f"book.epub written; {n} page anchors in page-list")

if __name__ == "__main__":
    main()
