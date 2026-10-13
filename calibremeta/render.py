"""Render plan.json as an HTML review page."""

import json
import sys
from html import escape as e

plan = json.load(open(sys.argv[1]))
out = sys.argv[2]


def fmt_idx(x):
    if x is None:
        return ""
    return str(int(x)) if float(x).is_integer() else str(x)


def table(head, rows, cls=""):
    h = "".join(f"<th>{e(c)}</th>" for c in head)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table></div>'


def arrow(old, new):
    return f'<span class="old">{e(old)}</span>', f'<span class="new">{e(new)}</span>'


A = plan["authors"]
special = [a for a in A if a["note"] or a.get("merge")]
routine = [a for a in A if not (a["note"] or a.get("merge"))]
total_books_authors = sum(a["books"] for a in A)

def author_rows(xs):
    rows = []
    for a in sorted(xs, key=lambda a: a["sort"].lower()):
        o, n = arrow(a["old"], a["new"])
        tag = '<span class="chip merge">merge</span> ' if a.get("merge") else ""
        rows.append([o, n, f'<span class="mono">{e(a["sort"])}</span>', str(a["books"]), tag + e(a["note"])])
    return rows

series_add = [s for s in plan["series"] if s["kind"] == "add"]
series_fix = [s for s in plan["series"] if s["kind"] == "fix"]

def series_rows(xs):
    rows = []
    for s in sorted(xs, key=lambda s: (s["series"], s["index"])):
        was = f'{e(s["old_series"])} #{fmt_idx(s["old_index"])}' if s["old_series"] else '<span class="dim">none</span>'
        rows.append([e(s["title"]), e(s["author"]), was,
                     f'<span class="new">{e(s["series"])} #{fmt_idx(s["index"])}</span>', e(s["note"])])
    return rows

counts = {
    "Authors renamed": len(A),
    "Books re-credited": len(plan["book_authors"]),
    "Series added or fixed": len(plan["series"]),
    "Bogus series removed": len(plan["series_clear"]),
    "Titles cleaned": len(plan["titles"]),
}

html = f"""<title>Calibre Cleanup Plan</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,650&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400&display=swap">
<style>
/* Layout: single reading column, wide tables scroll in their own frames; summary strip, then one section per change type. */
:root {{
  --bg: #f6f6f3; --surface: #ffffff; --fg: #1f2421; --muted: #5d6660; --line: #dcdfd9;
  --accent: #2f6b55; --accent-soft: #e3efe9; --old: #9a4b3c; --warn: #8a6410; --warn-soft: #f6ecd3;
  --display: "Fraunces", Georgia, serif; --body: "Source Sans 3", system-ui, sans-serif;
  --mono: "JetBrains Mono", ui-monospace, Menlo, monospace;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #161a18; --surface: #1e2321; --fg: #e4e8e5; --muted: #9aa49e; --line: #313834;
  --accent: #7cc4a6; --accent-soft: #23352d; --old: #e39a88; --warn: #e2bd63; --warn-soft: #3a3220; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --bg: #161a18; --surface: #1e2321; --fg: #e4e8e5; --muted: #9aa49e; --line: #313834;
  --accent: #7cc4a6; --accent-soft: #23352d; --old: #e39a88; --warn: #e2bd63; --warn-soft: #3a3220; color-scheme: dark; }}
body {{ background: var(--bg); color: var(--fg); font: 16px/1.55 var(--body); }}
main {{ max-width: 1100px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 80px; display: grid; gap: 40px; }}
h1, h2 {{ font-family: var(--display); text-wrap: balance; margin: 0; }}
h1 {{ font-size: 2.2rem; font-weight: 650; }}
h2 {{ font-size: 1.45rem; font-weight: 500; }}
p {{ margin: 0; max-width: 68ch; }}
section {{ display: grid; gap: 14px; }}
.lede {{ color: var(--muted); font-size: 1.05rem; }}
.stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }}
.stat {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; }}
.stat b {{ display: block; font: 650 1.6rem var(--display); font-variant-numeric: tabular-nums; color: var(--accent); }}
.stat span {{ color: var(--muted); font-size: .9rem; }}
.decide {{ background: var(--warn-soft); border-radius: 8px; padding: 16px 18px; display: grid; gap: 8px; }}
.decide h2 {{ color: var(--warn); }}
.decide ol {{ margin: 0; padding-left: 1.3em; display: grid; gap: 6px; max-width: 80ch; }}
.tw {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }}
table {{ border-collapse: collapse; width: 100%; font-size: .92rem; }}
th, td {{ text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top; }}
th {{ font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); font-weight: 600; background: var(--bg); position: sticky; top: 0; }}
tr:last-child td {{ border-bottom: 0; }}
td:nth-child(4) {{ font-variant-numeric: tabular-nums; }}
.old {{ color: var(--old); }}
.new {{ color: var(--accent); font-weight: 600; }}
.dim {{ color: var(--muted); }}
.mono {{ font-family: var(--mono); font-size: .85em; }}
.chip {{ display: inline-block; font-size: .72rem; padding: 1px 7px; border-radius: 99px; background: var(--accent-soft); color: var(--accent); font-weight: 600; }}
details {{ display: grid; gap: 10px; }}
summary {{ cursor: pointer; color: var(--accent); font-weight: 600; }}
summary:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
code {{ font-family: var(--mono); font-size: .88em; background: var(--accent-soft); padding: 1px 5px; border-radius: 4px; }}
ul.plain {{ margin: 0; padding-left: 1.2em; display: grid; gap: 4px; max-width: 80ch; }}
</style>
<main>
<header style="display:grid;gap:10px">
  <h1>Calibre Cleanup Plan</h1>
  <p class="lede">Proposed metadata changes for <code>~/Calibre Library</code> (669 books). Nothing has been changed yet. I ran the full plan against a copy of <code>metadata.db</code> and it applied cleanly.</p>
</header>

<div class="stats">{"".join(f'<div class="stat"><b>{v}</b><span>{e(k)}</span></div>' for k, v in counts.items())}</div>

<section class="decide">
  <h2>Decisions for you</h2>
  <ol>
    <li><b>Title cleanup (273 books)</b> goes beyond what you asked for. It strips series tags and store noise such as <em>"(The Dresden Files, Book 1)"</em> and <em>"(Kindle Single)"</em> out of titles, since the series field now carries that information. It can be skipped as a unit.</li>
    <li><b>Folder renames.</b> Calibre names book folders after the author, so <code>Butcher, Jim/…</code> becomes <code>Jim Butcher/…</code>. This happens automatically and Calibre keeps track of it. It only matters if something outside Calibre points at those paths.</li>
    <li><b>Delete the "Quick Start Guide"</b> sample book that Calibre ships. It goes to Calibre's trash, so it can be recovered.</li>
    <li><b>Escape from Yokai Land</b> gets Laundry Files #11.5. Library catalogs call it #12, but that number is already taken by Season of Skulls in your numbering.</li>
    <li><b>J.M.R. Higgs</b> is merged into <b>John Higgs</b>. They are the same person, but the KLF book is credited as J.M.R.</li>
  </ol>
</section>

<section>
  <h2>Authors: display “First Last”, sort “Last, First”</h2>
  <p>Calibre keeps a separate <em>sort</em> value for each author, so this works natively. Today 417 of 469 authors are stored as <code>Last| First</code>, which is Calibre's escaped comma, so they display as “Butcher, Jim”. Every one gets flipped for display, and its sort value is kept or corrected. Duplicates that collapse to the same name are merged; the merges below are mostly the KFX/EPUB import pairs, such as “Ann Leckie” and “Leckie, Ann”. The result is 459 authors.</p>
  <h3 style="margin:0;font-size:1rem">Special cases and merges ({len(special)})</h3>
  {table(["Current", "New display", "Sort", "Books", "Note"], author_rows(special))}
  <details><summary>Show the other {len(routine)} straightforward flips</summary>
  {table(["Current", "New display", "Sort", "Books", "Note"], author_rows(routine))}
  </details>
</section>

<section>
  <h2>Per-book author fixes</h2>
  <p>Co-authors in the wrong order (I used the order on the cover), wrong people credited, and the 8 books currently credited to “Unknown”.</p>
  {table(["Book", "Current", "Proposed", "Why"], [[e(b["title"]), f'<span class="old">{e("; ".join(b["old"]))}</span>', f'<span class="new">{e("; ".join(b["new"]))}</span>', e(b["note"])] for b in plan["book_authors"]])}
</section>

<section>
  <h2>Series: missing entries</h2>
  <p>These books belong to a series but had no series set. Indexes follow the usual Goodreads/ISFDB convention, where x.5 marks a novella or side story between numbered books. I checked the uncertain ones on the web (Murderbot, Dresden novellas, Laundry Files, Lady Astronaut, Dead Djinn, Radiant Star).</p>
  {table(["Book", "Author", "Was", "Proposed", "Note"], series_rows(series_add))}
</section>

<section>
  <h2>Series: wrong numbers</h2>
  {table(["Book", "Author", "Was", "Proposed", "Note"], series_rows(series_fix))}
</section>

<section>
  <h2>Series: remove bogus entries</h2>
  <p>Most of these came from store metadata that put a publisher imprint in the series field, such as “Allen Lane History #46”.</p>
  {table(["Book", "Current series", "Why"], [[e(s["title"]), f'<span class="old">{e(s["old_series"])} #{fmt_idx(s["old_index"])}</span>', e(s["note"])] for s in plan["series_clear"]])}
  <h3 style="margin:0;font-size:1rem">Rename series</h3>
  {table(["Current", "New", "Books"], [[*arrow(s["old"], s["new"]), str(s["books"])] for s in plan["series_rename"]])}
</section>

<section>
  <h2>Titles</h2>
  <p>This removes series tags and retail noise. Real subtitles (mostly nonfiction) are kept. Bare <em>“: A Novel”</em> suffixes are dropped. A few titles get proper capitalization or have a typo fixed.</p>
  <details><summary>Show all {len(plan["titles"])} title changes</summary>
  {table(["Current", "Proposed"], [list(arrow(t["old"], t["new"])) for t in plan["titles"]])}
  </details>
</section>

<section>
  <h2>Smaller fixes</h2>
  <h3 style="margin:0;font-size:1rem">Missing publication dates (currently shown as year 101)</h3>
  {table(["Book", "Current", "Proposed", "Note"], [[e(p["title"]), f'<span class="old">{e(p["old"])}</span>', f'<span class="new">{e(p["new"])}</span>', e(p["note"])] for p in plan["pubdates"]])}
  <h3 style="margin:0;font-size:1rem">Publisher variants merged</h3>
  {table(["Current", "New", "Books"], [[f'<span class="old">{e(p["old"])}</span>', f'<span class="new">{e(p["new"]) if p["new"] else "(cleared)"}</span>', str(p["books"])] for p in plan["publishers"]])}
  <h3 style="margin:0;font-size:1rem">Other</h3>
  <ul class="plain">{"".join(f"<li>{e(m.get('title', ', '.join(m.get('tags', []))))}: {e(m['note'])}</li>" for m in plan["misc"])}</ul>
</section>

<section>
  <h2>Not in this plan</h2>
  <ul class="plain">
    <li><b>Descriptions:</b> 608 of 669 books have no description. Calibre's metadata downloader could fill them, but that is a separate bulk job that pulls from the network.</li>
    <li><b>Genre tags:</b> only 9 books have any tags, so a genre or fiction/nonfiction pass could be worth doing as a follow-up.</li>
    <li><b>Polyamory and Jealousy</b> is listed as “More Than Two Essentials #10”. I couldn't verify that, so it is unchanged.</li>
    <li>Ann Leckie's <em>Provenance</em>, <em>Translation State</em> and <em>Radiant Star</em> are set in the Radch universe but aren't numbered entries, so they stay standalone.</li>
  </ul>
</section>

<section>
  <h2>How it gets applied</h2>
  <p>Close Calibre first. I'll back up <code>metadata.db</code>, then run <code>calibremeta/apply.py</code> under <code>calibre-debug</code>. It uses Calibre's own API, so renames, merges, sort values and folder moves stay consistent. Any section can be skipped, e.g. <code>--skip titles</code>.</p>
</section>
</main>
"""
open(out, "w").write(html)
print("wrote", out)
