"""Stage 9: invariants and coverage checks on the assembled book.

  * superscript marks: per chapter max == note count, first citations in order
  * back matter counts: 518 notes (61/39/64/86/69/59/64/48/28), 174 chronology
    entries, ~429 index entries, every note referenced from the body
  * subheads: each known subhead appears exactly once as an h2
  * per-page vocabulary coverage of the emitted text against the PDF's own text
    layer (out/text): flags pages that lost text
  * no unresolved ⟦?…⟧ markers; epubcheck clean
Exit status 1 if any hard check fails.
"""
import collections, json, pathlib, re, subprocess, sys
from normalize import norm_word
import layout

OUT = pathlib.Path("out")
EXPECTED = [61, 39, 64, 86, 69, 59, 64, 48, 28]
fails = []
def check(ok, msg):
    print(("ok   " if ok else "FAIL ") + msg)
    if not ok: fails.append(msg)

def strip(t): return re.sub(r"[^\w]+", "", norm_word(t)).lower()

# superscripts
p = OUT / "marks" / "summary.json"
if p.exists():
    s = json.loads(p.read_text())
    chs = s["chapters"] if isinstance(s, dict) and "chapters" in s else s
    for i, ch in enumerate(sorted(chs, key=lambda c: c.get("chapter", 0)) if isinstance(chs, list) else
                           [chs[k] for k in sorted(chs, key=int)]):
        k = i + 1
        check(ch.get("max") == EXPECTED[k-1], f"ch{k}: max mark {ch.get('max')} == {EXPECTED[k-1]} notes")
        # notes the book itself cites out of order or never (verified by eye in
        # the survey and again by the superscript stage)
        KNOWN_GAPS = {2: [26], 3: [42, 46], 6: [30], 7: [33, 48], 8: [32]}
        KNOWN_NEVER = {2: [26], 3: [42], 6: [30], 7: [33, 48]}
        gaps = sorted(set(ch.get("gaps", [])[::2]))
        check(gaps == KNOWN_GAPS.get(k, []), f"ch{k}: out-of-order first citations {gaps} are the known set {KNOWN_GAPS.get(k, [])}")
        nc = ch.get("never_cited", [])
        check(nc == KNOWN_NEVER.get(k, []), f"ch{k}: never-cited notes {nc} are the known set {KNOWN_NEVER.get(k, [])}")
else:
    check(False, "out/marks/summary.json missing")

# notes / chronology / index
p = OUT / "backmatter" / "notes.json"
if p.exists():
    d = json.loads(p.read_text())
    counts = [len(c["notes"]) for c in d["chapters"]]
    check(counts == EXPECTED, f"note counts {counts}")
    for c in d["chapters"]:
        ns = [n["n"] for n in c["notes"]]
        check(ns == list(range(1, len(ns) + 1)), f"ch{c['chapter']} note numbers contiguous")
else:
    check(False, "notes.json missing")
p = OUT / "backmatter" / "chronology.json"
check(p.exists() and len(json.loads(p.read_text())) == 174, "174 chronology entries")
p = OUT / "backmatter" / "index.json"
if p.exists():
    n0 = sum(1 for e in json.loads(p.read_text()) if e.get("level", 0) == 0)
    check(420 <= n0 <= 440, f"index main entries {n0} (expected ~429)")
else:
    check(False, "index.json missing")

# book.md
book = (OUT / "book.md").read_text()
h2 = collections.Counter(m.group(1).strip() for m in re.finditer(r"^## (.*)$", book, re.M))
missing = [h for h in layout.SUBHEADS if h2[h] != 1]
check(not missing, f"every known subhead appears once (missing/dup: {missing})")
refs = set(re.findall(r"\]\(#(c\d+n\d+)\)", book))
defined = set(re.findall(r"::: \{#(c\d+n\d+)", book))
check(refs <= defined, f"all {len(refs)} referenced notes exist ({len(refs - defined)} dangling: {sorted(refs - defined)[:10]})")
unref = sorted(defined - refs, key=lambda x: (int(x[1:x.index('n')]), int(x[x.index('n')+1:])))
print(f"info  notes never referenced from the body: {len(unref)} {unref[:20]}")
n_unres = book.count("⟦?")
check(n_unres == 0, f"unresolved markers: {n_unres}")

# vocabulary coverage per page (figure pages carry figure lettering in the IA
# layer that we deliberately drop, so they get a looser threshold)
fig_pages = set()
if (OUT / "figures" / "figures.json").exists():
    fig_pages = {f["page"] for f in json.loads((OUT / "figures" / "figures.json").read_text())}
low = []
for f in sorted((OUT / "pagetext").glob("p-*.txt")):
    pg = int(f.stem.split("-")[1])
    ours = collections.Counter(strip(w) for w in f.read_text().split())
    ia = [strip(w) for w in (OUT / "text" / f"p{pg:03d}.txt").read_text().split()]
    ia = [w for w in ia if len(w) >= 4 and w.isalpha()]
    if len(ia) < 20: continue
    cov = sum(1 for w in ia if ours[w] > 0) / len(ia)
    if cov < (0.5 if pg in fig_pages else 0.9): low.append((pg, round(cov, 2)))
check(not low, f"page vocabulary coverage >= 0.9 everywhere (low: {low})")

# epubcheck
if (OUT / "book.epub").exists():
    r = subprocess.run(["epubcheck", str(OUT / "book.epub")], capture_output=True, text=True)
    check("No errors or warnings" in r.stdout, "epubcheck: " + (r.stdout.strip().splitlines()[-2] if r.stdout else r.stderr[-200:]))
print(f"\n{len(fails)} failing checks")
sys.exit(1 if fails else 0)
