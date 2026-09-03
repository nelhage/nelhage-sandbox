"""Per-page OCR quality stats from tesseract hOCR output in out/hocr/.

Prints a TSV: page, n_words, mean_conf, frac_lt60, frac_lt80, n_lines, first_line
and a histogram of page mean confidence."""
import glob, re, statistics, sys
from lxml import html

rows = []
for f in sorted(glob.glob("out/hocr/p-*.hocr")):
    page = int(re.search(r"p-(\d+)", f).group(1))
    t = html.parse(f)
    words = t.xpath('//span[@class="ocrx_word"]')
    confs = []
    for w in words:
        m = re.search(r"x_wconf (\d+)", w.get("title", ""))
        if m and w.text_content().strip():
            confs.append(int(m.group(1)))
    lines = t.xpath('//span[@class="ocr_line"]')
    first = lines[0].text_content().strip().replace("\n", " ") if lines else ""
    if confs:
        rows.append((page, len(confs), statistics.mean(confs),
                     sum(c < 60 for c in confs)/len(confs), sum(c < 80 for c in confs)/len(confs),
                     len(lines), first[:60]))
    else:
        rows.append((page, 0, 0, 0, 0, len(lines), first[:60]))
for r in rows:
    print("%d\t%d\t%.1f\t%.3f\t%.3f\t%d\t%s" % r)
body = [r for r in rows if 17 <= r[0] <= 280 and r[1] > 50]
print("# body pages:", len(body), file=sys.stderr)
for lo in range(60, 100, 5):
    n = sum(1 for r in body if lo <= r[2] < lo+5)
    print(f"# mean conf {lo}-{lo+5}: {n}", file=sys.stderr)
worst = sorted(body, key=lambda r: r[2])[:15]
print("# worst body pages:", [(r[0], round(r[2],1)) for r in worst], file=sys.stderr)
