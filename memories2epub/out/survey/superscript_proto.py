#!/usr/bin/env python3
"""Prototype detector for superscript endnote references.

Pipeline per page:
  1. render page (pymupdf) at --dpi, or use the PDF's native 360-ppi JBIG2 text mask
     (--native; extracted with `pdfimages -png`), binarise.
  2. tesseract hOCR (--psm 1) for line geometry (bbox, baseline, x_size, ascender,
     descender) and word boxes / text.
  3. per text line: connected components (run-length union-find, pure python) in the
     line box extended upward; a candidate superscript is a component whose bottom is
     well above the baseline and whose top is above the x-height line, with a height
     of roughly 0.5-1.1 x-height.  Adjacent candidates are grouped into a "mark".
  4. per mark: crop, upscale, tesseract --psm 8 digits-only; also a simple
     nearest-template classifier built from templates/ (optional).
  5. classify mark as REF (preceded by punctuation / at word end) vs OTHER (e.g. M^2).

Usage:
  superscript_proto.py --pages 20 21 22 60 130 210 [--dpi 300|--native] [--gt gt.json]
Writes out/thumb/ss/det-<page>.json and prints a per-page report.
"""
import argparse, json, os, re, subprocess, sys, tempfile, math
from collections import defaultdict
from PIL import Image, ImageOps, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SCRATCH = os.path.join(ROOT, 'out', 'thumb', 'ss')
PDF = os.path.join(ROOT, 'book.pdf')

# --------------------------------------------------------------------------- images

def render_page(page, dpi):
    """Render with pymupdf, return grayscale PIL image and path."""
    import pymupdf
    path = os.path.join(SCRATCH, f'r{dpi}-{page:03d}.png')
    if not os.path.exists(path):
        doc = pymupdf.open(PDF)
        pix = doc[page - 1].get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        pix.save(path)
    return Image.open(path).convert('L'), path


def native_mask(page):
    """Extract the JBIG2 text mask (360 ppi, 1-bit; white = ink) and return it as a
    normal black-on-white grayscale image."""
    path = os.path.join(SCRATCH, f'n360-{page:03d}.png')
    if not os.path.exists(path):
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(['pdfimages', '-png', '-f', str(page), '-l', str(page), PDF,
                            os.path.join(td, 'x')], check=True)
            cands = [f for f in os.listdir(td) if f.endswith('.png')]
            best = None
            for f in cands:
                im = Image.open(os.path.join(td, f))
                if im.mode == '1' and (best is None or im.size[0] > best.size[0]):
                    best = im
            assert best is not None, 'no 1-bit mask on page %d' % page
            ImageOps.invert(best.convert('L')).save(path)
    return Image.open(path).convert('L'), path

# --------------------------------------------------------------------------- hOCR

def run_tesseract(png, base, psm=1, extra=()):
    hocr = base + '.hocr'
    if not os.path.exists(hocr):
        subprocess.run(['tesseract', png, base, '-l', 'eng', '--psm', str(psm), *extra, 'hocr'],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return hocr


def parse_hocr(path):
    from lxml import etree
    t = etree.parse(path, etree.HTMLParser())
    lines = []
    par_of = {}
    for par in t.iter():
        if par.get('class') == 'ocr_par':
            for el in par.iter():
                par_of[el] = par.get('id')
    for el in t.iter():
        if el.get('class') not in ('ocr_line', 'ocr_header', 'ocr_caption', 'ocr_textfloat'):
            continue
        title = el.get('title') or ''
        m = re.search(r'bbox (\d+) (\d+) (\d+) (\d+)', title)
        bb = tuple(map(int, m.groups()))
        g = lambda k, d: float(re.search(k + r' ([-\d.]+)', title).group(1)) if re.search(k + r' ([-\d.]+)', title) else d
        a, b = 0.0, 0.0
        mb = re.search(r'baseline ([-\d.]+) ([-\d.]+)', title)
        if mb:
            a, b = float(mb.group(1)), float(mb.group(2))
        words = []
        for w in el.iter():
            if w.get('class') != 'ocrx_word':
                continue
            wt = w.get('title') or ''
            mw = re.search(r'bbox (\d+) (\d+) (\d+) (\d+)', wt)
            mc = re.search(r'x_wconf (\d+)', wt)
            words.append(dict(bbox=tuple(map(int, mw.groups())), text=''.join(w.itertext()).strip(),
                              conf=int(mc.group(1)) if mc else 0))
        lines.append(dict(bbox=bb, a=a, b=b, x_size=g('x_size', 0), desc=g('x_descenders', 0),
                          asc=g('x_ascenders', 0), words=words, par=par_of.get(el)))
    lines.sort(key=lambda l: (l['bbox'][1], l['bbox'][0]))
    return lines

# --------------------------------------------------------------------------- CC

def components(img, box, thresh=128):
    """Connected components (8-conn) of ink pixels inside box=(x0,y0,x1,y1).
    Returns list of dicts with bbox (absolute coords), area, ink runs."""
    x0, y0, x1, y1 = box
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(img.width, x1); y1 = min(img.height, y1)
    if x1 <= x0 or y1 <= y0:
        return []
    crop = img.crop((x0, y0, x1, y1))
    W, H = crop.size
    data = crop.tobytes()
    # run-length encode each row
    rows = []
    for y in range(H):
        off = y * W
        runs = []
        x = 0
        while x < W:
            if data[off + x] < thresh:
                s = x
                while x < W and data[off + x] < thresh:
                    x += 1
                runs.append((s, x - 1))
            else:
                x += 1
        rows.append(runs)
    # union-find over runs
    parent = []
    run_index = []  # per row: list of run ids
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri
    rid = 0
    for y, runs in enumerate(rows):
        ids = []
        for r in runs:
            parent.append(rid); ids.append(rid); rid += 1
        run_index.append(ids)
        if y == 0:
            continue
        prev = rows[y - 1]; pids = run_index[y - 1]
        j = 0
        for k, (s, e) in enumerate(runs):
            # 8-connectivity: overlap allowing 1px diagonal
            while j < len(prev) and prev[j][1] < s - 1:
                j += 1
            jj = j
            while jj < len(prev) and prev[jj][0] <= e + 1:
                union(ids[k], pids[jj]); jj += 1
    comps = {}
    for y, runs in enumerate(rows):
        for k, (s, e) in enumerate(runs):
            r = find(run_index[y][k])
            c = comps.get(r)
            if c is None:
                comps[r] = c = dict(x0=s, y0=y, x1=e, y1=y, area=0)
            c['x0'] = min(c['x0'], s); c['x1'] = max(c['x1'], e)
            c['y0'] = min(c['y0'], y); c['y1'] = max(c['y1'], y)
            c['area'] += e - s + 1
    out = []
    for c in comps.values():
        out.append(dict(x0=c['x0'] + x0, y0=c['y0'] + y0, x1=c['x1'] + x0 + 1, y1=c['y1'] + y0 + 1,
                        area=c['area']))
    out.sort(key=lambda c: c['x0'])
    return out

# --------------------------------------------------------------------------- detection

PUNCT_TAIL = re.compile(r'[.,;:?!\'"”’)\]]+$')


def detect_line(img, line, scale, dbg=None):
    """Return list of marks in a line. scale = image px per hOCR px (1.0 if same image)."""
    x0, y0, x1, y1 = [v * scale for v in line['bbox']]
    xs = line['x_size'] * scale
    desc = line['desc'] * scale
    asc = line['asc'] * scale
    xh = xs - desc - asc
    if xh <= 4 or xs <= 0:
        return []
    def baseline_at(x):
        return y1 + line['b'] * scale + line['a'] * (x - x0)
    box = (int(x0 - 4), int(y0 - 0.5 * xs), int(x1 + 6), int(y1 + 2))
    comps = components(img, box)
    # per-component features relative to the baseline
    for c in comps:
        base = baseline_at((c['x0'] + c['x1']) / 2)
        c['h'] = c['y1'] - c['y0']
        c['w'] = c['x1'] - c['x0']
        c['bot_up'] = (base - c['y1']) / xh      # how far bottom sits above baseline (x-heights)
        c['top_up'] = (base - c['y0']) / xh      # how far top sits above baseline
        c['hx'] = c['h'] / xh
        c['fill'] = c['area'] / max(1, c['h'] * c['w'])
    # main-text components: those that touch the baseline band
    main = [c for c in comps if c['bot_up'] < 0.25 and c['top_up'] > 0.5]
    cands = []
    for c in comps:
        if c['area'] < 6:
            continue
        # superscript: bottom well above baseline, top above the x-height line, height ~ 0.45..1.15 x-height
        if 0.25 <= c['bot_up'] <= 0.75 and c['top_up'] >= 1.2 and 0.75 <= c['hx'] <= 1.25 and c['w'] >= 0.25 * xh:
            cands.append(c)
    # group horizontally adjacent candidates (gap <= 0.6 xh, vertical overlap)
    cands.sort(key=lambda c: c['x0'])
    groups = []
    seps = [c for c in comps if c['hx'] < 0.5 and 0.1 < c['bot_up'] < 0.9 and c['area'] >= 6 and c['w'] < 0.5 * xh]
    for c in cands:
        if groups and c['x0'] - groups[-1][-1]['x1'] <= 0.6 * xh:
            last = groups[-1][-1]
            # a raised small comma between two digits separates chained refs "60,61"
            if any(sp['x0'] >= last['x1'] - 1 and sp['x1'] <= c['x0'] + 1 for sp in seps):
                groups.append([c])
            else:
                groups[-1].append(c)
        else:
            groups.append([c])
    marks = []
    for g in groups:
        gx0 = min(c['x0'] for c in g); gx1 = max(c['x1'] for c in g)
        gy0 = min(c['y0'] for c in g); gy1 = max(c['y1'] for c in g)
        # nearest main-text component to the left / right
        # context to the left: ignore specks and anything sitting below the baseline (descender
        # fragments / noise from the next line)
        left = [c for c in comps if c['x1'] <= gx0 + 2 and c['area'] >= 15 and c['y1'] > gy0 + 2 and c['bot_up'] > -0.7]
        right = [c for c in comps if c['x0'] >= gx1 - 2 and c['area'] >= 6]
        lgap = (gx0 - max(c['x1'] for c in left)) / xh if left else 9
        rgap = (min(c['x0'] for c in right) - gx1) / xh if right else 9
        lc = max(left, key=lambda c: c['x1']) if left else None
        # what precedes: a low small component (period/comma) or a full letter
        # period/comma: small, sits on the baseline. quote/apostrophe: small, narrow, high.
        # (colon/semicolon: the upper dot is what sits nearest, bot_up ~0.7)
        prev_is_punct = bool(lc and lc['hx'] < 0.5 and lc['top_up'] < 1.0 and lc['bot_up'] > -0.4 and lc['w'] < 0.6 * xh)
        prev_is_quote = bool(lc and lc['hx'] < 0.9 and lc['bot_up'] > 0.6 and lc['w'] < 0.45 * xh)
        # allow "quote after period": look one more component left
        if prev_is_quote:
            left2 = [c for c in left if c['x1'] <= lc['x0'] + 1]
            lc2 = max(left2, key=lambda c: c['x1']) if left2 else None
            if lc2 and lc2['bot_up'] > 0.6 and lc2['hx'] < 0.9 and lc2['w'] < 0.45 * xh:   # second quote stroke
                left3 = [c for c in left2 if c['x1'] <= lc2['x0'] + 1]
                lc2 = max(left3, key=lambda c: c['x1']) if left3 else None
            prev_is_punct = prev_is_punct or bool(lc2 and lc2['hx'] < 0.5 and lc2['top_up'] < 0.6 and lc2['w'] < 0.6 * xh)
        m = dict(x0=gx0, y0=gy0, x1=gx1, y1=gy1, n=len(g), lgap=round(lgap, 2), rgap=round(rgap, 2),
                 prev_is_punct=prev_is_punct, prev_is_quote=prev_is_quote, hx=round(max(c['hx'] for c in g), 2),
                 bot_up=round(min(c['bot_up'] for c in g), 2), top_up=round(max(c['top_up'] for c in g), 2),
                 widths=[c['w'] for c in g], fills=[round(c['fill'], 2) for c in g], xh=round(xh, 1),
                 comps=[(c['x0'], c['y0'], c['x1'], c['y1']) for c in g])
        # quote marks: narrow and thin, sit high; drop groups made only of such
        thin = all(c['w'] < 0.35 * xh and c['fill'] > 0.5 and c['hx'] < 0.6 for c in g)
        m['thin'] = thin
        marks.append(m)
    return marks


def word_before(line, x, scale):
    """hOCR word whose right edge is nearest to (and left of / containing) x."""
    best = None
    for w in line['words']:
        wx0, _, wx1, _ = [v * scale for v in w['bbox']]
        if wx0 <= x + 2:
            if best is None or wx1 > best[0]:
                best = (wx1, w)
    return best[1] if best else None


def ocr_digits(img, m, pad=3, up=4, mode='psm8'):
    x0, y0, x1, y1 = m['x0'] - pad, m['y0'] - pad, m['x1'] + pad, m['y1'] + pad
    crop = img.crop((x0, y0, x1, y1)).point(lambda v: 255 if v >= 128 else 0)
    crop = crop.resize(((x1 - x0) * up, (y1 - y0) * up), Image.LANCZOS)
    # add white margin so tesseract sees a "word"
    W, H = crop.size
    canvas = Image.new('L', (W + 2 * H, 3 * H), 255)
    canvas.paste(crop, (H, H))
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'c.png'); canvas.save(p)
        # NB: psm 7 (single line) works far better than psm 8 (single word) on these crops
        r = subprocess.run(['tesseract', p, '-', '-l', 'eng', '--psm', '7', '-c',
                            'tessedit_char_whitelist=0123456789', 'tsv'],
                           capture_output=True, text=True)
    txt, conf = '', -1
    for row in r.stdout.splitlines()[1:]:
        f = row.split('\t')
        if len(f) >= 12 and f[11].strip():
            txt += f[11].strip(); conf = max(conf, float(f[10]))
    return txt, conf, canvas

# --------------------------------------------------------------------------- templates (nearest neighbour digit classifier)

def norm_crop(img, box, size=(16, 24)):
    x0, y0, x1, y1 = box
    c = img.crop((x0, y0, x1, y1)).resize(size, Image.LANCZOS)
    return [1.0 - p / 255.0 for p in c.tobytes()]


def template_classify(img, comps_boxes, templates):
    """Classify each component box by nearest template (L2 on normalised crop)."""
    out = []
    for b in comps_boxes:
        v = norm_crop(img, b)
        best = None
        for d, tv in templates:
            dist = sum((a - b2) ** 2 for a, b2 in zip(v, tv))
            if best is None or dist < best[0]:
                best = (dist, d)
        out.append(best)
    return out

# --------------------------------------------------------------------------- main

def process_page(page, dpi=300, native=False, psm=1, ocr=True, templates=None, verbose=True, mask_reocr=False):
    if native:
        img, png = native_mask(page)
        scale_note = 'native360'
    else:
        img, png = render_page(page, dpi)
        scale_note = f'{dpi}dpi'
    base = os.path.join(SCRATCH, f'ss-{scale_note}-psm{psm}-{page:03d}')
    hocr = run_tesseract(png, base, psm=psm)
    lines = parse_hocr(hocr)
    # page statistics: median x_size of confident multi-word lines; mean conf per paragraph
    body = [l['x_size'] for l in lines if len(l['words']) >= 4 and sum(w['conf'] for w in l['words']) / len(l['words']) >= 60]
    body.sort()
    med_xs = body[len(body) // 2] if body else (lines[0]['x_size'] if lines else 0)
    par_conf = {}
    for l in lines:
        ws = [w for w in l['words'] if re.search(r'[A-Za-z]{2,}', w['text'])]
        par_conf.setdefault(l['par'], []).extend(w['conf'] for w in ws)
    par_conf = {k: (sum(v) / len(v) if v else 0) for k, v in par_conf.items()}
    dets = []
    for li, line in enumerate(lines):
        # only real text lines (skips figure lettering): >=1 alphabetic word, x_size near the
        # page's body-text x_size, and decent mean word confidence in the line OR its paragraph
        # (a one-word paragraph-final line like "division.15" has low line confidence).
        words = [w for w in line['words'] if re.search(r'[A-Za-z]{2,}|\d{2,}', w['text'])]
        if len(words) < 1 or not (0.75 * med_xs <= line['x_size'] <= 1.25 * med_xs):
            continue
        lconf = sum(w['conf'] for w in words) / len(words)
        if lconf < 50 and par_conf.get(line['par'], 0) < 50:
            continue
        prev_ref = None
        for m in detect_line(img, line, 1.0):
            # chained refs "25,26": second group follows a REF mark closely
            if prev_ref is not None and (m['x0'] - prev_ref['x1']) <= 1.3 * m['xh'] and not m['thin']:
                m['prev_is_punct'] = True; m['chained'] = True
            w = word_before(line, m['x0'], 1.0)
            m['line'] = li
            m['word'] = w['text'] if w else ''
            m['word_conf'] = w['conf'] if w else -1
            m['page'] = page
            # classification
            body = PUNCT_TAIL.sub('', m['word'])
            m['after_punct'] = (m['prev_is_punct'] or m['prev_is_quote']) and (m['lgap'] <= 1.5 or m.get('chained'))
            if m['after_punct'] and not m['thin']:
                prev_ref = m
            # collect per-component boxes for template classifier
            if ocr:
                txt, conf, canvas = ocr_digits(img, m)
                m['ocr'] = txt; m['ocr_conf'] = conf
                canvas.save(os.path.join(SCRATCH, f'mark-{page:03d}-{li:02d}-{m["x0"]}.png'))
            dets.append(m)
    with open(os.path.join(SCRATCH, f'det-{scale_note}-{page:03d}.json'), 'w') as f:
        json.dump(dets, f, indent=1)
    if mask_reocr:
        masked = img.copy(); d = ImageDraw.Draw(masked)
        for m in dets:
            if m['after_punct'] and not m['thin']:
                d.rectangle((m['x0'] - 1, m['y0'] - 1, m['x1'] + 1, m['y1'] + 1), fill=255)
        mp = os.path.join(SCRATCH, f'masked-{scale_note}-{page:03d}.png'); masked.save(mp)
        mb = os.path.join(SCRATCH, f'masked-{scale_note}-psm{psm}-{page:03d}')
        if os.path.exists(mb + '.hocr'): os.remove(mb + '.hocr')
        mh = run_tesseract(mp, mb, psm=psm)
        mlines = parse_hocr(mh)
        # for each ref mark, find the word in the masked OCR that ends nearest the mark's left edge
        for m in dets:
            if not (m['after_punct'] and not m['thin']):
                continue
            best = None
            for ln in mlines:
                for w in ln['words']:
                    bx0, by0, bx1, by1 = w['bbox']
                    if by0 <= m['y1'] and by1 >= m['y0'] and bx1 <= m['x1'] + 2:
                        if best is None or bx1 > best[0]:
                            best = (bx1, w)
            m['masked_word'] = best[1]['text'] if best else ''
            m['masked_conf'] = best[1]['conf'] if best else -1
            if verbose:
                print(f"  masked: L{m['line']:02d} raw={m['word']!r} -> masked={m['masked_word']!r} ({m['masked_conf']})")
        with open(os.path.join(SCRATCH, f'det-{scale_note}-{page:03d}.json'), 'w') as f:
            json.dump(dets, f, indent=1)
    if verbose:
        print(f'--- page {page} ({scale_note}, {len(lines)} lines)')
        for m in dets:
            flag = 'REF' if (m['after_punct'] and not m['thin']) else ('quote?' if m['thin'] else 'other')
            print(f"  L{m['line']:02d} x={m['x0']:5d} n={m['n']} hx={m['hx']} bot={m['bot_up']} top={m['top_up']} "
                  f"lgap={m['lgap']} rgap={m['rgap']} w={m['widths']} fill={m['fills']} prev='{m['word']}' "
                  f"ocr='{m.get('ocr','')}'({m.get('ocr_conf','')}) {flag}")
    return dets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pages', type=int, nargs='+', default=[20, 21, 22, 60, 130, 210])
    ap.add_argument('--dpi', type=int, default=300)
    ap.add_argument('--native', action='store_true')
    ap.add_argument('--psm', type=int, default=1)
    ap.add_argument('--no-ocr', action='store_true')
    ap.add_argument('--mask-reocr', action='store_true', help='mask detected marks and re-OCR the page; report the body word')
    ap.add_argument('--gt', default=os.path.join(HERE, 'superscript_gt.json'))
    args = ap.parse_args()
    os.makedirs(SCRATCH, exist_ok=True)
    gt = json.load(open(args.gt)) if os.path.exists(args.gt) else None
    tp = fp = fn = 0; num_ok = 0; num_tot = 0
    for p in args.pages:
        dets = process_page(p, dpi=args.dpi, native=args.native, psm=args.psm, ocr=not args.no_ocr, mask_reocr=args.mask_reocr)
        if gt is None:
            continue
        refs = [m for m in dets if m['after_punct'] and not m['thin']]
        g = gt.get(str(p), [])
        used = set()
        for gi in g:
            # match by preceding word (prefix) and line proximity
            hit = None
            for k, m in enumerate(refs):
                if k in used:
                    continue
                if m['word'].lower().startswith(gi['word'].lower()[:5]) and abs(m["line"] - gi["line"]) <= 3:
                    hit = k; break
            if hit is None:
                fn += 1; print(f'  MISS p{p} L{gi["line"]} {gi["word"]} -> {gi["n"]}')
            else:
                used.add(hit); tp += 1
                if gi.get('legible', True):
                    num_tot += 1
                    if refs[hit].get('ocr') == str(gi['n']):
                        num_ok += 1
                    else:
                        print(f'  DIGIT p{p} L{gi["line"]} {gi["word"]} gt={gi["n"]} ocr={refs[hit].get("ocr")!r} conf={refs[hit].get("ocr_conf")}')
        for k, m in enumerate(refs):
            if k not in used:
                fp += 1; print(f'  FALSE p{p} L{m["line"]} {m["word"]!r} ocr={m.get("ocr")!r}')
    if gt is not None:
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        print(f'\nREF detection: TP={tp} FP={fp} FN={fn} precision={prec:.3f} recall={rec:.3f}')
        print(f'digit OCR on detected legible refs: {num_ok}/{num_tot}')


if __name__ == '__main__':
    main()
