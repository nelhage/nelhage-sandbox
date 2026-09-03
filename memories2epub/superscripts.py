#!/usr/bin/env python3
"""Stage 3: superscript endnote references (see PLAN.md, CONTRACT.md).

Per body page (PDF 17-280):
  1. geometric detection of superscript marks on out/pages/p-NNN.png using the
     hOCR line geometry in out/hocr/p-NNN.hocr + connected components (the rules
     of the survey prototype out/survey/superscript_proto.py, unchanged);
  2. reference marks are painted paper-colour -> out/pages2/p-NNN.png and the page
     is re-OCR'd (--psm 1) -> out/hocr2/p-NNN.hocr, so the body word comes out clean;
     each mark is attached to the hocr2 word immediately to its left on its line;
  3. the digits are read twice (tesseract --psm 7 on a 4x binarised crop, and a
     nearest-template classifier bootstrapped from the book's own high-confidence
     glyphs) and checked against the per-chapter numbering prior
     (value <= max_so_far + 1; every 1..N first-cited in increasing order);
     "ok" only when both readers agree and the prior holds, else "flag";
  4. a text-pattern check on the OCR text finds scan-dropped glyphs
     (terminal punctuation + one stray ' ’ ® ° ! * ¢ - with no detection there).

Outputs: out/marks/p-NNN.json (every body page), out/marks/summary.json,
out/marks/flags.json (+ out/marks/crops/*.png line strips), out/pages2, out/hocr2.

Usage: python3 superscripts.py [pages...] [--gt] [--workers N]
  --gt   compare the detections with out/survey/superscript_gt.json
Pages not given on the command line are taken from their existing out/marks json
(if any) so the chapter sequence check still sees the whole chapter.
"""
import argparse, bisect, io, json, os, re, subprocess, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import ndimage
from PIL import Image
from lxml import etree

CHAPTERS = [17, 50, 78, 109, 145, 176, 203, 229, 264, 281]     # PDF page of ch 1..9, end
NOTES = {1: 61, 2: 39, 3: 64, 4: 86, 5: 69, 6: 59, 7: 64, 8: 48, 9: 28}
BODY = range(17, 281)

PAGES = 'out/pages'; HOCR = 'out/hocr'; PAGES2 = 'out/pages2'; HOCR2 = 'out/hocr2'
MARKS = 'out/marks'; CROPS = os.path.join(MARKS, 'crops'); CACHE = os.path.join(MARKS, 'cache')
GT = 'out/survey/superscript_gt.json'

# scan-dropped glyph pattern: terminal punctuation + one stray glyph at word end
DROPPED = re.compile(r'[.,;:?!][\'’®°!*¢-]$')   # '-' : the p55 speck comes out as a hyphen


def chapter_of(p):
    return bisect.bisect_right(CHAPTERS, p) if 17 <= p < 281 else 0


def pnum(p):
    return f'p-{p:03d}'

# --------------------------------------------------------------------------- hOCR


def parse_hocr(path):
    """Text lines (ocr_line/header/caption/textfloat) in reading order with their
    geometry, words and paragraph id."""
    t = etree.parse(path, etree.HTMLParser())
    lines = []
    for el in t.iter():
        if el.get('class') not in ('ocr_line', 'ocr_header', 'ocr_caption', 'ocr_textfloat'):
            continue
        title = el.get('title') or ''
        m = re.search(r'bbox (\d+) (\d+) (\d+) (\d+)', title)
        bb = tuple(map(int, m.groups()))

        def g(k, d=0.0):
            mm = re.search(k + r' ([-\d.]+)', title)
            return float(mm.group(1)) if mm else d
        a, b = 0.0, 0.0
        mb = re.search(r'baseline ([-\d.]+) ([-\d.]+)', title)
        if mb:
            a, b = float(mb.group(1)), float(mb.group(2))
        par = None
        for anc in el.iterancestors():
            if anc.get('class') == 'ocr_par':
                par = anc.get('id'); break
        words = []
        for w in el.iter():
            if w.get('class') != 'ocrx_word':
                continue
            wt = w.get('title') or ''
            mw = re.search(r'bbox (\d+) (\d+) (\d+) (\d+)', wt)
            mc = re.search(r'x_wconf (\d+)', wt)
            words.append(dict(bbox=tuple(map(int, mw.groups())), text=''.join(w.itertext()).strip(),
                              conf=int(mc.group(1)) if mc else 0))
        lines.append(dict(bbox=bb, a=a, b=b, x_size=g('x_size'), desc=g('x_descenders'),
                          asc=g('x_ascenders'), words=words, par=par))
    lines.sort(key=lambda l: (l['bbox'][1], l['bbox'][0]))
    for i, l in enumerate(lines):
        l['idx'] = i
        l['text'] = ' '.join(w['text'] for w in l['words'])
    return lines

# --------------------------------------------------------------------------- connected components


def components(arr, box, thresh=128):
    """8-connected ink components inside box=(x0,y0,x1,y1); absolute coords, x1/y1 exclusive."""
    x0, y0, x1, y1 = box
    H, W = arr.shape
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(W, x1); y1 = min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return []
    sub = arr[y0:y1, x0:x1] < thresh
    lab, n = ndimage.label(sub, structure=np.ones((3, 3), int))
    if n == 0:
        return []
    areas = np.bincount(lab.ravel())
    out = []
    for i, (sy, sx) in enumerate(ndimage.find_objects(lab), 1):
        out.append(dict(x0=sx.start + x0, y0=sy.start + y0, x1=sx.stop + x0, y1=sy.stop + y0,
                        area=int(areas[i])))
    out.sort(key=lambda c: c['x0'])
    return out

# --------------------------------------------------------------------------- detection (prototype rules)


def detect_line(arr, line):
    x0, y0, x1, y1 = line['bbox']
    xs = line['x_size']; desc = line['desc']; asc = line['asc']
    xh = xs - desc - asc
    if xh <= 4 or xs <= 0:
        return []

    def baseline_at(x):
        return y1 + line['b'] + line['a'] * (x - x0)
    box = (int(x0 - 4), int(y0 - 0.5 * xs), int(x1 + 6), int(y1 + 2))
    comps = components(arr, box)
    for c in comps:
        base = baseline_at((c['x0'] + c['x1']) / 2)
        c['h'] = c['y1'] - c['y0']; c['w'] = c['x1'] - c['x0']
        c['bot_up'] = (base - c['y1']) / xh      # bottom above baseline, in x-heights
        c['top_up'] = (base - c['y0']) / xh      # top above baseline
        c['hx'] = c['h'] / xh
        c['fill'] = c['area'] / max(1, c['h'] * c['w'])
    cands = [c for c in comps if c['area'] >= 6 and 0.25 <= c['bot_up'] <= 0.75 and c['top_up'] >= 1.2
             and 0.75 <= c['hx'] <= 1.25 and c['w'] >= 0.25 * xh]
    cands.sort(key=lambda c: c['x0'])
    seps = [c for c in comps if c['hx'] < 0.5 and 0.1 < c['bot_up'] < 0.9 and c['area'] >= 6 and c['w'] < 0.5 * xh]
    groups = []
    for c in cands:
        if groups and c['x0'] - groups[-1][-1]['x1'] <= 0.6 * xh:
            last = groups[-1][-1]
            if any(sp['x0'] >= last['x1'] - 1 and sp['x1'] <= c['x0'] + 1 for sp in seps):
                groups.append([c])          # raised comma between digits: chained refs "60,61"
            else:
                groups[-1].append(c)
        else:
            groups.append([c])
    marks = []
    for g in groups:
        gx0 = min(c['x0'] for c in g); gx1 = max(c['x1'] for c in g)
        gy0 = min(c['y0'] for c in g); gy1 = max(c['y1'] for c in g)
        left = [c for c in comps if c['x1'] <= gx0 + 2 and c['area'] >= 15 and c['y1'] > gy0 + 2 and c['bot_up'] > -0.7]
        right = [c for c in comps if c['x0'] >= gx1 - 2 and c['area'] >= 6]
        lgap = (gx0 - max(c['x1'] for c in left)) / xh if left else 9
        rgap = (min(c['x0'] for c in right) - gx1) / xh if right else 9
        lc = max(left, key=lambda c: c['x1']) if left else None
        # what precedes: period/comma/colon-dot (small, on the baseline) or a quote stroke (narrow, high)
        prev_is_punct = bool(lc and lc['hx'] < 0.5 and lc['top_up'] < 1.0 and lc['bot_up'] > -0.4 and lc['w'] < 0.6 * xh)
        prev_is_quote = bool(lc and lc['hx'] < 0.9 and lc['bot_up'] > 0.6 and lc['w'] < 0.45 * xh)
        if prev_is_quote:
            left2 = [c for c in left if c['x1'] <= lc['x0'] + 1]
            lc2 = max(left2, key=lambda c: c['x1']) if left2 else None
            if lc2 and lc2['bot_up'] > 0.6 and lc2['hx'] < 0.9 and lc2['w'] < 0.45 * xh:   # second quote stroke
                left3 = [c for c in left2 if c['x1'] <= lc2['x0'] + 1]
                lc2 = max(left3, key=lambda c: c['x1']) if left3 else None
            prev_is_punct = prev_is_punct or bool(lc2 and lc2['hx'] < 0.5 and lc2['top_up'] < 0.6 and lc2['w'] < 0.6 * xh)
        thin = all(c['w'] < 0.35 * xh and c['fill'] > 0.5 and c['hx'] < 0.6 for c in g)
        marks.append(dict(x0=int(gx0), y0=int(gy0), x1=int(gx1), y1=int(gy1), n=len(g),
                          lgap=round(lgap, 2), rgap=round(rgap, 2), prev_is_punct=prev_is_punct,
                          prev_is_quote=prev_is_quote, hx=round(max(c['hx'] for c in g), 2),
                          bot_up=round(min(c['bot_up'] for c in g), 2), top_up=round(max(c['top_up'] for c in g), 2),
                          xh=round(xh, 1), thin=thin,
                          comps=[[int(c['x0']), int(c['y0']), int(c['x1']), int(c['y1'])] for c in g]))
    return marks


def word_before(line, x):
    best = None
    for w in line['words']:
        wx0, _, wx1, _ = w['bbox']
        if wx0 <= x + 2 and (best is None or wx1 > best[0]):
            best = (wx1, w)
    return best[1] if best else None


def detect_page(arr, lines):
    """All superscript-shaped marks on text lines of the page (refs, others, thin)."""
    body = sorted(l['x_size'] for l in lines if len(l['words']) >= 4
                  and sum(w['conf'] for w in l['words']) / len(l['words']) >= 60)
    med_xs = body[len(body) // 2] if body else (lines[0]['x_size'] if lines else 0)
    par_conf = {}
    for l in lines:
        ws = [w for w in l['words'] if re.search(r'[A-Za-z]{2,}', w['text'])]
        par_conf.setdefault(l['par'], []).extend(w['conf'] for w in ws)
    par_conf = {k: (sum(v) / len(v) if v else 0) for k, v in par_conf.items()}
    dets = []
    prev = None          # previous accepted text line
    for line in lines:
        # text lines only (skips figure lettering): an alphabetic word, x_size within +-25 % of
        # the body size, and mean word confidence >= 50 in the line or its paragraph. A one- or
        # two-word paragraph-final line like "division.15" has low confidence precisely because
        # of the mark (and tesseract sometimes gives it a paragraph of its own), so it is also
        # accepted when it hangs directly under an accepted line at the same left margin.
        words = [w for w in line['words'] if re.search(r'[A-Za-z]{2,}|\d{2,}', w['text'])]
        if len(words) < 1 or not (0.75 * med_xs <= line['x_size'] <= 1.25 * med_xs):
            continue
        lconf = sum(w['conf'] for w in words) / len(words)
        if lconf < 50 and par_conf.get(line['par'], 0) < 50:
            hangs = (prev is not None and len(line['words']) <= 2
                     and 0 <= line['bbox'][1] - prev['bbox'][3] <= 1.2 * line['x_size']
                     and abs(line['bbox'][0] - prev['bbox'][0]) <= 40)
            if not hangs:
                continue
        prev = line
        prev_ref = None
        for m in detect_line(arr, line):
            if prev_ref is not None and (m['x0'] - prev_ref['x1']) <= 1.3 * m['xh'] and not m['thin']:
                m['prev_is_punct'] = True; m['chained'] = True
                # the raised separator comma of "25,26" is painted out with the marks
                m['sep_box'] = [prev_ref['x1'] + 1, min(prev_ref['y0'], m['y0']) - 1, m['x0'] - 1,
                                max(prev_ref['y1'], m['y1']) + int(0.5 * m['xh'])]
            w = word_before(line, m['x0'])
            m['line'] = line['idx']; m['line_bbox'] = list(line['bbox']); m['line_text'] = line['text']
            m['raw_word'] = w['text'] if w else ''
            m['after_punct'] = (m['prev_is_punct'] or m['prev_is_quote']) and (m['lgap'] <= 1.5 or m.get('chained', False))
            m['is_ref'] = bool(m['after_punct'] and not m['thin'])
            if m['is_ref']:
                prev_ref = m
            dets.append(m)
    return dets

# --------------------------------------------------------------------------- digit readers


def ocr_digits(arr, m, pad=3, up=4):
    """tesseract --psm 7, digits only, on a 4x binarised crop with white margins.
    The upscaled crop is re-binarised with a slightly fat threshold (170): the thin vertical
    stroke of the superscript 5 otherwise breaks and tesseract reads 5 as 3 (book-wide the
    fat threshold removes about a quarter of the psm-7 misreads; the rest are inherent)."""
    x0, y0, x1, y1 = m['x0'] - pad, m['y0'] - pad, m['x1'] + pad, m['y1'] + pad
    crop = Image.fromarray(arr).crop((x0, y0, x1, y1)).point(lambda v: 255 if v >= 128 else 0)
    crop = crop.resize(((x1 - x0) * up, (y1 - y0) * up), Image.LANCZOS).point(lambda v: 255 if v >= 170 else 0)
    W, H = crop.size
    canvas = Image.new('L', (W + 2 * H, 3 * H), 255)
    canvas.paste(crop, (H, H))
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, 'c.png'); canvas.save(p)
        r = subprocess.run(['tesseract', p, '-', '-l', 'eng', '--psm', '7', '-c',
                            'tessedit_char_whitelist=0123456789', 'tsv'], capture_output=True, text=True)
    txt, conf = '', -1.0
    for row in r.stdout.splitlines()[1:]:
        f = row.split('\t')
        if len(f) >= 12 and f[11].strip():
            txt += f[11].strip(); conf = max(conf, float(f[10]))
    return txt, conf


def norm_crop(arr, box, size=(16, 24)):
    x0, y0, x1, y1 = box
    c = Image.fromarray(arr).crop((x0, y0, x1, y1)).resize(size, Image.LANCZOS)
    return 1.0 - np.asarray(c, dtype=np.float32).ravel() / 255.0


def three_or_five(arr, box):
    """Structural 3-vs-5 discriminator. In the upper-middle band of the glyph (rows 25-45 %) a 5
    has only its left vertical stroke while a 3 has only its right upper bowl, so the band's ink
    centroid is < 0.5 of the width for a 5 and > 0.5 for a 3 (book-wide: 5s <= 0.48, 3s >= 0.51).
    This is the one confusion both readers make (5 -> 3) and it survives broken strokes."""
    x0, y0, x1, y1 = box
    g = (arr[y0:y1, x0:x1] < 128).astype(np.float32)
    h, w = g.shape
    band = g[int(0.25 * h):int(0.45 * h)]
    if band.sum() == 0 or w == 0:
        return None
    cx = (band * np.arange(w)).sum() / band.sum() / w
    return '5' if cx < 0.5 else '3'


class Templates:
    """Nearest-template digit classifier bootstrapped from the book's own confident marks.
    Labels 3/5 are decided by the structural discriminator, both when building the template set
    (psm 7 mislabels ~10 % of the 5s as 3, which would poison the templates) and when reading."""

    def __init__(self):
        self.vecs = []; self.labels = []; self.M = None

    def add(self, arr, mark):
        for d, c in zip(mark['psm7'], mark['comps']):
            if d in '35':
                d = three_or_five(arr, c) or d
            self.vecs.append(norm_crop(arr, c)); self.labels.append(d)

    def finish(self):
        self.M = np.stack(self.vecs) if self.vecs else None

    def classify(self, arr, mark):
        if self.M is None:
            return ''
        out = ''
        for c in mark['comps']:
            v = norm_crop(arr, c)
            d = ((self.M - v) ** 2).sum(axis=1)
            lab = self.labels[int(d.argmin())]
            if lab in '35':
                lab = three_or_five(arr, c) or lab
            out += lab
        return out

    def inventory(self):
        return {d: self.labels.count(d) for d in '0123456789'}

# --------------------------------------------------------------------------- per-page stage A


def run_tesseract(png, base):
    subprocess.run(['tesseract', png, base, '-l', 'eng', '--psm', '1', 'hocr', 'txt'],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def remove_stale(p):
    for f in (f'{PAGES2}/{pnum(p)}.png', f'{HOCR2}/{pnum(p)}.hocr', f'{HOCR2}/{pnum(p)}.txt'):
        if os.path.exists(f):
            os.remove(f)


def line_of(lines2, m):
    """The (re-OCR'd) line the mark sits on: largest vertical overlap with the original line box,
    breaking ties by horizontal containment."""
    lx0, ly0, lx1, ly1 = m['line_bbox']
    best = None
    for ln in lines2:
        bx0, by0, bx1, by1 = ln['bbox']
        ov = min(ly1, by1) - max(ly0, by0)
        if ov <= 0 or bx1 < m['x0'] - 400 or bx0 > m['x1'] + 400:
            continue
        key = (ov, -abs((bx0 + bx1) / 2 - (lx0 + lx1) / 2))
        if best is None or key > best[0]:
            best = (key, ln)
    return best[1] if best else None


def attach_words(refs, lines2):
    """Word in the (re-OCR'd) hOCR immediately left of each mark on the same line.
    Chained marks ("25,26") share the word of the first mark; a bare punctuation token left
    over from the reference notation is skipped in favour of the word before it."""
    prev = None
    for m in sorted(refs, key=lambda m: (m['line'], m['x0'])):
        if m.get('chained') and prev is not None and prev['line'] == m['line'] and prev.get('word_bbox'):
            for k in ('word_bbox', 'word_text', 'word_conf', 'line2_text', 'line2_bbox'):
                m[k] = prev[k]
            prev = m
            continue
        prev = m
        ln = line_of(lines2, m)
        best = None
        if ln is not None:
            cands = sorted((w for w in ln['words'] if w['bbox'][2] <= m['x1'] + 2), key=lambda w: w['bbox'][2])
            while cands and not re.search(r'[A-Za-z0-9]', cands[-1]['text']) and len(cands) >= 2:
                cands.pop()
            if cands:
                best = (cands[-1]['bbox'][2], cands[-1])
        if best:
            m['word_bbox'] = list(best[1]['bbox']); m['word_text'] = best[1]['text']
            m['word_conf'] = best[1]['conf']; m['line2_text'] = ln['text']; m['line2_bbox'] = list(ln['bbox'])
        else:
            m['word_bbox'] = None; m['word_text'] = ''; m['word_conf'] = -1
            m['line2_text'] = m['line_text']; m['line2_bbox'] = m['line_bbox']


def attach_other(m, lines2):
    """For a non-reference superscript (M², 2¹⁰): the hOCR word that contains the mark."""
    ln = line_of(lines2, m)
    if ln is None:
        return False
    best = None
    for w in ln['words']:
        bx0, _, bx1, _ = w['bbox']
        if bx0 <= m['x0'] + 2 and bx1 >= m['x1'] - 2:
            best = w
    if best is None:
        return False
    m['word_bbox'] = list(best['bbox']); m['word_text'] = best['text']; m['word_conf'] = best['conf']
    m['line2_text'] = ln['text']; m['line2_bbox'] = list(ln['bbox'])
    return True


def text_pattern_flags(lines2, refs):
    """Words ending in terminal punctuation + one stray glyph, with no detected mark there."""
    out = []
    for ln in lines2:
        for w in ln['words']:
            t = w['text']
            if len(t) <= 3 or not DROPPED.search(t) or re.search(r'\d,\d{3}|\$\d', t):
                continue
            x0, y0, x1, y1 = w['bbox']
            if any(m['y0'] <= y1 and m['y1'] >= y0 and x0 - 5 <= m['x0'] <= x1 + 5 for m in refs):
                continue
            out.append(dict(kind='ref', text_only=True, x0=x1 - 20, y0=y0, x1=x1 + 4, y1=y1,
                            line=ln['idx'], line_bbox=list(ln['bbox']), line_text=ln['text'],
                            line2_text=ln['text'], line2_bbox=list(ln['bbox']),
                            word_bbox=list(w['bbox']), word_text=t, raw_word=t, word_conf=w['conf'],
                            n=0, comps=[], hx=0.0))
    return out


def stage_a(p, cache, lock):
    """Detect, read (psm 7), paint, re-OCR, attach. Returns the page record."""
    arr = np.asarray(Image.open(f'{PAGES}/{pnum(p)}.png').convert('L'))
    lines = parse_hocr(f'{HOCR}/{pnum(p)}.hocr')
    dets = detect_page(arr, lines)
    refs = [m for m in dets if m['is_ref']]
    others = [m for m in dets if not m['is_ref'] and not m['thin']]
    for m in refs + others:
        key = f"{p}:{m['x0']},{m['y0']},{m['x1']},{m['y1']}"
        with lock:
            hit = cache.get(key)
        if hit is None:
            hit = list(ocr_digits(arr, m))
            with lock:
                cache[key] = hit
        m['psm7'], m['psm7_conf'] = hit[0], float(hit[1])
    if refs:
        paper = int(np.median(arr[arr >= 128])) if (arr >= 128).any() else 255
        painted = arr.copy()
        for m in refs:
            painted[max(0, m['y0'] - 1):m['y1'] + 2, max(0, m['x0'] - 1):m['x1'] + 2] = paper
            if m.get('sep_box'):
                sx0, sy0, sx1, sy1 = m['sep_box']
                if sx1 > sx0:
                    painted[max(0, sy0):sy1, sx0:sx1] = paper
        buf = io.BytesIO(); Image.fromarray(painted).save(buf, format='PNG'); data = buf.getvalue()
        png = f'{PAGES2}/{pnum(p)}.png'; base = f'{HOCR2}/{pnum(p)}'
        same = os.path.exists(png) and open(png, 'rb').read() == data and os.path.exists(base + '.hocr')
        if not same:
            with open(png, 'wb') as f:
                f.write(data)
            run_tesseract(png, base)
        lines2 = parse_hocr(base + '.hocr')
    else:
        remove_stale(p)
        lines2 = lines
    attach_words(refs, lines2)
    others = [m for m in others if attach_other(m, lines2)]
    textflags = text_pattern_flags(lines2, refs)
    return dict(page=p, refs=refs, others=others, textflags=textflags, arr=arr)

# --------------------------------------------------------------------------- existing results (pages not in this run)


def load_existing(p):
    f = f'{MARKS}/{pnum(p)}.json'
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    refs, others, textflags = [], [], []
    for m in d['marks']:
        r = dict(x0=m['bbox'][0], y0=m['bbox'][1], x1=m['bbox'][2], y1=m['bbox'][3],
                 line_bbox=m['line_bbox'], line2_bbox=m['line_bbox'], word_bbox=m['word_bbox'],
                 word_text=m['word_text'], line=m.get('line', 0), line_text=m.get('line_text', ''),
                 line2_text=m.get('line_text', ''), raw_word=m.get('raw_word', ''),
                 comps=m.get('comps', []), n=len(m.get('comps', [])), hx=m.get('hx', 1.0),
                 lgap=m.get('lgap', 0.0), xh=m.get('xh', 21.0), chained=m.get('chained', False),
                 psm7=m['readings'].get('psm7', ''),
                 psm7_conf=float(m.get('conf', -1)), word_conf=m.get('word_conf', -1))
        if m.get('text_only'):
            r['text_only'] = True; r['kind'] = 'ref'; textflags.append(r)
        elif m['kind'] == 'ref':
            r['is_ref'] = True; refs.append(r)
        else:
            others.append(r)
    return dict(page=p, refs=refs, others=others, textflags=textflags, arr=None, loaded=True)


def page_arr(rec):
    if rec['arr'] is None:
        rec['arr'] = np.asarray(Image.open(f'{PAGES}/{pnum(rec["page"])}.png').convert('L'))
    return rec['arr']

# --------------------------------------------------------------------------- sequence prior


CONFUSABLE = {'3': '5', '0': '6', '1': '7'}     # reading -> what the glyph may really be


def one_digit_off(read, truth):
    """Could `read` be a misreading of `truth` by one known glyph confusion?"""
    if len(read) != len(truth):
        return False
    diff = [(a, b) for a, b in zip(read, truth) if a != b]
    return len(diff) == 1 and CONFUSABLE.get(diff[0][0]) == diff[0][1]


def sequence_chapter(ch, recs):
    """Assign values/status to every ref mark of the chapter in reading order."""
    N = NOTES[ch]
    items = []
    for rec in recs:
        for m in rec['refs'] + rec['textflags']:
            items.append((rec['page'], m['line2_bbox'][1], m['x0'], m))
    items.sort(key=lambda t: t[:3])
    mx = 0; first = {}; seen = set(); recites = []

    def suspects(k):
        # earlier re-cites whose reading is one digit away from the never-first-cited k
        for r in recites:
            if r['status'] == 'ok' and one_digit_off(str(r['value']), str(k)):
                r['status'] = 'flag'
                r['reason'] = f'suspect: read {r["value"]} but note {k} is never first-cited (one digit off)'
                r['suspect_for'] = k

    last = None
    for page, _, _, m in items:
        m['max_before'] = mx; m['allowed'] = list(range(1, min(mx + 1, N) + 1))
        reasons = []
        if m.get('text_only'):
            m['value'] = None; m['status'] = 'flag'
            m['reason'] = 'text pattern: terminal punctuation + stray glyph, no mark detected (scan-dropped superscript?)'
            continue
        last = m
        t7, tm = m['psm7'], m['template']
        v = None
        if t7 == tm and t7.isdigit():
            v = int(t7)
        else:
            reasons.append(f'readers disagree (psm7={t7!r} template={tm!r})')
            cands = [c for c in (tm, t7) if c.isdigit() and len(c) == m['n'] and 1 <= int(c) <= mx + 1]
            if not cands:
                cands = [c for c in (tm, t7) if c.isdigit() and len(c) == m['n'] and 1 <= int(c) <= N]
            if not cands:
                cands = [c for c in (tm, t7) if c.isdigit()]
            v = int(cands[0]) if cands else None
        if v is None:
            reasons.append('unreadable')
        elif m['n'] != len(str(v)):
            reasons.append(f"component count {m['n']} != digit count")
        if m['n'] > 3 or not (0.8 <= m['hx'] <= 1.15):
            reasons.append('odd geometry')
        if not m['word_text']:
            reasons.append('no word to the left')
        role = ''
        if v is not None:
            if v == mx + 1:
                first[v] = m; mx = v; role = 'first'
            elif 1 <= v <= mx:
                role = 'recite'; recites.append(m)
            elif mx + 1 < v <= min(mx + 3, N):
                missing = list(range(mx + 1, v))
                reasons.append(f'gap: {missing} never first-cited before this {v} (max was {mx})')
                m['gap_missing'] = missing
                for k in missing:
                    suspects(k)
                mx = v; role = 'gap'
            else:
                reasons.append(f'violates prior: {v} > max_so_far {mx} + 1 (N={N})')
                role = 'viol'
            seen.add(v)
        m['value'] = v; m['role'] = role
        m['status'] = 'flag' if reasons else 'ok'
        m['reason'] = '; '.join(reasons)
    # chapter end: every 1..N must have been first-cited
    if mx < N and last is not None:
        for k in range(mx + 1, N + 1):
            suspects(k)
        r = f'chapter ends with max {mx} < {N} notes: {list(range(mx + 1, N + 1))} never cited'
        last['status'] = 'flag'; last['reason'] = (last['reason'] + '; ' if last['reason'] else '') + r
    gaps = [k for k in range(1, N + 1) if k not in first]
    return dict(chapter=ch, max=mx, notes_expected=N, first_cites_in_order=(not gaps and mx == N),
                gaps=gaps, never_cited=[k for k in gaps if k not in seen],
                n_marks=sum(1 for i in items if not i[3].get('text_only')), items=[i[3] for i in items])

# --------------------------------------------------------------------------- output


def mark_json(m, kind):
    d = dict(bbox=[m['x0'], m['y0'], m['x1'], m['y1']], kind=kind, value=m.get('value'),
             readings=dict(psm7=m.get('psm7', ''), template=m.get('template', '')),
             conf=round(m.get('psm7_conf', -1)), status=m.get('status', 'ok'), reason=m.get('reason', ''),
             line_bbox=m.get('line2_bbox') or m['line_bbox'], word_bbox=m.get('word_bbox'),
             word_text=m.get('word_text', ''),
             line=m.get('line'), line_text=m.get('line2_text') or m.get('line_text', ''),
             raw_word=m.get('raw_word', ''), comps=m.get('comps', []), hx=m.get('hx'),
             lgap=m.get('lgap'), xh=m.get('xh'), chained=bool(m.get('chained')), word_conf=m.get('word_conf', -1))
    if m.get('text_only'):
        d['text_only'] = True
    if m.get('allowed') is not None:
        d['allowed'] = m['allowed']
    return d


def write_crop(rec, m):
    os.makedirs(CROPS, exist_ok=True)
    arr = page_arr(rec)
    lx0, ly0, lx1, ly1 = m.get('line2_bbox') or m['line_bbox']
    x0 = max(0, min(lx0, m['x0']) - 10); y0 = max(0, min(ly0, m['y0']) - 10)
    x1 = min(arr.shape[1], max(lx1, m['x1']) + 10); y1 = min(arr.shape[0], max(ly1, m['y1']) + 10)
    path = f"{CROPS}/{pnum(rec['page'])}-L{m.get('line', 0):02d}-{m['x0']}.png"
    Image.fromarray(arr[y0:y1, x0:x1]).save(path)
    return path, [x0, y0, x1, y1]


def draft_text(m):
    line = m.get('line2_text') or m.get('line_text', '')
    w = m.get('word_text', '')
    if w and w in line:
        i = line.index(w) + len(w)
        line = line[:i] + '^' + (m.get('psm7') or '?') + line[i:]
    return line


def flag_entry(rec, m, kind):
    crop, _ = write_crop(rec, m)
    ctx = dict(chapter=chapter_of(rec['page']), kind=kind, line_bbox=m.get('line2_bbox') or m['line_bbox'],
               word_text=m.get('word_text', ''), word_bbox=m.get('word_bbox'),
               psm7=m.get('psm7', ''), template=m.get('template', ''), conf=round(m.get('psm7_conf', -1)),
               max_so_far=m.get('max_before', 0), value=m.get('value'),
               line_text=m.get('line2_text') or m.get('line_text', ''))
    if m.get('gap_missing'):
        ctx['gap_missing'] = m['gap_missing']
    if m.get('suspect_for'):
        ctx['suspect_for'] = m['suspect_for']
    return dict(page=rec['page'], bbox=[m['x0'], m['y0'], m['x1'], m['y1']], crop=crop, draft=draft_text(m),
                reason=m['reason'], context=ctx, allowed=m.get('allowed', []))

# --------------------------------------------------------------------------- ground truth check


def check_gt(recs, gt):
    tp = fp = fn = 0; val_ok = val_tot = 0
    for rec in recs:
        g = gt.get(str(rec['page']))
        if g is None:
            continue
        refs = sorted(rec['refs'], key=lambda m: (m['y0'], m['x0']))
        used = set()
        for gi in g:
            hit = None
            for k, m in enumerate(refs):
                if k in used:
                    continue
                if abs(m['line'] - gi['line']) <= 3 and any(
                        w.lower().startswith(gi['word'].lower()[:5]) for w in (m['raw_word'], m['word_text'])):
                    hit = k; break
            if hit is None:
                fn += 1; print(f"  MISS p{rec['page']} L{gi['line']} {gi['word']} -> {gi['n']}"
                               f"{'' if gi.get('legible', True) else ' (glyph missing in scan)'}")
                continue
            used.add(hit); tp += 1
            if gi.get('legible', True):
                val_tot += 1
                m = refs[hit]
                if m.get('value') == gi['n']:
                    val_ok += 1
                else:
                    print(f"  VALUE p{rec['page']} L{gi['line']} {gi['word']} gt={gi['n']} got={m.get('value')} "
                          f"psm7={m.get('psm7')!r} tmpl={m.get('template')!r}")
        for k, m in enumerate(refs):
            if k not in used:
                fp += 1; print(f"  FALSE p{rec['page']} L{m['line']} {m['raw_word']!r} -> {m.get('value')}")
        tf = [t['word_text'] for t in rec['textflags']]
        oth = [(o['word_text'], o['value']) for o in rec['others']]
        print(f"  p{rec['page']}: {len(refs)} refs {[(m['word_text'], m['value'], m['status']) for m in refs]}; "
              f"other {oth}; text-pattern flags {tf}")
    print(f'GT: TP={tp} FP={fp} FN={fn}; values right on detected legible refs: {val_ok}/{val_tot}')

# --------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pages', type=int, nargs='*')
    ap.add_argument('--gt', action='store_true', help='compare with out/survey/superscript_gt.json')
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()
    run_pages = [p for p in (args.pages or list(BODY)) if p in BODY]
    for d in (PAGES2, HOCR2, MARKS, CROPS, CACHE):
        os.makedirs(d, exist_ok=True)
    cache_file = f'{CACHE}/psm7.json'
    cache = json.load(open(cache_file)) if os.path.exists(cache_file) else {}
    lock = threading.Lock()

    # ---- stage A: detect / read / paint / re-OCR / attach, in parallel
    t0 = time.time()
    recs = {}
    with ThreadPoolExecutor(args.workers) as ex:
        for i, rec in enumerate(ex.map(lambda p: stage_a(p, cache, lock), run_pages), 1):
            recs[rec['page']] = rec
            if i % 25 == 0:
                print(f'  {i}/{len(run_pages)} pages', flush=True)
    with open(cache_file, 'w') as f:
        json.dump(cache, f)
    for p in BODY:
        if p not in recs:
            r = load_existing(p)
            if r:
                recs[p] = r
    print(f'stage A: {len(run_pages)} pages in {time.time() - t0:.0f}s; '
          f'{sum(len(r["refs"]) for r in recs.values())} ref marks on {sum(1 for r in recs.values() if r["refs"])} pages')

    # ---- stage B: template classifier from confident marks over everything we know
    tpl = Templates()
    for rec in recs.values():
        for m in rec['refs']:
            if m['psm7_conf'] >= 90 and m['psm7'].isdigit() and len(m['psm7']) == m['n'] and m['comps']:
                tpl.add(page_arr(rec), m)
    tpl.finish()
    print(f'templates: {len(tpl.labels)} glyphs {tpl.inventory()}')
    for rec in recs.values():
        for m in rec['refs'] + rec['others']:
            m['template'] = tpl.classify(page_arr(rec), m) if m['comps'] else ''

    # "other" superscripts (M², 2¹⁰): keep only when both readers agree and the mark sits inside
    # a word right after a full-height glyph (word_text is tesseract's rendering of the whole
    # word, e.g. "M2" or "21°=1024"; the assembler places <sup>value</sup> by word_bbox/bbox)
    for rec in recs.values():
        keep = []
        for m in rec['others']:
            t = m['psm7']
            # value 0 is a degree sign (500°F); the confidence bar keeps out figure lettering
            if t.isdigit() and int(t) > 0 and t == m['template'] and len(t) == m['n'] and m['psm7_conf'] >= 80 \
                    and m['word_bbox'] and m['word_bbox'][0] < m['x0'] - 0.5 * m.get('xh', 21) \
                    and m['lgap'] <= 0.5:
                m['value'] = int(t); m['status'] = 'ok'; m['reason'] = ''
                keep.append(m)
        rec['others'] = keep

    # ---- stage C: numbering prior per chapter, outputs
    summary = {}; flags = []
    for ch in range(1, 10):
        chrecs = [recs[p] for p in range(CHAPTERS[ch - 1], CHAPTERS[ch]) if p in recs]
        s = sequence_chapter(ch, chrecs)
        items = s.pop('items')
        s['n_flags'] = sum(1 for m in items if m['status'] == 'flag')
        s['n_recites'] = sum(1 for m in items if m.get('role') == 'recite')
        summary[str(ch)] = s
        for rec in chrecs:
            marks = []
            for m in rec['refs']:
                marks.append(mark_json(m, 'ref'))
                if m['status'] == 'flag':
                    flags.append(flag_entry(rec, m, 'ref'))
            for m in rec['others']:
                marks.append(mark_json(m, 'other'))
            for m in rec['textflags']:
                marks.append(mark_json(m, 'ref'))
                flags.append(flag_entry(rec, m, 'ref'))
            marks.sort(key=lambda d: (d['line_bbox'][1], d['bbox'][0]))
            with open(f'{MARKS}/{pnum(rec["page"])}.json', 'w') as f:
                json.dump(dict(page=rec['page'], chapter=ch, marks=marks), f, indent=1, ensure_ascii=False)
    flags.sort(key=lambda d: (d['page'], d['bbox'][1], d['bbox'][0]))
    with open(f'{MARKS}/flags.json', 'w') as f:
        json.dump(flags, f, indent=1, ensure_ascii=False)
    tot = dict(n_marks=sum(s['n_marks'] for s in summary.values()), n_flags=len(flags),
               n_other=sum(len(r['others']) for r in recs.values()),
               pages_with_marks=sum(1 for r in recs.values() if r['refs']),
               pages_known=len(recs), templates=len(tpl.labels))
    with open(f'{MARKS}/summary.json', 'w') as f:
        json.dump(dict(chapters=summary, total=tot), f, indent=1)

    if args.gt and os.path.exists(GT):
        check_gt([recs[p] for p in run_pages], json.load(open(GT)))

    print(f'\n{"ch":>2} {"pages":>8} {"marks":>5} {"max":>4} {"expected":>8} {"recites":>7} {"gaps":>22} {"flags":>5}')
    for ch, s in summary.items():
        ch = int(ch)
        print(f'{ch:>2} {CHAPTERS[ch-1]:>3}-{CHAPTERS[ch]-1:<4} {s["n_marks"]:>5} {s["max"]:>4} {s["notes_expected"]:>8} '
              f'{s["n_recites"]:>7} {str(s["gaps"]):>22} {s["n_flags"]:>5}')
    print(f'total: {tot["n_marks"]} ref marks on {tot["pages_with_marks"]} pages, {tot["n_other"]} other superscripts, '
          f'{tot["n_flags"]} flags on {len(set(fl["page"] for fl in flags))} pages')
    for fl in flags:
        c = fl['context']
        print(f"  p{fl['page']} {c['word_text']!r:28s} psm7={c['psm7']!r:5s} tmpl={c['template']!r:5s} conf={c['conf']:>3} "
              f"v={c['value']} max={c['max_so_far']}  {fl['reason']}")
    print(f'total time {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
