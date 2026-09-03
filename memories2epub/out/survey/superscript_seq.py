#!/usr/bin/env python3
"""Whole-book analysis of superscript_proto detections: re-OCR digits (psm 7), apply the
per-chapter numbering prior, and count what would need escalation.

Reads out/thumb/ss/det-300dpi-NNN.json (from superscript_proto.process_page) and the
300-dpi renders; writes out/thumb/ss/seq.json and prints a report."""
import json, os, re, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import superscript_proto as sp

CHAPTERS = [17, 50, 78, 109, 145, 176, 203, 229, 264, 281]   # PDF pages; last = end of body
# number of endnotes per chapter, read off the References section (PDF 281-316)
NOTES = {1: 61, 2: 39, 3: 64, 4: 86, 5: 69, 6: 59, 7: 64, 8: 48, 9: 28}
JUNK = re.compile(r'[.,;:?!"\'”’)]+[^A-Za-z\s(]{1,3}$|[.?!"\'”’]\d{1,2}$')

CONF_ESC = float(os.environ.get('CONF_ESC', 70))


def load_page(p):
    f = f'{sp.SCRATCH}/det-300dpi-{p:03d}.json'
    if not os.path.exists(f):
        return None
    return json.load(open(f))


def main():
    reocr = '--no-reocr' not in sys.argv
    allrefs = []
    junk_words = []
    missing_pages = []
    for ch in range(1, 10):
        p0, p1 = CHAPTERS[ch - 1], CHAPTERS[ch]
        for p in range(p0, p1):
            dets = load_page(p)
            if dets is None:
                missing_pages.append(p); continue
            refs = [m for m in dets if m['after_punct'] and not m['thin']]
            refs.sort(key=lambda m: (m['line'], m['x0']))
            img = None
            for m in refs:
                if reocr and m.get('ocr7') is None:
                    if img is None:
                        img = sp.render_page(p, 300)[0]
                    txt, conf, _ = sp.ocr_digits(img, m)
                    m['ocr7'] = txt; m['ocr7_conf'] = conf
                m['chapter'] = ch
                allrefs.append(m)
            with open(f'{sp.SCRATCH}/det-300dpi-{p:03d}.json', 'w') as f:
                json.dump(dets, f, indent=1)
            # text-layer junk words not explained by a detection (possible missed refs)
            hocr = f'{sp.SCRATCH}/ss-300dpi-psm1-{p:03d}.hocr'
            if os.path.exists(hocr):
                for li, ln in enumerate(sp.parse_hocr(hocr)):
                    for w in ln['words']:
                        if JUNK.search(w['text']) and len(w['text']) > 3:
                            x0, y0, x1, y1 = w['bbox']
                            if not any(abs(m['line'] - li) <= 1 and m['x0'] >= x0 - 5 and m['x0'] <= x1 + 5 for m in refs):
                                junk_words.append((p, li, w['text']))
    print(f'pages without detections file: {missing_pages}')
    # ---- second reader: nearest-template classifier bootstrapped from confident psm-7 readings
    imgs = {}
    def img_of(p):
        if p not in imgs:
            imgs[p] = sp.render_page(p, 300)[0]
        return imgs[p]
    templates = []
    for m in allrefs:
        txt = m.get('ocr7', '')
        if m.get('ocr7_conf', 0) >= 90 and txt.isdigit() and len(txt) == m['n']:
            for d, c in zip(txt, m['comps']):
                templates.append((d, sp.norm_crop(img_of(m['page']), c), m['page']))
    print(f'templates: {len(templates)} digits from confident marks; inventory',
          sorted(((d, sum(1 for t in templates if t[0] == d)) for d in '0123456789')))
    for m in allrefs:
        if m.get('tmpl') is not None:
            continue
        out = []
        for c in m['comps']:
            v = sp.norm_crop(img_of(m['page']), c)
            best = None
            for d, tv, tp in templates:
                if tp == m['page'] and False:
                    continue
                dist = sum((a - b) ** 2 for a, b in zip(v, tv))
                if best is None or dist < best[0]:
                    best = (dist, d)
            out.append(best[1] if best else '?')
        m['tmpl'] = ''.join(out)
    for p in sorted(set(m['page'] for m in allrefs)):
        dets = load_page(p)
        for m in dets:
            for r in allrefs:
                if r['page'] == p and r['line'] == m['line'] and r['x0'] == m['x0']:
                    m['tmpl'] = r['tmpl']
        with open(f'{sp.SCRATCH}/det-300dpi-{p:03d}.json', 'w') as f:
            json.dump(dets, f, indent=1)
    # ---- numbering prior per chapter
    print(f'\n{"ch":>2} {"pages":>8} {"marks":>5} {"maxseen":>7} {"notes":>5} {"new":>4} {"recite":>6} {"gaps":>4} {"viol":>4} {"lowconf":>7} {"empty":>5} {"odd":>4} {"escal":>5}')
    tot = dict(marks=0, viol=0, lowconf=0, empty=0, odd=0, escal=0, recite=0, new=0, gaps=0, disagree=0)
    seq_out = []
    for ch in range(1, 10):
        refs = [m for m in allrefs if m['chapter'] == ch]
        mx = 0; new = recite = viol = lowconf = empty = odd = escal = gaps = disagree = 0
        for m in refs:
            txt = m.get('ocr7', m.get('ocr', '')); conf = m.get('ocr7_conf', m.get('ocr_conf', -1))
            tm = m.get('tmpl', '')
            # consensus: agree -> value; disagree -> the reading that fits the prior and the component count
            if txt == tm:
                v = int(txt) if txt.isdigit() else None
            else:
                cands = [c for c in (tm, txt) if c.isdigit() and len(c) == m['n'] and int(c) <= mx + 1]
                if not cands:
                    cands = [c for c in (tm, txt) if c.isdigit() and len(c) == m['n'] and int(c) <= NOTES[ch]]
                v = int(cands[0]) if cands else (int(txt) if txt.isdigit() else None)
                txt = str(v) if v is not None else txt
            e = []
            if v is None:
                empty += 1; e.append('empty')
            if conf < CONF_ESC:
                lowconf += 1; e.append(f'conf{conf:.0f}')
            if m['n'] != len(txt) or m['n'] > 3 or not (0.8 <= m['hx'] <= 1.15):
                odd += 1; e.append('odd-geom')
            if m.get('tmpl') != txt:
                disagree += 1; e.append(f"tmpl={m.get('tmpl')}")
            status = ''
            mx_before = mx
            if v is not None:
                if v == mx + 1:
                    new += 1; mx = v; status = 'new'
                elif 1 <= v <= mx:
                    recite += 1; status = 'recite'
                elif mx + 1 < v <= min(mx + 3, NOTES[ch]):
                    # small gap: most likely a ref between was missed (or dropped by the scan);
                    # advance, but flag the span for escalation
                    gaps += 1; e.append(f'gap({mx}->{v})'); status = 'GAP'; mx = v
                else:
                    viol += 1; e.append(f'viol({v}>max{mx}+1)'); status = 'VIOL'
            if e:
                escal += 1
            seq_out.append(dict(ch=ch, page=m['page'], line=m['line'], x=m['x0'], word=m['word'], n=m['n'],
                                ocr=txt, ocr7=m.get('ocr7'), conf=conf, value=v, tmpl=m.get('tmpl'), max_before=mx_before,
                                status=status, escalate=e))
        print(f'{ch:>2} {CHAPTERS[ch-1]:>3}-{CHAPTERS[ch]-1:<4} {len(refs):>5} {mx:>7} {NOTES[ch]:>5} {new:>4} {recite:>6} {gaps:>4} {viol:>4} {lowconf:>7} {empty:>5} {odd:>4} {escal:>5}')
        print(f'      (reader disagreements: {disagree})')
        for k, val in dict(marks=len(refs), viol=viol, lowconf=lowconf, empty=empty, odd=odd, escal=escal, recite=recite, new=new, gaps=gaps, disagree=disagree).items():
            tot[k] += val
    print(f'   total: {tot}')
    print('\ngaps (missing first-citations) and whether the number is cited anywhere later in the chapter:')
    for i, r in enumerate(seq_out):
        if r['status'] == 'GAP':
            missing = list(range(r['max_before'] + 1, r['value']))
            later = {k: any(q['value'] == k for q in seq_out[i + 1:] if q['ch'] == r['ch']) for k in missing}
            prev = [q for q in seq_out[:i] if q['ch'] == r['ch']]
            pv = prev[-1] if prev else None
            print(f"   ch{r['ch']} missing {missing} between p{pv['page'] if pv else '?'} L{pv['line'] if pv else '?'} and p{r['page']} L{r['line']}; cited later: {later}")
    # tight junk-suffix pattern in the page OCR text: does it point at anything the detector did not find?
    JUNK2 = re.compile(r'[.,;:?!"\'”’)]+[!°®*¢?\'’”"]{1,2}$|[.?!"\'”’)]\d{1,2}$')
    hits = 0; unexplained = []
    for p in range(17, 281):
        hocr = f'{sp.SCRATCH}/ss-300dpi-psm1-{p:03d}.hocr'
        dets = load_page(p) or []
        refs = [m for m in dets if m['after_punct'] and not m['thin']]
        for li, ln in enumerate(sp.parse_hocr(hocr)):
            for w in ln['words']:
                if JUNK2.search(w['text']) and len(w['text']) > 3 and not re.search(r'\d,\d{3}|\$\d', w['text']):
                    hits += 1
                    x0, y0, x1, y1 = w['bbox']
                    if not any(abs(m['line'] - li) <= 1 and x0 - 5 <= m['x0'] <= x1 + 5 for m in refs):
                        unexplained.append((p, li, w['text']))
    print(f'\ntight junk-suffix words in page OCR: {hits}; without a detection nearby: {len(unexplained)}')
    for u in unexplained:
        print('   ', u)
    pages_esc = len(set(r['page'] for r in seq_out if r['escalate']))
    print(f'   marks flagged for escalation: {tot["escal"]} on {pages_esc} pages (CONF_ESC={CONF_ESC})')
    print(f'\ntext-layer junk-suffix words with no detection nearby (possible missed refs): {len(junk_words)}')
    for j in junk_words[:80]:
        print('   ', j)
    json.dump(dict(refs=seq_out, junk=junk_words), open(f'{sp.SCRATCH}/seq.json', 'w'), indent=1)
    # confidence distribution
    confs = sorted(r['conf'] for r in seq_out)
    if confs:
        print('\nconf percentiles (5/25/50/75): ', [round(confs[int(len(confs) * q)], 1) for q in (0.05, 0.25, 0.5, 0.75)])
    print('\nescalations (all flagged marks):')
    for r in seq_out:
        if r['escalate']:
            print(f"   p{r['page']} L{r['line']:02d} {r['word']!r:28s} ocr={r['ocr']!r:5s} tmpl={r['tmpl']!r:5s} conf={r['conf']:.0f} n={r['n']} max_before={r['max_before']} {r['status']} {r['escalate']}")
    # hand-labelled hard cases (read from out/thumb/ss/montage-esc*.png and rng-*.png)
    HARD = {(20,15):'6',(22,16):'13',(38,19):'41',(44,1):'56',(59,23):'15',(64,23):'21',(64,32):'22',(68,19):'25',(70,1):'25',
            (72,2):'27',(77,32):'39',(78,15):'1',(92,1):'26',(100,18):'44',(101,29):'47',(104,14):'55',(104,36):'57',(105,3):'55',
            (105,18):'58',(107,2):'62',(110,36):'6',(110,33):'5',(114,3):'18',(118,7):'14',(122,24):'40',(123,24):'46',(124,5):'41',
            (124,30):'48',(124,36):'49',(128,6):'53',(128,12):'55',(128,1):'52',(128,15):'56',(134,10):'71',(135,21):'75',(146,27):'5',
            (168,9):'54',(168,13):'55',(168,15):'56',(168,33):'57',(169,3):'58',(170,5):'55',(173,19):'66',(178,21):'1',(178,23):'5',
            (179,36):'5',(180,24):'9',(189,37):'31',(191,28):'36',(191,36):'37',(192,18):'38',(192,24):'39',(192,31):'38',(168,2):'53'}
    ok7 = okt = okboth = n = 0
    print('\nhard cases (hand-labelled): page line truth ocr7 tmpl')
    for r in seq_out:
        k = (r['page'], r['line'])
        if k in HARD:
            n += 1; t = HARD[k]
            ok7 += r['ocr'] == t; okt += r['tmpl'] == t; okboth += (r['ocr'] == t or r['tmpl'] == t)
            print(f"   p{r['page']} L{r['line']:02d} truth={t:3s} ocr7={r['ocr']!r:6s}({r['conf']:.0f}) tmpl={r['tmpl']!r:6s} {'' if r['ocr']==t else 'OCR7-WRONG'} {'' if r['tmpl']==t else 'TMPL-WRONG'}")
    print(f'   hard cases: {n}; psm7 right {ok7}; template right {okt}; at least one right {okboth}')


if __name__ == '__main__':
    main()
