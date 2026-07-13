"""Recover the KFX content key from a Kindle heap dump.

The content key is a raw 16-byte AES key resident in native heap after a book is
opened.  We slide a 16-byte window over the dump and test each candidate against
the book's on-disk encrypted pages: the correct key makes the last CBC block
decrypt to valid PKCS7 padding.  Survivors are confirmed by fully decrypting a
page and parsing it as Ion.
"""
import sys, glob, os
import numpy as np
from Crypto.Cipher import AES

import androidvoucher as A
from amazon.ion import simpleion

HEAP = sys.argv[1] if len(sys.argv) > 1 else "heap.bin"
BOOKDIR = sys.argv[2] if len(sys.argv) > 2 else "files-4k"


def get_test_pages(bookdir, want=5):
    """Return up to `want` (cipher_text, cipher_iv) EncryptedPages + a fragment name."""
    found = []
    frag_name = None
    for f in sorted(glob.glob(os.path.join(bookdir, "CR!*.kfx"))):
        data = open(f, "rb").read()
        if not A.is_drmion(data):
            continue
        doc = simpleion.loads(data[8:-8], catalog=A._CATALOG, single_value=False)
        def walk(v):
            a = [x.text for x in getattr(v, 'ion_annotations', ())]
            tn = a[0] if a else ''
            if hasattr(v, 'keys'):
                if 'EncryptedPage' in tn:
                    ct = A._get(v, 'cipher_text'); iv = A._get(v, 'cipher_iv')
                    if ct is not None and iv is not None and len(bytes(ct)) % 16 == 0:
                        found.append((bytes(ct), bytes(iv)))
                for k in v: walk(v[k])
            elif isinstance(v, (list, tuple)):
                for x in v: walk(x)
        for v in doc:
            walk(v)
        if found:
            frag_name = f
            break
    return found[:want], frag_name


def pkcs7_ok(block16):
    n = block16[-1]
    return 1 <= n <= 16 and block16[-n:] == bytes([n]) * n


def make_multitest(pages):
    """Return key -> True iff it CBC-decrypts every test page to valid PKCS7
    padding.  This needs NO book-specific known plaintext (DCC books start
    `005d..`, others like 1984 start `CONT..`) — with 3+ pages the ~1/256-per-page
    padding fluke makes false positives ~10^-7 or rarer."""
    def pkcs7_ok(b):
        n = b[-1]
        return 1 <= n <= 16 and b[-n:] == bytes([n]) * n
    def test(key):
        for ct, iv in pages:
            if not pkcs7_ok(AES.new(key, AES.MODE_CBC, iv).decrypt(ct)):
                return False
        return True
    return test


# --- gather several test pages per book (multi-page test kills false positives) ---
books = []
for d in sorted(glob.glob(os.path.join(BOOKDIR, "B0*"))):
    pages, frag = get_test_pages(d)
    if len(pages) >= 3:   # 3+ pages keeps PKCS7-only false positives negligible
        books.append((os.path.basename(d), make_multitest(pages), pages[0], frag))
        print(f"test vector: {os.path.basename(d)} pages={len(pages)}")
print(f"{len(books)} books with >=2 test pages\n")

# --- load heap, vectorized pre-filter ---
buf = np.fromfile(HEAP, dtype=np.uint8)
n16 = (len(buf) // 16) * 16
print(f"heap {len(buf)/1e6:.0f} MB")

hits = []
for align in (16, 8, 4, 1):
    print(f"--- scanning alignment {align} ---")
    base = buf[:((len(buf) - 16) // align) * align + 16]
    idx = np.arange(0, len(base) - 15, align)
    W = np.stack([base[idx + j] for j in range(16)], axis=1)  # (N,16)
    # high-entropy filter: >=11 distinct byte values, not text-like
    srt = np.sort(W, axis=1)
    uniq = (np.diff(srt, axis=1) != 0).sum(axis=1) + 1
    printable = ((W >= 0x20) & (W < 0x7f)).sum(axis=1)
    keep = (uniq >= 11) & (printable <= 11)
    cand = np.unique(W[keep], axis=0)
    print(f"  {len(idx)} windows -> {len(cand)} unique high-entropy candidates")
    for row in cand:
        key = row.tobytes()
        for asin, test, page, frag in books:
            if test(key):                       # valid PKCS7 on ALL pages of this book
                hits.append((asin, key, page, frag))
                print(f"  *** KEY FOUND: {asin} key={key.hex()} ***")
    if hits:
        break

# --- confirm by full decrypt ---
print("\n=== confirming hits ===")
for asin, key, (ct, iv), frag in hits:
    pt = AES.new(key, AES.MODE_CBC, iv).decrypt(ct)
    pad = pt[-1]
    body = pt[:-pad] if 1 <= pad <= 16 else pt
    ion = body[:4] == b'\xe0\x01\x00\xea'
    print(f"{asin} key={key.hex()} first_page_plain[:8]={body[:8].hex()} ion_bvm={ion}")
