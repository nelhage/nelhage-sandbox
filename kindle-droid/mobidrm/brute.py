"""Recover a Mobipocket/Kindle content key (`found_key`) from a device heap dump.

Same idea as the KFX brute (harvest.brute_one): the Kindle app derives the
16-byte per-book key from the account PID and leaves it in anonymous heap after
the book renders.  We can't derive it offline (this app version wraps the
account secrets in AES-GCM, out of DeDRM androidkindlekey's reach), so we scan
the heap for a 16-byte window that correctly decrypts the text records.

Known-plaintext test — the hard part, because HUFF/CDIC dictionaries hold real
text fragments, so a *wrong* key still decompresses to plausible word-salad.  A
weak "looks like text" test false-positives.  What word-salad reliably fails:
  * a full decompressed record being *strictly* valid in the book's codepage
    (concatenated dictionary phrases routinely break a UTF-8 multibyte run), and
  * well-formed HTML tag structure at real density,
checked across *two* records.  The winner is still confirmed end-to-end by the
downstream calibre conversion.
"""
import re
import struct
import numpy as np

from .mobidedrm import MobiBook, getSizeOfTrailingDataEntries
from .pc1 import PC1
from .uncompress import make_reader

_TAG = re.compile(rb"<[a-zA-Z!/][^<>\x00]{0,60}>")


def _dict_sections(book):
    """HUFF record followed by its CDIC records, located by magic (only needed
    for HUFF/CDIC-compressed books)."""
    huff, cdics = None, []
    for i in range(book.num_sections):
        d = book.loadSection(i)
        if d[:4] == b"HUFF":
            huff = d
        elif d[:4] == b"CDIC":
            cdics.append(d)
    return [huff] + cdics if huff is not None else []


def _record(book, i):
    """Encrypted text record i with its (unencrypted) trailing entries removed."""
    data = book.loadSection(i)
    extra = getSizeOfTrailingDataEntries(data, len(data), book.extra_data_flags)
    return data[:len(data) - extra]


def _codepage_ok(out, codepage):
    """Is `out` a fully valid, near-all-printable text record in this codepage?"""
    if codepage == 65001:
        try:
            out.decode("utf-8")
        except UnicodeDecodeError:
            return False
    good = sum(1 for b in out if b in (9, 10, 13) or 0x20 <= b < 0x7F or b >= 0x80)
    if good < len(out) * 0.99:
        return False
    return len(_TAG.findall(out)) >= 2


def _pc1_prefix_batch(keys, src, nbytes):
    """Vectorised PC1: decrypt the first `nbytes` of `src` under every key in
    `keys` (an (N,16) uint8 array) at once.  PC1 is byte-sequential but its
    per-byte state update is identical across keys, so the whole candidate set
    decrypts in a handful of numpy passes instead of N pure-Python PC1 runs.
    Returns (N, nbytes) uint8 of decrypted bytes."""
    N = keys.shape[0]
    wkey = (keys[:, 0::2].astype(np.int64) << 8) | keys[:, 1::2]   # (N, 8)
    sum1 = np.zeros(N, dtype=np.int64)
    sum2 = np.zeros(N, dtype=np.int64)
    out = np.empty((N, nbytes), dtype=np.uint8)
    for i in range(nbytes):
        temp1 = np.zeros(N, dtype=np.int64)
        byteXor = np.zeros(N, dtype=np.int64)
        for j in range(8):
            temp1 ^= wkey[:, j]
            sum2 = (sum2 + j) * 20021 + sum1
            sum1 = (temp1 * 346) & 0xFFFF
            sum2 = (sum2 + sum1) & 0xFFFF
            temp1 = (temp1 * 20021 + 1) & 0xFFFF
            byteXor ^= temp1 ^ sum2
        cur = ((src[i] ^ (byteXor >> 8)) ^ byteXor) & 0xFF   # decryption
        out[:, i] = cur
        wkey ^= (cur * 257)[:, None]                          # key feedback
    return out


def _build_reader(comp, dict_secs, expand=False):
    r = make_reader(comp)
    if dict_secs:
        r.load_dicts(dict_secs)
    if expand and hasattr(r, "dictionary"):
        # HuffcdicReader expands dictionary phrases lazily, mutating self as it
        # decodes.  For the cheap per-candidate loop we reuse ONE reader across
        # millions of decodes, so eagerly expand every phrase up front: then each
        # candidate decode is read-only (and a garbage candidate that raises can't
        # corrupt shared state for the next one).
        for i in range(len(r.dictionary)):
            entry = r.dictionary[i]
            if entry is not None and not entry[1]:
                r.dictionary[i] = None
                r.dictionary[i] = (r.unpack(entry[0]), 1)
    return r


def make_full_test(book, nrec=8):
    """Return test(key)->bool doing the *decisive* check on the first `nrec`
    text records.

    The decisive signal is the DECOMPRESSED RECORD LENGTH.  A MOBI text record
    is a fixed-size (record_size, normally 4096) slab of the HTML stream, so a
    correct decrypt decompresses every non-final record to (just under) that
    size.  A WRONG key feeds garbage bits to the PalmDOC/HUFF-CDIC decoder, whose
    bitstream then terminates at an essentially random point — overrunning the
    record size (HUFF/CDIC dictionaries expand wildly: we see 10k–50k for a 4k
    record) or falling well short.  This is key-, format- and version-independent,
    unlike the old "valid codepage + tag density" test, which HUFF/CDIC word-salad
    trivially passes (the dictionary is full of real tagged text fragments — the
    exact false-positive that shipped wrong keys for KF8 books like B00SEFAIRI).

    `_codepage_ok` is kept as a cheap secondary gate.  The final text record may
    legitimately be short, so we only length-check records 1..num_text_records-1
    (capped at `nrec`); tiny books with a single record fall back to record 1."""
    import struct
    comp = book.compression
    codepage = book.mobi_codepage
    dict_secs = _dict_sections(book) if comp == 17480 else []
    rsize, = struct.unpack('>H', book.sect[0x0A:0x0C])   # PalmDOC record size
    if rsize <= 0:
        rsize = 4096
    ntext = book.records
    last_full = min(nrec, ntext - 1)               # # of non-final records to test
    idx = list(range(1, last_full + 1)) or [1]     # at least record 1 (tiny books)
    recs = {i: _record(book, i) for i in idx}

    def reader():
        r = make_reader(comp)
        if dict_secs:
            r.load_dicts(dict_secs)
        return r

    hi = rsize + 64        # correct decodes land at rsize (HUFF) or rsize+1
    lo = rsize - 256       # (uncompressed/PalmDOC trailing byte); wrong keys blow past

    def test(key):
        lens = []
        for i in idx:
            try:
                full = reader().unpack(PC1(key, recs[i]))
            except Exception:
                return False
            if len(full) > hi:                     # decisive: overran the record
                return False
            if not _codepage_ok(full, codepage):
                return False
            lens.append(len(full))
        # Non-final records must be near-full; allow one short one (chapter flush).
        near_full = sum(1 for L in lens if L >= lo)
        return near_full >= len(lens) - 1

    return test


def scan_heap(heap_path, book, log=lambda *_: None, prefix=24,
              min_uniq=13, max_printable=10, aligns=(16, 8, 4, 1)):
    """Find every 16-byte heap window that decrypts the book's text records.

    Two-stage: (1) a vectorised PC1 + short HUFF/CDIC decode of record 1's prefix
    cheaply rejects the vast majority (record 1 begins the book's HTML, so its
    decompressed prefix starts with a tag); (2) survivors face the decisive
    full-record validation.  `min_uniq`/`max_printable` bound the candidate set:
    a real 16-byte content key is high-entropy (~15 distinct bytes, ~6 printable),
    unlike ASCII/pointer noise."""
    buf = np.fromfile(heap_path, dtype=np.uint8)
    log(f"scan_heap: {len(buf)} heap bytes")

    comp = book.compression
    dict_secs = _dict_sections(book) if comp == 17480 else []
    rec1_prefix = _record(book, 1)[:prefix]
    src = np.frombuffer(rec1_prefix, dtype=np.uint8).astype(np.int64)

    cheap = _build_reader(comp, dict_secs, expand=True)   # reused, read-only
    full_test = make_full_test(book)
    hits, seen = [], set()
    for align in aligns:
        idx = np.arange(0, len(buf) - 15, align)
        W = np.stack([buf[idx + j] for j in range(16)], axis=1)
        srt = np.sort(W, axis=1)
        uniq = (np.diff(srt, axis=1) != 0).sum(axis=1) + 1
        printable = ((W >= 0x20) & (W < 0x7F)).sum(axis=1)
        cand = np.unique(W[(uniq >= min_uniq) & (printable <= max_printable)], axis=0)
        log(f"scan_heap: align={align:2d} -> {len(cand)} candidate window(s)")
        dec = _pc1_prefix_batch(cand, src, prefix)     # (N, prefix) decrypted
        for k in range(len(cand)):
            try:
                head = cheap.unpack(dec[k].tobytes())
            except Exception:
                continue
            if head[:1] != b"<" or not _TAG.search(head):
                continue
            key = cand[k].tobytes()
            if key in seen:
                continue
            seen.add(key)
            if full_test(key):
                log(f"scan_heap: key found at align={align}: {key.hex()}")
                hits.append(key)
        if hits:
            break
    return hits


def brute_mobi(heap_path, prc_path, log=lambda *_: None):
    """Return the 16-byte content key for the Mobipocket book at prc_path by
    scanning heap_path, or None.  Only crypto type 2 (Amazon MOBI DRM)."""
    book = MobiBook(prc_path)
    crypto_type, = struct.unpack(">H", book.sect[0xC:0xE])
    if crypto_type != 2:
        raise ValueError(f"crypto type {crypto_type} not supported (need 2)")
    hits = scan_heap(heap_path, book, log=log)
    if not hits:
        return None
    if len(hits) > 1:
        log(f"brute_mobi: {len(hits)} keys passed; using first: {hits[0].hex()}")
    return hits[0]
