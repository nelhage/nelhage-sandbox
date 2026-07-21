"""Recover the 8-byte Topaz bookKey from a heap dump.

Same shape as mobidrm.brute: the app derives the bookKey when it renders the
book and leaves it in the scudo heap; we scan the dump for the 8-byte window
that decrypts a known encrypted+compressed payload record to a valid zlib
stream.  Oracle = the book's own page[0] record (encrypted + zlib-compressed),
read straight from the .prc — so an unrelated heap simply yields no key.

Vectorised prefix filter (numpy) computes only the first two plaintext bytes for
every candidate window and keeps those whose decrypted prefix is a valid zlib
header (byte0 == 0x78 and (0x7800 | byte1) % 31 == 0), then full-verifies the
handful of survivors with the reference cipher + zlib.decompress.  Mirrors
mobidrm._pc1_prefix_batch.
"""
import zlib
import numpy as np
from .alfcrypto import Topaz_Cipher, _MASK, _C0, _MAGIC


# ------------------------------------------------------------- container oracle
def _reader(data):
    pos = [0]
    def rdnum():
        d = data[pos[0]]; pos[0] += 1; flag = False
        if d == 0xFF:
            flag = True; d = data[pos[0]]; pos[0] += 1
        if d >= 0x80:
            x = d & 0x7F
            while d >= 0x80:
                d = data[pos[0]]; pos[0] += 1; x = (x << 7) + (d & 0x7F)
            d = x
        return -d if flag else d
    def rdstr():
        n = rdnum(); s = data[pos[0]:pos[0] + n]; pos[0] += n; return s
    return pos, rdnum, rdstr


def parse_headers(data):
    """Return (payload_offset, {name: [[offset,declen,complen],...]})."""
    if data[:4] != b'TPZ0':
        raise ValueError("not a Topaz (TPZ0) file")
    pos, rdnum, rdstr = _reader(data)
    pos[0] = 4
    hdr = {}
    for _ in range(rdnum()):
        if data[pos[0]] != 0x63:
            raise ValueError("bad Topaz header record")
        pos[0] += 1
        tag = rdstr()
        hdr[tag] = [[rdnum(), rdnum(), rdnum()] for _ in range(rdnum())]
    if data[pos[0]] != 0x64:
        raise ValueError("bad Topaz header terminator")
    pos[0] += 1
    return pos[0], hdr


def oracle_record(prc_path, name=b'page', index=0):
    """The raw (still-encrypted) ciphertext of one compressed payload record."""
    data = open(prc_path, "rb").read()
    payoff, hdr = parse_headers(data)
    if name not in hdr:
        # some books have no 'page'; fall back to glyphs
        name = b'glyphs'
    off, declen, complen = hdr[name][index]
    pos, rdnum, rdstr = _reader(data)
    pos[0] = payoff + off
    tag = rdstr()
    if tag != name:
        raise ValueError(f"record name mismatch {tag!r} != {name!r}")
    ri = rdnum()
    if ri >= 0:
        raise ValueError("oracle record is not encrypted")
    if complen <= 0:
        raise ValueError("oracle record is not compressed (no zlib oracle)")
    return data[pos[0]:pos[0] + complen]


# ------------------------------------------------------------------ verify one
def _try_key(key, oracle):
    try:
        rec = Topaz_Cipher().decrypt(oracle, Topaz_Cipher().ctx_init(key))
        zlib.decompress(rec)
        return True
    except Exception:
        return False


# --------------------------------------------------------- vectorised prefix2
def _prefix_mask(cols, c0, c1):
    n = cols[0].shape[0]
    ctx1 = np.full(n, _C0, dtype=np.uint64)
    ctx2 = np.empty(n, dtype=np.uint64)
    two, three, seven = np.uint64(2), np.uint64(3), np.uint64(7)
    mask, magic, ff = np.uint64(_MASK), np.uint64(_MAGIC), np.uint64(0xFF)
    for j in range(8):
        ctx2 = ctx1.copy()
        kb = cols[j]
        ctx1 = (((ctx1 >> two) * (ctx1 >> seven)) & mask) ^ ((kb * kb * magic) & mask)
    ks0 = ((ctx1 >> three) & ff) ^ ((ctx2 << three) & ff)
    m0 = (np.uint64(c0) ^ ks0) & ff
    cand = m0 == np.uint64(0x78)
    if not cand.any():
        return cand
    ctx2b = ctx1
    ctx1b = (((ctx1 >> two) * (ctx1 >> seven)) & mask) ^ ((m0 * m0 * magic) & mask)
    ks1 = ((ctx1b >> three) & ff) ^ ((ctx2b << three) & ff)
    m1 = (np.uint64(c1) ^ ks1) & ff
    hdr_ok = ((np.uint64(0x7800) | m1) % np.uint64(31)) == np.uint64(0)
    return cand & hdr_ok


def scan_heap(buf, oracle, aligns=(1,), chunk=8_000_000, log=None):
    """Scan buf for the 8-byte key. Returns bytes or None.

    The Topaz bookKey is NOT 16-aligned (observed at odd byte offsets, like the
    Mobipocket key), so the default is a full align=1 sweep — a superset of every
    coarser alignment, so there is no point also scanning 8/4."""
    a = np.frombuffer(buf, dtype=np.uint8)
    c0, c1 = oracle[0], oracle[1]
    n = len(buf) - 8
    for al in aligns:
        starts = np.arange(0, n, al, dtype=np.int64)
        surv_total = 0
        for i in range(0, len(starts), chunk):
            s = starts[i:i + chunk]
            cols = [a[s + j].astype(np.uint64) for j in range(8)]
            surv = s[np.asarray(_prefix_mask(cols, c0, c1))]
            surv_total += len(surv)
            for off in surv:
                key = bytes(buf[int(off):int(off) + 8])
                if _try_key(key, oracle):
                    if log:
                        log(f"topaz key at off {int(off)} (align {al}): {key.hex()}")
                    return key
        if log:
            log(f"topaz brute align={al}: {surv_total} prefix-survivors, no key")
    return None


def brute_topaz(heap, prc_path, aligns=(1,), log=None):
    """heap: path to a dump .bin (or bytes). prc_path: the book's .prc (oracle)."""
    buf = open(heap, "rb").read() if isinstance(heap, str) else heap
    oracle = oracle_record(prc_path)
    if log:
        log(f"topaz brute: {len(buf)/1e6:.0f} MB heap, oracle {len(oracle)}B "
            f"first2={oracle[:2].hex()}")
    return scan_heap(buf, oracle, aligns=aligns, log=log)
