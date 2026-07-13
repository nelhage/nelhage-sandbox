"""Find the 32-byte voucher-unwrap key in a heap dump.

For no-lock books the derivation input `shared` is a fixed public constant, so
every no-lock voucher is unwrapped by the SAME 32-byte AES key
(= HMAC-SHA256(obfuscate(shared), "PIDv3")).  We locate it by testing heap
windows as the AES-256 key on a known no-lock voucher's cipher_text and checking
the result is a valid KeySet.  That one key then decrypts every no-lock voucher.
"""
import sys, glob
import numpy as np
from Crypto.Cipher import AES
import androidvoucher as A

HEAP = sys.argv[1]
REFBOOK = sys.argv[2] if len(sys.argv) > 2 else "files-4k/B08V4QSV6W"

v = A.AndroidVoucher(open(glob.glob(REFBOOK + "/amzn1.drm-voucher.v1.*.ast")[0], "rb").read())
ct0 = v.cipher_text[:16]
iv = v.cipher_iv[:16]
# The decrypted KeySet begins with the same ProtectedData Ion header as the
# voucher files -> known plaintext for CBC block 0.  P0 = D(key, C0) XOR IV.
EXPECT = bytes.fromhex('e00100eaee9e8183de9a86be97de9584')

buf = np.fromfile(HEAP, dtype=np.uint8)
found = None
for align in (16, 8, 4):
    base = buf[:((len(buf) - 32) // align) * align + 32]
    idx = np.arange(0, len(base) - 31, align)
    W = np.stack([base[idx + j] for j in range(32)], axis=1)
    srt = np.sort(W, axis=1)
    uniq = (np.diff(srt, axis=1) != 0).sum(axis=1) + 1
    keep = (uniq >= 18)
    cand = np.unique(W[keep], axis=0)
    print(f"align {align}: {len(idx)} windows -> {len(cand)} candidates", flush=True)
    for row in cand:
        key = row.tobytes()
        p0 = AES.new(key, AES.MODE_ECB).decrypt(ct0)
        if bytes(a ^ b for a, b in zip(p0, iv)) == EXPECT:   # exact known-plaintext match
            found = key
            print(f"*** UNWRAP KEY: {key.hex()}", flush=True)
            break
    if found:
        break

if not found:
    print("unwrap key not found in dump")
    sys.exit(1)

# Apply to every no-lock voucher we have
print("\n=== decrypting all no-lock vouchers with this unwrap key ===")
for d in sorted(glob.glob("files-4k/B0*")):
    vp = glob.glob(d + "/amzn1.drm-voucher.v1.*.ast")
    if not vp:
        continue
    vv = A.AndroidVoucher(open(vp[0], "rb").read())
    if vv.lock_parameters:
        print(f"{d.split('/')[-1]}: lock={vv.lock_parameters} (needs DSN, skipped)")
        continue
    pt = AES.new(found, AES.MODE_CBC, vv.cipher_iv[:16]).decrypt(vv.cipher_text)
    n = pt[-1]
    ck = A._extract_keyset_key(pt[:-n]) if 1 <= n <= 16 else None
    print(f"{d.split('/')[-1]}: content_key={ck.hex() if ck else 'FAILED'}")
