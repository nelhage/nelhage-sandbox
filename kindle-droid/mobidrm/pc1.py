"""Pukall Cipher 1 (PC1) — the stream cipher Mobipocket/Kindle DRM uses.

Vendored (Python-3-only, trimmed) from DeDRM_tools' alfcrypto.py
(noDRM/DeDRM_tools, GPL v3; orig. "The Dark Reverser", Apprentice Harper et al.).
The 16-byte `found_key` recovered from device memory feeds PC1(found_key, record)
to decrypt each Mobipocket text record.
"""

def PC1(key, src, decryption=True):
    """PC1-decrypt (or encrypt) `src` under the 16-byte `key`. Pure Python."""
    if len(key) != 16:
        raise ValueError("PC1: bad key length %d (want 16)" % len(key))
    sum1 = sum2 = keyXorVal = 0
    wkey = [key[i * 2] << 8 | key[i * 2 + 1] for i in range(8)]
    dst = bytearray(len(src))
    for i in range(len(src)):
        temp1 = 0
        byteXorVal = 0
        for j in range(8):
            temp1 ^= wkey[j]
            sum2 = (sum2 + j) * 20021 + sum1
            sum1 = (temp1 * 346) & 0xFFFF
            sum2 = (sum2 + sum1) & 0xFFFF
            temp1 = (temp1 * 20021 + 1) & 0xFFFF
            byteXorVal ^= temp1 ^ sum2
        curByte = src[i]
        if not decryption:
            keyXorVal = curByte * 257
        curByte = ((curByte ^ (byteXorVal >> 8)) ^ byteXorVal) & 0xFF
        if decryption:
            keyXorVal = curByte * 257
        for j in range(8):
            wkey[j] ^= keyXorVal
        dst[i] = curByte
    return bytes(dst)
