"""Topaz stream cipher (the only crypto Topaz DRM uses).

Clean Python-3 port of DeDRM's alfcrypto Topaz_Cipher pure-python path (the
upstream one is Python-2 era: ord()/chr() on a str).  Validated byte-for-byte
against the reference and against real book records (see topazdrm/brute.py
selftest).  Operates on/returns bytes.

The record cipher is a plaintext-feedback stream cipher keyed by the 8-byte
bookKey: no AES, no external key material — magic constant 0x0F902007.
"""

_MASK = 0xFFFFFFFF
_C0 = 0x0CAFFE19E
_MAGIC = 0x0F902007


class Topaz_Cipher(object):
    def ctx_init(self, key):
        # key: bytes (8-byte bookKey, or an 8-char PID for the dkey path)
        ctx1 = _C0
        ctx2 = ctx1
        for kb in bytearray(key):
            ctx2 = ctx1
            ctx1 = (((ctx1 >> 2) * (ctx1 >> 7)) & _MASK) ^ ((kb * kb * _MAGIC) & _MASK)
        self._ctx = [ctx1, ctx2]
        return [ctx1, ctx2]

    def decrypt(self, data, ctx=None):
        ctx1, ctx2 = ctx if ctx is not None else self._ctx
        out = bytearray(len(data))
        for i, c in enumerate(bytearray(data)):
            m = (c ^ ((ctx1 >> 3) & 0xFF) ^ ((ctx2 << 3) & 0xFF)) & 0xFF
            ctx2 = ctx1
            ctx1 = (((ctx1 >> 2) * (ctx1 >> 7)) & _MASK) ^ ((m * m * _MAGIC) & _MASK)
            out[i] = m
        return bytes(out)
