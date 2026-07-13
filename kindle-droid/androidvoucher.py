"""Decrypt Amazon Kindle *Android* KFX DRM vouchers and content.

Background
----------
The public KFX reverse-engineering (calibre "KFX Input", DeDRM) covers the KFX
*container* format and the desktop (Kindle-for-PC/Mac) DRM voucher.  The Android
app stores each book as a directory of ``CR!*.kfx`` fragments plus an
``amzn1.drm-voucher.v1.*.ast`` voucher.  That voucher is an Amazon Ion document
of type ``com.amazon.drm.VoucherEnvelope`` whose ``strategy`` is
``com.amazon.drm.PIDv3`` locked on ``CLIENT_ID`` (the device serial number / DSN)
-- i.e. the same PIDv3 crypto core as the desktop scheme, but wrapped in the
``ProtectedData`` container and keyed *only* by the device serial.

Given the device serial (DSN), this module:
  1. parses the voucher, extracting cipher_iv / cipher_text / algorithm params;
  2. derives the wrapping key exactly as the app does:
        shared        = "PIDv3" + alg + transformation + hash
                        + <sorted lock params, each: NAME + value>
        shared_secret = obfuscate(shared, version)      # per-version scramble
        key           = HMAC-SHA256(shared_secret, "PIDv3")[:32]
        keyset_ion    = pkcs7_unpad(AES-256-CBC(key, iv).decrypt(cipher_text))
  3. reads the AES content key out of the resulting ``com.amazon.drm.KeySet``.

The per-version obfuscation is brute-forced across every scramble variant known
to DeDRM, so the exact VoucherEnvelope version number is not required.

The content (``\\xeaDRMION\\xee`` fragments) is then decrypted with the content
key: each ``com.amazon.drm.EncryptedPage`` is AES-128-CBC(key[:16], iv), pkcs7
unpadded, optionally LZMA-decompressed.
"""

import sys
import lzma
from io import BytesIO

from amazon.ion import simpleion
from amazon.ion.symbols import shared_symbol_table, SymbolTableCatalog
from Crypto.Cipher import AES

import _ionobf as obf

# The shared "ProtectedData" symbol table shipped by the Kindle Ion runtime.
# (Verbatim from DeDRM ion.py -- needed so field names / type annotations in the
# voucher and DRMION documents resolve to text.)
SYM_NAMES = [
    'com.amazon.drm.Envelope@1.0', 'com.amazon.drm.EnvelopeMetadata@1.0', 'size', 'page_size',
    'encryption_key', 'encryption_transformation', 'encryption_voucher', 'signing_key',
    'signing_algorithm', 'signing_voucher', 'com.amazon.drm.EncryptedPage@1.0', 'cipher_text',
    'cipher_iv', 'com.amazon.drm.Signature@1.0', 'data', 'com.amazon.drm.EnvelopeIndexTable@1.0',
    'length', 'offset', 'algorithm', 'encoded', 'encryption_algorithm', 'hashing_algorithm',
    'expires', 'format', 'id', 'lock_parameters', 'strategy', 'com.amazon.drm.Key@1.0',
    'com.amazon.drm.KeySet@1.0', 'com.amazon.drm.PIDv3@1.0', 'com.amazon.drm.PlainTextPage@1.0',
    'com.amazon.drm.PlainText@1.0', 'com.amazon.drm.PrivateKey@1.0', 'com.amazon.drm.PublicKey@1.0',
    'com.amazon.drm.SecretKey@1.0', 'com.amazon.drm.Voucher@1.0', 'public_key', 'private_key',
    'com.amazon.drm.KeyPair@1.0', 'com.amazon.drm.ProtectedData@1.0', 'doctype',
    'com.amazon.drm.EnvelopeIndexTableOffset@1.0', 'enddoc', 'license_type', 'license',
    'watermark', 'key', 'value', 'com.amazon.drm.License@1.0', 'category', 'metadata',
    'categorized_metadata', 'com.amazon.drm.CategorizedMetadata@1.0',
    'com.amazon.drm.VoucherEnvelope@1.0', 'mac', 'voucher', 'com.amazon.drm.ProtectedData@2.0',
    'com.amazon.drm.Envelope@2.0', 'com.amazon.drm.EnvelopeMetadata@2.0',
    'com.amazon.drm.EncryptedPage@2.0', 'com.amazon.drm.PlainText@2.0', 'compression_algorithm',
    'com.amazon.drm.Compressed@1.0', 'page_index_table',
] + ['com.amazon.drm.VoucherEnvelope@%d.0' % n
     for n in list(range(2, 29)) + [9708, 1031, 2069, 9041, 3646, 6052, 9479, 9888, 4648, 5683]]

_CATALOG = SymbolTableCatalog()
_CATALOG.register(shared_symbol_table('ProtectedData', 1, SYM_NAMES))


def _pkcs7unpad(msg, blocklen=16):
    if len(msg) % blocklen != 0:
        raise ValueError("not block-aligned")
    n = msg[-1]
    if not (1 <= n <= blocklen) or msg[-n:] != bytes([n]) * n:
        raise ValueError("bad padding (wrong key)")
    return msg[:-n]


def _load(data):
    return simpleion.loads(data, catalog=_CATALOG, single_value=True)


def _fields(struct):
    """Yield (text_name, value) for an Ion struct, resolving symbol keys."""
    for k in struct:
        name = k.text if hasattr(k, 'text') else k
        yield name, struct[k]


def _get(struct, want):
    for name, val in _fields(struct):
        if name == want:
            return val
    return None


class AndroidVoucher:
    """Parse and decrypt a Kindle Android ``amzn1.drm-voucher`` (.ast) file."""

    def __init__(self, voucher_bytes):
        self.raw = voucher_bytes
        self.encryption_algorithm = None
        self.encryption_transformation = None
        self.hashing_algorithm = None
        self.lock_parameters = []
        self.cipher_iv = None
        self.cipher_text = None
        self.license_type = None
        self.content_key = None
        self._parse()

    def _parse(self):
        env = _load(self.raw)
        strategy = _get(env, 'strategy')
        if strategy is None:
            raise ValueError("voucher has no 'strategy' (not a VoucherEnvelope?)")
        self.encryption_algorithm = str(_get(strategy, 'encryption_algorithm'))
        self.encryption_transformation = str(_get(strategy, 'encryption_transformation'))
        self.hashing_algorithm = str(_get(strategy, 'hashing_algorithm'))
        self.lock_parameters = [str(x) for x in _get(strategy, 'lock_parameters')]

        voucher = _load(bytes(_get(env, 'voucher')))
        self.cipher_iv = bytes(_get(voucher, 'cipher_iv'))
        self.cipher_text = bytes(_get(voucher, 'cipher_text'))
        lic = _get(voucher, 'license')
        if lic is not None:
            self.license_type = str(_get(lic, 'license_type'))

    def _shared_base(self, dsn, secret):
        shared = ("PIDv3" + self.encryption_algorithm
                  + self.encryption_transformation + self.hashing_algorithm).encode('ascii')
        for param in sorted(self.lock_parameters):
            if param == "CLIENT_ID":
                shared += param.encode('ascii') + _b(dsn)
            elif param == "ACCOUNT_SECRET":
                shared += param.encode('ascii') + _b(secret)
            else:
                raise ValueError("unknown lock parameter: %s" % param)
        return shared

    def _candidate_secrets(self, shared):
        """Every shared-secret scramble variant DeDRM knows about."""
        cands = []
        for v in range(1, 29):
            for fn in (obf.obfuscate, obf.obfuscate2, obf.obfuscate3):
                try:
                    cands.append(fn(shared, v))
                except Exception:
                    pass
        for fn in (obf.process_V9708, obf.process_V1031, obf.process_V2069, obf.process_V9041,
                   obf.process_V3646, obf.process_V6052, obf.process_V9479, obf.process_V9888,
                   obf.process_V4648, obf.process_V5683):
            try:
                cands.append(fn(shared))
            except Exception:
                pass
        return cands

    def decrypt_content_key(self, dsn, secret=b""):
        """Return the raw AES content key, trying every obfuscation variant."""
        shared = self._shared_base(dsn, secret)
        for sharedsecret in self._candidate_secrets(shared):
            key = obf.hmac.new(bytes(sharedsecret), b"PIDv3",
                               digestmod=obf.hashlib.sha256).digest()
            try:
                pt = AES.new(key[:32], AES.MODE_CBC, self.cipher_iv[:16]).decrypt(self.cipher_text)
                pt = _pkcs7unpad(pt, 16)
                ck = _extract_keyset_key(pt)
                if ck is not None:
                    self.content_key = ck
                    return ck
            except Exception:
                continue
        return None


def _extract_keyset_key(plaintext):
    """Parse a decrypted ``com.amazon.drm.KeySet`` and return SecretKey.encoded."""
    try:
        keyset = _load(plaintext)
    except Exception:
        return None
    anns = [a.text for a in getattr(keyset, 'ion_annotations', ())]
    if not any(a and 'KeySet' in a for a in anns):
        return None
    for entry in keyset:                       # list of SecretKey structs
        enc = _get(entry, 'encoded')
        if enc is not None:
            return bytes(enc)
    return None


# ---------------------------------------------------------------------------
# Content (DRMION) decryption
# ---------------------------------------------------------------------------

DRMION_MAGIC = b'\xeaDRMION\xee'


def is_drmion(data):
    return data[:8] == DRMION_MAGIC


class _StubVoucher:
    def __init__(self, content_key):
        self.secretkey = content_key


def decrypt_drmion(data, content_key):
    """Decrypt a ``\\xeaDRMION\\xee`` fragment to its plaintext KFX bytes.

    Delegates page assembly/decompression to DeDRM's proven DrmIon parser.
    """
    from kfxdrm.ion import DrmIon
    if is_drmion(data):
        data = data[8:-8]          # strip 8-byte magic header and trailer
    out = BytesIO()
    DrmIon(BytesIO(data), lambda name: _StubVoucher(content_key)).parse(out)
    return out.getvalue()


def _walk_envelope(value, key, out):
    anns = [a.text for a in getattr(value, 'ion_annotations', ())]
    if not isinstance(value, (list, tuple)):
        return
    for item in value:
        _process_page(item, key, out)


def _process_page(item, key, out):
    anns = [a.text for a in getattr(item, 'ion_annotations', ())]
    tn = anns[0] if anns else ''
    if not isinstance(item, (list, tuple, dict)) and not hasattr(item, 'keys'):
        return
    if hasattr(item, 'keys'):                      # a struct (EncryptedPage / PlainText)
        ct = _get(item, 'cipher_text')
        iv = _get(item, 'cipher_iv')
        data = _get(item, 'data')
        compressed = _has_compressed(item)
        if ct is not None and iv is not None:
            msg = AES.new(key[:16], AES.MODE_CBC, bytes(iv)[:16]).decrypt(bytes(ct))
            msg = _pkcs7unpad(msg, 16)
            out.write(_maybe_decompress(msg, compressed))
        elif data is not None:
            out.write(_maybe_decompress(bytes(data), compressed))
    else:                                          # nested list (Envelope body)
        for sub in item:
            _process_page(sub, key, out)


def _has_compressed(struct):
    for _, v in _fields(struct):
        anns = [a.text for a in getattr(v, 'ion_annotations', ())]
        if any(a and 'Compressed' in a for a in anns):
            return True
    return False


def _maybe_decompress(msg, compressed):
    if not compressed:
        return msg
    if msg[:1] != b'\x00':
        raise ValueError("LZMA UseFilter not supported")
    d = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE)
    return d.decompress(msg[1:])


def _b(x):
    if isinstance(x, str):
        return x.encode('ascii')
    return bytes(x)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description="Inspect / decrypt a Kindle Android DRM voucher")
    ap.add_argument('voucher')
    ap.add_argument('--dsn', help="device serial number (CLIENT_ID)")
    args = ap.parse_args()
    v = AndroidVoucher(open(args.voucher, 'rb').read())
    print("encryption_algorithm     :", v.encryption_algorithm)
    print("encryption_transformation:", v.encryption_transformation)
    print("hashing_algorithm        :", v.hashing_algorithm)
    print("lock_parameters          :", v.lock_parameters)
    print("license_type             :", v.license_type)
    print("cipher_iv                : %s (%d bytes)" % (v.cipher_iv.hex(), len(v.cipher_iv)))
    print("cipher_text              : %d bytes" % len(v.cipher_text))
    if args.dsn:
        ck = v.decrypt_content_key(args.dsn)
        if ck:
            print("CONTENT KEY              :", ck.hex(), "(%d bytes)" % len(ck))
        else:
            print("CONTENT KEY              : FAILED (wrong DSN or unknown obfuscation)")
