"""MOBI record decompressors — PalmDoc (LZ77) and HUFF/CDIC.

Vendored (Python-3-only, de-`compatibility_utils`'d) from KindleUnpack's
mobi_uncompress.py (kevinhendricks/KindleUnpack, GPL v3).  We only need these to
*validate* a candidate content key during the memory brute-force: the real
DRM-strip (mobidedrm) leaves records compressed for calibre to expand.
"""
import struct


class UnpackError(Exception):
    pass


class UncompressedReader:
    def load_dicts(self, sections):
        pass

    def unpack(self, data):
        return data


class PalmdocReader:
    def load_dicts(self, sections):
        pass

    def unpack(self, i):
        o, p = b"", 0
        while p < len(i):
            c = i[p]
            p += 1
            if 1 <= c <= 8:
                o += i[p:p + c]
                p += c
            elif c < 128:
                o += bytes([c])
            elif c >= 192:
                o += b" " + bytes([c ^ 128])
            else:
                if p < len(i):
                    c = (c << 8) | i[p]
                    p += 1
                    m = (c >> 3) & 0x07FF
                    n = (c & 7) + 3
                    if m > n:
                        o += o[-m:n - m]
                    else:
                        for _ in range(n):
                            o += o[-m:] if m == 1 else o[-m:-m + 1]
        return o


class HuffcdicReader:
    q = struct.Struct(b">Q").unpack_from

    def load_dicts(self, dict_sections):
        """dict_sections: the raw bytes of the HUFF record followed by each CDIC
        record, in order (as found in the PalmDB)."""
        self.load_huff(dict_sections[0])
        for cdic in dict_sections[1:]:
            self.load_cdic(cdic)

    def load_huff(self, huff):
        if huff[0:8] != b"HUFF\x00\x00\x00\x18":
            raise UnpackError("invalid huff header")
        off1, off2 = struct.unpack_from(b">LL", huff, 8)

        def dict1_unpack(v):
            codelen, term, maxcode = v & 0x1F, v & 0x80, v >> 8
            assert codelen != 0
            if codelen <= 8:
                assert term
            maxcode = ((maxcode + 1) << (32 - codelen)) - 1
            return (codelen, term, maxcode)
        self.dict1 = list(map(dict1_unpack, struct.unpack_from(b">256L", huff, off1)))

        dict2 = struct.unpack_from(b">64L", huff, off2)
        self.mincode, self.maxcode = (), ()
        for codelen, mincode in enumerate((0,) + dict2[0::2]):
            self.mincode += (mincode << (32 - codelen),)
        for codelen, maxcode in enumerate((0,) + dict2[1::2]):
            self.maxcode += (((maxcode + 1) << (32 - codelen)) - 1,)
        self.dictionary = []

    def load_cdic(self, cdic):
        if cdic[0:8] != b"CDIC\x00\x00\x00\x10":
            raise UnpackError("invalid cdic header")
        phrases, bits = struct.unpack_from(b">LL", cdic, 8)
        n = min(1 << bits, phrases - len(self.dictionary))
        h = struct.Struct(b">H").unpack_from

        def getslice(off):
            blen, = h(cdic, 16 + off)
            sl = cdic[18 + off:18 + off + (blen & 0x7FFF)]
            return (sl, blen & 0x8000)
        self.dictionary += list(map(getslice,
                                    struct.unpack_from(b">%dH" % n, cdic, 16)))

    def unpack(self, data):
        q = HuffcdicReader.q
        bitsleft = len(data) * 8
        data += b"\x00\x00\x00\x00\x00\x00\x00\x00"
        pos = 0
        x, = q(data, pos)
        n = 32
        s = b""
        while True:
            if n <= 0:
                pos += 4
                x, = q(data, pos)
                n += 32
            code = (x >> n) & ((1 << 32) - 1)
            codelen, term, maxcode = self.dict1[code >> 24]
            if not term:
                while code < self.mincode[codelen]:
                    codelen += 1
                maxcode = self.maxcode[codelen]
            n -= codelen
            bitsleft -= codelen
            if bitsleft < 0:
                break
            r = (maxcode - code) >> (32 - codelen)
            sl, flag = self.dictionary[r]
            if not flag:
                self.dictionary[r] = None
                sl = self.unpack(sl)
                self.dictionary[r] = (sl, 1)
            s += sl
        return s


def make_reader(compression):
    """Return a fresh decompressor for a PalmDB compression id."""
    if compression == 1:
        return UncompressedReader()
    if compression == 2:
        return PalmdocReader()
    if compression == 17480:
        return HuffcdicReader()
    raise UnpackError("unknown compression %r" % compression)
