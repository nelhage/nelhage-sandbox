"""Topaz (TPZ0) DRM removal for kindle-droid.

The app derives an 8-byte bookKey when it renders a Topaz book and leaves it in
the scudo heap — recover it exactly like the KFX/Mobipocket content keys
(brute_topaz), then decrypt + convert with vendored DeDRM code (topazextract +
genbook: Topaz FlatXML/glyphs -> HTML/SVG -> .htmlz -> calibre EPUB).

Vendored GPLv3 from apprenticeharper/DeDRM_tools: topazextract, genbook,
convert2xml, flatxml2html, flatxml2svg, stylexml2css (alfcrypto is a clean py3
re-port of just the Topaz cipher).
"""
import os
import sys

# The vendored DeDRM modules import each other by bare name (`import genbook`,
# `import convert2xml`, ...), so put this package dir on sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from .brute import brute_topaz, oracle_record, parse_headers  # noqa: E402


def is_topaz(path):
    """True if `path` is a Topaz container (magic 'TPZ0')."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b'TPZ0'
    except OSError:
        return False


def book_title(prc_path):
    """Title from the Topaz metadata record (unencrypted), or '' if absent.
    Cheap: parses headers + the plaintext metadata record only."""
    import topazextract
    tb = topazextract.TopazBook(prc_path)
    try:
        return tb.getBookTitle()
    finally:
        tb.cleanup()


def decrypt_to_htmlz(prc_path, key, out_htmlz, svg_zip=None, quiet=True):
    """Decrypt a Topaz .prc with the 8-byte bookKey and build an .htmlz archive
    (book.html + book.opf + style.css + img/) that calibre can convert.  Returns
    out_htmlz.  Raises on failure.  genbook is very chatty; quiet=True swallows
    its stdout (errors still surface via exceptions / stderr)."""
    import contextlib
    import topazextract  # resolved via _HERE on sys.path
    import genbook

    if isinstance(key, str):
        key = bytes.fromhex(key)
    if len(key) != 8:
        raise ValueError(f"Topaz bookKey must be 8 bytes, got {len(key)}")

    tb = topazextract.TopazBook(prc_path)
    try:
        with open(os.devnull, "w") as dn, \
                (contextlib.redirect_stdout(dn) if quiet else contextlib.nullcontext()):
            tb.setBookKey(key)
            tb.createBookDirectory()
            tb.extractFiles()          # prints a dot per record
            rv = genbook.generateBook(tb.outdir, 0, True)  # very chatty
        if rv != 0:
            raise RuntimeError(f"genbook.generateBook failed (rv={rv})")
        tb.getFile(out_htmlz)
        if svg_zip:
            tb.getSVGZip(svg_zip)
    finally:
        tb.cleanup()
    return out_htmlz
