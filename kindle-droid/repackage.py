"""Build a DRM-free .kfx-zip from a Kindle Android book directory + content key.

Replaces each encrypted DRMION fragment with its decrypted KFX container, and
zips all the book's KFX containers + sidecars into a .kfx-zip that calibre's
KFX Input plugin can convert to EPUB.

Usage: python repackage.py <book_dir> <content_key_hex> <out.kfx-zip>
"""
import sys, os, glob, zipfile
import androidvoucher as A

bookdir, keyhex, outpath = sys.argv[1], sys.argv[2], sys.argv[3]
key = bytes.fromhex(keyhex)

n_dec = n_plain = 0
with zipfile.ZipFile(outpath, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(glob.glob(os.path.join(bookdir, "*"))):
        name = os.path.basename(f)
        if name.endswith(".kfx") or name.endswith(".azw") or "CR!" in name:
            data = open(f, "rb").read()
            if A.is_drmion(data):
                data = A.decrypt_drmion(data, key)   # -> plaintext CONT container
                n_dec += 1
            else:
                n_plain += 1
            z.writestr(name, data)
        elif name.endswith((".phl", ".metadata", ".asc")):
            z.writestr(name, open(f, "rb").read())    # sidecars

print(f"wrote {outpath}: {n_dec} decrypted + {n_plain} plaintext containers")
