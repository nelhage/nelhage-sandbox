"""Mobipocket/Kindle (AZW/MOBI) DRM support for kindle-droid.

Some Kindle titles download as legacy Mobipocket/KF8 (`<asin>_EBOK.prc`) rather
than KFX (see harvest.book_format).  Their DRM is Amazon's older crypto-type-2
MOBI scheme, publicly reverse-engineered in DeDRM_tools — but this app version
AES-GCM-wraps the account secrets DeDRM's androidkindlekey needs, so we instead
recover the 16-byte content key from a device heap dump (brute_mobi), then
DRM-strip with the vendored DeDRM record-decrypt (decrypt_with_key).

Vendored code: mobidedrm.py (DeDRM_tools, GPL v3), uncompress.py (KindleUnpack,
GPL v3), pc1.py (DeDRM alfcrypto, GPL v3).
"""
from .brute import brute_mobi
from .mobidedrm import decrypt_with_key, MobiBook, DrmException

__all__ = ["brute_mobi", "decrypt_with_key", "MobiBook", "DrmException"]
