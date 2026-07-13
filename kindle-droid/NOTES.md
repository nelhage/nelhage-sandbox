# Kindle Android KFX DRM — reverse-engineering notes

Goal: decrypt personally-purchased Kindle books (pulled off an Android emulator)
for personal archival / cross-device use.

## What the files are

Each book lives in `files/<ASIN>/`:
- `amzn1.drm-voucher.v1.<uuid>.ast` — the DRM **voucher** (Amazon Ion binary).
- `CR!*.kfx` — content fragments, two kinds:
  - `CONT…` magic → KFX **container** (publicly documented format).
  - `\xeaDRMION\xee…` → `com.amazon.drm.Envelope` = **encrypted content pages**.
- `.metadata` — JSON manifest (delivery URLs, resource list). Not crypto.

## Voucher format (the previously-undocumented Android piece)

The `.ast` is Ion using Amazon's shared symbol table `ProtectedData` (v1).
Structure (`com.amazon.drm.VoucherEnvelope@<N>`):

```
{ strategy: com.amazon.drm.PIDv3@1.0 {
      lock_parameters: [ "CLIENT_ID" ]  // or []  — see below
      encryption_algorithm:      "AES"
      encryption_transformation: "AES/CBC/PKCS5Padding"
      hashing_algorithm:         "HmacSHA256" }
  mac:     <32 bytes HmacSHA256>
  voucher: <Ion doc> com.amazon.drm.Voucher@1.0 {
      id: "amzn1.drm-voucher.v1.<uuid>"
      cipher_iv:   <16 bytes>
      cipher_text: <512 bytes>          // AES-256-CBC( wrapping_key ) of a KeySet
      <64-byte signature>
      license: { license_type: "Purchase", watermark: "atv:kin:2:/…" }
      categorized_metadata: … } }
```

Key derivation (same PIDv3 core as desktop DeDRM):
```
shared        = "PIDv3" + "AES" + "AES/CBC/PKCS5Padding" + "HmacSHA256"
                + for each sorted lock param: NAME + value
                    CLIENT_ID     -> device serial number (DSN)
                    ACCOUNT_SECRET -> account secret     (not used by these books)
shared_secret = obfuscate(shared, envelope_version)     // per-version byte scramble
key           = HMAC-SHA256(shared_secret, "PIDv3")[:32]
keyset        = pkcs7_unpad( AES-256-CBC(key, cipher_iv).decrypt(cipher_text) )
              = Ion com.amazon.drm.KeySet -> SecretKey{ algorithm:AES, format:RAW, encoded:<content key> }
```

Content pages: `AES-128-CBC(content_key[:16], page.cipher_iv)`, pkcs7-unpadded,
optionally LZMA (FORMAT_ALONE, first byte 0x00 filter flag).

## Per-book lock status (from `androidvoucher.py`)

| ASIN | lock_parameters |
|------|-----------------|
| B003418518 | ["CLIENT_ID"] |
| B0FPCCYTL1 | ["CLIENT_ID"] |
| B0G3S65GF9 | ["CLIENT_ID"] |
| B08PBCD9Y7 | **[]** (no device binding) |
| B093DJ7F3C | **[]** |
| B09R6C5X88 | **[]** |

## The two remaining unknowns

1. **Envelope obfuscation for this version.** The top-level type is symbol SID
   126 in the real `ProtectedData` table; DeDRM's reverse-engineered table only
   defines up to SID 110, so we don't know the version number, and none of the
   ~13 known obfuscation scrambles (`obfuscate/2/3` v1–v28 + process_V####)
   produce a valid KeySet even for the no-lock books (where `shared` is fully
   known). => this VoucherEnvelope version's scramble is not in the public tools.
2. **The device serial (DSN)** — needed for the 3 CLIENT_ID books. Stored
   AES-GCM-encrypted in `map_data_storage.db` under an Android-Keystore key.

## How it was actually solved (see README.md for the operational guide)

Neither unknown above had to be solved. The `libcrypto` hook approach
(`archive/hook_crypto.js`) never fired because the app's BoringSSL is statically linked
and stripped, and the obfuscation strings aren't in the binary in plaintext.

Instead we recover the 16-byte **content key directly from app memory**:

1. Open the book in the Kindle app so KRF derives the content key and renders.
2. Dump anonymous rw heap with frida, attaching *after* the reader has drawn
   (attaching during open hangs the render). `dump_mem.js` / `dump_driver.py`.
3. Brute-force which 16-byte heap window is the AES content key using a
   **known-plaintext** test: every book's first `EncryptedPage` decrypts to the
   fixed prefix `00 5d 00 00 40 00 00 28`. `brute_key.py`.
4. Decrypt the DRMION fragments with that key (DeDRM `DrmIon`), rebuild a
   `.kfx-zip`, `ebook-convert` → EPUB. `repackage.py`.

This works identically for no-lock and CLIENT_ID books — **the device serial was
never needed**, because the app itself already derived the key.

Notes:
- The **16 KB-page** AVD (`sdk_gphone16k_arm64`) crashed the app and broke frida;
  moved to a **4 KB-page** AVD (`sdk_gphone64_arm64`). Both rooted via Magisk.
- All 6 no-lock books share one 32-byte voucher-unwrap key, but it is freed from
  memory shortly after open, so it hasn't been captured (`archive/find_unwrap_key.py`).
  The per-book content keys persist (KRF caches ~2–3), so we recover those.

Reversing the obfuscation from `libKindleAndroidNativeBundlerJNI.so` (anchored on
the strings `DRM voucher decryption failed due to incorrect lock parameters`,
`Checking device serial number`, `Missing DRM lock parameters`, `CLIENT_ID`,
`ACCOUNT_SECRET`) remains the path to a fully-offline tool, but was not needed.

## Tooling

See `README.md` "Tooling index". Key files: `androidvoucher.py`,
`dump_driver.py`+`dump_mem.js`, `brute_key.py`, `repackage.py`, `kfxdrm/`.
