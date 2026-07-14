# Kindle Android KFX DRM — archival toolkit

Goal: decrypt personally-purchased Kindle books (pulled off a rooted Android
emulator) into DRM-free EPUBs for personal archival / cross-device use.

## Dependencies

You will need a running rooted Android emulator. Install Android Studio and create an AVD. Note that as-of this writing, the Kindle application does not support [16kb pages](https://developer.android.com/guide/practices/page-sizes), and so I used an older image that still uses 4kb pages -- I used specifically `system-images/android-36.1/google_apis_playstore/arm64-v8a/` aka "Google Play ARM 64 v8a System Image" for Android 16.

Root the AVD using [rootAVD.sh](https://gitlab.com/newbit/rootAVD). Note that the bundled Magisk is old; Follow [these instructions](https://falasi.prose.sh/Rooting-an-Android-17-Emulator-in-2026) and use the `FAKEBOOTIMG` flow for newer Android versions.

Once you have Magisk installed and the device is cold-booted, verify that `adb shell su -c id` works (shows UID 0 and a context containing "magisk"). You may need to manually enable it in the Magisk app inside the emulator at first.

Install the Kindle app on the emulator. If you don't want to log in to the Play Store in the emulator, you can use [Aurora](https://f-droid.org/en/packages/com.aurora.store/) to access the store without logging in.

Make sure `adb` is in your path, and the Python dependencies are installed, and then you should be able to use `harvest.py`. Sample usage:

```
# List all books in your library
python harvest.py --list
# Download one or more books, and find the encryption key
python harvest.py --asins BXXXX,BYYYY

# In addition, decrypt the book and repackage as an EPUB
# Decrypted books will end up in `out/`
python harvest.py --asins BXXXX,BYYYY --repackage

# Operate on all books in your kindle library
python harvest.py --all --repackage

# If you want to skip specific books from `--all`, place their ASINs in the `SKIP` file, one per line, with an optional comment after a `#`
```

If you have any issues or questions, ask Claude (I've been using Claude Code with Opus 4.8).

# Claude-written notes

## TL;DR of what we learned

1. The Android voucher (`amzn1.drm-voucher.v1.*.ast`) is an Amazon **Ion**
   document, `com.amazon.drm.VoucherEnvelope` with a `PIDv3` strategy — the same
   crypto core as the public desktop DeDRM scheme, but a different container and
   keyed *only* on the device serial (`CLIENT_ID`), or on *nothing* (empty lock).
   Full structure in `NOTES.md`.
2. Deriving the content key purely offline is blocked by **one unknown**: this
   VoucherEnvelope version's byte-scramble ("obfuscation") is not in any public
   tool, and its symbol/version isn't in DeDRM's tables. Even the no-lock books
   (whose `shared` input is fully known) don't decrypt with any known scramble.
   Reversing that scramble from `libKindleAndroidNativeBundlerJNI.so` is possible
   but the crypto there is statically-linked BoringSSL, stripped, with the DRM
   strings (`PIDv3`, the scramble "words") not present in plaintext.
3. **So we recover the content key from app memory instead.** When a book is
   open in the Kindle app, its 16-byte AES content key is resident in native
   heap. We dump the heap (frida, attaching *after* the book renders) and
   brute-force which 16-byte window is the key by testing it against the book's
   on-disk encrypted pages (known-plaintext: page 0 decrypts to the fixed prefix
   `00 5d 00 00 40 00 00 28`). This works identically for **both** the no-lock
   and CLIENT_ID-locked books — we never actually needed the device serial.
4. Once we have the content key, decryption + repackaging use the **public** KFX
   tooling: decrypt the `\xeaDRMION\xee` fragments (DeDRM's `DrmIon`), rebuild a
   `.kfx-zip`, and convert to EPUB with calibre's **KFX Input** plugin.

Two facts worth reusing:
- All 6 **no-lock** books share the *same* 32-byte voucher-unwrap key (identical
  `shared` input). If that key were captured it would decrypt all their vouchers
  offline — but it is freed from memory shortly after book-open, so we haven't
  caught it (`archive/find_unwrap_key.py` scanned dumps, not present). The per-book
  16-byte *content* key persists (KRF keeps it for rendering), so we recover that.
- KRF keeps only ~2–3 content keys cached at once (older books get evicted), so
  dump right after opening each book.

---

## Environment / setup

```sh
# Python env: the flake devShell provides python3 + uv; deps are declared in
# pyproject.toml and installed into a uv-managed .venv. `.envrc` uses a
# `layout uv` helper that runs `uv sync` and activates .venv on `cd`, so
# `python harvest.py` just works — no manual venv activation.
cd kindle-droid && direnv allow      # first entry auto-runs `uv sync`
# (outside direnv:  uv sync && . .venv/bin/activate)
# NOTE: frida is pinned in pyproject.toml to 17.15.4 to match the on-device
# frida-server; a host/server version mismatch breaks agent injection.

# frida-server on device (arm64, matching host frida version, e.g. 17.15.4)
adb push frida-server /data/local/tmp/frida-server
adb shell "su -c 'chmod 755 /data/local/tmp/frida-server'"
adb shell "su -c 'setenforce 0'"                 # permissive: needed for agent injection
# run frida-server in the FOREGROUND (a detached `&`/setsid launch dies on inject):
adb shell su -c '/data/local/tmp/frida-server'   # leave running in its own shell

# calibre + KFX Input plugin already installed on host (ebook-convert on PATH)
```

Gotchas learned the hard way:
- The **16 KB-page** AVD (`sdk_gphone16k_arm64`) crashes the Kindle native libs
  and breaks frida injection. Use a **4 KB-page** image.
- frida 17 API: no `Java` global, no `Module.findExportByName`/`Memory.readByteArray`;
  use `ptr.readByteArray(n)`, `module.findExportByName(name)`, `send(meta, buf)`.
- The app's crypto is static/stripped, so hooking `libcrypto` EVP/HMAC catches
  nothing (`archive/hook_crypto.js` is kept for reference but did not fire).
- Attaching frida **while a book is opening hangs the render**; attach only
  *after* the reader has drawn text.

---

## Workflow (per book)

```sh
. .venv/bin/activate
# 1. open the book in the Kindle app so it renders (see Open-books problem below)
# 2. dump heap (attach AFTER it has rendered)
python dump_driver.py heap.bin
# 3. recover the content key (prints "*** KEY FOUND: <ASIN> <hex> ***")
python brute_key.py heap.bin files-4k
#    -> append the ASIN + key to keys.txt
# 4. build a DRM-free .kfx-zip and convert to EPUB
python repackage.py files-4k/<ASIN> <hexkey> out/<ASIN>.kfx-zip
ebook-convert out/<ASIN>.kfx-zip out/<ASIN>.epub
```

Batch step 4 for everything in `keys.txt`:
```sh
while read asin key; do
  python repackage.py "files-4k/$asin" "$key" "out/$asin.kfx-zip" &&
  ebook-convert "out/$asin.kfx-zip" "out/$asin.epub"
done < keys.txt
```

---

## Tooling index

| File | Purpose |
|------|---------|
| `androidvoucher.py` | Parse the Android voucher; derive content key *given a DSN + obfuscation* (works if the scramble ever gets reversed); decrypt DRMION via `kfxdrm.DrmIon`. Also a CLI to inspect a voucher. |
| `dump_mem.js` / `dump_driver.py` | frida heap dumper (anonymous rw regions → `heap.bin` + `.idx`). Attach after render. |
| `brute_key.py` | Recover the 16-byte content key from a heap dump via the page-0 known-plaintext test. |
| `repackage.py` | Decrypt DRMION fragments with a key, rebuild a `.kfx-zip`. |
| `kfxdrm/` | DeDRM `ion.py` + `kfxtables.py` as an importable package (`DrmIon`, `DrmIonVoucher`, obfuscation helpers). GPLv3. |
| `_ionobf.py`, `kfxtables.py` | Standalone copies of DeDRM obfuscation/scramble helpers used by `androidvoucher.py`. GPLv3. |
| `mobidrm/` | Legacy **Mobipocket/KF8** (`.prc`/`.azw`) support for titles that don't deliver as KFX. `brute_mobi()` recovers the crypto-type-2 MOBI key from a heap dump (same memory approach as KFX — see below); `decrypt_with_key()` strips it. Vendored DeDRM/KindleUnpack (PC1, `MobiBook`, HUFF/CDIC). GPLv3. |
| `NOTES.md` | Deep-dive on the voucher/DRMION format. |

Exploration dead-ends that are no longer part of the pipeline live in `archive/`
(kept for reference — none of them fired, and nothing imports them):

| File | Purpose |
|------|---------|
| `archive/find_unwrap_key.py` | Scan a dump for the shared 32-byte no-lock voucher-unwrap key (not found — it's transient). |
| `archive/scan_keyset.js` / `archive/scan_driver.py` | Scan memory for decrypted Ion `ProtectedData` headers (didn't surface keys). |
| `archive/hook_crypto.js` / `archive/capture.py` | frida hooks on libcrypto EVP/HMAC (didn't fire — static crypto). |

---

## Non-KFX books: legacy Mobipocket/KF8 (`.prc`)

Not every title delivers as KFX. Some (e.g. `B00KVI76ZS`, the Ashlee Vance *Elon
Musk* bio) download as a legacy **Mobipocket/KF8** container `files/<ASIN>/<ASIN>_EBOK.prc`
(PalmDB `BOOK`/`MOBI`, HUFF/CDIC-compressed, EXTH cdetype `EBOK`, **encryption
type 2** = Amazon's old MOBI DRM). `harvest.py` classifies on-disk content with
`book_format()` and routes these to `mobidrm/`.

That DRM was reverse-engineered by DeDRM long ago, and normally you'd strip it
*offline* with a device/account key. **That offline path is dead on this app
version**: the account serial + `kindle.account.tokens` in
`databases/map_data_storage.db` are `AES-GCM+…`-wrapped (key in the Android
keystore), so DeDRM's `androidkindlekey.py` finds nothing to decrypt.

So we recover the key the *same* way as KFX — from memory. When the book is open,
the app holds the 16-byte crypto-type-2 `found_key` in the anonymous heap;
`mobidrm.brute_mobi()` scans the dump for it. The catch vs. KFX: HUFF/CDIC
dictionaries are full of real text, so a *wrong* key still decompresses to
plausible word-salad — the KFX-style "looks like text" test false-positives. The
decisive test is a **full** record decompressing to strictly-valid codepage text
with real HTML-tag density, checked across two records (the winner is then
confirmed by the calibre conversion). A vectorised (numpy) PC1 + a shared,
eagerly-expanded decompressor keep the scan tractable; the key is often at a
non-16-aligned offset, so the align=1 pass (millions of windows) can take several
minutes. Decryption uses vendored DeDRM (`mobidrm.decrypt_with_key`) → a DRM-free
`.mobi` → calibre → EPUB. `harvest.py --asins <mobi-ASIN> --repackage` does it all.

---

## Whole-library harvest (open any book by ASIN — SOLVED)

The old blocker was "reliably open a specific book" — the Library grid reorders,
and `kindle://` deep links only reach Home because the reader activity
(`StandAloneBookReaderActivity`) is **not exported** and loads its book from an
internal `ReaderController` (so a raw `am start` won't render it). Solved by
driving the app's own **KRX open API in-process via frida** — deterministic,
UI-free, keyed on ASIN, and it scales to the full library (~757 ebooks, all
enumerable from `kindle_library.db`).

**How the opener works** (`krx_agent.js` + `harvest.py`):

1. frida 17 ships **no** built-in `Java` bridge, so `krx_agent.js` is a bundle of
   `frida-java-bridge` compiled with modern `frida-compile` (v16+, *not* the old
   browserify `frida-compile@10`; source is ESM `import Java from 'frida-java-bridge'`).
   Rebuild: `cd scratch && npm i frida-compile@latest frida-java-bridge &&
   frida-compile agent_src.js -o krx_agent.js`.
2. The agent grabs the live SDK instance (`com.amazon.kindle.krx.a`) with
   `Java.choose`, then mirrors what the app's own notification-tap handler does:
   - `sdk.getLibraryManager().getContentFromAsin(asin, false)` → `IBook`
   - `sdk.getStoreManager().downloadBook(book)` — for REMOTE books; `.kfx`
     fragments land in `files/<ASIN>/` within ~10 s.
   - `sdk.getReaderManager().openBook(book, null, null, currentActivity)` — opens
     the reader. (`currentActivity` is pulled from `ActivityThread`.)
3. **CRITICAL:** frida attached *during* the async book-load blocks the render.
   So the flow is **attach → `open(asin)` → detach *immediately*** — only then
   does `StandAloneBookReaderActivity` foreground and KRF actually decrypt +
   cache the content key. (Proven: an open-only dump has **0** key occurrences;
   after a real render the key is resident.) Then re-attach to dump.

**Per-book pipeline** (all in `harvest.py`): force-stop→launch (clean slate) →
download if `files/<ASIN>/` has no `.kfx` → `open(asin)` + detach → poll for
`StandAloneBookReaderActivity` + ~8 s render wait → pull `files/<ASIN>/` into
`files-4k/` → dump heap → `brute_one` (targeted, stops at first alignment hit) →
append key to `keys.txt` → optional repackage → EPUB.

```sh
. .venv/bin/activate
python harvest.py --list                       # inventory + which have keys
python harvest.py --asins B003JTHWKU --repackage   # one book, end to end
python harvest.py --all --repackage            # every book missing a key
```

Notes:
- 32-char non-`B0` ASINs (older `AMZNID0/<id>/4/` items — imported/personal docs)
  may not resolve via `getContentFromAsin`; the bulk of the library is `B0…`.
- KRF caches only ~2–3 content keys at once, so `harvest.py` dumps per book right
  after its render; `brute_one` tests only that book's pages.
- `ebook-meta out/<ASIN>.epub` confirms each decrypted book's real title/author.
- Tooling: `krx_agent.js` (compiled agent), `open_dump.py` (open+dump one book),
  `harvest.py` (batch). The APK is decompiled on demand with
  `nix build nixpkgs#jadx` when the open path needs re-investigation.
