# Kindle Android DRM — archival toolkit

Goal: decrypt personally-purchased Kindle books (pulled off a rooted Android
emulator) into DRM-free EPUBs for personal archival / cross-device use.

Supports all three formats Amazon delivers to the Android app: **KFX** (the
modern format, DRMION-encrypted), legacy **Mobipocket / KF8** (`.prc`/`.azw`),
and legacy **Topaz** (`TPZ0`). One command — `harvest.py` — handles the whole
library, auto-detecting each book's format.

## Dependencies

You will need a running **rooted Android emulator**. Install Android Studio and
create an AVD. As of this writing the Kindle app does not support
[16 KB pages](https://developer.android.com/guide/practices/page-sizes) — the
16 KB image crashes the Kindle native libraries and breaks frida injection — so
use an older **4 KB-page** image. I used
`system-images/android-36.1/google_apis_playstore/arm64-v8a/` ("Google Play ARM
64 v8a System Image" for Android 16).

Root the AVD with [rootAVD.sh](https://gitlab.com/newbit/rootAVD). The bundled
Magisk is old; follow [these
instructions](https://falasi.prose.sh/Rooting-an-Android-17-Emulator-in-2026)
and use the `FAKEBOOTIMG` flow for newer Android versions. After Magisk is
installed and the device cold-booted, verify `adb shell su -c id` shows UID 0
and a "magisk" context (you may need to enable root for shell in the Magisk app
first).

Install the Kindle app and sign in to your Amazon account. If you'd rather not
log in to the Play Store, [Aurora](https://f-droid.org/en/packages/com.aurora.store/)
can install it without a Google account.

On the host you need:
- `adb` on your `PATH`.
- The Python environment: the flake devShell provides `python3` + `uv`, and
  `.envrc` (`layout uv`) auto-runs `uv sync` and activates `.venv` on `cd`, so
  `python harvest.py` just works. **frida is pinned to 17.15.4** in
  `pyproject.toml` — the host frida version must match the on-device
  `frida-server`, or agent injection breaks.
- **calibre** with the **KFX Input** plugin (`ebook-convert` on `PATH`) — used
  to convert every decrypted book to EPUB.
- An **Android NDK** (provided by the flake devShell as `ANDROID_NDK_ROOT`) —
  only needed to build the persistent hook (`persist_hook.py build`).

### frida-server on the device

Push a `frida-server` binary matching the host frida version (17.15.4) and run
it **as root, in the foreground** (a detached `&`/`setsid` launch dies on
inject):

```sh
adb push frida-server /data/local/tmp/frida-server
adb shell "su -c 'chmod 755 /data/local/tmp/frida-server'"
adb shell "su -c 'setenforce 0'"                    # permissive: needed for injection
adb shell su -c '/data/local/tmp/frida-server -D'   # leave running in its own shell
```

The persistent hook writes captured keys to `/data/local/tmp/loot.bin`; that
directory must be world-writable so the app uid can create the file:
`adb shell su -c 'chmod 777 /data/local/tmp'`.

### Build the persistent hook

```sh
python persist_hook.py build     # compiles hook/hook.c -> hook.so (needs the NDK)
python persist_hook.py deploy    # pushes hook.so + primes /data/local/tmp
```

## Usage

```sh
# List every book in your library (+ which already have a recovered key)
python harvest.py --list

# Recover the content key for one or more books
python harvest.py --asins BXXXX,BYYYY

# ...and decrypt + repackage them as EPUBs (land in out/<ASIN>.epub)
python harvest.py --asins BXXXX,BYYYY --repackage

# Do the whole library (every book missing a key)
python harvest.py --all --repackage

# Skip specific books from --all: put their ASINs in the SKIP file, one per
# line, with an optional "# comment" after each.
```

Recovered keys are cached in `keys.txt` (gitignored — regenerable). Add `-v` for
verbose per-step timing. `--offline` re-runs repackaging from already-recovered
keys + already-pulled files without touching the device.

If you hit issues, ask Claude — I built this with Claude Code (Opus 4.8).

---

# How it works

Deriving Amazon's content key purely offline is blocked by unknowns specific to
this app version (see [NOTES.md](NOTES.md)): the voucher's per-version byte
"obfuscation" isn't in any public DeDRM table, and the account secrets in the
app's databases are Android-Keystore-wrapped, so `androidkindlekey.py` /
`androidvoucher.py` find nothing to work with. Hooking `libcrypto` also catches
nothing — the app's crypto is statically-linked, stripped BoringSSL plus its own
bundled software ciphers.

So instead of cracking the DRM, **we let the Kindle app derive each key and lift
it out of the running app.** Once we have the per-book key, decryption and
repackaging use ordinary vendored DeDRM code + calibre.

The whole library is driven **UI-free and deterministically by ASIN**: the app's
reader activity isn't exported (a raw `am start` won't render a book), so
`harvest.py` drives the app's own KRX API in-process via frida — download a book,
open it, query the current book — all keyed on ASIN. The full inventory (~750
titles) is enumerable from the app's `kindle_library.db`.

## Two ways to lift the key

### 1. Render-hook (primary path, all formats)

Each format's decryption runs through a specific native routine, and the key is
sitting in a register argument when it's called. We install a tiny **persistent
inline hook** on that routine and read the key off as the book renders.

The wrinkle: KFX rendering is protected by an **anti-tamper check** that scans
`/proc/self/maps` for frida's footprint and, if frida is attached, stalls the
book on the cover splash forever (fully reverse-engineered in
[FRIDA-DETECTION.md](FRIDA-DETECTION.md)). The trick that beats it: the hook
lives in *our own* `dlopen`'d `.so` (`hook/hook.c` → `hook.so`), installed by
frida (`install_hook.js`) but designed to **survive frida detaching**. With
frida gone, the `/proc/self/maps` scan passes, the book renders normally, and the
resident hook still fires — appending captured keys to `loot.bin`. We then filter
the captured keys through a per-format **oracle** (below) to pick the real one.

The three hook targets (`HOOK_TARGET` in `persist_hook.py`):

| Format | Routine | Key | Cipher |
|--------|---------|-----|--------|
| KFX/DRMION | `libKindleAndroidNativeBundlerJNI.so` (KRF) `+0x3302714` (`AES_set_encrypt_key`) | 16-byte, reg `x0` | AES-128-CBC |
| Mobipocket/KF8 | `libKRF.so` `+0x37a548` (PC1 setup) | 16-byte at `x0`, stored as byte-swapped u16 words | PC1 stream |
| Topaz | `libKRF.so` `+0x3f6790` (cipher decrypt) | 8-byte, libc++ `std::string` at `x0` | custom stream |

This path recovers even the "hard" books whose key never lands in the heap
(voucher-locked KFX, Harry Potter MOBI, all Topaz).

### 2. Heap dump + brute-force (fallback)

The original method, kept as a robust fallback. Open the book, let it render,
dump the native heap with frida (`dump_mem.js`), and brute-force which window is
the key by testing candidates against the book's own on-disk encrypted content.
The dump is narrowed to the `[anon:scudo:primary]` region (where KRF's C++
mallocs live) and, when possible, to just the pages that *changed* during the
render (delta dump) — a ~4× cheaper brute.

Brute oracles (also used to validate render-hook captures — a wrong key that
"looks like text" is a real hazard for the compressed formats):
- **KFX**: candidate must produce PKCS7-valid padding across several CBC test
  pages.
- **Mobipocket**: a wrong HUFF/CDIC key still decompresses to plausible
  word-salad, so the decisive test is a **decompressed-record-length invariant** —
  every non-final text record must decode to ~`record_size` bytes (a wrong key's
  bitstream halts early or overruns).
- **Topaz**: candidate 8-byte key must make page 0 decrypt to a valid zlib
  stream (header byte `0x78`, `%31==0`) that fully inflates.

## The per-book pipeline (`harvest_one` in `harvest.py`)

1. Skip non-store TYPEs (personal docs / newspapers / samples) up front.
2. **Download** if the content isn't on disk (poll `kindle_library.db`
   `STATE=LOCAL` — non-KFX formats never produce a `.kfx`, so a file poll would
   time out).
3. **Detect format** by content magic (`book_format`): `kfx` / `mobi` / `topaz`.
   DRM-free Mobipocket (crypto type 0) skips straight to repackage.
4. **KFX fast path** — the KFX key is derived at *download* time, so a plain
   scudo dump + align-16 brute often finds it with no render at all (~18 s).
5. **Render-hook** — the primary path for MOBI/Topaz and for voucher KFX that the
   fast path misses.
6. **Fallback** — open + render + delta dump + brute, then a full-scudo dump,
   then (KFX only) an automated **remove-download + re-download** to force a
   fresh key derivation.
7. **Repackage** — decrypt the fragments with the key and convert to EPUB
   (`repackage`): KFX → `.kfx-zip`, Mobipocket → `.mobi`, Topaz → `.htmlz`
   (stamping title/author from the library DB, which Topaz files omit) → calibre
   → `out/<ASIN>.epub`.

## What doesn't work here

- **Personal documents / Send-to-Kindle** (`BT_EBOOK_PDOC`): `downloadBook()` is
  a no-op for these and a manual re-send lands the file in a shared dir under an
  unrelated name, so `harvest.py` skips them.
- A handful of transient emulator failures (download stalls, wedged frida) — just
  retry; the tool retries each book once itself. Heavy force-stop/attach churn
  can wedge the AVD into a crash loop that only `adb reboot` clears (app
  login/data survive).

## Tooling index

| File | Purpose |
|------|---------|
| `harvest.py` | The driver. Inventory, download, key recovery (all paths), repackage, `--all`. |
| `agent_src.js` → `krx_agent.js` | frida agent bundling `frida-java-bridge` (frida 17 has no `Java` global). Drives the app's KRX API: `download`, `open`, `curasin`, `removeDownload`. `frida-compile`d — see `CLAUDE.md`. |
| `persist_hook.py` | Build/deploy/drive the persistent render-hook; `capture_key()` is the render-hook path; `keyscan`/`m1`/`m2`/`m3` are the standalone experiments. |
| `hook/hook.c` → `hook.so` | The PIC hook payload: captures key args to `loot.bin`, detach-surviving, file-backed so the maps scan can't see it. |
| `install_hook.js` | frida-side installer: `Module.load` + build an Arm64 trampoline into the hook and patch the target prologue. Hooks any module/offset per `HOOK_TARGET`. |
| `dump_mem.js` / `dump_driver.py` | frida heap dumper (scudo-narrowed, baseline + delta). Fallback path. |
| `brute_key.py` / `open_dump.py` | Standalone KFX brute / open-and-dump-one-book helpers. |
| `repackage.py` | Decrypt KFX DRMION fragments with a key → `.kfx-zip`. |
| `kfxdrm/` | DeDRM `ion.py` + `kfxtables.py` (`DrmIon`, `DrmIonVoucher`). GPLv3. |
| `mobidrm/` | Mobipocket/KF8 support: `pc1.py`, `mobidedrm.py`, `uncompress.py` (HUFF/CDIC), `brute.py` (record-length oracle). Vendored DeDRM/KindleUnpack, GPLv3. |
| `topazdrm/` | Topaz support: `alfcrypto.py` (cipher), `topazextract.py`/`genbook.py`/`convert2xml.py`/`flatxml2*`/`stylexml2css.py`, `brute.py` (zlib oracle). Vendored DeDRM, GPLv3. |
| `androidvoucher.py`, `_ionobf.py`, `kfxtables.py` | Offline voucher parsing / obfuscation helpers — the fully-offline crack that isn't finished (see NOTES.md). |
| `NOTES.md` | Deep-dive on the KFX voucher / DRMION format and why the offline crack is blocked. |
| `FRIDA-DETECTION.md` | Reverse-engineering of the anti-tamper render-block and how the persistent hook defeats it. |
| `archive/` | Dead-end explorations (libcrypto hooks, keyset scanners) — nothing imports them. |
