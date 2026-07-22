# CLAUDE.md — kindle-droid

Toolkit for decrypting the user's own purchased Kindle books off a rooted Android
emulator (personal archival). See `README.md` (operational), `NOTES.md` (KFX
voucher/DRMION format), `FRIDA-DETECTION.md` (anti-tamper render-block).

## The big picture (so you don't re-derive it)
- **Three formats, one driver.** `harvest.py` auto-detects each book's on-disk
  format by content magic (`book_format`): **KFX** (DRMION, AES-128-CBC),
  **Mobipocket/KF8** (`.prc`/`.azw`, PC1 stream, crypto-type 2), **Topaz**
  (`TPZ0`, custom 8-byte stream cipher). DRM-free Mobipocket (crypto type 0) just
  gets repackaged. DB `CONTENT_TYPE` is UNRELIABLE for the delivered format — use
  the on-disk magic sniff, and DB `TYPE` (`BT_EBOOK` = harvestable store book;
  PDOC/NEWSPAPER/SAMPLE are skipped).
- **Offline crack is DEAD on this app version** (voucher obfuscation not in public
  DeDRM tables; account secrets are Android-Keystore-wrapped). Don't chase
  `androidkindlekey.py`/`androidvoucher.py` or `libcrypto`/JCE hooks — all ruled
  out. Keys come from the running app.
- **Render-hook is the PRIMARY key source for all formats** (committed a5f05e4).
  A persistent inline native hook (`hook/hook.c` → `hook.so`, installed by
  `install_hook.js`) on each format's decrypt routine, which SURVIVES frida
  detaching — so the KRF anti-tamper `/proc/self/maps` scan passes, the book
  renders, and the hook reads the key off a register arg into `loot.bin`. Targets
  (`persist_hook.HOOK_TARGET`): KFX = `libKindleAndroidNativeBundlerJNI.so`
  (JNI KRF) `+0x3302714` (`AES_set_encrypt_key`, key=x0/16B); MOBI = `libKRF.so`
  `+0x37a548` (PC1, key=x0/16B stored byte-swapped as u16 words — `_swap_pairs`);
  Topaz = `libKRF.so` `+0x3f6790` (key = libc++ `std::string` at x0/8B). NOTE
  `libKRF.so`'s text is at `file_offset + 0x4000` (p_offset≠p_vaddr), UNLIKE the
  JNI lib where file offset == module offset.
- **Heap dump + brute is the FALLBACK** (scudo-narrowed, delta-then-full). Each
  render yields ~25-1000 candidate keys; the real one is chosen by a per-format
  ORACLE. KFX = PKCS7-valid across CBC test pages. MOBI = decompressed-record-
  length invariant (a wrong HUFF/CDIC key still yields plausible word-salad, so
  the old "looks like text" oracle FALSE-POSITIVED and shipped wrong keys — commit
  5ad0b27; now require every non-final text record to decode to ~record_size).
  Topaz = page-0 decrypts to a valid, fully-inflating zlib stream.
- **`harvest_one` ordering:** skip non-`BT_EBOOK` → download (poll STATE=LOCAL) →
  format route → KFX fast-path (key derives at DOWNLOAD time, plain scudo dump +
  align-16 brute, no render) → render-hook → open+render+delta/full dump+brute →
  (KFX) auto remove-download+re-download. `--no-renderhook` forces the legacy
  path; `--offline` repackages from cached keys/files without the device.
- **Anti-tamper render-block:** an idle frida ATTACHMENT (not our hooks) makes KRF
  `createBook` detect frida's memfd footprint via inline-syscall `/proc/self/maps`
  scan and stall forever on the cover splash. Renaming frida-server strings is
  insufficient (it keys on the agent's in-process footprint). The detach-surviving
  persistent hook is the working bypass. Full write-up in `FRIDA-DETECTION.md`.

## frida gotchas
- **frida-python lowercases/snake_cases RPC export names.** A JS agent export
  `rpc.exports = { removeDownload() {} }` is called from Python as
  `script.exports_sync.remove_download(...)` — NOT `.removeDownload(...)`, which
  fails with `RPCException: unable to find method 'removedownload'`. So in
  `harvest.py`, `agent_call(dev, "remove_download", ...)` (snake_case), even
  though the JS name is `removeDownload`. Single-word exports (`download`,
  `open`, `curasin`) are unaffected.
- Rebuilding the agent: `krx_agent.js` is `frida-compile`d from `agent_src.js`
  (ESM `import Java from 'frida-java-bridge'`). Build from `scratch/` (has the
  npm deps): `cp agent_src.js scratch/ && cd scratch && ./node_modules/.bin/frida-compile agent_src.js -o ../krx_agent.js`.
  frida-compile requires the entrypoint to live inside the project root, hence
  the copy into `scratch/`.
- A method that reflects fine (shows up in `getDeclaredMethods()`) can still be
  un-callable on the wrapper frida binds from an interface getter — cast to the
  concrete class first, e.g. `Java.cast(sdk.getLibraryManager(), Java.use('com.amazon.kindle.krx.library.LibraryManager'))`.
