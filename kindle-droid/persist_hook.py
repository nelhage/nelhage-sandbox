"""persist_hook.py — install a native hook that SURVIVES Frida detach, to beat
the KRF anti-tamper render-block.

The block (see FRIDA-DETECTION.md): while a Frida agent is attached, KRF's
createBook detects the agent's /proc/self/maps footprint and sleeps forever, so
the book never renders and the content key is never derived.  Idea: use Frida to
dlopen our own hook.so and inline-patch a target function to jump into it, then
DETACH — with Frida gone the check passes, the book renders, and our resident
hook still fires (it lives in our file-backed .so + a COW'd KRF text page, both
owned by the process, not by Frida).

Milestones (this file covers M1 + M2; M3 = real DRM-AES key capture is later):
  smoke          dlopen hook.so from the app uid (settles linker-namespace/DAC).
  m1 [asin]      install no-op logging hook on KRF storage AES (base+0x3302714),
                 trigger a fresh download, DETACH, pull loot.bin — captures land
                 with Frida ABSENT => persistence + trampoline proven.
  m2 [asin]      install the (benign) hook, open the book, DETACH, then poll
                 WITHOUT Frida for the reader activity => the resident footprint
                 does NOT re-trip the anti-tamper (make-or-break).
  build          compile hook.so with the NDK.
  deploy         push hook.so to the device.

Usage:
  python persist_hook.py build
  python persist_hook.py deploy
  python persist_hook.py smoke
  python persist_hook.py m1 B00KVI76ZS
  python persist_hook.py m2 B01LXW2IUQ
"""
import argparse, glob, os, struct, subprocess, sys, time
import frida

import harvest as H

STORAGE_AES_OFF = 0x3302714          # KRF storage AES_set_encrypt_key(key=x0,bits=x1)
SO_LOCAL = os.path.join(os.path.dirname(__file__), "hook.so")
SO_TMP = "/data/local/tmp/hook.so"
SO_APPDIR = f"/data/data/{H.PKG}/hook.so"
LOOT = "/data/local/tmp/loot.bin"


# ---- build / deploy --------------------------------------------------------
def find_clang():
    ndk = os.environ.get("ANDROID_NDK_ROOT") or os.environ.get("ANDROID_NDK_HOME")
    if not ndk:
        sys.exit("ANDROID_NDK_ROOT not set (add the NDK to the kindle-droid devShell)")
    # Search recursively: the NDK may sit directly at $ANDROID_NDK_ROOT or nested
    # (e.g. .../libexec/android-sdk/ndk/<ver>/) depending on how it was installed.
    hits = glob.glob(f"{ndk}/**/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android24-clang",
                     recursive=True) or \
           glob.glob(f"{ndk}/toolchains/llvm/prebuilt/*/bin/aarch64-linux-android24-clang")
    if not hits:
        sys.exit(f"no aarch64-linux-android24-clang under {ndk}")
    return hits[0]


def build():
    cc = find_clang()
    src = os.path.join(os.path.dirname(__file__), "hook", "hook.c")
    cmd = [cc, "--target=aarch64-linux-android24", "-shared", "-fPIC", "-O2",
           "-fvisibility=hidden", "-Wl,-soname,hook.so", "-Wl,-z,noexecstack",
           "-llog", "-o", SO_LOCAL, src]
    print("[build]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[build] wrote {SO_LOCAL} ({os.path.getsize(SO_LOCAL)} bytes)")


def deploy(appdir=False):
    if not os.path.exists(SO_LOCAL):
        build()
    subprocess.run(["adb", "push", SO_LOCAL, SO_TMP], check=True)
    H.adb(f"chmod 755 {SO_TMP}", su=True)
    # World-writable so the app uid (not shell) can create loot.bin here. DAC
    # still applies under setenforce 0; /data/local/tmp is shell-owned, so the
    # app can't write a new file unless the dir is o+w.
    H.adb("chmod 777 /data/local/tmp", su=True)
    path = SO_TMP
    if appdir:
        # Copy into the app's own data dir, owned by the app uid — always inside
        # the app's default linker namespace (fallback if /data/local/tmp is
        # refused by the linker namespace / SELinux).
        owner = H.adb(f"stat -c %u:%g /data/data/{H.PKG}", su=True).strip()
        H.adb(f"cp {SO_TMP} {SO_APPDIR} && chown {owner} {SO_APPDIR} && chmod 755 {SO_APPDIR}", su=True)
        path = SO_APPDIR
    print(f"[deploy] hook.so at {path}")
    return path


# ---- frida session helpers -------------------------------------------------
def attach(dev, retries=5):
    # frida's kindle_pid matches the process *name* "Kindle", which only holds
    # while the app is foreground; fall back to adb pidof (robust either way).
    last = None
    for i in range(retries):
        pid = H.kindle_pid(dev)
        if pid is None:
            out = H.adb(f"pidof {H.PKG}").strip()
            pid = int(out.split()[0]) if out else None
        if pid is None:
            raise RuntimeError("no Kindle pid (is the app running?)")
        try:
            return dev.attach(pid)
        except (frida.NotSupportedError, *H.TRANSIENT) as e:
            # frida injection can race right after app launch; back off + retry.
            last = e
            print(f"[attach] transient {type(e).__name__}: {e} (retry {i+1}/{retries})")
            time.sleep(2)
    raise last


def load(s, name):
    js = s.create_script(open(name).read())
    js.on("message", lambda m, d: print(f"  [{name}]", m.get("description") or m.get("payload"))
          if m.get("type") == "error" or m.get("payload") else None)
    js.load()
    return js


# Per-format native decrypt/key-setup routine to hook, and the capture config.
#   kfx   -> KRF (JNI) AES_set_encrypt_key @0x3302714, key=x0, len=x1 bits
#   mobi  -> libKRF.so PC1 @vaddr 0x37a548, key ptr=x0 (ctx wkey), fixed 16 bytes
#   topaz -> libKRF.so cipher decrypt @vaddr 0x3f6790, key = libc++ std::string @x0
HOOK_TARGET = {
    "kfx":   (STORAGE_AES_OFF, {}),
    "mobi":  (0x37a548, {"module": "libKRF.so", "keyReg": 0, "lenMode": 0, "lenVal": 16}),
    "topaz": (0x3f6790, {"module": "libKRF.so", "keyReg": 0, "lenMode": 3, "lenVal": 0}),
}


def install_persistent(dev, off, path, opts=None, retries=5):
    """Attach, install the hook, detach — retried through this emulator's frequent
    transient frida/app failures. install() is idempotent, so a retry after a
    partial patch is safe."""
    last = None
    for i in range(retries):
        s = attach(dev)
        try:
            inst = load(s, "install_hook.js")
            res = inst.exports_sync.install(path, hex(off), opts or {})
            if res.get("ok"):
                print("[install] ok:", res)
                return res
            print("[install] failed:", res)
            last = RuntimeError(res.get("err"))
        except (frida.InvalidOperationError, frida.NotSupportedError, *H.TRANSIENT) as e:
            print(f"[install] transient {type(e).__name__}: {e} (retry {i+1}/{retries})")
            last = e
            H.relaunch(dev)
        finally:
            try: s.detach()
            except Exception: pass
        time.sleep(1)
    raise last or RuntimeError("install failed")


def reader_focused():
    out = H.adb("dumpsys window")
    for ln in out.splitlines():
        if "mCurrentFocus" in ln and H.READER_ACT in ln:
            return True
    return H.top_activity() == H.READER_ACT


def clear_loot():
    H.adb(f"rm -f {LOOT}", su=True)


def pull_loot(dest):
    # loot.bin is written by the app uid; copy world-readable (root) then pull.
    exists = H.adb(f"test -f {LOOT} && echo yes", su=True).strip()
    if exists != "yes":
        print(f"[loot] {LOOT} does not exist on device (hook wrote nothing)")
        return None
    H.adb(f"cp {LOOT} /data/local/tmp/loot_pull.bin && chmod 666 /data/local/tmp/loot_pull.bin", su=True)
    subprocess.run(["adb", "pull", "/data/local/tmp/loot_pull.bin", dest],
                   check=True, capture_output=True)
    return dest


def parse_loot(path):
    """Yield captured keys from the length-prefixed { u32 len; bytes } records."""
    with open(path, "rb") as f:
        data = f.read()
    out, i = [], 0
    while i + 4 <= len(data):
        (n,) = struct.unpack_from("<I", data, i); i += 4
        if n == 0 or n > 64 or i + n > len(data):
            break
        out.append(data[i:i + n]); i += n
    return out


# ---- subcommands -----------------------------------------------------------
def cmd_smoke(dev, args):
    """dlopen hook.so from the app process. Tries /data/local/tmp, then the app
    data dir. Prints the loaded base + resolved exports, or the link error."""
    for appdir in (False, True):
        path = deploy(appdir=appdir)
        s = attach(dev)
        try:
            inst = load(s, "install_hook.js")
            try:
                info = inst.exports_sync.smoke(path)
                print(f"[smoke] OK from {path}:")
                print(f"        base={info['base']} hook_fn={info['hook_fn']} tramp_slot={info['tramp_slot']}")
                if not info["hook_fn"] or not info["tramp_slot"]:
                    print("[smoke] WARNING: exports missing — check the build's -fvisibility/default symbols")
                return
            except Exception as e:
                print(f"[smoke] Module.load FAILED from {path}: {e}")
        finally:
            try: s.detach()
            except Exception: pass
    print("[smoke] both paths failed — investigate linker namespace / SELinux for dlopen")


def cmd_m1(dev, args):
    """Prove the hook fires with Frida DETACHED. Install on storage AES, trigger a
    fresh download, detach, then pull loot.bin (written entirely post-detach)."""
    asin = args.asin
    path = deploy(appdir=args.appdir)
    clear_loot()
    s = attach(dev)
    try:
        krx = load(s, "krx_agent.js")
        inst = load(s, "install_hook.js")
        before = inst.exports_sync.peek(hex(STORAGE_AES_OFF), 16)
        res = inst.exports_sync.install(path, hex(STORAGE_AES_OFF))
        print("[m1] install:", res)
        if not res.get("ok"):
            print("[m1] install failed; aborting"); return
        after = inst.exports_sync.peek(hex(STORAGE_AES_OFF), 16)
        print(f"[m1] target bytes  before={before}  after={after}")
        # Trigger a fresh parse (storage AES fires at download-time), then DETACH
        # immediately so all captures happen with Frida gone.
        print(f"[m1] removeDownload + download {asin} (async) ...")
        krx.exports_sync.remove_download(asin)
        krx.exports_sync.download(asin)
    finally:
        try: s.detach()
        except Exception: pass
    print("[m1] DETACHED. waiting for the download to run + hook to capture ...")
    time.sleep(args.wait)
    dest = os.path.join(os.path.dirname(__file__), "loot.bin")
    keys = parse_loot(dest) if pull_loot(dest) else []
    print(f"[m1] captured {len(keys)} key(s) into {dest} (Frida was DETACHED):")
    for k in keys:
        print(f"       {len(k):2d}B  {k.hex()}")
    print("[m1] PASS — hook survived detach" if keys else
          "[m1] FAIL — no captures (hook didn't fire post-detach)")


def cmd_m2(dev, args):
    """Make-or-break: with the .so + inline patch resident and Frida DETACHED,
    does the book actually render? Verified WITHOUT Frida via the reader window."""
    asin = args.asin
    path = deploy(appdir=args.appdir)
    # Kindle must be FOREGROUND for open()'s currentActivity() to resolve — a
    # backgrounded app (launcher foreground) yields activity=None and open() no-ops.
    _wake()
    H.launch_home()
    if not H.wait_for(lambda: H.PKG in (H.top_activity() or ""), 30, 1.0, label="Kindle foreground"):
        print(f"[m2] Kindle never came foreground (top={H.top_activity()!r}); aborting"); return
    _wake()
    if args.no_install:
        print("[m2] BASELINE: no hook installed")
    else:
        # Install the persistent hook in its own attach; it survives the detach.
        install_persistent(dev, STORAGE_AES_OFF, path)
    # Trigger the render via harvest's robust attach->open->detach (with retries);
    # the resident hook is already in place.  Prompt detach is what unblocks it.
    _wake()
    print(f"[m2] open({asin}) + detach via agent_call ...")
    print("[m2] open ->", H.agent_call(dev, "open", asin))
    _wake()
    print("[m2] DETACHED. polling for the reader window (no Frida attached) ...")
    ok = H.wait_for(reader_focused, args.wait, 1.0, label="reader window focused")
    if ok:
        print(f"[m2] PASS — {H.READER_ACT} reached with hook resident: anti-tamper bypassed")
    else:
        print(f"[m2] FAIL — still not rendering after {args.wait}s (top={H.top_activity()!r})")


def cmd_m3(dev, args):
    """Locate the DRM content-AES the direct way: install the persistent hook on
    KRF's AES key-setup (0x3302714 — set_decrypt_key routes through it too),
    render a VOUCHER book UNBLOCKED (Frida detached), and test every captured key
    against the book's DRMION pages. A hit proves the content decrypt IS the KRF
    AES, only ever exercised at render (which was always blocked before)."""
    import harvest  # for A (kfxdrm) via test_pages_for/make_test
    asin = args.asin
    path = deploy(appdir=args.appdir)
    _wake(); H.launch_home()
    H.wait_for(lambda: H.PKG in (H.top_activity() or ""), 30, 1.0, label="Kindle foreground")

    # Ensure the book is downloaded with DRMION content.
    if H.book_format(asin) != "kfx":
        print(f"[m3] {asin} not local as kfx; downloading ...")
        H.agent_call(dev, "download", asin)
        if not H.wait_for(lambda: H.book_format(asin) == "kfx", 90, 2.0, label="download kfx"):
            print(f"[m3] download did not land kfx (format={H.book_format(asin)}); aborting"); return
    bookdir = H.pull_book(asin)
    pages = H.test_pages_for(bookdir)
    print(f"[m3] {len(pages)} DRMION test page(s) from {bookdir}")
    if len(pages) < 3:
        print("[m3] <3 test pages — can't validate; aborting"); return
    test = H.make_test(pages)
    # Sanity: does the stored key still work (i.e. content wasn't re-vouchered)?
    known = None
    for line in open(os.path.join(os.path.dirname(__file__), "keys.txt"), errors="ignore"):
        if line.startswith(asin):
            known = bytes.fromhex(line.split()[1])
    print(f"[m3] stored key {'VALID' if (known and test(known)) else 'stale/absent'} for current on-disk content")

    clear_loot()
    install_persistent(dev, STORAGE_AES_OFF, path)
    _wake()
    print(f"[m3] open({asin}) + detach; rendering unblocked ...")
    print("[m3] open ->", H.agent_call(dev, "open", asin))
    _wake()
    H.wait_for(reader_focused, 40, 1.0, label="reader focused")
    # Turn pages to force fresh content-page decrypts.
    for _ in range(int(args.pages)):
        time.sleep(1.5)
        H.adb("input keyevent 22")  # DPAD_RIGHT → next page
    time.sleep(args.wait)

    dest = os.path.join(os.path.dirname(__file__), "loot.bin")
    keys = parse_loot(dest) if pull_loot(dest) else []
    uniq16 = list(dict.fromkeys(k for k in keys if len(k) == 16))
    print(f"[m3] {len(keys)} captures, {len(uniq16)} unique 16-byte keys; testing vs DRMION ...")
    for k in uniq16:
        if test(k):
            print(f"\n[m3] *** DRM CONTENT KEY CAPTURED via 0x3302714: {k.hex()} ***")
            print("[m3] => the DRM content-AES IS the KRF AES; it only runs at render.")
            return
    print("\n[m3] no captured key decrypts the DRMION.")
    print(f"[m3]   (16-byte keys seen: {[k.hex() for k in uniq16]})")
    print("[m3] => content decrypt does NOT route through 0x3302714; DRM AES is a separate impl.")


def _swap_pairs(b):
    """Swap each adjacent byte pair — PC1 stores its key as 8 little-endian u16
    words, so the in-memory wkey is the found_key with each 2-byte pair swapped."""
    return bytes(b[i ^ 1] for i in range(len(b) & ~1))


def _format_oracle(fmt, bookdir):
    """Return (check(captured)->canonical_key_or_None, note): given a raw captured
    key, does it (in some canonical framing) decrypt this format's content, and if
    so what is the content key?  KFX→DRMION PKCS7, MOBI→PC1 full-record (try the
    key and its pair-swap), Topaz→8-byte stream-cipher zlib oracle."""
    import glob as _glob
    if fmt == "kfx":
        pages = H.test_pages_for(bookdir)
        if len(pages) < 3:
            return None, "too few DRMION test pages"
        t = H.make_test(pages)
        return (lambda k: k if (len(k) == 16 and t(k)) else None), f"{len(pages)} DRMION pages"
    prc = next(iter(_glob.glob(os.path.join(bookdir, "*_EBOK.prc"))
                    or _glob.glob(os.path.join(bookdir, "*.prc"))), None)
    if not prc:
        return None, "no .prc found"
    if fmt == "mobi":
        from mobidrm.mobidedrm import MobiBook
        from mobidrm.brute import make_full_test
        t = make_full_test(MobiBook(prc))

        def check(k):
            if len(k) != 16:
                return None
            for cand in (k, _swap_pairs(k)):   # PC1 wkey is byte-pair-swapped
                if t(cand):
                    return cand
            return None
        return check, f"PC1 oracle on {os.path.basename(prc)}"
    if fmt == "topaz":
        from topazdrm.brute import oracle_record, _try_key
        orc = oracle_record(prc)
        return (lambda k: k if _try_key(k, orc) else None), f"Topaz oracle on {os.path.basename(prc)}"
    return None, f"unknown format {fmt}"


def cmd_keyscan(dev, args):
    """Render a book with the 0x3302714 hook and test EVERY captured AES key-setup
    against the book's content with its format-appropriate oracle — answering
    whether Mobipocket (PC1) and Topaz (custom stream cipher) route their content
    decrypt through the same KRF AES, or a different routine."""
    asin = args.asin
    path = deploy(appdir=args.appdir)
    _wake(); H.launch_home()
    H.wait_for(lambda: H.PKG in (H.top_activity() or ""), 30, 1.0, label="Kindle foreground")

    fmt = H.book_format(asin)
    if fmt is None:
        print(f"[keyscan] {asin} not local; downloading ...")
        H.agent_call(dev, "download", asin)
        H.wait_for(lambda: H.book_format(asin) is not None, 90, 2.0, label="download")
        fmt = H.book_format(asin)
    bookdir = H.pull_book(asin)
    test, note = _format_oracle(fmt, bookdir)
    off, opts = HOOK_TARGET.get(fmt, (None, None))
    tgt = opts.get("module", "KRF(JNI)") if opts else "?"
    print(f"[keyscan] {asin} format={fmt}; hooking {tgt}+{hex(off) if off else '?'}; oracle: {note}")
    if test is None or off is None:
        print("[keyscan] no usable oracle/target; aborting"); return

    clear_loot()
    install_persistent(dev, off, path, opts=opts)
    _wake()
    print(f"[keyscan] open({asin}) + detach; rendering unblocked ...")
    print("[keyscan] open ->", H.agent_call(dev, "open", asin))
    _wake()
    H.wait_for(reader_focused, 40, 1.0, label="reader focused")
    for _ in range(int(args.pages)):
        time.sleep(1.5)
        H.adb("input keyevent 22")
    time.sleep(args.wait)

    dest = os.path.join(os.path.dirname(__file__), "loot.bin")
    keys = parse_loot(dest) if pull_loot(dest) else []
    uniq = list(dict.fromkeys(keys))
    from collections import Counter
    lens = Counter(len(k) for k in uniq)
    print(f"[keyscan] {len(keys)} captures, {len(uniq)} unique keys; lengths={dict(lens)}")
    for k in uniq:
        canon = test(k)
        if canon:
            print(f"\n[keyscan] *** {fmt} CONTENT KEY RECOVERED: {canon.hex()} "
                  f"(from captured {k.hex()}) ***")
            return canon.hex()
    print(f"\n[keyscan] NO captured key decrypts the {fmt} content at {tgt}+{hex(off)} "
          f"(the hook fired {len(keys)}x but none validate).")
    return None


def _wake():
    H.adb("input keyevent KEYCODE_WAKEUP")
    H.adb("svc power stayon true")
    H.adb("wm dismiss-keyguard")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    dp = sub.add_parser("deploy"); dp.add_argument("--appdir", action="store_true")
    for name in ("smoke", "m1", "m2", "m3", "keyscan"):
        p = sub.add_parser(name)
        default_asin = "B01LXW2IUQ" if name == "m3" else "B00KVI76ZS"
        p.add_argument("asin", nargs="?", default=default_asin)
        p.add_argument("--appdir", action="store_true", help="load from the app data dir")
        p.add_argument("--no-install", action="store_true", help="(m2) baseline: open+detach with NO hook")
        p.add_argument("--key", help="(keyscan) known content key hex to search for (else keys.txt)")
        p.add_argument("--pages", type=int, default=6, help="page-turns to force content decrypts")
        p.add_argument("--wait", type=float, default=(25.0 if name == "m1" else 40.0))
    args = ap.parse_args()
    H.VERBOSE = True

    if args.cmd == "build":
        build(); return
    if args.cmd == "deploy":
        deploy(appdir=args.appdir); return

    H.setenforce_permissive()
    dev = frida.get_usb_device()
    print(f"device={dev}")
    {"smoke": cmd_smoke, "m1": cmd_m1, "m2": cmd_m2, "m3": cmd_m3,
     "keyscan": cmd_keyscan}[args.cmd](dev, args)


if __name__ == "__main__":
    main()
