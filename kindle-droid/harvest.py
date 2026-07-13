"""Batch-harvest Kindle content keys for the whole library.

For each ASIN: (download if remote) -> open in reader via KRX so the app caches
the 16-byte content key -> dump native heap -> brute-force the key -> (optional)
repackage to a DRM-free EPUB.

Handles both delivered formats.  KFX (the common case): the key is the DRMION
AES content key; brute_one + repackage.py -> .kfx-zip -> calibre.  Legacy
Mobipocket/KF8 (some titles land as <asin>_EBOK.prc): the key is the crypto-
type-2 MOBI key, recovered from heap the same way (mobidrm.brute_mobi, since this
app version's account secrets are AES-GCM-wrapped and out of DeDRM's offline
reach) and stripped with vendored DeDRM (mobidrm.decrypt_with_key) -> calibre.

The reliable open-by-ASIN mechanism (see README "Whole-library harvest"):
  attach frida, call krx_agent open(asin), DETACH IMMEDIATELY (frida attached
  during the async load blocks the render), then the reader foregrounds + renders
  on its own and KRF caches the key.  Only then is the heap dump useful.

Usage:
  python harvest.py --list                      # print library inventory + status
  python harvest.py --asins B003JTHWKU,B0...    # harvest specific ASINs
  python harvest.py --all [--limit N]           # harvest everything missing a key
  python harvest.py --all --repackage           # also build EPUBs as we go
  python harvest.py -v --asins B0...            # -v/--verbose: log every command run

Env assumptions (see README): rooted 4KB-page AVD, frida-server running, host
frida venv active, `setenforce 0`.  krx_agent.js + dump_mem.js in cwd.
"""
import argparse, glob, os, shlex, subprocess, sys, threading, time
import numpy as np
import frida
from Crypto.Cipher import AES

import androidvoucher as A
from amazon.ion import simpleion
from mobidrm import brute_mobi, decrypt_with_key

PKG = "com.amazon.kindle"
DEVICE_FILES = f"/data/media/0/Android/data/{PKG}/files"
READER_ACT = "com.amazon.kcp.reader.StandAloneBookReaderActivity"
KEYS_TXT = "keys.txt"

# ------------------------------------------------------------- verbose logging
# `--verbose/-v` turns these on.  vcmd() logs every shell / subprocess / frida
# command as it runs (`+ …`, à la `set -x`); vlog() logs incidental detail
# (`[v] …`).  Both go to stderr so `--verbose` output stays separable from the
# normal per-book progress on stdout.
VERBOSE = False
_DIM, _RST = ("\033[2m", "\033[0m") if sys.stderr.isatty() else ("", "")

def vlog(msg):
    if VERBOSE:
        print(f"{_DIM}[v] {msg}{_RST}", file=sys.stderr, flush=True)

def vcmd(argv, note=""):
    if VERBOSE:
        line = argv if isinstance(argv, str) else " ".join(shlex.quote(str(a)) for a in argv)
        print(f"{_DIM}+ {line}{('   # ' + note) if note else ''}{_RST}",
              file=sys.stderr, flush=True)

def _summarize(text):
    """One-line preview of command output for verbose logging."""
    if text is None:
        return ""
    s = text.strip()
    first = s.splitlines()[0] if s else ""
    if len(first) > 120:
        first = first[:117] + "…"
    return f"{len(text)}B" + (f", first line: {first!r}" if first else "")

def run(argv, **kw):
    """subprocess.run wrapper that logs the command + timing when verbose."""
    vcmd(argv)
    t0 = time.time()
    r = subprocess.run(argv, **kw)
    if VERBOSE:
        rc = getattr(r, "returncode", 0)
        out = _summarize(getattr(r, "stdout", None))
        vlog(f"  ↳ {time.time()-t0:.2f}s"
             + (f" rc={rc}" if rc else "") + (f"  [{out}]" if out else ""))
    return r

# ------------------------------------------------------------------ adb helpers
def adb(*args, su=False, timeout=60):
    cmd = ["adb", "shell"]
    inner = " ".join(args)
    cmd.append(f"su -c '{inner}'" if su else inner)
    return run(cmd, capture_output=True, text=True, timeout=timeout).stdout

def setenforce_permissive():
    adb("setenforce 0", su=True)

def force_stop():
    adb("am force-stop " + PKG)

def launch_home():
    adb(f"monkey -p {PKG} -c android.intent.category.LAUNCHER 1")

def top_activity():
    out = adb("dumpsys activity activities")
    for line in out.splitlines():
        if "topResumedActivity" in line:
            line = line.strip()
            vlog(f"top activity: {line}")
            return line
    vlog("top activity: <none found>")
    return ""

# Content-file extensions we recognise.  Only KFX is harvestable by this tool
# (DRMION content keys live in CR!*.kfx); the app also delivers legacy
# Mobipocket/KF8 books as <asin>_EBOK.prc / .azw / .azw3, which use a different
# (older, separately-reversed) DRM scheme this tool doesn't handle.
_MOBI_EXTS = {"prc", "azw", "azw3", "mobi"}

def book_format(asin):
    """Classify the on-disk content in files/<asin>/ as 'kfx', 'mobi', or None
    (no recognised content file present — not downloaded, or some other format).
    Sidecars (.apnx/.phl/.asc/.db/.ser/.ast/.metadata) are ignored."""
    out = adb(f"ls {DEVICE_FILES}/{asin}/ 2>/dev/null", su=True)
    exts = {name.rsplit(".", 1)[-1].lower() for name in out.split() if "." in name}
    fmt = "kfx" if "kfx" in exts else "mobi" if exts & _MOBI_EXTS else None
    vlog(f"book_format({asin}) = {fmt!r} (exts={sorted(exts)})")
    return fmt

def pull_book(asin, dest_parent="files-4k"):
    """Pull the device book dir into files-4k/<asin>/ (needed for brute + repackage)."""
    dest = os.path.join(dest_parent, asin)
    os.makedirs(dest, exist_ok=True)
    # copy to a world-readable staging dir first (app dir is private)
    stage = f"/data/local/tmp/kh_{asin}"
    adb(f"rm -rf {stage}; cp -r {DEVICE_FILES}/{asin} {stage}; chmod -R 777 {stage}", su=True)
    run(["adb", "pull", "-a", stage + "/.", dest],
        capture_output=True, text=True, timeout=300)
    adb(f"rm -rf {stage}", su=True)
    vlog(f"pulled {asin} -> {dest}/ ({len(glob.glob(os.path.join(dest, '*')))} files)")
    return dest

# --------------------------------------------------------------- frida helpers
def kindle_pid(dev):
    return next((p.pid for p in dev.enumerate_processes() if p.name == "Kindle"), None)

TRANSIENT = (frida.ProcessNotRespondingError, frida.InvalidOperationError,
             frida.TransportError, frida.ProcessNotFoundError, frida.ServerNotRunningError)

def relaunch(dev):
    """Bring Kindle back up after a crash and give it time to init the SDK."""
    vlog("relaunch: setenforce 0 + launch home + wait for pid")
    setenforce_permissive()
    launch_home()
    wait_for(lambda: kindle_pid(dev) is not None, 40, label="Kindle pid")
    time.sleep(8)

def agent_call(dev, method, *margs, script_path="krx_agent.js", retries=3):
    """attach, call one krx_agent rpc, DETACH immediately (prompt detach is what
    unblocks the render), return result. Resilient to the flaky emulator killing
    the app mid-call: relaunch + retry on transient frida/process errors."""
    argstr = ", ".join(repr(a) for a in margs)
    for attempt in range(retries):
        pid = kindle_pid(dev)
        if pid is None:
            vlog(f"agent_call {method}: no Kindle pid, relaunching"); relaunch(dev); continue
        s = None
        try:
            vcmd(f"frida[pid={pid}] {method}({argstr})",
                 note=f"attach+rpc+detach, try {attempt+1}/{retries}")
            s = dev.attach(pid)
            js = s.create_script(open(script_path).read())
            js.load()
            r = getattr(js.exports_sync, method)(*margs)
            vlog(f"  ↳ {method} -> {r!r}")
            return r
        except TRANSIENT as e:
            print(f"    (transient {type(e).__name__} on {method}; relaunch+retry)")
            relaunch(dev)
        finally:
            if s is not None:
                vlog(f"  detach pid={pid}")
                try: s.detach()
                except Exception: pass
    raise RuntimeError(f"agent_call({method}) failed after {retries} tries")

def verify_and_dump(dev, asin, outpath, retries=3):
    """One attach: confirm the reader's current book == asin, and if so dump the
    heap. Returns bytes dumped, or -1 if the wrong/no book is loaded."""
    for attempt in range(retries):
        pid = kindle_pid(dev)
        if pid is None:
            vlog("verify_and_dump: no Kindle pid"); relaunch(dev); return -1  # caller re-opens
        s = None
        try:
            vcmd(f"frida[pid={pid}] curasin()", note="confirm target book is open")
            s = dev.attach(pid)
            chk = s.create_script(open("krx_agent.js").read()); chk.load()
            cur = chk.exports_sync.curasin()
            if cur != asin:
                vlog(f"  ↳ current book is {cur!r}, want {asin!r} — not dumping")
                return -1
            vcmd(f"frida[pid={pid}] dump_all() -> {outpath}", note="dump native heap")
            dmp = s.create_script(open("dump_mem.js").read())
            f = open(outpath, "wb"); state = {"off": 0, "regions": 0}; done = threading.Event()
            def on_message(msg, data):
                if msg.get("type") == "send":
                    if msg["payload"].get("done"): done.set(); return
                    if data: f.write(data); state["off"] += len(data); state["regions"] += 1
            dmp.on("message", on_message); dmp.load()
            dmp.exports_sync.dump_all(); done.wait(timeout=300); f.close()
            vlog(f"  ↳ dumped {state['off']} bytes in {state['regions']} region(s)")
            return state["off"]
        except TRANSIENT as e:
            print(f"    (transient {type(e).__name__} during dump; retry)")
            relaunch(dev); return -1
        finally:
            if s is not None:
                vlog(f"  detach pid={pid}")
                try: s.detach()
                except Exception: pass
    return -1

# ------------------------------------------------------------- key brute force
def test_pages_for(bookdir, want=5):
    found = []
    for f in sorted(glob.glob(os.path.join(bookdir, "CR!*.kfx"))):
        data = open(f, "rb").read()
        if not A.is_drmion(data):
            continue
        doc = simpleion.loads(data[8:-8], catalog=A._CATALOG, single_value=False)
        def walk(v):
            ann = [x.text for x in getattr(v, "ion_annotations", ())]
            tn = ann[0] if ann else ""
            if hasattr(v, "keys"):
                if "EncryptedPage" in tn:
                    ct = A._get(v, "cipher_text"); iv = A._get(v, "cipher_iv")
                    if ct is not None and iv is not None and len(bytes(ct)) % 16 == 0:
                        found.append((bytes(ct), bytes(iv)))
                for k in v: walk(v[k])
            elif isinstance(v, (list, tuple)):
                for x in v: walk(x)
        for v in doc: walk(v)
        if found:
            break
    return found[:want]

def pkcs7_ok(block):
    n = block[-1]
    return 1 <= n <= 16 and block[-n:] == bytes([n]) * n

def make_test(pages):
    """Known-plaintext-FREE test: the correct key CBC-decrypts every test page to
    valid PKCS7 padding.  Per page that's a ~1/256 fluke, so 3+ pages together
    give a ~10^-7..-12 false-positive rate — no book-specific header assumption
    (the DCC books start `005d..`, 1984 starts `CONT..`; both just need padding)."""
    def test(key):
        for ct, iv in pages:
            pt = AES.new(key, AES.MODE_CBC, iv).decrypt(ct)
            if not pkcs7_ok(pt):
                return False
        return True
    return test

def brute_one(heap_path, bookdir):
    """Return the 16-byte content key for the single book in bookdir, or None."""
    pages = test_pages_for(bookdir)
    vlog(f"brute_one: {len(pages)} test page(s) from {bookdir}")
    if len(pages) < 3:
        vlog("brute_one: <3 test pages, cannot brute — giving up")
        return None
    test = make_test(pages)
    buf = np.fromfile(heap_path, dtype=np.uint8)
    vlog(f"brute_one: scanning {len(buf)} heap bytes")
    for align in (16, 8, 4, 1):
        idx = np.arange(0, len(buf) - 15, align)
        W = np.stack([buf[idx + j] for j in range(16)], axis=1)
        srt = np.sort(W, axis=1)
        uniq = (np.diff(srt, axis=1) != 0).sum(axis=1) + 1
        printable = ((W >= 0x20) & (W < 0x7f)).sum(axis=1)
        cand = np.unique(W[(uniq >= 11) & (printable <= 11)], axis=0)
        vlog(f"brute_one: align={align:2d} -> {len(cand)} candidate window(s)")
        for row in cand:
            key = row.tobytes()
            if test(key):
                vlog(f"brute_one: key found at align={align}")
                return key
    return None

# ------------------------------------------------------------------- inventory
def pull_library_db():
    """Copy the app's kindle_library.db off-device and return the local path."""
    dbp = "/tmp/kh_library.db"
    adb(f"cp /data/data/{PKG}/databases/kindle_library.db /data/local/tmp/kl.db; "
        f"chmod 666 /data/local/tmp/kl.db", su=True)
    run(["adb", "pull", "/data/local/tmp/kl.db", dbp], capture_output=True)
    return dbp

def book_state(asin):
    """DB download state for one ASIN (LOCAL once fully downloaded, else REMOTE /
    a transient state), or None if the ASIN isn't in the library db."""
    import sqlite3
    con = sqlite3.connect(pull_library_db())
    row = con.execute("SELECT STATE FROM KindleContent WHERE ID LIKE ? "
                      "ORDER BY STATE='LOCAL' DESC LIMIT 1",
                      (f"%/{asin}/%",)).fetchone()
    con.close()
    st = row[0] if row else None
    vlog(f"book_state({asin}) = {st!r}")
    return st

# The only library entries this KFX harvester can handle are ordinary purchased
# books, TYPE == 'BT_EBOOK'.  Other TYPEs share the '%EBOOK%' shape but must be
# skipped (see book_type / is_harvestable):
#   BT_EBOOK_PDOC      personal documents ("Send to Kindle").  32-char base32 id
#                      (AMZNID0/<id>/4/, vs /0/ for store books).  The KRX
#                      getStoreManager().downloadBook() call is a NO-OP for these
#                      (returns ok, STATE never leaves REMOTE, nothing lands) --
#                      only a manual re-send from Amazon delivers them, and then
#                      the CR!*.kfx lands in the SHARED files/kindle/ dir, not in
#                      files/<asin>/.  They also don't show in the app's Books
#                      browse (separate Docs view).  So they can't be driven
#                      by-ASIN like store books -- skip them.
#   BT_EBOOK_NEWSPAPER periodical subscriptions;  BT_EBOOK_SAMPLE  free samples.
HARVESTABLE_TYPE = "BT_EBOOK"

def book_type(asin):
    """DB catalog TYPE for one ASIN (e.g. 'BT_EBOOK', 'BT_EBOOK_PDOC'), or None
    if the ASIN isn't in the library db."""
    import sqlite3
    con = sqlite3.connect(pull_library_db())
    row = con.execute("SELECT TYPE FROM KindleContent WHERE ID LIKE ? "
                      "ORDER BY STATE='LOCAL' DESC LIMIT 1",
                      (f"%/{asin}/%",)).fetchone()
    con.close()
    t = row[0] if row else None
    vlog(f"book_type({asin}) = {t!r}")
    return t

def is_harvestable(btype):
    """True for the TYPEs this KFX harvester can drive by-ASIN (store books only;
    personal docs / newspapers / samples are skipped -- see HARVESTABLE_TYPE)."""
    return btype == HARVESTABLE_TYPE

def load_inventory():
    """Return list of (asin, title, state, btype) for ebook-shaped entries, from a
    pulled library db.  Non-harvestable TYPEs (PDOC/NEWSPAPER/SAMPLE) are included
    so --list can show them, but callers should filter with is_harvestable()."""
    import sqlite3
    con = sqlite3.connect(pull_library_db())
    rows = con.execute(
        "SELECT ID, TITLE, STATE, TYPE FROM KindleContent "
        "WHERE TYPE LIKE '%EBOOK%' ORDER BY STATE, TITLE").fetchall()
    con.close()
    out, seen = [], {}
    for cid, title, state, btype in rows:
        parts = cid.split("/")
        asin = parts[1] if len(parts) > 1 else cid
        # an ASIN can appear as several rows (e.g. a stale FAILED_RETRYABLE +
        # the real LOCAL/REMOTE one); keep one, preferring a downloaded state.
        if asin in seen:
            if state == "LOCAL" and out[seen[asin]][2] != "LOCAL":
                out[seen[asin]] = (asin, title, state, btype)
            continue
        seen[asin] = len(out)
        out.append((asin, title, state, btype))
    n_harv = sum(is_harvestable(b) for _, _, _, b in out)
    vlog(f"inventory: {len(out)} ebook entries ({n_harv} harvestable BT_EBOOK, "
         f"{len(out)-n_harv} pdoc/newspaper/sample skipped, "
         f"{sum(s=='LOCAL' for _,_,s,_ in out)} LOCAL) from {len(rows)} db rows")
    return out

def loaded_keys():
    if not os.path.exists(KEYS_TXT):
        return {}
    d = {}
    for line in open(KEYS_TXT):
        p = line.split()
        if len(p) == 2:
            d[p[0]] = p[1]
    return d

# ---------------------------------------------------------------- per-book job
def wait_for(pred, timeout, interval=2.0, label=None):
    if label:
        vlog(f"wait_for {label} (<= {timeout:.0f}s)")
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            if label:
                vlog(f"  ↳ {label} satisfied after {time.time()-t0:.1f}s")
            return True
        time.sleep(interval)
    if label:
        vlog(f"  ↳ {label} TIMED OUT after {timeout:.0f}s")
    return False

def ensure_app_home(dev):
    """Make sure Kindle is running; if a reader is showing, back out to Home.

    We deliberately DON'T force-stop: a cold launch auto-reopens the last-read
    book, and that races with our open() (leaving the wrong book rendered)."""
    if kindle_pid(dev) is None:
        vlog("ensure_app_home: Kindle not running, launching")
        launch_home()
        wait_for(lambda: kindle_pid(dev) is not None, 30, label="Kindle pid")
        time.sleep(6)   # app init so the KRX SDK is live
    # back out of any open reader so open() starts from a clean state
    for _ in range(3):
        if READER_ACT not in top_activity():
            break
        adb("input keyevent KEYCODE_BACK")
        time.sleep(1.5)

def harvest_one(dev, asin, render_wait=15.0, dl_timeout=300, open_tries=3):
    """Download+open+dump+brute one book. Return hex key or None."""
    # 0. bail on non-harvestable TYPEs before touching the app.  --all already
    #    filters these out, but an explicit --asins <pdoc> would otherwise sink a
    #    full dl_timeout waiting for a download that never comes: downloadBook()
    #    is a no-op for personal docs, and even a manual re-send lands the file in
    #    the shared files/kindle/ dir (not files/<asin>/) under an unrelated name.
    btype = book_type(asin)
    if btype is not None and not is_harvestable(btype):
        kind = btype.replace("BT_EBOOK_", "").lower()
        print(f"  [{asin}] TYPE is {btype} ({kind}, not a store KFX book) — "
              f"skipping; this harvester only handles {HARVESTABLE_TYPE}")
        return None

    setenforce_permissive()
    ensure_app_home(dev)

    # 1. download if the content isn't on disk yet.  Poll the library db's STATE
    #    (LOCAL == fully downloaded) rather than the content dir: non-KFX formats
    #    never produce a .kfx, so a kfx-only poll would spuriously time out on
    #    them (and we want to reach the format check below to skip them cleanly).
    if book_format(asin) is None:
        print(f"  [{asin}] downloading…")
        agent_call(dev, "download", asin)
        if not wait_for(lambda: book_state(asin) == "LOCAL", dl_timeout, 3.0,
                        label=f"{asin} download (STATE=LOCAL)"):
            print(f"  [{asin}] download did not land in {dl_timeout}s — skipping")
            return None

    # 1b. route by on-disk format.  KFX carries the DRMION content key (brute_one
    #     + repackage.py); the app delivers some titles as legacy Mobipocket/KF8
    #     (<asin>_EBOK.prc/.azw) instead, whose crypto-type-2 MOBI key we recover
    #     the same way from heap (brute_mobi) and strip with vendored DeDRM.
    #     Both open+dump+brute identically below; only the brute + repackage differ.
    fmt = book_format(asin)
    if fmt not in ("kfx", "mobi"):
        print(f"  [{asin}] format is {fmt or 'unknown'!r} — skipping; this "
              f"harvester handles KFX and Mobipocket/KF8")
        return None

    # 2. open, wait for render, then verify-the-right-book + dump in ONE attach
    #    (fewer attaches = less chance of tripping the flaky app's anti-debug).
    heap = f"/tmp/kh_{asin}.bin"
    dumped = -1
    for attempt in range(open_tries):
        print(f"  [{asin}] opening (try {attempt+1}/{open_tries})…")
        r = agent_call(dev, "open", asin)          # attach/open/detach
        if not r.get("ok"):
            print(f"  [{asin}] open failed: {r}"); ensure_app_home(dev); continue
        wait_for(lambda: READER_ACT in top_activity(), 40, label="reader activity")
        time.sleep(render_wait)                     # first draw + key caching
        dumped = verify_and_dump(dev, asin, heap)   # curasin==asin? then dump
        if dumped > 0:
            break
        print(f"  [{asin}] target book not confirmed open; re-opening")
        ensure_app_home(dev)
    if dumped <= 0:
        print(f"  [{asin}] could not open+dump target book — skipping"); return None
    print(f"  [{asin}] dumped {dumped/1e6:.0f} MB; brute-forcing…")

    # 3. pull content (for brute + repackage) and brute the key
    bookdir = pull_book(asin)
    if fmt == "kfx":
        key = brute_one(heap, bookdir)
    else:                                    # mobi: brute the crypto-type-2 key
        prc = mobi_content_file(bookdir)
        key = brute_mobi(heap, prc, log=vlog) if prc else None
    try: os.remove(heap)
    except OSError: pass
    if key is None:
        print(f"  [{asin}] KEY NOT FOUND"); return None
    hexkey = key.hex()
    print(f"  [{asin}] *** KEY {hexkey} ***")
    return hexkey

def mobi_content_file(bookdir):
    """The single Mobipocket/KF8 content file in a pulled book dir, or None."""
    for f in sorted(glob.glob(os.path.join(bookdir, "*"))):
        if f.rsplit(".", 1)[-1].lower() in _MOBI_EXTS:
            return f
    return None

def repackage(asin, hexkey):
    """DRM-strip a recovered book to EPUB.  Dispatches on the pulled content's
    format: KFX -> repackage.py (.kfx-zip) -> calibre; Mobipocket/KF8 ->
    mobidrm.decrypt_with_key (.mobi) -> calibre."""
    bookdir = f"files-4k/{asin}"
    os.makedirs("out", exist_ok=True)
    epub = f"out/{asin}.epub"
    prc = mobi_content_file(bookdir)
    if prc:                                  # Mobipocket/KF8
        stripped = f"out/{asin}.mobi"
        decrypt_with_key(prc, bytes.fromhex(hexkey), stripped)
        run(["ebook-convert", stripped, epub], check=True)
    else:                                    # KFX
        kfxzip = f"out/{asin}.kfx-zip"
        run([sys.executable, "repackage.py", bookdir, hexkey, kfxzip], check=True)
        run(["ebook-convert", kfxzip, epub], check=True)
    return epub

# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--asins", help="comma-separated ASINs")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repackage", action="store_true")
    ap.add_argument("--render-wait", type=float, default=8.0)
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="log every shell/frida command as it runs, plus extra detail (to stderr)")
    args = ap.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    have = loaded_keys()

    if args.list:
        inv = load_inventory()
        for asin, title, state, btype in inv:
            mark = ("KEY" if asin in have else
                    "   " if is_harvestable(btype) else "SKP")
            # tag non-harvestable rows with their TYPE (PDOC/NEWSPAPER/SAMPLE)
            tag = "" if is_harvestable(btype) else f" [{btype.replace('BT_EBOOK_','')}]"
            print(f"{mark} {asin} {state:6} {title[:60]}{tag}")
        n_harv = sum(is_harvestable(b) for _, _, _, b in inv)
        print(f"\n{len(inv)} ebook entries ({n_harv} harvestable); "
              f"{len(have)} keys recovered")
        return

    if args.asins:
        targets = args.asins.split(",")
    elif args.all:
        # only real store books (BT_EBOOK) are drivable by-ASIN; skip personal
        # docs / newspapers / samples up front so we don't waste a 300s download
        # timeout on each of them (they never deliver via downloadBook()).
        targets = [a for a, t, s, bt in load_inventory()
                   if a not in have and is_harvestable(bt)]
    else:
        ap.error("specify --list, --asins, or --all")

    if args.limit:
        targets = targets[: args.limit]
    print(f"harvesting {len(targets)} book(s)")
    vlog(f"targets: {targets}")

    dev = frida.get_usb_device()
    vlog(f"frida device: {dev}")
    for i, asin in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {asin}")
        if asin in have:
            print("  already have key"); continue
        key = None
        for book_try in range(2):            # one whole-book retry for flaky crashes
            try:
                key = harvest_one(dev, asin, render_wait=args.render_wait)
                break
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                relaunch(dev)
        if key:
            with open(KEYS_TXT, "a") as f:
                f.write(f"{asin} {key}\n")
            have[asin] = key
            if args.repackage:
                try:
                    print("  ->", repackage(asin, key))
                except Exception as e:
                    print(f"  repackage failed: {e}")

if __name__ == "__main__":
    main()
