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

Two paths to get the key into (and out of) the native heap:

  FAST PATH (KFX, the common case): the content key is derived and cached at
  DOWNLOAD time, not at render time — so for a freshly-downloaded book we can
  skip opening it entirely: just dump the scudo heap and brute (align=16 only,
  since KFX keys are 16-aligned).  download -> dump -> brute, ~2 attaches, no
  render wait.

  FALLBACK (open-by-ASIN, see README "Whole-library harvest"): when the key
  isn't already resident (e.g. a book that was LOCAL from a prior app session,
  so nothing this process derived it), actually render it: attach frida, call
  krx_agent open(asin), DETACH IMMEDIATELY (frida attached during the async load
  blocks the render), then the reader foregrounds + renders on its own and KRF
  caches the key; dump the delta and brute.  Mobipocket/KF8 always uses this
  path (its brute is minutes-long, not worth running speculatively).

Usage:
  python harvest.py --list                      # print library inventory + status
  python harvest.py --asins B003JTHWKU,B0...    # harvest specific ASINs
  python harvest.py --all [--limit N]           # harvest everything missing a key
                                                #   (ASINs in ./SKIP are excluded)
  python harvest.py --all --repackage           # also build EPUBs as we go
  python harvest.py --offline --all --repackage # no device: build EPUBs from
                                                #   every already-harvested key
  python harvest.py -v --asins B0...            # -v/--verbose: log every command run

Env assumptions (see README): rooted 4KB-page AVD, frida-server running, host
frida venv active, `setenforce 0`.  krx_agent.js + dump_mem.js in cwd.
"""
import argparse, glob, os, shlex, shutil, subprocess, sys, threading, time
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
SKIP_TXT = "SKIP"                           # one ASIN per line; excluded from --all
# Sentinel "key" for a DRM-free (Mobipocket crypto type 0) book: there's no key
# to recover, so harvest_one returns this and repackage() converts the raw .prc.
# Stored in keys.txt like a real key so re-runs skip re-harvesting it.
NODRM = "nodrm"
LIBRARY_DB = "files-4k/kindle_library.db"   # cached copy of the app's library db

# `--offline` never touches the device: it works off the cached library db and
# the already-pulled files-4k/<asin>/ dirs, so --list / --repackage keep working
# with no AVD.  harvest_one() short-circuits to None under it (see there).
OFFLINE = False

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

# Content-file extensions that carry the actual book.  A .prc / .azw / .azw3
# extension is NOT enough to know the format: the app delivers both legacy
# Mobipocket/KF8 (PalmDB "BOOKMOBI", crypto type 2 — handled by mobidrm) AND
# Topaz ("TPZ0" magic — a wholly different, unsupported DRM) under a .prc name.
# So we sniff the file magic, not just the extension.
_MOBI_EXTS = {"prc", "azw", "azw3", "mobi"}

def book_format(asin):
    """Classify the on-disk content in files/<asin>/ as one of:
      'kfx'   — KFX container (DRMION content key; brute_one + repackage.py)
      'mobi'  — Mobipocket/KF8, PalmDB 'BOOKMOBI' (brute_mobi + mobidrm)
      'topaz' — Topaz 'TPZ0' container (a .prc/.azw, but NOT Mobipocket; this
                tool has no Topaz DRM support, so callers skip it cleanly)
      None    — no recognised content file (not downloaded, or something else)
    Sidecars (.apnx/.phl/.asc/.db/.ser/.ast/.metadata) are ignored.  For the
    ambiguous mobi-ish extensions we read the file's first bytes to tell a real
    Mobipocket from a Topaz book (both ship as <asin>_EBOK.prc)."""
    d = f"{DEVICE_FILES}/{asin}"
    names = adb(f"ls {d}/ 2>/dev/null", su=True).split()
    exts = {n.rsplit(".", 1)[-1].lower() for n in names if "." in n}
    if "kfx" in exts:
        fmt = "kfx"
    elif exts & _MOBI_EXTS:
        cf = next((n for n in names if n.rsplit(".", 1)[-1].lower() in _MOBI_EXTS), None)
        # Topaz starts with 'TPZ0'; Mobipocket has 'BOOKMOBI' at offset 0x3C.
        hexs = adb(f"head -c 64 {shlex.quote(d + '/' + cf)} | xxd -p", su=True) if cf else ""
        head = bytes.fromhex("".join(hexs.split())) if hexs.strip() else b""
        fmt = "topaz" if head[:4] == b"TPZ0" else "mobi"
    else:
        fmt = None
    vlog(f"book_format({asin}) = {fmt!r} (exts={sorted(exts)})")
    return fmt

def pull_book(asin, dest_parent="files-4k"):
    """Pull the device book dir into files-4k/<asin>/ (needed for brute + repackage)."""
    dest = os.path.join(dest_parent, asin)
    # Clean any prior pull first.  A re-download (removeDownload + download)
    # issues content fragments under NEW CR! ids, so a plain additive pull would
    # leave the previous download's stale CR!*.kfx alongside the fresh ones —
    # repackage then feeds those stale fragments the new key and fails with
    # "Incorrect padding - Wrong key".
    shutil.rmtree(dest, ignore_errors=True)
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

def open_with_baseline(dev, asin, retries=3):
    """attach → hash the scudo heap as a baseline → open(asin) → DETACH, in one
    attach.  The baseline (per-page fingerprints taken BEFORE the book loads)
    lets the later dump ship only the pages that changed as the book rendered —
    where the freshly-derived content key lives — for a ~4x cheaper brute.
    Folding it into the open attach keeps the attach count unchanged; the
    baseline runs before open() so it never delays the render-unblocking detach.
    Returns (baseline, open_result); baseline is {} if it couldn't be taken."""
    for attempt in range(retries):
        pid = kindle_pid(dev)
        if pid is None:
            vlog("open_with_baseline: no Kindle pid, relaunching"); relaunch(dev); continue
        s = None
        try:
            vcmd(f"frida[pid={pid}] baseline()+open({asin!r})",
                 note=f"attach, hash baseline, open, detach, try {attempt+1}/{retries}")
            s = dev.attach(pid)
            dm = s.create_script(open("dump_mem.js").read()); dm.load()
            baseline = dm.exports_sync.baseline()
            vlog(f"  ↳ baseline: {sum(len(v) for v in baseline.values())} pages")
            op = s.create_script(open("krx_agent.js").read()); op.load()
            r = op.exports_sync.open(asin)
            vlog(f"  ↳ open -> {r!r}")
            return baseline, r
        except TRANSIENT as e:
            print(f"    (transient {type(e).__name__} on baseline/open; relaunch+retry)")
            relaunch(dev)
        finally:
            if s is not None:
                vlog(f"  detach pid={pid}")
                try: s.detach()
                except Exception: pass
    raise RuntimeError(f"open_with_baseline({asin}) failed after {retries} tries")

def verify_and_dump(dev, asin, outpath, baseline=None, verify=True, retries=3):
    """One attach: optionally confirm the reader's current book == asin, then dump
    the native heap. Returns bytes dumped, or -1 on a wrong/absent book or failure.

    verify=True (post-open path): the dump is only meaningful if the freshly-
    rendered book is `asin`, so curasin() must match first (else return -1).
    verify=False (fast path): no open() happened — we just want whatever key the
    download already cached — so skip the check and dump unconditionally.  (brute
    uses THIS book's test pages, so an unrelated heap simply yields no key.)

    If `baseline` (from open_with_baseline) is given, dump only the scudo pages
    that changed since it was taken — the cheap delta that almost always still
    contains the key.  With baseline=None, dump the full scudo heap (the safe
    fallback the caller upgrades to when a delta brute comes up empty)."""
    delta = baseline is not None
    for attempt in range(retries):
        pid = kindle_pid(dev)
        if pid is None:
            vlog("verify_and_dump: no Kindle pid"); relaunch(dev); return -1  # caller re-opens
        s = None
        try:
            s = dev.attach(pid)
            if verify:
                vcmd(f"frida[pid={pid}] curasin()", note="confirm target book is open")
                chk = s.create_script(open("krx_agent.js").read()); chk.load()
                cur = chk.exports_sync.curasin()
                if cur != asin:
                    vlog(f"  ↳ current book is {cur!r}, want {asin!r} — not dumping")
                    return -1
            what = "dump_delta()" if delta else "dump_all()"
            vcmd(f"frida[pid={pid}] {what} -> {outpath}",
                 note=("dump changed scudo pages" if delta else "dump full scudo heap"))
            dmp = s.create_script(open("dump_mem.js").read())
            f = open(outpath, "wb"); state = {"off": 0, "regions": 0}; done = threading.Event()
            def on_message(msg, data):
                if msg.get("type") == "send":
                    if msg["payload"].get("done"): done.set(); return
                    if data: f.write(data); state["off"] += len(data); state["regions"] += 1
            dmp.on("message", on_message); dmp.load()
            if delta: dmp.exports_sync.dump_delta(baseline)
            else:     dmp.exports_sync.dump_all()
            done.wait(timeout=300); f.close()
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

def brute_one(heap_path, bookdir, aligns=(16, 8, 4, 1)):
    """Return the 16-byte content key for the single book in bookdir, or None.

    `aligns` controls which byte-alignments to scan.  KFX content keys are scudo
    mallocs and so are always 16-byte-aligned in practice — pass aligns=(16,) for
    a cheap speculative scan (the fast path) where a miss should bail fast instead
    of grinding the ~10M-window align=1 pass; the default tries every alignment
    for the thorough fallback."""
    pages = test_pages_for(bookdir)
    vlog(f"brute_one: {len(pages)} test page(s) from {bookdir}")
    if len(pages) < 3:
        vlog("brute_one: <3 test pages, cannot brute — giving up")
        return None
    test = make_test(pages)
    buf = np.fromfile(heap_path, dtype=np.uint8)
    vlog(f"brute_one: scanning {len(buf)} heap bytes (aligns={aligns})")
    for align in aligns:
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
    """Copy the app's kindle_library.db off-device into files-4k/ and return the
    local path.  With --offline, skip the device and reuse the cached copy."""
    if OFFLINE:
        if not os.path.exists(LIBRARY_DB):
            raise SystemExit(f"--offline: no cached library db at {LIBRARY_DB}; "
                             "run once online (e.g. --list) to populate it first")
        return LIBRARY_DB
    os.makedirs(os.path.dirname(LIBRARY_DB), exist_ok=True)
    adb(f"cp /data/data/{PKG}/databases/kindle_library.db /data/local/tmp/kl.db; "
        f"chmod 666 /data/local/tmp/kl.db", su=True)
    run(["adb", "pull", "/data/local/tmp/kl.db", LIBRARY_DB], capture_output=True)
    return LIBRARY_DB

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

def skip_asins():
    """ASINs listed in SKIP_TXT (one per line, '#' comments allowed) to exclude
    from --all — e.g. books that reliably crash or that we don't want."""
    if not os.path.exists(SKIP_TXT):
        return set()
    out = set()
    for line in open(SKIP_TXT):
        a = line.split("#", 1)[0].strip()
        if a:
            out.add(a)
    return out

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

def harvest_one(dev, asin, render_wait=5.0, dl_timeout=300, open_tries=3,
                no_redownload=False):
    """Download+open+dump+brute one book. Return hex key or None."""
    # --offline never drives the device: this whole path (download/open/dump)
    # needs the AVD, so there's nothing to do — callers fall back to any
    # pre-existing key + already-pulled files-4k/<asin>/ for repackaging.
    if OFFLINE:
        print(f"  [{asin}] --offline: no cached key, skipping device harvest")
        return None
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
        if not wait_for(lambda: book_state(asin) == "LOCAL", dl_timeout, 1.5,
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
        why = ("Topaz (TPZ0) — no Topaz DRM support" if fmt == "topaz"
               else f"format is {fmt or 'unknown'!r}")
        print(f"  [{asin}] {why} — skipping; this harvester handles KFX and "
              f"Mobipocket/KF8")
        return None

    # content is on disk now — pull it (needed for the brute's test pages and for
    # repackage) and wire up the format-appropriate brute once, shared by both the
    # fast path and the open+render fallback below.
    bookdir = pull_book(asin)
    prc = mobi_content_file(bookdir) if fmt == "mobi" else None
    def brute(heap):
        if fmt == "kfx":
            return brute_one(heap, bookdir)
        return brute_mobi(heap, prc, log=vlog) if prc else None

    # 1c. Mobipocket books can be DRM-free.  The encryption flag (0=none,
    #     1=legacy, 2=Amazon DRM) decides: only type 2 needs a heap key.  Type 0
    #     is unencrypted — there's nothing to brute, no need to even open/render;
    #     signal NODRM so repackage() converts the .prc straight to EPUB.  (This
    #     is why public-domain titles like the Divine Comedy showed up as a hard
    #     "crypto type 0 not supported" crash from brute_mobi before.)  Any other
    #     type we can't handle — skip cleanly instead of crashing.
    if fmt == "mobi":
        ct = mobi_crypto_type(prc) if prc else -1
        if ct == 0:
            print(f"  [{asin}] Mobipocket, no DRM (crypto type 0) — no key needed")
            return NODRM
        if ct != 2:
            print(f"  [{asin}] Mobipocket crypto type {ct} unsupported — skipping")
            return None

    # 2. FAST PATH (KFX only) — no render needed.  The content key is derived and
    #    cached in the native heap at DOWNLOAD time, not at render time (see NOTES):
    #    a freshly-downloaded book already has its key resident.  So try a plain
    #    full-scudo dump + brute BEFORE the slower open+render+delta dance.  There
    #    is no open() to verify against (verify=False); brute keys off THIS book's
    #    own test pages, so a heap that happens not to hold this key just yields
    #    None (another resident book's key can't pass this book's padding test).
    #    The speculative brute scans align=16 ONLY (KFX keys are always 16-aligned)
    #    so a miss bails in ~1s instead of grinding the align=1 pass; the thorough
    #    fallback below still covers the (theoretical) unaligned case.  Mobi is
    #    skipped here entirely — its brute is minutes-long and not worth running
    #    speculatively, and it may sit at a non-16 offset (see mobidrm notes).
    if fmt == "kfx":
        fast_heap = f"/tmp/kh_{asin}.fast.bin"
        key = None
        n = verify_and_dump(dev, asin, fast_heap, baseline=None, verify=False)
        if n > 0:
            print(f"  [{asin}] fast-path dump {n/1e6:.0f} MB; brute-forcing…")
            key = brute_one(fast_heap, bookdir, aligns=(16,))
        try: os.remove(fast_heap)
        except OSError: pass
        if key is not None:
            hexkey = key.hex()
            print(f"  [{asin}] *** KEY {hexkey} (fast path, no render) ***")
            return hexkey
        print(f"  [{asin}] fast-path miss — falling back to open+render")

    # 3. FALLBACK — open, render, delta-dump.  Reaches here when the key wasn't
    #    already resident (e.g. a book that was LOCAL from a PRIOR app session, so
    #    nothing this process ever derived it): actually rendering the book is what
    #    triggers the derivation.  Open + wait for render, then verify-the-right-
    #    book + dump in ONE attach (fewer attaches = less anti-debug risk).  We
    #    hash a heap baseline in the open attach (before the book loads) so the
    #    dump can ship only the pages that changed as it rendered — the delta that
    #    holds the freshly-derived key — for a ~4x cheaper brute.
    delta_heap = f"/tmp/kh_{asin}.delta.bin"
    baseline = {}
    dumped = -1
    for attempt in range(open_tries):
        print(f"  [{asin}] opening (try {attempt+1}/{open_tries})…")
        baseline, r = open_with_baseline(dev, asin)  # attach: baseline+open, detach
        if not r.get("ok"):
            print(f"  [{asin}] open failed: {r}"); ensure_app_home(dev); continue
        wait_for(lambda: READER_ACT in top_activity(), 40, 0.5, label="reader activity")
        time.sleep(render_wait)                     # first draw + key caching
        dumped = verify_and_dump(dev, asin, delta_heap, baseline=baseline)
        if dumped > 0:
            break
        print(f"  [{asin}] target book not confirmed open; re-opening")
        ensure_app_home(dev)
    if dumped <= 0:
        print(f"  [{asin}] could not open+dump target book — skipping"); return None
    print(f"  [{asin}] delta dump {dumped/1e6:.0f} MB; brute-forcing…")

    # try the cheap delta first; on a miss (e.g. the key was resident before the
    # baseline, so it sits in unchanged pages) upgrade to a full scudo dump.
    key = brute(delta_heap)
    try: os.remove(delta_heap)
    except OSError: pass
    if key is None:
        print(f"  [{asin}] delta miss — re-dumping full scudo heap and retrying")
        full_heap = f"/tmp/kh_{asin}.bin"
        full = verify_and_dump(dev, asin, full_heap)   # baseline=None => full dump
        if full > 0:
            print(f"  [{asin}] full dump {full/1e6:.0f} MB; brute-forcing…")
            key = brute(full_heap)
        try: os.remove(full_heap)
        except OSError: pass
    # last resort (KFX): the key was never derived into this process's heap —
    # e.g. the book was downloaded in a PRIOR session and re-opening it doesn't
    # re-derive the key.  Remove the stale download and re-fetch it FRESH; the key
    # is derived at download time, so a plain dump then finds it (the fast path).
    if key is None and fmt == "kfx" and not no_redownload:
        hexkey = redownload_and_recover(dev, asin)
        if hexkey:
            print(f"  [{asin}] *** KEY {hexkey} (after re-download) ***")
            return hexkey

    if key is None:
        print(f"  [{asin}] KEY NOT FOUND"); return None
    hexkey = key.hex()
    print(f"  [{asin}] *** KEY {hexkey} ***")
    return hexkey

def redownload_and_recover(dev, asin, dl_timeout=300):
    """Remove a book's (stale) local download and re-fetch it fresh so the app
    re-derives the content key at download time, then dump + brute.  Returns the
    hex key or None.  This is the automated form of the manual 'Remove Download +
    re-download' recovery for KFX books whose key the open+render path can't find
    (the key was only ever derived at some prior session's download)."""
    print(f"  [{asin}] removing download + re-fetching to re-derive the key…")
    # frida-python maps the snake_case accessor to the JS removeDownload export.
    agent_call(dev, "remove_download", asin)
    wait_for(lambda: book_format(asin) is None, 60, 2.0, label="download removed")
    agent_call(dev, "download", asin)
    if not wait_for(lambda: book_state(asin) == "LOCAL", dl_timeout, 1.5,
                    label="re-download (STATE=LOCAL)"):
        print(f"  [{asin}] re-download did not land — giving up"); return None
    bookdir = pull_book(asin)                       # fresh content (new CR! id)
    heap = f"/tmp/kh_{asin}.redl.bin"
    key = None
    n = verify_and_dump(dev, asin, heap, baseline=None, verify=False)
    if n > 0:
        print(f"  [{asin}] re-download dump {n/1e6:.0f} MB; brute-forcing…")
        key = brute_one(heap, bookdir, aligns=(16,))
    try: os.remove(heap)
    except OSError: pass
    return key.hex() if key else None

def mobi_content_file(bookdir):
    """The single Mobipocket/KF8 content file in a pulled book dir, or None."""
    for f in sorted(glob.glob(os.path.join(bookdir, "*"))):
        if f.rsplit(".", 1)[-1].lower() in _MOBI_EXTS:
            return f
    return None

def mobi_crypto_type(prc):
    """Mobipocket encryption flag from the PalmDB record-0 header: 0=none,
    1=legacy Mobipocket, 2=Amazon DRM.  -1 if the file isn't parseable as
    Mobipocket (e.g. a Topaz .prc, which book_format already routes away)."""
    from mobidrm.mobidedrm import MobiBook
    import struct
    try:
        ct, = struct.unpack(">H", MobiBook(prc).sect[0xC:0xE])
        vlog(f"mobi_crypto_type({os.path.basename(prc)}) = {ct}")
        return ct
    except Exception as e:
        vlog(f"mobi_crypto_type({os.path.basename(prc)}): {type(e).__name__}: {e}")
        return -1

def repackage(asin, hexkey):
    """Turn a recovered book into an EPUB.  Dispatches on the pulled content's
    format: KFX -> repackage.py (.kfx-zip) -> calibre; Mobipocket/KF8 ->
    mobidrm.decrypt_with_key (.mobi) -> calibre.  A NODRM 'key' means the .prc
    is unencrypted (crypto type 0) — convert it straight through, no stripping."""
    bookdir = f"files-4k/{asin}"
    os.makedirs("out", exist_ok=True)
    epub = f"out/{asin}.epub"
    if os.path.exists(epub):                 # already built — idempotent re-runs
        return epub + " (cached)"
    prc = mobi_content_file(bookdir)
    if prc:                                  # Mobipocket/KF8
        if hexkey == NODRM:                  # unencrypted — no key to strip
            run(["ebook-convert", prc, epub], check=True)
            return epub
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
    ap.add_argument("--offline", action="store_true",
                    help="never touch the device: work off the cached library db "
                         "and already-pulled books (composes with --list/--repackage)")
    ap.add_argument("--render-wait", type=float, default=5.0)
    ap.add_argument("--no-redownload", action="store_true",
                    help="don't try removing + re-fetching a KFX book whose key "
                         "the open+render path can't find (the re-download re-"
                         "derives the key, but costs a full fresh download)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="log every shell/frida command as it runs, plus extra detail (to stderr)")
    args = ap.parse_args()

    global VERBOSE, OFFLINE
    VERBOSE = args.verbose
    OFFLINE = args.offline

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
        # Normally skip books we already cracked; with --repackage keep them —
        # they're exactly the ones we want to (re)build an EPUB for.
        skip = skip_asins()
        targets = [a for a, t, s, bt in load_inventory()
                   if is_harvestable(bt) and a not in skip
                   and (args.repackage or a not in have)]
        if skip:
            print(f"skipping {len(skip)} ASIN(s) from {SKIP_TXT}")
    else:
        ap.error("specify --list, --asins, or --all")

    if args.limit:
        targets = targets[: args.limit]
    print(f"harvesting {len(targets)} book(s)")
    vlog(f"targets: {targets}")

    dev = None if OFFLINE else frida.get_usb_device()
    vlog(f"frida device: {dev}")
    for i, asin in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {asin}")
        key = have.get(asin)
        if key:
            print("  already have key")
        else:                                # no cached key — go get one
            for book_try in range(2):        # one whole-book retry for flaky crashes
                try:
                    key = harvest_one(dev, asin, render_wait=args.render_wait,
                                      no_redownload=args.no_redownload)
                    break
                except Exception as e:
                    print(f"  ERROR: {type(e).__name__}: {e}")
                    relaunch(dev)
            if key:
                with open(KEYS_TXT, "a") as f:
                    f.write(f"{asin} {key}\n")
                have[asin] = key
        # repackage any book we hold a key for, freshly harvested or pre-existing
        if key and args.repackage:
            try:
                print("  ->", repackage(asin, key))
            except Exception as e:
                print(f"  repackage failed: {e}")

if __name__ == "__main__":
    main()
