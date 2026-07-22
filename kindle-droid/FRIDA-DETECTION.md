# Kindle Android — anti-frida / render-block investigation

Reference notes for the anti-tamper mechanism that blocks book rendering while
frida is attached. Written 2026-07-20/21. Everything here is either **directly
observed** (disassembly, stack dumps, live experiments) or a **deduction** with
its supporting evidence noted. Addresses are runtime **module-relative offsets**
into `libKindleAndroidNativeBundlerJNI.so` (KRF) unless stated. That library's
first LOAD segment has `p_offset == p_vaddr == 0`, so **ELF file offset == module
offset**; a runtime hook target is `module.base + <offset>`.

Everything below was reproduced against book **B01LXW2IUQ** (*Revenger*, Alastair
Reynolds), the representative "voucher KEY-NOT-FOUND" title, but the block is
generic to book-open.

---

## 1. Observed behaviour (the symptom)

- With **no frida attached**, books open and render normally.
- With a frida session **attached** (even an idle script — `rpc.exports={}` — no
  hooks), opening a book **hangs on the cover-splash** `BookOpenActivity` forever;
  it never reaches `StandAloneBookReaderActivity`, `curasin()` stays `None`, and
  no page content is decrypted.
- **Detaching** frida lets a *fresh* open proceed. This is the origin of the old
  working recipe: `attach → openBook() → DETACH IMMEDIATELY → the reader renders
  on its own`. Detaching works because you leave *before* the stall loop starts.
- Once the stall loop **has** started, detaching does **not** visibly recover it
  within ~18 s (see §3 — the loop re-checks only every 30 s because of a real
  `sleep(30)`, so recovery may take up to 30 s; not conclusively retested).
- The screen must be awake for a real render (`mWakefulness=Asleep` on the AVD
  otherwise); irrelevant to the block itself but confounded earlier tests. Wake
  with `input keyevent KEYCODE_WAKEUP; svc power stayon true`.

## 2. Where the stall lives (call chain)

Captured with an ART thread dump while stalled:

```
adb shell su -c "kill -3 <pid>"          # SIGQUIT → ART dumps all thread stacks
adb shell su -c "cat /data/anr/trace_00" # (also /data/anr/anr_*)
```

The book-open worker thread (named `kar-shared-3`, a `ThreadPoolExecutor` worker)
was stuck here:

```
native: #00 nanosleep+8                              (libc.so)
native: #01 sleep+44                                 (libc.so)
native: #02 krf+0x31ff210                            <-- returns into the stall fn
native: #03 krf+0x3241c70                            <-- obfuscated (CFF) caller
native: #04 krf+0x323d54c
native: #05 krf+0x323b098
native: #06 krf+0x3136b4c
native: #07 krf+0x13fcf08
native: #08 krf+0x13fc18c
native: #09 krf+0x13f91d8
native: #10 krf+0x13e8f78
native: #11 krf+0x15c780c   Java_com_amazon_krf_internal_KRFBookImpl_createBook+1616
  at com.amazon.krf.internal.KRFBookImpl.createBook(Native method)
  at com.amazon.krf.platform.KRF.loadBookInfo
  at com.amazon.kindle.rendering.BaseKRIFBookItem.*
  at com.amazon.kindle.content.format.KindleContentFormat.c
  at ...ThreadPoolExecutor$Worker.run
```

So the block is **inside native `KRFBookImpl.createBook`** (KRF content-load), not
in the Java/UI layer. Every other thread during the stall was in state `S`
(sleeping) — nothing spinning, nothing kernel-blocked (`D`). The book-open is
simply *waiting in a sleep loop*.

## 3. The stall primitive: `sleep(30)` in a loop

Disassembly of the stall function (frame #02 returns to `0x31ff210`):

```
0x31ff1a4: bl   #0x31ff368            ; ... build/cleanup locals (std::string dtors
0x31ff1b0: tbz  w8, #0, #0x31ff1bc    ;     via bl #0x3538830 a few times) ...
   ... several `bl #0x3538830` (free) calls ...
0x31ff20c: mov  w0, #0x1e             ; w0 = 30
0x31ff210: bl   #0x353a060            ; **sleep(30)**   (backtrace: sleep+44 → nanosleep)
0x31ff214: tbz  w26, #0, #0x31ff220
   ...
0x31ff250: ret
```

The function does cleanup → `sleep(30)` → return, **once per call**. The *loop*
is in the caller.

**Live experiment — `sleep` is not the gate.** Replacing libc `sleep`
(and `nanosleep`) with a no-op that returns 0:

```
Interceptor.replace(libc.findExportByName('sleep'),
  new NativeCallback(function(s){ return 0; }, 'uint', ['uint']));
```

→ the loop spins ~**50,000 iterations/second** and the book **still never opens**.
Therefore the loop is `while (detected) { report_telemetry(); sleep(30); }`, and
the detection is **re-evaluated every iteration** and stays true while frida is
present. NOPing the sleep alone is useless.

## 4. The loop is control-flow-flattened (CFF) — no clean branch to NOP

With `sleep` no-op'd (loop hot), a Stalker **basic-block** trace of the stalling
thread showed a tight cycle (period ≈ 4 blocks) through a dispatcher:

```
krf+0x3241a44 .. +0x3241a54
krf+0x3241a54 .. +0x3241a64
krf+0x3242920 .. +0x3242978
krf+0x3243800 .. +0x3243808      <-- CFF dispatcher head
```

Disassembly reveals a classic **flattened state machine**, not ordinary control
flow:

- **State variable** lives at `[x28, #0x190]`.
- **Dispatcher** `0x32437f0..0x3243808`:
  ```
  0x32437f0: mul  w8, w22, w8
  0x32437f4: lsr  w8, w8, #0x11
  0x32437f8: sub  w8, w9, w8
  0x32437fc: and  w8, w8, #0x7f          ; next-state = f(...) & 0x7f
  0x3243800: cmp  w8, #0x68
  0x3243804: b.ne #0x3241a44
  0x3243808: b    #0x32419e0
  ```
- **Jump-table dispatch** at `0x3241a44`:
  ```
  0x3241a44: ldr   w9, [x28, #0x190]     ; load state
  0x3241a48: add   w9, w9, w8
  0x3241a4c: cmp   w9, #0x67
  0x3241a50: b.hi  #0x3243800
  0x3241a54: adr   x8, #0x3241a54
  0x3241a58: ldrsw x10, [x24, x9, lsl #2] ; x24 = jump-table base, x9 = state idx
  0x3241a5c: add   x8, x8, x10
  0x3241a60: br    x8                      ; computed jump to next state block
  ```
- **State transitions are XOR-computed with magic constants** — e.g. the loop-body
  block `0x3242920..0x3242978`:
  ```
  0x3242924: ldr  w8, [sp, #0x184]
  0x3242944: eor  w8, w8, w9              ; w9 = 0x181fc21 (built via mov/movk/sub)
  0x3242948: ldr  w9, [sp, #0x188]
  0x3242954: eor  w9, w9, w10             ; w10 = 0xf11b4476-ish
  0x3242958: and  x8, x11, x8             ; x11 = [sp,#0x28]
  0x324295c: add  x8, x8, w9, sxtw
  0x3242960: ldr  x9, [sp, #0xe8]
  0x3242964: cmp  x9, x8                  ; **decision compare**
  0x3242968: mov  w8, #0x44               ; candidate next-state 0x44
  0x324296c: mov  w9, #0x1e               ; candidate next-state 0x1e
  0x3242970: csel w8, w9, w8, lo          ; next-state = (x9<x8)? 0x1e : 0x44
  0x3242974: b    #0x3243800              ; back to dispatcher
  ```

State numbers observed: `0x1e, 0x44, 0x67, 0x68` — a VM with ~100+ states. The
caller at `0x3241c70` (frame #03) is the same flavour: `blr x8` where
`x8 = [sp,#0x168] ^ ([sp,#0x18c] + 0x3666b373)` (XOR-computed indirect call),
plus computed branches (`b #0x3243800`, `#0x32436e8`, `#0x3243294`) and magic
constants (`0xa604`, `0x245b`, `0x1e0cdca0`).

**Conclusion:** there is **no single `if(detected) goto stall` branch to NOP**.
Control flow is data-driven through the state var `[x28,#0x190]` and jump tables.
The only obvious lever is the `csel` at `0x324296c` (picks next-state `0x1e` vs
`0x44` from a `cmp` of computed values), *if* that comparison turns out to be the
detection gate — unconfirmed (see §8).

## 5. What triggers detection — the frida footprint (deductions + evidence)

The detection is **live** (clears when frida detaches) and targets the **agent's
in-process footprint**, NOT frida-server and NOT a book/account property:

- **Evidence it's the agent, not the server/port:** detaching removes the agent's
  mappings and lets the book render, *while frida-server keeps listening on port
  27042 the entire time*. So it is not a port-27042 connect check and not
  frida-server's mere existence.
- **The only footprint present while attached** (from `/proc/<pid>/maps`): the
  frida agent, mapped from a **memfd**:
  ```
  788664a000-7887d16000 r-xp ... /memfd:frida-agent-64.so (deleted)
  ... 9 segments total (r-xp code, r--p rodata, rw-p data) ...
  ```
  These vanish on detach. No suspicious thread names, `TracerPid=0`.

**Detection vectors RULED OUT** (each hooked/observed live, all negative):

| Vector | Test | Result |
|---|---|---|
| Java `Debug.isDebuggerConnected()` / `waitingForDebugger()` | called via agent | both already **false** while attached; hooking them false didn't help |
| `ptrace` / `TracerPid` | `/proc/self/status`, hooked `ptrace` | `TracerPid=0`; never called |
| `dl_iterate_phdr` (module enum) | hooked | **0 calls** — and frida's manually-mapped agent isn't even in `link_map`/`_r_debug`, so `Process.enumerateModules()` and a `link_map` `l_name` rename find nothing |
| libc `open`/`openat`/`fopen`/`__openat` on `/proc/self/maps` | hooked all | never opens maps; only `/proc/<pid>/stat` (a metrics thread) |
| libc `syscall()` wrapper | hooked, counted by number | hot syscall in loop = `futex(98)`; **no `openat(56)`** |
| libc `strstr/strcmp/strncmp/memmem` with `frida`/`memfd`/`gum` needle | hooked, filtered | **0 hits** |

Because maps is clearly the footprint yet no libc file/parse/compare call fires,
the detection almost certainly reads `/proc/self/maps` (or scans its own memory)
via **inline `svc` syscalls + a custom (non-libc) byte scan**, all inside the CFF
code — invisible to `Interceptor`. (Not yet confirmed by an svc-level trace.)

## 6. Stealth frida-server attempt (name scrub) — necessary but INSUFFICIENT

Patched `/data/local/tmp/frida-server` (v17.15.4) with equal-length byte edits:

- `frida-agent` → `media-agent` (9×) — this is the memfd name; the runtime name
  `frida-agent-64.so` is built as `frida-agent-<bits>.so` from this prefix.
- `gum-js-loop` → `gjs-mn-loop` (2×), `pool-spawn` → `pool-sporn` (4×).
- **Left intact:** `frida:rpc` (RPC protocol — client depends on it; breaking it
  kills RPC) and `re.frida.*` service names.

Result: `/proc/<pid>/maps` shows **0 `frida`** lines (agent now maps as
`/memfd:media-agent-64.so`) — **but the book still stalls.**

So the detector is **not** matching the literal string "frida" in the map name.
A follow-up in-process scan found:

- **4 executable `r-xp` regions** backed by memfd / anonymous / `(deleted)` memory
  (the agent). `/memfd:` and `(deleted)` are **kernel-assigned**, unpatchable.
- The string **`frida` is still present in the agent's executable memory** (1
  `Memory.scanSync` hit) — the agent's own code/rodata.

**Therefore the detector keys on one (or both) of:**
1. the **structural signature** — an executable mapping backed by
   memfd/anonymous/`(deleted)` memory (a generic code-injection tell), or
2. **`frida`/`gum` string content** inside that executable region.

Neither is defeated by renaming frida-server strings.

### frida-server deployment gotchas (for repro)

- Must run as **root** (uid 0). Working invocation:
  `adb shell su -c "/data/local/tmp/frida-server -D"` — the **`-D`** (daemonize)
  flag is REQUIRED; without it the process exits immediately.
- Adding a `setenforce 0;` prefix or launching via `nohup … &` inside `su -c`
  drops it to **uid 2000 (shell)** → `frida.PermissionDeniedError` on attach.
  Verify with `grep Uid /proc/$(pidof frida-server)/status`.
- `cp` over a **running** frida-server silently fails (text-busy). Always
  `pkill -9 -f frida-server`, confirm dead, then copy.
- Original backed up on device: `/data/local/tmp/frida-server.orig`. Local copies
  in scratchpad: `frida-server` (pristine) and `frida-server.patched`.

## 7. Why this matters for the DRM goal

Content-page decryption (the DRMION AES-128-CBC that yields/uses the 16-byte
content key) happens **during the render**, which is exactly what this block
prevents while attached. That is why the content key could never be observed for
these voucher titles: earlier successes (e.g. B09HZH2SWB) only worked because the
key happened to still be resident from a genuine *un*-attached render. Defeating
this block is the single unlock: once a book renders while attached, the content
key becomes catchable the normal way (dump during active decryption + brute, or a
broad AES-key hook). See the main memory note / `NOTES.md` for the crypto side
(the DRM AES is a bundled software AES, not libcrypto/JCE/KRF-storage-AES).

## 8. Open questions / concrete next steps

1. **Confirm the detection vector** with an `svc`-level trace (Stalker `exec`
   events over the loop, or `Stalker` with an instruction probe) to see the actual
   `openat("/proc/self/maps")` / memory-scan and what it matches (`frida` string
   vs `memfd`/`(deleted)`/exec-anon structure). This decides which bypass is
   viable.
2. **Test the `csel` gate hypothesis** (cheapest patch probe): log the operands of
   `cmp x9, x8` at `0x3242964` across iterations (attached-stall vs a clean run),
   and see whether forcing `csel` at `0x324296c` to always pick the "proceed"
   next-state releases the loop. If yes → a **one-instruction on-disk patch** to
   KRF is possible.
3. **Injection-side bypass** (if detection is structural/content, §6): inject via a
   benignly-named **real-file Frida Gadget** (no memfd) with the agent binary's
   `frida`/`gum` strings scrubbed — removes both the exec-memfd structure and the
   content signature.
4. **CFF devirtualization** (heavy): recover the state-machine graph of the
   `createBook` sub-function (state var `[x28,#0x190]`, jump table at `x24`) to
   identify the state(s) leading to `sleep(30)` and reroute them on-disk, then
   redeploy the `.so` via a Magisk mount.

## 9. Key addresses & artifacts (quick reference)

Library: `libKindleAndroidNativeBundlerJNI.so` (KRF), pulled to scratchpad as
`krf.so` (58 MB, stripped, arm64). File offset == module offset.

| Offset | What |
|---|---|
| `0x15c780c` | `Java_..._KRFBookImpl_createBook+1616` (return site into the stall path) |
| `0x31ff20c` | `mov w0,#0x1e` (sleep arg = 30) |
| `0x31ff210` | `bl sleep` — the stall call |
| `0x3241c70` | obfuscated caller (frame #03), XOR-computed `blr x8` |
| `0x3243800` | CFF dispatcher head |
| `0x3241a44` | CFF jump-table dispatch (state at `[x28,#0x190]`, table base `x24`) |
| `0x3242964` | decision `cmp x9, x8` |
| `0x324296c` | `csel w8, #0x1e, #0x44, lo` — candidate detection gate (unconfirmed) |
| `0x3302714` | (unrelated) KRF `AES_set_encrypt_key(key=x0,bits=x1,ctx=x2)` — its own storage AES, NOT the DRM AES |

Scratchpad tooling produced this session:
- `anrdump.py` — SIGQUIT ART stack dump + pull.
- `detect_probe.js` — hooks syscall/file/string ops, filters frida needles.
- `stalk.js` — Stalker **call-summary** of the stall loop.
- `looptrace.js` — Stalker **basic-block** trace → found the CFF cycle.
- `hidefrida.js` — `_r_debug`/`link_map` `l_name` rename (no-op here; agent absent
  from link_map).
- `blockprobe.py`, `threadstate.py`, `footprint.py` — state/thread/maps probes.
- `frida-server.patched` — name-scrubbed frida-server (`frida-agent`→`media-agent`).

### Repro skeleton (get to the stall, hot)

```python
import frida, subprocess, harvest as H
dev = frida.get_usb_device(); pid = H.kindle_pid(dev)
H.adb("input keyevent KEYCODE_WAKEUP"); H.ensure_app_home(dev)
s = dev.attach(pid)
# optional: no-op sleep to make the loop hot for tracing
s.create_script("""
var libc=Process.findModuleByName('libc.so');
Interceptor.replace(libc.findExportByName('sleep'),
  new NativeCallback(function(x){return 0;},'uint',['uint']));
""").load()
subprocess.run(["adb","shell","input","tap","150","760"])   # tap a book cover on Home
# now stalled in BookOpenActivity; attach Stalker / probes to the calling thread
```
