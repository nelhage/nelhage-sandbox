// install_hook.js — Frida agent that installs a PERSISTENT inline hook and then
// lets us detach.  Loaded directly (no frida-compile, no Java bridge), like
// dump_full.js.
//
// The persistence trick: we dlopen our own hook.so (Module.load), then build a
// trampoline INTO a reserved slot inside that .so's .text (tramp_slot), and
// overwrite the target function's prologue with a 16-byte absolute jump to the
// trampoline.  All of this is owned by the process, not by Frida — so after we
// detach, the KRF anti-tamper /proc/self/maps scan sees no frida footprint, the
// book renders, and our hook still fires.  Crucially the trampoline lives in the
// file-backed .so (NOT an mmap'd anon page), so it never presents as the
// anonymous-executable signature the detector flags.
//
// Codegen is done by Frida's own Arm64Writer/Arm64Relocator (the same machinery
// Interceptor uses) — the relocator handles PC-relative prologue instructions
// (ADRP/ADR/B/BL/LDR-literal).  Only the fixed register save/restore glue is
// hand-encoded (below), because its encodings are constant and need no fixups.

'use strict';

// ---- fixed trampoline glue (AArch64, little-endian instruction words) ------
// Save the live entry state into a frame matching C `struct regframe`:
//   [sp+0x00]=x0 [+0x08]=x1 [+0x10]=x2 [+0x18]=x3 [+0x20]=x4 [+0x28]=x5
//   [+0x30]=x6 [+0x38]=x7 [+0x40]=x8 [+0x48]=lr        (0x50 bytes total)
// We save x0-x8 (args + indirect-result ptr) and lr (our bl clobbers it, but the
// target still needs its original return address).  Callee-saved x19-x28/fp are
// the caller's and untouched at a prologue; hook_fn preserves them per AAPCS.
const SAVE = [
  0xD10143FF, // sub  sp, sp, #0x50
  0xA90007E0, // stp  x0, x1,  [sp, #0x00]
  0xA9010FE2, // stp  x2, x3,  [sp, #0x10]
  0xA90217E4, // stp  x4, x5,  [sp, #0x20]
  0xA9031FE6, // stp  x6, x7,  [sp, #0x30]
  0xA9047BE8, // stp  x8, x30, [sp, #0x40]
  0x910003E0, // mov  x0, sp            ; x0 = &frame (arg to hook_fn)
];
const RESTORE = [
  0xA94007E0, // ldp  x0, x1,  [sp, #0x00]
  0xA9410FE2, // ldp  x2, x3,  [sp, #0x10]
  0xA94217E4, // ldp  x4, x5,  [sp, #0x20]
  0xA9431FE6, // ldp  x6, x7,  [sp, #0x30]
  0xA9447BE8, // ldp  x8, x30, [sp, #0x40]
  0x910143FF, // add  sp, sp, #0x50
];

function words(arr) {
  return new Uint32Array(arr).buffer; // native LE == AArch64 LE
}

const KRF = 'libKindleAndroidNativeBundlerJNI.so';

// Resolve an export from a Module robustly: findExportByName is unreliable on an
// already-loaded (persistent-dlopen'd) module, so fall back to enumerateExports.
function resolveExport(m, name) {
  let p = m.findExportByName(name);
  if (p) return p;
  const e = m.enumerateExports().find((x) => x.name === name);
  return e ? e.address : null;
}

function loadSo(path) {
  Module.load(path); // dlopen; idempotent + refcounted, persists past detach
  let m = Process.findModuleByName('hook.so');
  if (!m) m = Process.enumerateModules().find((x) => x.path === path) || null;
  if (!m) return { base: null, err: 'hook.so not in module list after load' };
  const hook = resolveExport(m, 'hook_fn');
  const tramp = resolveExport(m, 'tramp_slot');
  return { base: m.base.toString(), path: m.path, hook_fn: hook && hook.toString(),
           tramp_slot: tramp && tramp.toString() };
}

// Build the trampoline into `tramp` and repoint `target`'s prologue at it.
// Returns { ok, target, tramp, hookFn, reloc } (reloc = bytes of prologue moved).
function buildAndPatch(target, hookFn, tramp) {
  // 1) Emit the trampoline body into the reserved (file-backed) slot.
  let reloc = 0;
  Memory.patchCode(tramp, 512, (code) => {
    const w = new Arm64Writer(code, { pc: tramp });
    w.putBytes(words(SAVE));            // save frame + x0 = &frame
    w.putBlImm(hookFn);                 // call C hook_fn(&frame) (relative, intra-.so)
    w.putBytes(words(RESTORE));         // restore frame

    // Relocate the original prologue instructions that our 16-byte patch will
    // clobber (>= 16 bytes; instructions are 4-byte aligned).
    const rel = new Arm64Relocator(target, w);
    let off = 0;
    while (off < 16) {
      const r = rel.readOne();
      if (r === 0) break;              // relocator stalled (unexpected instr)
      off = r;
    }
    rel.writeAll();
    reloc = off;

    // Jump back to the remainder of the original function.
    w.putLdrRegAddress('x16', target.add(off)); // ldr x16, =target+off (pooled literal)
    w.putBrReg('x16');                           // br  x16
    w.flush();
    rel.dispose();
    w.dispose();
  });

  if (reloc < 16) {
    return { ok: false, err: 'relocator moved only ' + reloc + ' bytes (<16); pick a hook a few bytes in', reloc };
  }

  // 2) Overwrite the target prologue with a 16-byte absolute jump to the tramp.
  //    putLdrRegAddress + putBrReg emits: ldr x16,#8 ; br x16 ; .quad tramp  (16 B).
  Memory.patchCode(target, 16, (code) => {
    const w = new Arm64Writer(code, { pc: target });
    w.putLdrRegAddress('x16', tramp);
    w.putBrReg('x16');
    w.flush();
    w.dispose();
  });

  return { ok: true, target: target.toString(), tramp: tramp.toString(),
           hookFn: hookFn.toString(), reloc };
}

rpc.exports = {
  // Pre-M1 smoke test: can we dlopen the .so from the app's uid at all?
  // Returns the loaded base + resolved export addresses (throws on link failure).
  smoke(path) { return loadSo(path); },

  // Full install: load hook.so, resolve KRF base, target = base+offset, build the
  // trampoline, patch the prologue.  Call BEFORE detaching.
  install(soPath, targetOffset) {
    const info = loadSo(soPath);
    if (!info.hook_fn || !info.tramp_slot) {
      return { ok: false, err: 'hook.so missing exports (hook_fn/tramp_slot)', info };
    }
    const krf = Process.findModuleByName(KRF);
    if (!krf) return { ok: false, err: KRF + ' not loaded' };
    const target = krf.base.add(ptr(targetOffset));
    // Idempotent: if the prologue already starts with our `ldr x16,#8` stub, the
    // hook is installed — re-patching would relocate our OWN jump (corruption).
    // This makes the driver's retry-on-transient safe.
    if ((target.readU32() >>> 0) === 0x58000050) {
      return { ok: true, already: true, target: target.toString(), krfBase: krf.base.toString(), so: info };
    }
    const res = buildAndPatch(target, ptr(info.hook_fn), ptr(info.tramp_slot));
    res.krfBase = krf.base.toString();
    res.so = info;
    return res;
  },

  // Debug: read N bytes at module_base+offset as hex (to eyeball the patch).
  peek(offset, n) {
    const krf = Process.findModuleByName(KRF);
    if (!krf) return null;
    const bytes = krf.base.add(ptr(offset)).readByteArray(n || 16);
    const u = new Uint8Array(bytes);
    let s = '';
    for (let i = 0; i < u.length; i++) s += ('0' + u[i].toString(16)).slice(-2);
    return s;
  },
};
console.log('[install_hook ready]');
