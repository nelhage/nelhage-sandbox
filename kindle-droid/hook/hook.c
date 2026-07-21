// hook.c — persistent native hook payload for the KRF anti-tamper bypass.
//
// This .so is dlopen'd into the Kindle process by Frida, its install() work is
// done from JS (Frida's Arm64Relocator builds the trampoline into tramp_slot,
// and patches the target's prologue to jump here), and then Frida DETACHES.
// Everything here is owned by the *process*, not by Frida, so it survives the
// detach — which is the whole point: with Frida gone the KRF anti-tamper check
// (which scans /proc/self/maps for the agent's memfd/anon exec footprint)
// passes, the book renders, and our hook still fires.
//
// Two exported symbols matter to the installer:
//   * hook_fn     — the C hook, called from the trampoline with a pointer to a
//                   saved register frame (see struct regframe).
//   * tramp_slot  — a reserved, file-backed executable slot in .text that the
//                   installer overwrites with the generated trampoline. It MUST
//                   live in an r-x file-backed segment (NOT an mmap'd anon page),
//                   because an anonymous executable mapping is exactly the
//                   injected-code signature the detector flags. A file-backed
//                   page stays attributed to this .so in /proc/self/maps even
//                   after the copy-on-write that patching triggers.
//
// Build (see CLAUDE.md / plan): aarch64-linux-android, -shared -fPIC, link
// bionic libc (do NOT -nostdlib), no C++/libc++_shared.

#include <stdint.h>
#include <fcntl.h>
#include <unistd.h>
#include <android/log.h>

// Saved register frame the trampoline hands us. Layout MUST match the stores
// the installer emits (stp x0,x1 ... stp x8,lr). x[0..7] = args, x[8] =
// indirect-result location, lr = the target's original return address.
struct regframe {
    uint64_t x[9];   // x0..x8
    uint64_t lr;     // x30
};

// ---- milestone configuration -------------------------------------------
// M1 target is KRF's storage AES_set_encrypt_key(key=x0, bits=x1, ctx=x2):
// the key pointer is in x0 and the key length in bits is in x1 (128 or 256).
// For an M3 (real DRM AES) target, redefine these to match that routine's ABI.
#ifndef KEY_REG
#define KEY_REG 0        // register index holding the key pointer
#endif
#ifndef LEN_MODE_BITS
#define LEN_MODE_BITS 1  // 1: LEN_REG holds key length in BITS; 0: in BYTES
#endif
#ifndef LEN_REG
#define LEN_REG 1        // register index holding the key length
#endif
#ifndef LOOT_PATH
#define LOOT_PATH "/data/local/tmp/loot.bin"
#endif
#ifndef LOG_TAG
#define LOG_TAG "kdroidhook"
#endif

// Cap so a bogus length never makes us read/write unbounded memory.
#define MAX_KEY_LEN 64

// Append a length-prefixed record { u32 len; key bytes } to the loot file.
// Length-prefixed so repeated captures accumulate and the host can split them.
static void loot_append(const uint8_t *key, uint32_t len) {
    int fd = open(LOOT_PATH, O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (fd < 0) return;
    // Best-effort; short writes on a small local file are not a concern here.
    (void)write(fd, &len, sizeof(len));
    (void)write(fd, key, len);
    close(fd);
}

__attribute__((visibility("default")))
void hook_fn(struct regframe *r) {
    const uint8_t *key = (const uint8_t *)(uintptr_t)r->x[KEY_REG];
    uint64_t raw = r->x[LEN_REG];
#if LEN_MODE_BITS
    uint32_t len = (uint32_t)(raw / 8);
#else
    uint32_t len = (uint32_t)raw;
#endif
    if (!key || len == 0 || len > MAX_KEY_LEN) {
        __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                            "hook_fn: skip (key=%p len=%u)", (void *)key, len);
        return;
    }
    loot_append(key, len);
    __android_log_print(ANDROID_LOG_INFO, LOG_TAG,
                        "hook_fn: captured %u bytes", len);
}

// Reserved trampoline slot, forced into .text (file-backed r-x). The installer
// mprotects it +w, writes the generated trampoline here, and mprotects it back.
// 128 instructions (512 B) is ample for: 9-reg save + bl + restore + up to a
// few relocated prologue instructions + the back-branch.
__attribute__((naked, used, visibility("default")))
void tramp_slot(void) {
    __asm__(
        ".rept 128\n\t"
        "nop\n\t"
        ".endr\n\t"
    );
}
