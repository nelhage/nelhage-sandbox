# V8 Bug Analysis

A working taxonomy of bugs in Chrome's V8 JavaScript engine, built up one bug
at a time from `data/sample_patches`.

---

## Summary

Ten V8 security fixes, analyzed one at a time. Per-bug write-ups follow; this
section is the synthesis.

Columns reflect the analytical axes that emerged: **locus** (where it lives —
treated as load-bearing, since "inside the optimizer" vs. "bytecode-gen with no
real CFG" vs. "runtime" vs. "hand-written assembly" behave very differently);
**primary defect class**; **cross-cutting axes** it also touches; and whether a
**memory-safe language / richer type discipline would have prevented it**.

| # | CVE | Sev | Locus | Primary defect class | Cross-cutting axes | Safer lang prevents? |
|---|-----|-----|-------|----------------------|--------------------|----------------------|
| 01 | 2023-2724 | High | Public API → internal fast-path | Spec-invariant bypass on a web-reachable surface | semantics gap (API vs spec); footgun-fix | **No** — spec/logic |
| 02 | 2026-7999 | Low | Turboshaft (Wasm load-elim) | Optimizer immutability assumption wrong | repr↔value; analysis-unsoundness; lifecycle (repr rewrite) | **No** — optimizer modeling |
| 03 | 2026-10936 | High | Turboshaft (graph builder) | Dataflow/CFG merge over wrong edge set | invariant/soundness; non-mainline edge (deopt) | **No** — compiler logic |
| 04 | 2021-4061 | High | TurboFan (JS→Wasm inliner) | Semantics mismatch: compiler special-case vs runtime (wrong object type) | semantics gap; wrong-type arg | **Maybe** — a typed IR could reject it |
| 05 | 2025-11215 | Med | Runtime (regexp, C++) | Classic memory unsafety (off-by-one) | memory-safety | **Yes** — bounds-checked iterator |
| 06 | 2025-12433 | High | Bytecode generator (no real CFG) | Dataflow unsoundness — elision omitted an edge | invariant/soundness; non-mainline edge (break) | **No** — compiler logic |
| 07 | 2025-13042 | High | Maglev (register allocator) | Invariant violation — stale SSA-location metadata | invariant/soundness; non-mainline edge (exception); stale-state | **No** — compiler invariant |
| 08 | 2026-5871 | High | Maglev (type inference) | Type-lattice unsoundness — two representation axes out of sync | repr↔value; invariant/soundness | **No** — compiler soundness |
| 09 | 2021-4078 | High | Runtime (`DefineClass`, C++) | Aliased shared-state mutation without restore | aliasing/mutability; footgun-fix | **Yes** — borrow/aliasing rules |
| 10 | 2026-7936 | Med | Builtins / hand-written asm | Lifecycle (bytecode flush) + missing guard in compiled code | lifecycle; missing-check; stale-state | **Maybe** — sum-type/`Option` forces the case |

### Cross-cutting analysis

**Type confusion is the destination, not the cause.** Eight of ten cash out as
type confusion (7/10 so labeled by Chrome), yet the *mechanisms* share almost
nothing across the "primary defect class" column. "Type confusion" describes how
V8 bugs *exploit* — almost any broken invariant about "what kind of value lives
here" lands there — so it's nearly useless as a *root-cause* taxonomy. Classify
by the invariant that broke instead; that's what the rest of these axes do.

**The single biggest class is analysis / invariant unsoundness inside the
compiler** (03, 06, 07, 08 — four of ten, all High). These are bugs where some
compiler pass computes a fact that isn't true: a CFG merge over the wrong edge
set (03, 06), stale SSA-location metadata (07), an unsound type lattice (08).
Two sub-distinctions matter. (a) *Real CFG vs. structural approximation*: 03 is a
genuine dataflow over a CFG; 06 is in the bytecode generator, which is an
AST→bytecode walk with **no real CFG**, so non-lexical edges (`break`) have to be
re-injected by hand — a structurally more error-prone setting. (b) *Direction of
the error*: 03 wrongly **included** a not-yet-valid back-edge (too conservative,
dropped a live frame state); 06 wrongly **excluded** a `break` edge (too
aggressive, kept an unsound elision). Same "wrong edge set" family, opposite
signs.

**Representation vs. logical value** (02, 08) is the second cross-cutting axis:
an immutable string whose *heap representation* is mutable (02), and a `Smi`-typed
number flipped to `HeapNumber` by an orthogonal *machine-representation* decision
(08). In both, the optimizer's abstraction silently lost track of a
representation change underneath it.

**A recurring sub-theme spanning several classes: state changed underneath code
that assumed stability** — the deopt frame (03), the reassigned register (07),
the rewritten string representation (02), the flushed bytecode (10), the aliased
caller frame (09). The sharpest version is the **non-mainline edge**: the happy
path is correct and the bug lives on an exceptional / early-exit / lifecycle edge
— deopt (03), exception handler (07), code-flush self-heal (10) — precisely the
rarely-exercised paths.

**What a safer language would, and wouldn't, catch — the most striking split.**
Only **two of ten** (05 off-by-one, 09 aliased mutation) are the classic kind a
memory-safe language eliminates outright — bounds-checked iteration for 05,
borrow/aliasing rules for 09. The other **eight are logic/soundness bugs that no
amount of memory safety or borrow-checking prevents**: they're wrong *facts*
computed by correct-looking code (a compiler pass that miscomputes a frame state
or a type is perfectly memory-safe while doing so). Two cases (04, 10) sit in a
middle ground that a richer **type discipline** — not memory safety per se —
could plausibly catch: a strongly-typed compiler IR that distinguished
`WasmInstanceObject` from `WasmApiFunctionRef` would reject 04, and a sum
type / `Option` forcing the flushed-field case to be handled would catch 10. The
practical upshot: "rewrite the JIT in Rust" would have stopped 2 of these 10;
the dominant risk in an optimizing engine is *semantic* correctness of the
analyses, which is a verification problem, not a memory-safety one.

**The optimizing compilers are the richest and most severe source.** Five of ten
(02, 03, 04, 07, 08) are in Turboshaft/TurboFan/Maglev, and all the High ones
among them are type confusions; Maglev alone contributes two (07, 08),
consistent with newer, fast-moving tiering machinery. The non-compiler bugs skew
toward simpler root causes (off-by-one, aliasing, missing guard) even when High.

**Fix shapes cluster, and some fixes are deliberately *not* the root fix.**
*Footgun removal* — delete/rename the unsafe primitive (`set_at` in 09;
`DefineAccessor`→`DefineOwnAccessorIgnoreAttributes` in 01). *Make the model
match reality* — teach the analysis the missing fact (02, 03, 06, 07). And
*incomplete-by-design* — the kill-switch flag flip (04, whose real fix landed in
a separate CL) and the self-described band-aid (08, which patches one site while
acknowledging latent unsoundness elsewhere). Lesson for reading these: **the fix
patch and the bug are not always the same artifact** — a flag flip (04) hides the
interesting code entirely; a band-aid (08) signals more of the class is still out
there.

**Reachability is broad.** Seven are reachable from ordinary web JS with no
special flags; the Wasm-string bug (02) needs Wasm + `js-string` builtins, and
the flush bug (10) needs memory pressure (reliably via `--stress-flush-code`).
Even the API bug (01) is web-reachable because Blink calls the unsafe API on
objects web content can shape.

> **Method/confidence note.** Patches came from `data/sample_patches`; several
> root causes and exploit mechanisms were corroborated against commit messages
> and (for Maglev internals) V8 source via web fetches, and a few are explicitly
> flagged inline as inference rather than confirmed (the crbug entries are
> login-gated). Notably, Bug 04's real root cause was found only by following a
> separate CL, and Bug 03's "type confusion" exploitation path is the standard
> mechanism for the bug class rather than something the patch documents.

---

## Bug 01 — CVE-2023-2724 (High): `v8::Object::SetAccessorProperty`

- **CL:** [4458947](https://chromium-review.googlesource.com/c/v8/v8/+/4458947) — "[api] Fix v8::Object::SetAccessorProperty"
- **Component:** API surface / object property definition (`src/api/api.cc`)
- **Chrome's label:** Type Confusion

### What the code did

`v8::Object::SetAccessorProperty` is a **public C++ API** that embedders (and
V8 internals) call to install a getter/setter accessor pair on an object. The
old implementation called the internal helper `JSObject::DefineAccessor(...)`.

That helper is a low-level "just install the accessor" path: it does **not**
enforce the ECMA-262 `[[DefineOwnProperty]]` invariants. In particular it does
not check whether the property already exists as **non-configurable**, nor
whether the receiver is **non-extensible**. It simply overwrites whatever is
there with an accessor pair.

### The bug

Because the public API bypassed the spec-mandated checks, it was possible to
**redefine a property that should have been locked down** — e.g. replace a
non-configurable data property (or an existing accessor) with an
attacker-controlled getter/setter. Code elsewhere in the engine relies on the
invariant that non-configurable properties don't change shape; violating it
leads to **type confusion** (the engine assumes a field holds one kind of
value/representation, but the accessor swap makes that assumption false).

### The fix

Two coordinated changes:

1. **Behavioral fix** (`api.cc`): `SetAccessorProperty` now builds a proper
   `i::PropertyDescriptor` (translating the `DontEnum`/`DontDelete` attribute
   bits into `enumerable`/`configurable`) and routes through
   `i::JSReceiver::DefineOwnProperty(...)`, the spec-compliant path that
   honors configurability/extensibility (and can legitimately throw/fail on a
   Proxy or a locked property).

2. **Hygiene / footgun rename** (everywhere else): the dangerous helper
   `JSObject::DefineAccessor` is renamed to
   `JSObject::DefineOwnAccessorIgnoreAttributes`, making its attribute- and
   invariant-ignoring behavior explicit at every call site so future callers
   don't reach for it by accident. The remaining call sites (bootstrapper,
   runtime, internal `DefineOwnProperty` slow paths, tests) are legitimately
   internal and keep the ignore-attributes behavior. Doc comments in
   `v8-object.h` are also refreshed to point at current spec URLs.

### Taxonomy notes (draft)

- **Root cause class:** missing spec-invariant enforcement on a public API
  entry point (the fast/internal path was wired to an externally reachable
  surface).
- **Exploit primitive:** redefinition of a non-configurable/locked property →
  type confusion.
  - In particular, Blink calls the public API in a way that can be triggered from user code, which allows a web page to trigger the confusion.
- **Fix shape:** re-route to the spec-compliant operation + rename the unsafe
  primitive to discourage misuse.

### Nelson's notes
- C++ API called into an internal fast-path which could violate invariants; Blink called into that API in a way that was triggerable via the web platform.

---

## Bug 02 — CVE-2026-7999 (Low): Wasm string `PrepareForGetCodeUnit` load elimination

- **CL:** [7693134](https://chromium-review.googlesource.com/c/v8/v8/+/7693134) — "[wasm][strings] Fix PrepareForGetCodeUnit elimination"
- **Component:** Turboshaft optimizing compiler — Wasm load-elimination pass (`src/compiler/turboshaft/wasm-load-elimination-reducer.h`)
- **Chrome's label:** Inappropriate implementation

### What the code did

The Wasm load-elimination reducer caches "load-like" operations so redundant
ones can be eliminated. One such operation is `PrepareForGetCodeUnit` — the
setup step behind the `wasm:js-string` builtin `charCodeAt`, which resolves a
string down to a concrete backing-store pointer + encoding so subsequent code
units can be read cheaply.

Both `FindLoadLike` and `InsertLoadLike` hard-coded `mutability = false`,
telling the pass these cached values never change. So once a string's prepared
pointer was computed, the pass would reuse it for later `charCodeAt`s on the
"same" string — including **across intervening calls**.

### The bug

A JS string's *contents* are immutable, but its **heap representation is not**.
A string can be silently rewritten in place into:
- a **ThinString** (e.g. after it's interned by being used as a property key), or
- an **ExternalString** (after externalization).

Both change where the character data lives. Marking `PrepareForGetCodeUnit` as
immutable meant the pass would **not invalidate the cached backing-store
pointer at a call** that performs one of these rewrites. After the call, the
eliminated re-preparation reused a **stale pointer/encoding** → reads of code
units from the wrong/freed backing store (OOB / type-confused read of string
data).

The regression test makes this concrete: it calls `charCodeAt`, then a
`call_indirect` to either `thin(s)` (interns `s` into a ThinString via
`o[s]`) or `ext(s)` (`externalizeString(s)`), then reads more code units — and
checks the results still match before and after tier-up.

### The fix

Add `LoadLikeMutability(offset_sentinel)` returning `true` for
`kStringPrepareForGetCodeunitIndex` (and `false` otherwise). `FindLoadLike` /
`InsertLoadLike` now pass that instead of a hard-coded `false`, so prepared
string pointers are treated as **mutable and invalidated at calls** — while
genuinely-immutable load-likes keep being eliminated.

### Taxonomy notes (draft)

- **Root cause class:** compiler optimization with an incorrect immutability
  assumption — a value treated as invariant whose *underlying representation*
  can change out from under the cached result.
- **Exploit primitive:** stale string backing-store pointer reused after the
  string is rewritten to Thin/External → OOB / wrong-memory read of string data.
- **Why Low (likely):** requires Wasm + the `js-string` builtins and the
  primitive is a constrained read of string contents, not a general r/w.
- **Fix shape:** make the optimizer's mutability model match reality for the
  one operation whose representation can mutate (per-sentinel mutability).

### Nelson's notes
- "representation" vs "logical value" confusion -- in particular, an immutable logical value still had a mutable representation

---

## Bug 03 — CVE-2026-10936 (High): Turboshaft loop-header `dominating_frame_state`

- **CL:** [7761805](https://chromium-review.googlesource.com/c/v8/v8/+/7761805) — "[compiler] Skip loop back-edges when computing dominating_frame_state"
- **Component:** Turboshaft optimizing compiler — graph builder (`src/compiler/turboshaft/graph-builder.cc`)
- **Chrome's label:** Type Confusion

### What the code did

When the Turboshaft graph builder visits a basic block, it computes a single
`dominating_frame_state` for the block by walking its predecessors: take the
first predecessor's `final_frame_state`, and if all the others agree, use it;
if any disagree, fall back to `Invalid`. That frame state is what
deopt-carrying operations in the block deopt *to* (it reconstructs the
interpreter state).

Blocks are visited in **RPO order**. For a **loop header**, one predecessor is
the **loop back-edge**, which comes from a block with a *higher* RPO number
that hasn't been visited yet — so its `final_frame_state` slot is still
**default-initialized to `Invalid`** at header-visit time.

### The bug

The old merge treated that not-yet-computed `Invalid` back-edge entry as a
*disagreeing* predecessor. So the loop header's `dominating_frame_state`
**collapsed to `Invalid`, and that `Invalid` propagated through the loop** —
even when a perfectly good frame state dominated the loop.

Per the commit message, the concrete failure: a **pre-loop `Checkpoint` (#28)
genuinely dominates** a post-loop `CheckedTaggedToTaggedSigned`, but the
(side-effect-free) copy loop "wipes it out" during the RPO merge. The
downstream checked conversion is thus **stranded with no valid dominating
frame state**. In debug builds this trips a DCHECK (what the regression test
originally hit); in release builds a deopt-guarded type check
(`CheckedTaggedToTaggedSigned` — "this tagged value is really a Smi, else
deopt") loses its safety net → a non-Smi can slip through as a Smi → **type
confusion**.

Note the direction: the bug is **not** that a *wrong* frame state was used —
it's that a *valid dominating* frame state was incorrectly **discarded**,
breaking the deopt that protects a downstream type assumption.

### The fix

Iterate predecessors and **skip any whose `rpo_number() >= block`'s** (i.e. the
loop back-edges, guarded by `DCHECK(block->IsLoopHeader())`), since their
`final_frame_state` isn't computed yet. Only genuinely-dominating (forward)
predecessors contribute to the merge, so the real pre-loop checkpoint is
preserved. (Trigger: `[...].sort(Array)` — a *non-inlined* builtin comparator —
in a loop, which lowers `Array.prototype.sort` into a graph with a copy loop
followed by a checked conversion.)

### Taxonomy notes (draft)

- **Root cause class:** compiler dataflow merge over a CFG that mishandles the
  loop back-edge — an uninitialized/forward-referenced value (the back-edge's
  not-yet-computed frame state) folded into a merge before it's valid.
- **Exploit primitive:** a deopt-guarded type check (`CheckedTaggedToTagged-
  Signed`) loses its dominating frame state → guard can't deopt → type
  confusion (non-Smi treated as Smi).
- **Contrast with Bug 02:** both are optimizer correctness bugs, but here the
  optimizer was *too conservative* (dropped a valid frame state) rather than
  *too aggressive* (kept a stale value) — and conservativeness was still
  unsafe because a later phase relied on that frame state existing.
- **Fix shape:** correct the CFG iteration to exclude back-edges from the
  dominator merge.

### Nelson's notes
- Broken invariant **inside** the compiler/optimizer; incorrect dataflow pass over CFG violated an assumption about deoptimization state.

---

## Bug 04 — CVE-2021-4061 (High): JS→Wasm call inlining (kill-switch fix)

- **CL:** [3290583](https://chromium-review.googlesource.com/c/v8/v8/+/3290583) — "[wasm][turbofan] Disable inlining of JS->Wasm calls by default"
- **Component:** TurboFan optimizing compiler — JS→Wasm call inlining (flag only: `src/flags/flag-definitions.h`)
- **Chrome's label:** Type Confusion

### What the patch actually is

This patch is a **one-line feature kill switch**: it flips the default of
`turbo_inline_js_wasm_calls` from `true` to `false`. That's the entire diff —
no test, no logic change.

So the patch tells us almost nothing about the *root cause*. The actual buggy
code lives in TurboFan's JS→Wasm call-inlining machinery, which this CL simply
stops exercising by default. The real fix (correcting the inlining) came
later; this is the fast "stop the bleeding" mitigation shipped to close the
CVE.

### What the underlying bug was (from context, not the patch)

When TurboFan optimizes a JS function that calls an exported **Wasm** function,
the inliner can splice the Wasm callee's body directly into the JS graph. That
requires bridging two type worlds at the boundary: tagged JS values vs. raw
Wasm machine types (`i32`/`i64`/`f64`/refs). Public CVE data confirms this is
a **type confusion** reachable from a crafted HTML page leading to heap
corruption (CVSS 8.8).

### The actual root cause and real fix (CL [9324d7fd](https://chromium.googlesource.com/v8/v8/+/9324d7fd21278d382dbfe1892f4fbb64241fb9f0))

The genuine correctness fix (separate CL, "[wasm][turbofan] Pass correct
instance when inlining JsToWasm wrappers"; regression test
`regress-1271456.js` ties it to this same bug id) shows the bug was **not** a
deopt/representation mismatch at all — it was passing the **wrong heap object
type as the Wasm instance argument**.

Setup: a Wasm module **imports a JS function and re-exports it**. Calling that
export goes JS → (inlined `JSToWasm` wrapper) → and the "Wasm function" it
targets is itself a **`WasmToJS` wrapper** (because the callee is really an
imported JS function). Every Wasm call threads an "instance" argument that the
callee uses to find its context. For a normal Wasm function that's a
`WasmInstanceObject`; but for this re-exported-import case the `WasmToJS`
wrapper expects a **`WasmApiFunctionRef`** (the object that carries the
imported callable + its context).

The buggy inliner passed **`nullptr`** as the instance node to
`BuildWasmCall`, which made it fall back to the **ambient
`WasmInstanceObject`** of the compilation. The `WasmToJS` wrapper then
interpreted that `WasmInstanceObject` *as if* it were a `WasmApiFunctionRef`,
reading fields at `WasmApiFunctionRef` offsets out of a differently-laid-out
object → **type confusion** (wrong pointers → heap corruption).

The fix explicitly loads the correct ref from the function data
(`WasmFunctionData::kRefOffset`) and passes that as the instance:

```cpp
Node* instance_node = gasm_->LoadFromObject(
    MachineType::TaggedPointer(), function_data,
    wasm::ObjectAccess::ToTagged(WasmFunctionData::kRefOffset));
BuildWasmCall(sig_, ..., instance_node, frame_state);   // was: nullptr
```

So my earlier "deopt boundary" guess was **wrong** for this specific bug — the
real issue is supplying the wrong object type for the instance/context
argument. (The deopt-boundary mechanism is a real but *separate* JS↔Wasm bug
class, e.g. CVE-2026-6307.)

### Taxonomy notes (draft)

- **Root cause class:** **wrong-object-type passed for a context/instance
  argument** during compiler inlining — a `WasmInstanceObject` supplied where a
  `WasmToJS` wrapper expects a `WasmApiFunctionRef`. (Plain "wrong type in,
  reinterpreted by the callee" confusion — *not* a representation/deopt issue,
  which was my initial wrong guess.)
- **Fix shape — a new category:** *feature kill switch*. Rather than fixing the
  faulty code, the CVE was closed by **disabling the optimization by default**.
  The patch is a flag flip; the real correctness fix is elsewhere/later.
- **Analysis lesson:** "the fix patch" and "the bug" are not always the same
  artifact. A flag-flip CVE fix means the interesting code is *not* in the diff —
  you have to go to the feature it disables. Worth flagging these separately in
  any taxonomy so they don't get mis-bucketed as "trivial."

### Nelson's notes

- Incorrect semantics in the inliner for JS->Wasm calls where the "wasm function" is actually a re-exported JS function.
- Category: semantics mismatch between runtime and a special-case implementation in the compiler.

---

## Bug 05 — CVE-2025-11215 (Medium): off-by-one OOB read in `RegExpMatchGlobalAtom_OneCharPattern`

- **CL:** [6871496](https://chromium-review.googlesource.com/c/v8/v8/+/6871496) — "[regexp] Fix OOB read in RegExpMatchGlobalAtom_OneCharPattern"
- **Component:** Regexp runtime fast path (`src/runtime/runtime-regexp.cc`)
- **Chrome's label:** Off-by-one error

### What the code did

`RegExpMatchGlobalAtom_OneCharPattern` is a fast path for a global regexp whose
pattern is a **single literal character** (`/x/g`): it scans the subject
string counting occurrences. The scan used a C-style loop:

```cpp
for (SChar c = *block; block < end; c = *(++block)) {
  if (c != static_cast<const SChar>(pattern)) continue;
  ...
}
```

### The bug

Both the init and the increment **dereference `block` before the `block < end`
bound is checked** — the loop reads, *then* tests:

- **Init:** `SChar c = *block` reads `block[0]` before the condition runs at
  all → an OOB read if the range is empty (`block == end` on entry).
- **Increment:** `c = *(++block)` pre-increments and dereferences in the loop's
  update step, which runs **before** the condition is re-tested. On the last
  valid element, `++block` makes `block == end` and `*(++block)` reads
  **`*end`** — one `SChar` past the buffer — *then* the `block < end` test
  fails and the loop exits. The OOB read has already happened.

So it reads one character beyond the string's backing store (the classic
off-by-one). It's a **read**, so the impact is info-leak / crash, not a write
— consistent with the Medium rating.

### The fix

Restructure so the dereference happens **only after** the bound check passes —
move the read into the body and make the update a bare `++block`:

```cpp
for (; block < end; ++block) {
  SChar c = *block;
  ...
}
```

### Taxonomy notes (draft)

- **Root cause class:** classic off-by-one / read-before-bounds-check in a
  hand-written C++ loop — the loop's *update expression* dereferenced the
  pointer it had just advanced, before the guard re-ran.
- **Exploit primitive:** 1-element OOB **read** past a string backing store →
  potential info leak of adjacent heap bytes (surfaced via match count/results)
  or crash. No write primitive.
- **Reachability:** pure JS — any `str.match(/x/g)` / `replace` with a
  single-char global pattern that lands on this fast path.
- **Contrast with the compiler bugs (02–04):** this is a *plain memory-safety*
  bug in straight-line runtime C++ — no optimizer reasoning, no type system,
  just a loop idiom that reads one step ahead of its bound. The simplest class
  in the set so far.
- **Fix shape:** reorder so the bounds check strictly precedes every
  dereference.

### Nelson's notes
- Classic memory unsafety (off-by-one, fixable via Rust or a safe iterator pattern)

---

## Bug 06 — CVE-2025-12433 (High): TDZ hole-check elision misses `break` edges

- **CL:** [7026071](https://chromium-review.googlesource.com/c/v8/v8/+/7026071) — "[interpreter] Merge hole elision info on break"
- **Component:** Bytecode generator — TDZ hole-check elision (`src/interpreter/bytecode-generator.cc`)
- **Chrome's label:** Inappropriate implementation

### Background

`let`/`const` bindings have a **temporal dead zone (TDZ)**: reading one before
its initializer runs must throw `ReferenceError`. Internally an uninitialized
binding holds **the hole** (a special sentinel), and the bytecode generator
emits `ThrowReferenceErrorIfHole` before reads.

As an optimization, once a variable is proven initialized on a path, later hole
checks are **elided** — tracked in a `hole_check_bitmap_`. Across branch/merge
control flow (`if/else`, ternary, `switch` with `default`), elision must be
**sound**: a variable stays elided after the merge only if it was initialized
on **every** incoming path. `HoleCheckElisionMergeScope` does this by AND-ing
each branch's bitmap into `merge_value_`.

### The bug

A **`break` is an extra control-flow edge** straight to the merge point (the
statement after the block/switch), and that edge can carry a *less-initialized*
state than the structural branches — but the old merge logic only AND-ed the
structural branches and **ignored the break edges**. So a variable initialized
on every *case*/fall-through path but **not** on the `break` path was wrongly
treated as initialized at the merge, and its hole check was elided.

The regression test (`switch-labels.js`) nails it: a `break target` jumps out
of the switch *before* `x = 1` runs, but both the case fall-through and the
`default` initialize `x`, so the old elision concluded "`x` always
initialized" and dropped the check at `print(x)`. Result: `print(x)` reads the
**raw hole** instead of throwing `ReferenceError` — "crash here due to seeing a
hole." A leaked hole sentinel flowing into normal JS operations is a **type
confusion** primitive (the hole has its own map/representation that the rest of
the engine never expects to see in a value slot).

### The fix

Give `ControlScopeForBreakable` its own `HoleCheckElisionMergeScope
merge_elider_`, and on `CMD_BREAK` call `merge_elider_.MergeBranch(generator())`
— folding the hole-check state **at each break** into the merge. Now a variable
is elidable after the construct only if initialized on every path *including
every break*. The class definitions are hoisted earlier in the file so the
control scope can hold one; `switch` is rewired to use the breakable scope's
elider (so `break`-out-of-switch and case merges share one consistent merge),
and labeled (breakable) blocks now Branch+Merge instead of using a plain
save/restore scope.

### Taxonomy notes (draft)

- **Root cause class:** unsound **safety-check elision** in a
  control-flow-sensitive analysis — a dataflow merge that **omitted an
  early-exit (`break`) edge**, so an "initialized on all paths" fact was
  computed over too few paths.
- **Exploit primitive:** elided TDZ check → the **hole sentinel leaks** as an
  ordinary value → type confusion.
- **Strong parallel to Bug 03:** both are *control-flow-merge bugs that drop an
  edge* — Bug 03 wrongly **included** a not-yet-valid back-edge (too
  conservative, lost a frame state); Bug 06 wrongly **excluded** a break edge
  (too aggressive, kept an unsafe elision). Same family: "the optimizer's CFG
  merge didn't match the real edge set." Candidate top-level taxonomy bucket:
  **CFG/dataflow merge over the wrong edge set.**
- **Reachability:** pure JS — labeled `switch`/block + `break` + `let`/`const`.
- **Fix shape:** make the merge account for every edge into the join (treat
  each `break` as a branch).

### Nelson's notes
- Unsoundness in dataflow analysis (hole check elision)
- n.b. this one is in bytecode generation, which is AST->bytecode and doesn't **have** a real CFG to work over

---

## Bug 07 — CVE-2025-13042 (High): Maglev leftover register allocations

- **CL:** [7119379](https://chromium-review.googlesource.com/c/v8/v8/+/7119379) — "[maglev] Fix left over register allocations from regalloc"
- **Component:** Maglev (mid-tier) compiler — register allocator (`src/maglev/maglev-regalloc.cc`)
- **Chrome's label:** Inappropriate implementation

### Background

Maglev's `StraightForwardRegisterAllocator` tracks, per node, where the node's
value currently lives — its `regalloc_info()` records whether it's in a
register (`has_register()`). Code generation later reads this location metadata
to decide whether to use a register directly or load the value from its spill
slot in memory.

### The bug

After the allocator finished processing all blocks, it **left register
allocations dangling** — some nodes were still recorded as living in a register
even though allocation was done. Per the commit message: *"The regalloc should
clear the node allocations when it is done. Failing to do so can cause the
codegen to use stale register state."*

The concrete failure (from the commit): **exception-handler trampolines**.
When codegen emitted the trampoline for an exception edge, it consulted the
stale "this value is in register R" metadata and therefore **skipped the load
from the spill slot**. At runtime the exception handler then reads register R —
which no longer holds that value — and proceeds with whatever stale contents
are there. A value materialized from the wrong location is a **type-confusion /
UB primitive** (the handler treats arbitrary register contents as a value of
the expected type). That's why it rates a High CVE even though the commit frames
it neutrally as a miscompilation.

### The fix

Add a `ClearRegisters()` call at the very end of `AllocateRegisters()` so no
node retains register state into codegen. Mechanically, the existing
`SpillAndClearRegisters` is refactored into a templated
`ClearRegisters<RegisterT, bool spill>`: `SpillAndClearRegisters` is now
`ClearRegisters<…, /*spill=*/true>` (save to spill slot **and** clear), while
the new end-of-allocation `ClearRegisters()` uses `spill=false` (just clear the
bookkeeping — the values are already where they need to be). A `DEBUG`-only
`DCHECK(!node->regalloc_info()->has_register())` is added at the top of
codegen's per-node `Process` to assert the invariant going forward.

### Taxonomy notes (draft)

- **Root cause class:** **stale location metadata** left by the register
  allocator — value-location bookkeeping not reset, so a later phase (codegen)
  trusted an out-of-date "lives in register R" fact.
- **Exploit primitive:** codegen skips a spill-slot reload on an exception edge
  → handler reads a register that no longer holds the value → wrong-location
  value → type confusion.
- **Third "non-mainline edge" bug:** the failure is on the **exception-handler**
  path — joining Bug 03 (**deopt** edge) and Bug 06 (**break** edge). Recurring
  V8 theme: **the happy path is fine; the bug lives on an exceptional/early-exit
  control-flow edge** whose value/state bookkeeping is out of sync. These are
  exactly the edges that are rarely exercised and easy to get wrong.
- **Fix shape:** reset the transient state at phase boundary + add a `DCHECK` to
  pin the invariant.

### Nelson's notes
- Incorrect tracking of SSA variable location (register vs stack slot) due to invariant violation in the register allocator.

---

## Bug 08 — CVE-2026-5871 (High): Maglev phi-untagging Smi→HeapNumber type widening

- **CL:** [7701796](https://chromium-review.googlesource.com/c/v8/v8/+/7701796) — "[maglev] Account for phi smi type widening in BuildCheckHeapObject"
- **Component:** Maglev compiler — graph builder type inference (`src/maglev/maglev-graph-builder.cc`)
- **Chrome's label:** Type Confusion

### Background — two distinct representation axes

The key to this bug is that Maglev tracks **two orthogonal axes**, and `kSmi`
lives on the first:

1. **`NodeType` lattice** (`kSmi`, `kHeapNumber`, `kString`, `kAnyHeapObject`,
   `kNumber`, …) — a lattice over the space of **tagged values**, used to elide
   checks and specialize ops. Crucially, `kSmi` vs `kHeapNumber` is *itself a
   representation distinction*: both are the same number, differing only in
   **boxing** — an immediate **Smi** (a small integer encoded directly in the
   tagged word) vs a heap-allocated **HeapNumber**. So `kSmi` is **not** an
   "abstract value" fact; it asserts "this number is boxed as a Smi." (`kSmi`
   and `kAnyHeapObject` are *disjoint* — a Smi is not a heap object — which is
   why `IntersectType(kSmi, kAnyHeapObject)` is empty and forces a deopt.)
2. **`ValueRepresentation`** (`kTagged`, `kInt32`, `kFloat64`, …) — the
   **machine** representation. Note a Smi is *tagged*; the genuinely *untagged*
   integers are `Int32`/`Float64`. **Untagging** moves a node from `kTagged` to
   `kInt32`/`kFloat64`.

`BuildCheckHeapObject(object)` emits a runtime "not a Smi" check (deopt if Smi)
and then **refines** the `NodeType` by removing `kSmi` via `EnsureType(object,
kAnyHeapObject)` (metadata only — see the EnsureType discussion above).

Separately, Maglev **untags phis** on the machine axis (a numeric phi becomes
`Float64`). The hazard: re-tagging is decided *later*, and a use that requires a
**heap object** (here marked by `phi->SetUseRequiresHeapObject()`) cannot be
satisfied by a Smi — so the untagged `Float64` is **boxed as a `HeapNumber`**.
The number's *tagged representation* thus flips from Smi to HeapNumber across
the untag→retag round-trip, even though the `NodeType` lattice still says
`kSmi`.

### The bug — the two axes get out of sync

In `BuildCheckHeapObject`, when `initial_type` included `kSmi`, the code removed
`kSmi` (now "heap object") but **didn't record that `HeapNumber` is still
possible**. If the rest of `initial_type` was, say, `kString`, the refined type
collapsed to "definitely a String" — yet a phi the lattice typed `kSmi`, once
untagged to `Float64` and re-tagged for a heap-object use, materializes as a
**`HeapNumber`**. Downstream code trusting "this is a String" then operates on
a `HeapNumber`'s layout → **type confusion**.

The trap is subtle: removing `kSmi` *looks* like pure, safe narrowing. But the
`NodeType` `kSmi` fact (a *tagged-representation* claim) was silently
invalidated by a `ValueRepresentation` decision (untagging, on the orthogonal
machine axis) made in a later pass — so dropping `kSmi` discarded a `HeapNumber`
possibility that the lattice had no way to see coming.

### The fix

After `EnsureType(... kAnyHeapObject)`, if `object` is a `Phi` and
`initial_type` could be `kSmi`, **union `kHeapNumber` back into** the phi's type
info (`info->UnionType(NodeType::kHeapNumber)`), so the refined type keeps
HeapNumber as a possibility. The author flags it as a deliberate **band-aid**:
the deeper issue is that *any* `GetType(phi)` returning `kSmi` may actually be a
`HeapNumber`, so other sites could have the same latent unsoundness — this CL
only patches the `BuildCheckHeapObject` case.

### Taxonomy notes (draft)

- **Root cause class:** **two representation axes out of sync** — a `NodeType`
  fact (`kSmi`, the *tagged* Smi-vs-HeapNumber boxing) invalidated by a
  `ValueRepresentation` decision (machine-level untagging) taken in a later
  pass. The lattice refinement (`remove kSmi`) couldn't see that the untag→retag
  round-trip can box the value as a `HeapNumber`.
- **Note on terminology:** `kSmi` is *not* "abstract value, not representation"
  (an earlier mis-framing) — it **is** a representation fact, just on the tagged
  axis (Smi vs HeapNumber boxing). A Smi is *tagged*; untagged numbers are
  `Int32`/`Float64`. The bug is precisely the seam between the tagged-axis fact
  and the machine-axis decision.
- **Exploit primitive:** a value mis-typed as a non-`HeapNumber` heap object
  while actually a `HeapNumber` → downstream layout assumptions break → type
  confusion.
- **Echoes Bug 02:** both are **representation-vs-logical-value** bugs. Bug 02:
  an immutable string *value* with a mutable *heap representation* (Thin/
  External). Bug 08: a number whose *tagged representation* (Smi) is flipped to
  HeapNumber by an orthogonal *machine-representation* decision (untagging). In
  both, the optimizer's abstraction didn't track a representation change
  underneath it.
- **Self-described band-aid:** fix is narrow and acknowledged incomplete — worth
  flagging as a "spot fix, root cause still latent" in the taxonomy (cf. Bug 04's
  kill switch as another "not the real fix" shape, though for a different reason).
- **Reachability:** pure JS that drives Maglev to untag a numeric phi (numeric
  loops mixing Smi-range and out-of-range/float values) reaching a heap-object
  check.
- **Fix shape:** restore the dropped possibility in the type refinement.

### Nelson's notes
- Unsoundness in the type representation lattice (An Smi can be lifted to a HeapNumber, but the type information does not reflect that fact)
- Specifically, downstream of some details of Phi handling
- CFG invariant violation, and representation vs logical value

---

## Bug 09 — CVE-2021-4078 (High): clobbered argument frame in `Runtime_DefineClass`

- **CL:** [3283077](https://chromium-review.googlesource.com/c/v8/v8/+/3283077) — "[runtime] Reset clobbered argument in DefineClass"
- **Component:** Runtime — class definition (`src/runtime/runtime-classes.cc`), arguments helper (`src/execution/arguments.h`)
- **Chrome's label:** Type Confusion

### Background

A runtime function receives its arguments as an `Arguments` array. Critically,
that array is **not a private copy** — it aliases the caller's argument frame,
and **for a function called from the interpreter it aliases the interpreter's
register frame** (the caller's locals/registers live there). So mutating an
argument slot mutates the caller's register.

`Runtime_DefineClass` (the runtime behind a `class … {}` definition) needed the
freshly created prototype to be visible at a fixed argument index
(`kPrototypeArgumentIndex`) so its sub-helpers (`InitClassConstructor` /
`InitClassPrototype`) could read it — it reused the argument array as a scratch
channel to pass the prototype along.

### The bug

The old code did this with a **permanent** overwrite:

```cpp
args.set_at(ClassBoilerplate::kPrototypeArgumentIndex, *prototype);
```

and never restored the slot. Since that slot aliases a register in the calling
JS function's interpreter frame, the prototype object was **left sitting in one
of the caller's registers** after the runtime returned. The interpreter then
continued executing the caller's bytecode, reading that register expecting
whatever value the JS code had put there — but finding the class prototype
instead → **type confusion** (the bytecode operates on the prototype as if it
were the original, differently-typed value).

### The fix

Two changes:

1. **Scoped, restoring mutation.** Replace the permanent `set_at` with a new
   RAII `Arguments::ChangeValueScope` that saves the old slot value, writes the
   temporary one, and **restores the original in its destructor** — so the slot
   is only the prototype *during* DefineClass's internal work and is put back
   before returning, leaving the caller's register frame intact.
2. **Footgun removal.** The unsafe `set_at` (permanently write an argument
   slot) is **deleted** from `Arguments`, so no other runtime function can
   clobber the caller's frame the same way.

### Taxonomy notes (draft)

- **Root cause class:** **mutation of aliased caller state without restore** —
  a runtime function used its (shared, frame-aliasing) argument array as scratch
  space and failed to put it back. Not a compiler bug at all; a runtime
  state-management bug.
- **Exploit primitive:** an attacker-controlled register slot in the
  interpreter frame is overwritten with a known object (the class prototype) →
  type confusion on the value the bytecode expected there.
- **Footgun-removal parallel to Bug 01:** same fix *shape* — delete/replace the
  dangerous primitive (`set_at` here; `DefineAccessor` → `…IgnoreAttributes`
  there) so misuse isn't reachable, on top of the local behavioral fix.
- **"Interpreter register frame" theme:** touches the same structure as Bug 07
  (Maglev leaving values in registers) but from the runtime side — here the
  hazard is that runtime args *alias* interpreter registers. Reinforces that
  V8's frame/register aliasing is a recurring sharp edge.
- **Reachability:** pure JS — evaluating a `class` definition.
- **Fix shape:** RAII save/restore of the temporarily-mutated shared slot +
  remove the unsafe permanent-write primitive.

### Nelson's notes
- Mutating an aliased array passed as an argument. Relatively vanilla mutability/sharing bug, exacerbated by the bytecode VM being brittle. Plausibly solvable via Rust-style borrowck

---

## Bug 10 — CVE-2026-7936 (Medium): flushed-bytecode field in `InterpreterEntryTrampoline`

- **CL:** [7664937](https://chromium-review.googlesource.com/c/v8/v8/+/7664937) — "[sandbox] Fix crash in InterpreterEntryTrampoline with flushed bytecode"
- **Component:** Hand-written builtins / macro-assembler across **all 7 CPU backends** (`src/builtins/{arm,arm64,ia32,loong64,mips64,riscv,x64}-…`, `src/codegen/*/macro-assembler-*`)
- **Chrome's label:** Object lifecycle issue

### Background

When a JS function is invoked, the `InterpreterEntryTrampoline` (and friends)
loads the function's bytecode from
`SharedFunctionInfo::kTrustedFunctionDataOffset`, reads its `Map` /
instance-type to dispatch on what kind of code object it is, and jumps in.

V8 also performs **bytecode flushing**: under memory pressure (and aggressively
under `--stress-flush-code`) it can **discard a function's compiled bytecode**,
to be lazily recompiled on next call. After a flush the trusted-data field no
longer holds a code/bytecode object:
- **sandbox on:** it's a **null indirect-pointer handle** (`kNullIndirectPointerHandle == 0`).
- **sandbox off:** it's a **Smi** (a sentinel, not a heap pointer).

### The bug

The entry path **didn't check for the flushed state**. It loaded `data` and
went straight to `LoadMap(data)` / reading the instance type — i.e. it
**dereferenced a Smi (or null handle) as if it were a heap-object pointer**.
Reading the "map" of a Smi reads from a small-integer-as-address → a wrong-type
/ out-of-bounds read that then drives code dispatch (deciding *what code to
run* for the function) → crash, and a type-confusion primitive on a
security-critical path. This is an **object-lifecycle** bug: the bytecode
object's lifetime ended (flushed) but the entry code still treated the field as
live.

The shared helper `LoadTrustedUnknownPointerField` made it worse: on an
invalid/none-of-the-expected-types field it would silently **zero the
destination and fall through**, conflating "valid pointer, unknown type" with
"field is invalid" — so callers couldn't tell a flushed field apart.

### The fix

- **Detect the flushed state and divert to recompilation.** Add an explicit
  check right after the load: `JumpIfSmi(data, is_unavailable)` in the
  per-backend trampolines, and in `LoadTrustedUnknownPointerField` a new
  `is_unavailable` label — taken when the handle is null (sandbox) or the value
  is a Smi (no sandbox). The helper now distinguishes "invalid field → jump to
  `is_unavailable`" from "valid but unrecognized type → zero & fall through."
  The `is_unavailable` path leads to lazily recompiling the function.
- **Replicated across all 7 architectures** by hand (arm, arm64, ia32, loong64,
  mips64, riscv, x64) — the same guard added to each backend's assembly.
- Also a drive-by: `GenerateTailCallToReturnedCode(Runtime::kCompileLazy)` →
  `TailCallBuiltin(Builtin::kCompileLazy)`.
- Regression test (`regress-490485402.js`) runs under `--stress-flush-code`,
  uses a failed asm.js link to leave a function in a flushable state, GCs to
  flush, then re-invokes it.

### Taxonomy notes (draft)

- **Root cause class:** **missing validity/type guard in hand-written
  assembly** for an **object-lifecycle** transition (code flushing) — a field
  assumed to always hold a live heap object can legitimately become a
  Smi/null-handle sentinel, and the dereference (`LoadMap`) had no Smi check.
- **Exploit primitive:** Smi-as-pointer dereference feeding code dispatch →
  wrong-type read deciding which code to execute → crash / type confusion on a
  control-flow-critical path.
- **Lifecycle-change theme (cf. Bugs 02 & 07):** a recurring shape where *the
  object/state changed underneath code that assumed stability* — Bug 02 (string
  representation rewritten), Bug 07 (register reassigned), Bug 10 (bytecode
  flushed). Here the change is **deallocation/flush** of the bytecode itself.
- **"Same bug × N backends" characteristic:** because this lives in per-arch
  hand-written assembly, the identical logic flaw existed in **7 parallel
  implementations** and had to be fixed in each — a structural risk multiplier
  unique to the assembly/builtins layer (contrast the compiler bugs, which are
  single-source).
- **Sandbox framing:** tagged "[sandbox]" — part of hardening the trampoline so
  a flushed/null trusted-pointer field can't be confused for a valid one
  (defense-in-depth for the V8 sandbox), with the no-sandbox Smi case fixed in
  the same CL.
- **Reachability:** pure JS under memory pressure / code flushing (reliably with
  `--stress-flush-code`; in the wild whenever the GC flushes a function's
  bytecode between calls).
- **Fix shape:** add the missing guard + give the shared helper a way to signal
  "field invalid" distinctly from "unknown type."

### Nelson's notes
- Two lenses on this one:
    1. It's a lifetime issue -- the compiler assumed bytecode for an interpreted function is immortal, but actually it can be flushed under memory pressure
    2. It's a missing check in compiled code -- the bytecode-entry trampoline assumes `trusted_function_data_` points to the function bytecode, but it

<!-- Local Variables: -->
<!-- flycheck-disabled-checkers: (markdown-aspell-dynamic) -->
<!-- eval: (auto-revert-mode 1) -->
<!-- End: -->
