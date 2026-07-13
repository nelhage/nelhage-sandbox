// Dump the Kindle process's NATIVE heap (scudo) via send(meta, buffer) binary
// transfer.  The KFX/MOBI content key is a raw 16-byte AES key that KRF (native
// C++) mallocs; on Android's scudo allocator that lands in [anon:scudo:primary]
// (small allocations) — never in the Java/ART heap, thread stacks, WebView
// partition_alloc, .bss, etc.  Restricting the dump to scudo cut the search
// space ~4x (979MB all-anon -> 231MB primary / 424MB incl. secondary) in
// measurement, speeding up both the USB transfer and the offline numpy brute
// and keeping the align=1 pass off the OOM line.  Ground truth: for B00KVI76ZS
// the known key appeared 7x, all inside a single [anon:scudo:primary] region.
//
// enumerateRanges() does not expose the [anon:...] VMA name, so we parse
// /proc/self/maps (readable in-process) to find the scudo ranges.
//
// DELTA MODE (baseline/dumpDelta): the key is *derived* when the book renders,
// so it lands in memory that changed since before the open.  baseline() FNV-
// hashes every 4KB scudo page (call it at Home, before open()); dumpDelta()
// re-hashes and streams only the changed/new pages.  Opening a book dirties
// only ~35MB of the 231MB primary heap, so the delta brute is ~6x cheaper.
// Correctness caveat: if the book was opened earlier and its key is still
// resident at baseline, the key sits in UNCHANGED pages and the delta misses
// it — so the driver must fall back to a full dumpAll() when the delta brute
// comes up empty.  (Soft-dirty page tracking would be cleaner but this
// emulator's kernel zeroes /proc/pid/pagemap even for root.)

// Which scudo classes to dump.  'primary' (small allocations) held the key in
// every case observed.  'secondary' (large single allocations) is kept on as
// cheap insurance; drop it to primary-only for a ~4x dump once trusted.
var WANT = { primary: true, secondary: true };

function scudoRanges() {
  var out = [];
  var maps;
  try { maps = File.readAllText('/proc/self/maps'); }
  catch (e) { return null; }                    // signal: fall back to old behaviour
  maps.split('\n').forEach(function (ln) {
    // e.g. "7c6ca43000-7c6cf83000 rw-p 00000000 00:00 0   [anon:scudo:primary]"
    var m = ln.match(/^([0-9a-f]+)-([0-9a-f]+)\s+(\S{4})\s.*\[anon:scudo:(\w+)\]/);
    if (!m) return;
    var perms = m[3];
    if (perms[0] !== 'r' || perms[1] !== 'w') return;   // rw only (reserve is ---p)
    if (!WANT[m[4]]) return;
    var base = ptr('0x' + m[1]);
    var size = ptr('0x' + m[2]).sub(base).toUInt32();
    out.push({ base: base, key: base.toString(), size: size });
  });
  return out;
}

// Fallback: original behaviour (all anonymous rw- regions < 128MB) if we can't
// read/parse maps for some reason.
function fallbackRegions() {
  var out = [];
  Process.enumerateRanges('rw-').forEach(function (r) {
    if (r.size > 128 * 1024 * 1024) return;
    if (r.file) return;
    out.push({ base: r.base, size: r.size });
  });
  return out;
}

var PAGE = 4096;
var WORDS = PAGE / 4;         // 1024 uint32 words per page
var STRIDE = 16;             // sample every 16th word (64 bytes) -> 64 samples/page

// Per-page change fingerprint over a whole region read into one ArrayBuffer.
// Dense per-byte hashing across the JS bridge measured ~3 MB/s (unusably slow
// for a 100-400MB heap); sampling 64 words/page from a single bulk read is
// ~500 MB/s.  A page that received a freshly-written key allocation changes far
// more than one sampled word, so sampling reliably flags it; the rare miss
// (key overwritten into an otherwise-untouched page) is caught by the full-dump
// fallback in the driver.
function sparseHashRegion(ab, size) {
  var dv = new Uint32Array(ab);
  var np = Math.floor(size / PAGE), out = new Array(np);
  for (var p = 0; p < np; p++) {
    var base = p * WORDS, h = 2166136261;
    for (var k = 0; k < WORDS; k += STRIDE) { h ^= dv[base + k]; h = (h * 16777619) >>> 0; }
    out[p] = h;
  }
  return out;
}

function streamRegions(regions, mode) {
  var total = 0;
  regions.forEach(function (r) { total += r.size; });
  console.log('[dumper] mode=' + mode + ' regions=' + regions.length + ' bytes=' + total);
  var CHUNK = 1024 * 1024;
  regions.forEach(function (r) {
    var pos = 0;
    while (pos < r.size) {
      var n = Math.min(CHUNK, r.size - pos);
      var addr = r.base.add(pos);
      var buf = null;
      try { buf = addr.readByteArray(n); } catch (e) {}
      if (buf) send({ base: addr.toString(), size: n }, buf);
      pos += n;
    }
  });
  send({ done: true });
  return regions.length;
}

rpc.exports = {
  // Full native-heap dump (safe path / fallback).
  dumpAll: function () {
    var regions = scudoRanges();
    if (regions === null || regions.length === 0) return streamRegions(fallbackRegions(), 'fallback-all-anon');
    return streamRegions(regions, 'scudo-full');
  },

  // Per-page change fingerprints of the scudo heap, keyed by region base.  Call
  // BEFORE open() so the key isn't in it yet.  Returns { "<baseHex>": [h0,...] }.
  baseline: function () {
    var ranges = scudoRanges() || [];
    var out = {}, pages = 0;
    ranges.forEach(function (r) {
      var ab;
      try { ab = r.base.readByteArray(r.size); } catch (e) { return; }
      if (!ab) return;
      var arr = sparseHashRegion(ab, r.size);
      out[r.key] = arr; pages += arr.length;
    });
    console.log('[dumper] baseline: ' + ranges.length + ' ranges, ' + pages + ' pages');
    return out;
  },

  // Stream only the scudo pages that changed vs `baseline` (a map from
  // baseline()).  New regions / new pages count as changed.  Falls back to a
  // full dump if the baseline is empty (baseline() couldn't read maps).
  dumpDelta: function (baseline) {
    var ranges = scudoRanges();
    if (ranges === null || ranges.length === 0) return streamRegions(fallbackRegions(), 'fallback-all-anon');
    if (!baseline || Object.keys(baseline).length === 0) return streamRegions(ranges, 'scudo-full(no-baseline)');
    var sent = 0, changedPages = 0;
    ranges.forEach(function (r) {
      var ab;
      try { ab = r.base.readByteArray(r.size); } catch (e) { return; }
      if (!ab) return;
      var bl = baseline[r.key];                   // array or undefined (=> all changed)
      var cur = sparseHashRegion(ab, r.size), np = cur.length, runStart = -1;
      function flush(endPage) {                   // send pages [runStart,endPage) from ab
        if (runStart < 0) return;
        var off = runStart * PAGE, end = endPage * PAGE;
        send({ base: r.base.add(off).toString(), size: end - off }, ab.slice(off, end));
        sent += end - off; runStart = -1;
      }
      for (var i = 0; i < np; i++) {
        var changed = !bl || i >= bl.length || bl[i] !== cur[i];
        if (changed) { if (runStart < 0) runStart = i; changedPages++; }
        else flush(i);
      }
      flush(np);
    });
    console.log('[dumper] delta: ' + changedPages + ' changed pages, ' + sent + ' bytes');
    send({ done: true });
    return sent;
  }
};
console.log('[dumper ready]');
