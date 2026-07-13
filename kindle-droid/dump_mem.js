// Dump the Kindle process's NATIVE heap (scudo) via send(meta, buffer) binary
// transfer.  The KFX/MOBI content key is a raw 16-byte AES key that KRF (native
// C++) mallocs; on Android's scudo allocator that lands in [anon:scudo:primary]
// (small allocations) — never in the Java/ART heap, thread stacks, WebView
// partition_alloc, .bss, etc.  Restricting the dump to scudo regions cut the
// search space ~4x (979MB -> ~230MB primary; ~420MB incl. secondary) in
// measurement, which speeds up both the USB transfer and the offline brute and
// avoids OOM on the align=1 numpy pass.  Ground truth: for B00KVI76ZS the known
// key appeared 7x, all inside a single [anon:scudo:primary] region.
//
// enumerateRanges() does not expose the [anon:...] VMA name, so we parse
// /proc/self/maps (readable in-process) to find the scudo ranges, then read
// each with the enumerateRanges() protection as a backstop against stale maps.

// Which scudo classes to dump.  'primary' (small allocations) held the key in
// every case observed — B00KVI76ZS's key appeared only in [anon:scudo:primary].
// 'secondary' (large single allocations) is kept on by default as cheap
// insurance against a build embedding the key in a big structure; drop it to
// 'primary' only for a ~4x dump (231MB vs 424MB here) once you trust primary.
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
    out.push({ base: base, size: size });
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

rpc.exports = {
  dumpAll: function () {
    var regions = scudoRanges();
    var mode = 'scudo';
    if (regions === null || regions.length === 0) {
      regions = fallbackRegions();
      mode = 'fallback-all-anon';
    }
    var total = 0;
    regions.forEach(function (r) { total += r.size; });
    console.log('[dumper] mode=' + mode + ' regions=' + regions.length +
                ' bytes=' + total);
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
};
console.log('[dumper ready]');
