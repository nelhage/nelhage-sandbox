// Dump anonymous rw- heap of the (already-rendered) Kindle process via
// send(meta, buffer) binary transfer. The KFX content key is a raw 16-byte
// AES key in native heap; dumping it lets us brute-force it offline.

function heapRegions() {
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
    var regions = heapRegions();
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
