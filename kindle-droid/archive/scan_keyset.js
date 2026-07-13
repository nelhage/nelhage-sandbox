// Scan Kindle process memory for decrypted KFX DRM Ion documents.
// Every KFX DRM Ion doc (voucher, KeySet, decrypted content) begins with the
// same "ProtectedData" shared-symbol-table import header.  The decrypted
// *KeySet* (voucher plaintext) is small and carries the 16-byte content key,
// so scanning for the header surfaces content keys without needing any symbol.

var HEADER = 'e0 01 00 ea ee 9e 81 83 de 9a 86 be 97 de 95 84 8d 50 72 6f 74 65 63 74 65 64 44 61 74 61 85';

function scanOnce() {
  var results = [];
  var ranges = Process.enumerateRanges('rw-');
  var scanned = 0;
  ranges.forEach(function (r) {
    if (r.size > 96 * 1024 * 1024) return;   // skip giant graphics/JIT ranges
    scanned++;
    try {
      Memory.scanSync(r.base, r.size, HEADER).forEach(function (m) {
        var bytes = Memory.readByteArray(m.address, 320);
        results.push({ addr: m.address.toString(), hex: bufToHex(bytes) });
      });
    } catch (e) {}
  });
  console.log('[scanned ' + scanned + '/' + ranges.length + ' ranges, ' + results.length + ' hits]');
  return results;
}

function bufToHex(ba) {
  var u = new Uint8Array(ba), s = '';
  for (var i = 0; i < u.length; i++) s += ('0' + u[i].toString(16)).slice(-2);
  return s;
}

rpc.exports = {
  scan: function () { return scanOnce(); }
};
console.log('[scanner ready]');
