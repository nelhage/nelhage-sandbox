// Frida hook to capture Kindle Android KFX DRM keys on a stable (4k-page) AVD.
//
// The voucher key-derivation and content decryption run in
// libKindleAndroidNativeBundlerJNI.so via system BoringSSL (libcrypto).
// We hook the crypto primitives in libcrypto to capture:
//   * every AES-CBC key/iv  (EVP_CipherInit_ex / EVP_DecryptInit_ex)
//       -> the 32-byte key decrypts the 512-byte voucher cipher_text (=> content key)
//       -> the 16-byte key decrypts the DRMION content pages
//   * every HMAC(key,data)  -> reveals sharedsecret = obfuscate(shared) for a
//       known `shared` input (lets us reverse the per-version scramble offline)
//
// Usage:  frida -U -p <Kindle pid> -l hook_crypto.js   (then open each book)

'use strict';

function hexdump_bytes(ptr, len) {
  try { return Memory.readByteArray(ptr, len); } catch (e) { return null; }
}
function toHex(ba) {
  if (!ba) return '<unreadable>';
  var u = new Uint8Array(ba), s = '';
  for (var i = 0; i < u.length; i++) s += ('0' + u[i].toString(16)).slice(-2);
  return s;
}

var seen = {};
function once(tag, s) {
  var k = tag + ':' + s;
  if (seen[k]) return false;
  seen[k] = true;
  return true;
}

var LIBC = null;
function exp(name) {           // frida 17: exports are instance methods on Module
  return LIBC ? LIBC.findExportByName(name) : null;
}

function hookLibcrypto() {
  var mod = Process.findModuleByName('libcrypto.so');
  if (!mod) { console.log('[!] libcrypto.so not loaded yet'); return false; }
  LIBC = mod;
  console.log('[+] libcrypto.so @ ' + mod.base);

  // int EVP_DecryptInit_ex(ctx, const EVP_CIPHER *type, impl, const uint8_t *key, const uint8_t *iv)
  ['EVP_DecryptInit_ex', 'EVP_EncryptInit_ex'].forEach(function (name) {
    var p = exp(name);
    if (!p) return;
    Interceptor.attach(p, {
      onEnter: function (args) {
        var key = args[3], iv = args[4];
        // key length is unknown here; dump 32 bytes and record both 16/32 slices offline
        var kb = toHex(hexdump_bytes(key, 32));
        var ib = toHex(hexdump_bytes(iv, 16));
        if (once(name, kb + ib))
          console.log('[' + name + '] key32=' + kb + ' iv=' + ib);
      }
    });
    console.log('[+] hooked ' + name);
  });

  // int EVP_CipherInit_ex(ctx, type, impl, key, iv, int enc)
  var pc = exp('EVP_CipherInit_ex');
  if (pc) {
    Interceptor.attach(pc, {
      onEnter: function (args) {
        var enc = args[5].toInt32();
        var kb = toHex(hexdump_bytes(args[3], 32));
        var ib = toHex(hexdump_bytes(args[4], 16));
        if (once('EVP_CipherInit_ex', enc + kb + ib))
          console.log('[EVP_CipherInit_ex enc=' + enc + '] key32=' + kb + ' iv=' + ib);
      }
    });
    console.log('[+] hooked EVP_CipherInit_ex');
  }

  // uint8_t *HMAC(evp_md, key, key_len, data, data_len, md, md_len)
  var ph = exp('HMAC');
  if (ph) {
    Interceptor.attach(ph, {
      onEnter: function (args) {
        this.klen = args[2].toInt32();
        this.dlen = args[4].toInt32();
        this.key = toHex(hexdump_bytes(args[1], this.klen));
        var d = hexdump_bytes(args[3], Math.min(this.dlen, 96));
        // show data as ascii if printable (e.g. "PIDv3"), else hex
        this.data = toHex(d);
        try {
          var u = new Uint8Array(d), asc = '', ok = true;
          for (var i = 0; i < u.length; i++) {
            if (u[i] >= 32 && u[i] < 127) asc += String.fromCharCode(u[i]);
            else { ok = false; break; }
          }
          if (ok) this.data = '"' + asc + '"';
        } catch (e) {}
      },
      onLeave: function () {
        if (once('HMAC', this.key + this.data))
          console.log('[HMAC] key(' + this.klen + ')=' + this.key + ' data(' + this.dlen + ')=' + this.data);
      }
    });
    console.log('[+] hooked HMAC');
  }
  return true;
}

function globalExp(name) {
  try { return Module.getGlobalExportByName(name); } catch (e) {}
  var mods = Process.enumerateModules();
  for (var i = 0; i < mods.length; i++) {
    var p = mods[i].findExportByName(name);
    if (p) return p;
  }
  return null;
}

if (!hookLibcrypto()) {
  // libcrypto may load lazily; retry when a new module is dlopen'd
  var api = globalExp('android_dlopen_ext');
  if (api) Interceptor.attach(api, { onLeave: function () { hookLibcrypto(); } });
}
