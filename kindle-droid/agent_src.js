import Java from 'frida-java-bridge';

let SDK = null;                 // cached live IKindleReaderSDK instance
const SDK_CLASSES = [
  'com.amazon.kindle.krx.BaseKindleReaderSDK',
  'com.amazon.kindle.krx.a',
];

function findSDK() {
  if (SDK !== null) return SDK;
  for (const cn of SDK_CLASSES) {
    let found = null;
    try {
      Java.choose(cn, {
        onMatch(inst) { found = inst; return 'stop'; },
        onComplete() {},
      });
    } catch (e) { /* class not present */ }
    if (found) { SDK = found; return SDK; }
  }
  return null;
}

// Run f() on the Android main (UI) thread, resolve with its return value.
function onMain(f) {
  return new Promise((resolve, reject) => {
    Java.scheduleOnMainThread(() => {
      try { resolve(f()); } catch (e) { reject('' + e + (e.stack ? '\n' + e.stack : '')); }
    });
  });
}


// Find the currently-resumed Activity via ActivityThread (main thread only).
function currentActivity() {
  const ActivityThread = Java.use('android.app.ActivityThread');
  const at = ActivityThread.currentActivityThread();
  const map = at.mActivities.value;                 // ArrayMap<IBinder, ActivityClientRecord>
  const vals = map.values().toArray();
  for (let i = 0; i < vals.length; i++) {
    const rec = Java.cast(vals[i], Java.use('android.app.ActivityThread$ActivityClientRecord'));
    let paused = true;
    try { paused = rec.paused.value; } catch (e) {}
    if (!paused) return rec.activity.value;
  }
  // fallback: return any activity
  if (vals.length) {
    const rec = Java.cast(vals[0], Java.use('android.app.ActivityThread$ActivityClientRecord'));
    return rec.activity.value;
  }
  return null;
}

rpc.exports = {
  // Diagnostic: which SDK instance did we bind, and can we resolve a book?
  info(asin) {
    return new Promise((resolve) => {
      Java.perform(() => {
        const sdk = findSDK();
        if (!sdk) { resolve({ ok: false, err: 'no SDK instance found' }); return; }
        let book = null, binfo = {};
        try {
          book = sdk.getLibraryManager().getContentFromAsin(asin, false);
          if (book) {
            binfo.title = '' + book.getTitle();
            try { binfo.state = '' + book.getContentState(); } catch (e) { binfo.state = 'n/a'; }
          }
        } catch (e) { binfo.err = '' + e; }
        resolve({ ok: true, sdkClass: sdk.$className, hasBook: !!book, book: binfo });
      });
    });
  },

  download(asin) {
    return new Promise((resolve) => {
      Java.perform(() => onMain(() => {
        const sdk = findSDK();
        const book = sdk.getLibraryManager().getContentFromAsin(asin, false);
        if (!book) return { ok: false, err: 'no book for asin' };
        sdk.getStoreManager().downloadBook(book);
        return { ok: true, title: '' + book.getTitle() };
      }).then(resolve).catch((e) => resolve({ ok: false, err: '' + e })));
    });
  },

  open(asin) {
    return new Promise((resolve) => {
      Java.perform(() => onMain(() => {
        const sdk = findSDK();
        const book = sdk.getLibraryManager().getContentFromAsin(asin, false);
        if (!book) return { ok: false, err: 'no book for asin' };
        const act = currentActivity();
        const rm = sdk.getReaderManager();
        const r = rm.openBook(book, null, null, act);
        return { ok: true, opened: r, activity: act ? act.$className : null, title: '' + book.getTitle() };
      }).then(resolve).catch((e) => resolve({ ok: false, err: '' + e })));
    });
  },

  // ASIN of the book currently loaded in the reader (null if none). Used to
  // confirm the *right* book is open + rendered before dumping the heap.
  curasin() {
    return new Promise((resolve) => {
      Java.perform(() => {
        try {
          const sdk = findSDK();
          const rm = sdk.getReaderManager();
          const b = rm.getCurrentBook();
          resolve(b ? ('' + b.getASIN()) : null);
        } catch (e) { resolve(null); }
      });
    });
  },
};
