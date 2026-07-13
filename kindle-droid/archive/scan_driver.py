"""Attach the memory scanner to Kindle, scan repeatedly, save unique hits.

Usage: python scan_driver.py <iterations> <outfile>
Run it, and while it loops, open a book in the app.
"""
import sys, time, frida

iters = int(sys.argv[1]) if len(sys.argv) > 1 else 20
outpath = sys.argv[2] if len(sys.argv) > 2 else "scan_hits.txt"

dev = frida.get_usb_device()
pid = next((p.pid for p in dev.enumerate_processes() if p.name == "Kindle"), None)
if pid is None:
    print("Kindle not running"); sys.exit(1)
print("attaching to", pid)
session = dev.attach(pid)
script = session.create_script(open("scan_keyset.js").read())
script.set_log_handler(lambda level, text: print(text))
script.load()

seen = set()
out = open(outpath, "w")
for i in range(iters):
    try:
        hits = script.exports_sync.scan()
    except Exception as e:
        print("scan error:", e); break
    new = 0
    for h in hits:
        if h["hex"] not in seen:
            seen.add(h["hex"])
            out.write(h["addr"] + " " + h["hex"] + "\n")
            out.flush()
            new += 1
    print(f"iter {i}: {len(hits)} hits, {new} new, {len(seen)} unique total")
    time.sleep(1)
out.close()
session.detach()
print("wrote", len(seen), "unique buffers to", outpath)
