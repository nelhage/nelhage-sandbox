"""Open a book by ASIN (in-process via KRX) and dump the heap, in one frida session.

Usage: python open_dump.py <ASIN> <outfile.bin> [load_secs]

Loads krx_agent.js to call getReaderManager().openBook(asin) so KRF loads and
decrypts the book (caching its 16-byte content key in native heap), waits for the
load, then dumps anonymous rw heap via dump_mem.js.  No visible reader needed.
"""
import sys, threading, time, frida

asin = sys.argv[1]
outpath = sys.argv[2] if len(sys.argv) > 2 else "heap.bin"
load_secs = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0

dev = frida.get_usb_device()
pid = next((p.pid for p in dev.enumerate_processes() if p.name == "Kindle"), None)
if pid is None:
    print("Kindle not running"); sys.exit(1)
print("attaching to", pid)
session = dev.attach(pid)

# 1) open the book so KRF loads + decrypts it
opener = session.create_script(open("krx_agent.js").read())
opener.on("message", lambda m, d: print("  opener:", m.get("payload") or m))
opener.load()
print("open ->", opener.exports_sync.open(asin))
print(f"waiting {load_secs}s for KRF to load/decrypt…")
time.sleep(load_secs)

# 2) dump heap
dumper = session.create_script(open("dump_mem.js").read())
f = open(outpath, "wb"); idx = open(outpath + ".idx", "w")
state = {"off": 0, "n": 0}; done = threading.Event()

def on_message(msg, data):
    if msg.get("type") == "error":
        print("ERR", msg.get("stack")); return
    if msg.get("type") != "send":
        return
    p = msg["payload"]
    if p.get("done"):
        done.set(); return
    if data:
        f.write(data)
        idx.write(f"{state['off']} {p['base']} {len(data)}\n")
        state["off"] += len(data); state["n"] += 1

dumper.on("message", on_message)
dumper.load()
count = dumper.exports_sync.dump_all()
done.wait(timeout=300)
f.close(); idx.close(); session.detach()
print(f"dumped {count} regions, {state['off']/1e6:.1f} MB -> {outpath}")
