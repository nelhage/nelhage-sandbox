"""Dump Kindle heap to a file (binary send transfer) for offline key recovery.

Usage: python dump_driver.py <outfile.bin>
Attach AFTER the target book is fully rendered on screen.
"""
import sys, threading, frida

outpath = sys.argv[1] if len(sys.argv) > 1 else "heap.bin"
dev = frida.get_usb_device()
pid = next((p.pid for p in dev.enumerate_processes() if p.name == "Kindle"), None)
if pid is None:
    print("Kindle not running"); sys.exit(1)
print("attaching to", pid)
session = dev.attach(pid)
script = session.create_script(open("dump_mem.js").read())

f = open(outpath, "wb")
idx = open(outpath + ".idx", "w")
state = {"off": 0, "n": 0}
done = threading.Event()

def on_message(msg, data):
    if msg.get("type") != "send":
        if msg.get("type") == "error":
            print("ERR", msg.get("stack"))
        return
    p = msg["payload"]
    if p.get("done"):
        done.set(); return
    if data:
        f.write(data)
        idx.write(f"{state['off']} {p['base']} {len(data)}\n")
        state["off"] += len(data)
        state["n"] += 1
        if state["n"] % 200 == 0:
            print(f"  {state['n']} chunks, {state['off']/1e6:.1f} MB")

script.set_log_handler(lambda level, text: print(text))
script.on("message", on_message)
script.load()
count = script.exports_sync.dump_all()
done.wait(timeout=300)
f.close(); idx.close()
session.detach()
print(f"dumped {count} regions, {state['off']/1e6:.1f} MB -> {outpath}")
