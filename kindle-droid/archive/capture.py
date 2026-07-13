"""Attach hook_crypto.js to Kindle and log captured crypto to a file.

Usage: python capture.py <seconds> [outfile]
Stays attached for <seconds> while you open books, writing each captured line.
"""
import sys, time, frida

secs = int(sys.argv[1]) if len(sys.argv) > 1 else 60
outpath = sys.argv[2] if len(sys.argv) > 2 else "capture.log"

dev = frida.get_usb_device()
pid = None
for p in dev.enumerate_processes():
    if p.name == "Kindle":
        pid = p.pid
        break
if pid is None:
    print("Kindle not running")
    sys.exit(1)
print("attaching to Kindle pid", pid)

session = dev.attach(pid)
script = session.create_script(open("hook_crypto.js").read())
out = open(outpath, "w")

def on_message(msg, data):
    if msg.get("type") == "send":
        line = msg["payload"]
    elif msg.get("type") == "error":
        line = "ERROR " + msg.get("stack", str(msg))
    else:
        line = str(msg)
    print(line)
    out.write(line + "\n")
    out.flush()

# hook_crypto.js uses console.log -> arrives as log messages, not send
script.set_log_handler(lambda level, text: (print(text), out.write(text + "\n"), out.flush()))
script.on("message", on_message)
script.load()
print("loaded; capturing for", secs, "s")
time.sleep(secs)
print("done")
session.detach()
out.close()
