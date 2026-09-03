"""Stage 2: tesseract hOCR for every cleaned page image.

out/hocr/p-NNN.hocr  (+ .txt)  --psm 1 on out/pages/p-NNN.png
out/hocr4/p-NNN.hocr           --psm 4 for the Chronology pages (317-328), which
                                keeps the date and its text on one line.
Usage: python3 ocr.py [pages...]   (default: all)
"""
import pathlib, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

SRC = pathlib.Path("out/pages")
CHRONOLOGY = range(317, 329)

def run(args):
    png, outbase, psm = args
    subprocess.run(["tesseract", str(png), str(outbase), "-l", "eng", "--psm", str(psm), "hocr", "txt"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return outbase

if __name__ == "__main__":
    pages = [int(p) for p in sys.argv[1:]] or None
    jobs = []
    for png in sorted(SRC.glob("p-*.png")):
        n = int(png.stem.split("-")[1])
        if pages and n not in pages: continue
        pathlib.Path("out/hocr").mkdir(exist_ok=True)
        jobs.append((png, pathlib.Path("out/hocr") / png.stem, 1))
        if n in CHRONOLOGY:
            pathlib.Path("out/hocr4").mkdir(exist_ok=True)
            jobs.append((png, pathlib.Path("out/hocr4") / png.stem, 4))
    with ThreadPoolExecutor(12) as ex:
        for i, _ in enumerate(ex.map(run, jobs), 1):
            if i % 50 == 0: print(i, "/", len(jobs), flush=True)
    print("done", len(jobs))
