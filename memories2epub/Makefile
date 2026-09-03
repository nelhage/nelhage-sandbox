# Scanned book -> EPUB. Stages are idempotent scripts; `make` runs them in order.
# Sub-agent review results are dropped into out/corrections/*.json and picked up
# by `make epub`.

PDF := book.pdf

.PHONY: all render ocr marks layout figures backmatter flags italics epub verify clean

all: epub verify

out/png300/p-001.png: $(PDF)
	mkdir -p out/png300 && pdftoppm -r 300 -gray -png $(PDF) out/png300/p

render: out/png300/p-001.png
	python3 render.py

ocr: render
	python3 ocr.py > out/ocr.log

marks:
	python3 superscripts.py

figures:
	python3 figures.py

backmatter:
	python3 notes.py && python3 chronology.py && python3 index.py

layout:
	python3 layout.py

flags: layout
	python3 flags.py

italics: layout
	python3 italics_scan.py

epub: layout
	python3 assemble.py

verify:
	python3 verify.py

clean:
	rm -rf out/blocks out/flags out/book.md out/book.epub out/pagetext
