PDF   := $(wildcard pdfs/*.pdf)
NAME  := $(basename $(notdir $(PDF)))
MD    := out/$(NAME).md
EPUB  := out/$(NAME).epub

.PHONY: all md epub clean
all: epub

md: $(MD)
epub: $(EPUB)

$(MD): pdf2md.py $(PDF)
	uv run python pdf2md.py "$(PDF)" out

# --split-level=2 chapters on the `##` headings; the title comes from the
# Markdown YAML metadata block.
$(EPUB): $(MD)
	pandoc "$(MD)" -o "$@" \
	  --from=markdown+footnotes+yaml_metadata_block \
	  --resource-path=out --toc --toc-depth=3 --split-level=2

clean:
	rm -rf out
