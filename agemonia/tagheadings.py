"""Ensure every heading carries the page number printed in the book.

The transcribers tag headings as they go, but a few slipped through -- mostly
runs of short headings on the credits page, where the tag is repetitive. The
tag is derivable from the file, so fill in whatever is missing rather than
asking a model to redo the page.
"""

import json
import pathlib
import re

MD = pathlib.Path("out/md")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
TAGGED = re.compile(r"\((?:p\d+|cover|components|contents|back cover)\)$")

# The five unnumbered pages, by PDF page number.
UNNUMBERED = {1: "cover", 2: "components", 3: "components", 4: "contents", 5: "contents", 6: "contents", 68: "back cover"}


def main():
    index = {e["pdf_page"]: e for e in json.loads(pathlib.Path("out/index.json").read_text())}
    fixed = 0
    for path in sorted(MD.glob("p*.md")):
        pno = int(path.stem[1:])
        printed = index[pno]["printed"]
        tag = f"p{printed}" if printed is not None else UNNUMBERED.get(pno, "unnumbered")

        lines = path.read_text().splitlines()
        out = []
        for line in lines:
            m = HEADING.match(line)
            if m and not TAGGED.search(m.group(2)):
                out.append(f"{m.group(1)} {m.group(2)} ({tag})")
                fixed += 1
            else:
                out.append(line)
        path.write_text("\n".join(out) + "\n")
    print(f"added a page tag to {fixed} headings")


if __name__ == "__main__":
    main()
