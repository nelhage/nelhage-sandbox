#!/usr/bin/env python3
import json
import re
import shutil
import sqlite3
import subprocess
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from urllib.parse import urlparse
from xml.sax.saxutils import escape

from lxml.html import html5parser, tostring
from lxml.html.html5parser import XHTML_NAMESPACE

INFO_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>
    <string>{id}</string>
    <key>CFBundleName</key>
    <string>{name}</string>
    <key>DocSetPlatformFamily</key>
    <string>{platform}</string>
    <key>isDashDocset</key>
    <true/>
</dict>
</plist>
"""

HTML_DOCTYPE = "<!DOCTYPE html>"


@dataclass
class IndexEntry:
    name: str
    type: str
    path: str

    @classmethod
    def from_entry(cls, entry, *, key: str, type: str):
        path = entry_path(entry)
        data = entry.get("data", None)
        name = key
        if isinstance(data, dict):
            for nameKey in ["userName", "title", "term"]:
                v = data.get(nameKey, None)
                if v is not None:
                    name = v
        return cls(name=name, type=type, path=path)


def entry_path(entry):
    path = entry["address"].removeprefix("/")
    if path.endswith("/"):
        path += "index.html"
    if "id" in entry:
        path = path + "#" + entry["id"]
    return path


def classify_doc_entry(key: str, entries: list) -> list[IndexEntry]:
    bits = key.rsplit(".", 1)
    if len(bits) == 1:
        ns = ""
        localname = bits[0]
    else:
        ns, localname = bits

    if len(entries) == 2:
        for_name = [e for e in entries if e["id"].endswith(localname)]
        for_mk = [e for e in entries if e["id"].endswith("__mk")]
        if len(for_name) == 1 and len(for_mk) == 1:
            return [
                IndexEntry.from_entry(for_mk[0], key=key, type="Method"),
                IndexEntry.from_entry(for_name[0], key=key, type="Type"),
            ]

    entry = entries[0]

    if ns:
        type = "Method"
    else:
        type = "Function"

    if "id" in entry and not entry["id"].endswith(localname):
        if entry["id"].endswith("__mk"):
            type = "Type"
        if entry["id"].endswith("__intro"):
            type = "Struct"

    return [IndexEntry.from_entry(entry, key=key, type=type)]


def classify_default(key: str, entries: list, type: str) -> list[IndexEntry]:
    return [IndexEntry.from_entry(entries[0], key=key, type=type)]


def write_index(docdir: Path, dbpath: Path):
    db = sqlite3.connect(dbpath, autocommit=False)
    db.executescript("""\
    CREATE TABLE searchIndex(id INTEGER PRIMARY KEY, name TEXT, type TEXT, path TEXT);
    CREATE UNIQUE INDEX anchor ON searchIndex (name, type, path);
    """)
    db.commit()

    with (docdir / "xref.json").open("r") as fh:
        xref = json.load(fh)

    xref_map = {
        "Verso.Genre.Manual.section": "Section",
        "Verso.Genre.Manual.example": "Sample",
        "Verso.Genre.Manual.doc.tech": "Word",
        "Verso.Genre.Manual.doc.tactic": "Procedure",
        "Verso.Genre.Manual.doc": classify_doc_entry,
    }

    index_entries = []
    for xref_group, cat in xref_map.items():
        if callable(cat):
            gen_items = cat
        else:
            gen_items = partial(classify_default, type=cat)

        for key, es in xref.get(xref_group, {}).get("contents", {}).items():
            if not es:
                continue
            try:
                index_entries.extend(gen_items(key, es))
            except Exception as ex:
                ex.add_note(f"While processing entry: root[{xref_group!r}][{key!r}]")
                raise

    BATCH_SIZE = 1024
    insert_batch = []

    def send_batch():
        if not insert_batch:
            return
        db.executemany(
            "INSERT INTO searchIndex (name, type, path) VALUES(?, ?, ?)",
            insert_batch,
        )
        insert_batch.clear()

    seen = set()
    for ent in index_entries:
        if (ent.type, ent.name) in seen:
            print(f"Duplicate entries for {ent}!")
            continue
        seen.add((ent.type, ent.name))
        insert_batch.append((ent.name, ent.type, ent.path))
        if len(insert_batch) >= BATCH_SIZE:
            send_batch()
    send_batch()

    db.commit()


def rewrite_one_page(path: Path) -> int:
    with path.open("r") as fh:
        tree = html5parser.HTMLParser(namespaceHTMLElements=False).parse(fh)

    links = tree.xpath("//a")
    changed = 0
    for a in links:
        parsed = urlparse(a.attrib["href"])
        if (not parsed.scheme) and parsed.path.endswith("/"):
            newurl = parsed._replace(path=parsed.path + "index.html")
            a.attrib["href"] = newurl.geturl()
            changed += 1

    if changed:
        html = tostring(tree, pretty_print=True, doctype=HTML_DOCTYPE)
        path.write_bytes(html)
    return changed


def rewrite_links(docdir: Path):
    for dirpath, dirs, files in docdir.walk():
        for filename in files:
            if not filename.endswith(".html"):
                continue
            fullpath = dirpath / filename
            changed = rewrite_one_page(fullpath)
            print(f"Rewrote {fullpath.relative_to(docdir)} ({changed} links)...")


def main(
    html_zip: Path,
    outpath: Path,
    *,
    overwrite: bool = False,
    bundle_id: str | None = None,
    bundle_name: str | None = None,
    bundle_platform: str = "unknown",
):
    if not html_zip.is_file():
        raise ValueError(f"input zip file must exist!")
    if outpath.exists():
        if not overwrite:
            raise ValueError(
                f"Output path {outpath} exists, and --overwrite was not passed!"
            )
        shutil.rmtree(outpath)

    contents = outpath / "Contents"
    docdir = contents / "Resources/Documents/"
    docdir.parent.mkdir(parents=True)

    subprocess.check_call(["unzip", "-q", html_zip, "-d", str(docdir.parent)])
    (docdir.parent / "html-multi").rename(docdir)

    rewrite_links(docdir)

    bundle_id = bundle_id or html_zip.with_suffix("").name

    index_html = docdir / "index.html"
    if bundle_name is None and index_html.is_file():
        title_re = re.compile(r"<title>(.*)</title>")
        with index_html.open("r") as fh:
            for line in fh:
                if (m := title_re.search(line)) is not None:
                    bundle_name = m.group(1)
                    break
    if bundle_name is None:
        bundle_name = bundle_id

    info_dict = {
        "id": bundle_id,
        "name": bundle_name,
        "platform": bundle_platform,
    }

    (contents / "Info.plist").write_text(
        INFO_TEMPLATE.format(**{k: escape(v) for k, v in info_dict.items()})
    )

    print("Building symbol index...")
    dbpath = contents / "Resources/docSet.dsidx"
    write_index(docdir, dbpath)


if __name__ == "__main__":
    import cyclopts

    cyclopts.run(main)
