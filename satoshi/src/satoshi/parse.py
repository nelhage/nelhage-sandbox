"""Parse fetched mailing-list archives into the SQLite messages table.

Two input formats are handled:

* **Pipermail / mbox text**: concatenated RFC-822 messages prefixed by envelope
  ``From `` separator lines. Used for the Cryptography list (gzipped monthly
  files) and the Cypherpunks venona raw yearly files.
* **FreeLists HTML**: one post per page, with ``From``/``To``/``Date`` in a
  ``<ul>`` and the body inside a ``<pre>``. Used for the HashCash list.
"""

from __future__ import annotations

import email
import email.policy
import email.utils
import gzip
import re
import sqlite3
from email.message import Message
from pathlib import Path
from typing import Iterable, Iterator

from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from tqdm import tqdm

from .db import INSERT_MESSAGE_SQL

# Line at the start of each mbox message: "From user@host ctime-date"
FROM_LINE_RE = re.compile(rb"^From [^\r\n]*\r?\n", re.MULTILINE)

# Pipermail obfuscates ``user@host`` as ``user at host`` inside headers. We only
# de-obfuscate in the From line of the header section when extracting the
# email address, leaving the stored ``from_raw`` unchanged so the original
# bytes are preserved.
AT_OBFUSCATION_RE = re.compile(r"(\S+) at (\S+)")


def _deobfuscate(text: str) -> str:
    return AT_OBFUSCATION_RE.sub(r"\1@\2", text)


def _open_maybe_gzip(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as f:
            return f.read()
    return path.read_bytes()


def _split_mbox(data: bytes) -> list[bytes]:
    """Split a concatenated mbox byte string into individual message blobs.

    We look for lines matching ``^From \\S+ ...`` which is the standard mboxo
    separator used by both Pipermail and MHonArc archives.
    """
    starts = [m.start() for m in FROM_LINE_RE.finditer(data)]
    if not starts:
        return []
    # Exclude any "From " that isn't at the very start of a line (already
    # handled by MULTILINE), and drop matches that are clearly inside a
    # message body by requiring the previous byte to be a newline or BOF.
    messages: list[bytes] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(data)
        msg = data[start:end]
        # Strip the envelope line; email.parser will work on the remainder.
        nl = msg.find(b"\n")
        if nl == -1:
            continue
        messages.append(msg[nl + 1 :])
    return messages


def _parse_message(blob: bytes) -> Message:
    return email.message_from_bytes(blob, policy=email.policy.compat32)


def _extract_body(msg: Message) -> str:
    parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            if part.is_multipart():
                continue
            if ctype == "text/plain" or ctype.startswith("text/"):
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                charset = part.get_content_charset() or "utf-8"
                try:
                    parts.append(payload.decode(charset, errors="replace"))
                except (LookupError, UnicodeDecodeError):
                    parts.append(payload.decode("utf-8", errors="replace"))
    else:
        payload = msg.get_payload(decode=True)
        if payload is None:
            payload_obj = msg.get_payload()
            return payload_obj if isinstance(payload_obj, str) else ""
        charset = msg.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except (LookupError, UnicodeDecodeError):
            return payload.decode("utf-8", errors="replace")
    return "\n\n".join(parts)


def _parse_date(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        dt_obj = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        dt_obj = None
    if dt_obj is None:
        try:
            dt_obj = dateparser.parse(raw, fuzzy=True)
        except (ValueError, OverflowError, dateparser.ParserError):
            return None
    if dt_obj is None:
        return None
    if dt_obj.tzinfo is None:
        return dt_obj.isoformat()
    return dt_obj.astimezone(tz=None).isoformat()


def _coerce_str(value: object) -> str | None:
    """Coerce an email header value (possibly ``Header`` / bytes) to ``str``."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:  # pragma: no cover - malformed headers
        return None


def _from_fields(raw: object) -> tuple[str | None, str | None]:
    text = _coerce_str(raw)
    if not text:
        return None, None
    cleaned = _deobfuscate(text)
    name, addr = email.utils.parseaddr(cleaned)
    return (name or None), (addr.lower() or None if addr else None)


def _headers_text(msg: Message) -> str:
    return "\n".join(f"{k}: {_coerce_str(v) or ''}" for k, v in msg.items())


def _message_row(
    *,
    list_name: str,
    source: str,
    source_index: int,
    msg: Message,
    body_override: str | None = None,
) -> dict:
    from_raw = _coerce_str(msg.get("From"))
    from_name, from_email_addr = _from_fields(from_raw)
    date_raw = _coerce_str(msg.get("Date"))
    message_id = (_coerce_str(msg.get("Message-ID")) or "").strip() or None
    in_reply_to = (_coerce_str(msg.get("In-Reply-To")) or "").strip() or None
    refs = (_coerce_str(msg.get("References")) or "").strip() or None
    body = body_override if body_override is not None else _extract_body(msg)
    return {
        "list_name": list_name,
        "source": source,
        "source_index": source_index,
        "message_id": message_id,
        "in_reply_to": in_reply_to,
        "refs": refs,
        "from_raw": from_raw,
        "from_name": from_name,
        "from_email": from_email_addr,
        "to_raw": _coerce_str(msg.get("To")),
        "cc_raw": _coerce_str(msg.get("Cc")),
        "subject": _coerce_str(msg.get("Subject")),
        "date_raw": date_raw,
        "date_utc": _parse_date(date_raw),
        "headers": _headers_text(msg),
        "body": body,
    }


def iter_mbox_messages(path: Path, list_name: str) -> Iterator[dict]:
    data = _open_maybe_gzip(path)
    for idx, blob in enumerate(_split_mbox(data)):
        try:
            msg = _parse_message(blob)
        except Exception:  # pragma: no cover - malformed messages
            continue
        yield _message_row(
            list_name=list_name,
            source=path.name,
            source_index=idx,
            msg=msg,
        )


# -------- FreeLists HashCash HTML parser --------


def _extract_freelists_body(header, meta_ul) -> str:
    """Extract the post body from a FreeLists post DOM.

    Older templates wrap the body in a single ``<pre>``; newer templates emit
    one or more ``<p>`` elements with ``<br/>`` linebreaks and terminate at an
    ``<!--X-MsgBody-End-->`` HTML comment / ``<a name="footer">`` anchor.
    """
    from bs4 import Comment, NavigableString, Tag

    start = meta_ul if meta_ul is not None else header
    parts: list[str] = []
    for sibling in start.next_siblings:
        if isinstance(sibling, Comment):
            if "X-MsgBody-End" in sibling:
                break
            continue
        if isinstance(sibling, NavigableString):
            continue
        if not isinstance(sibling, Tag):
            continue
        if sibling.name == "a" and sibling.get("name") == "footer":
            break
        if sibling.name in {"script", "ins", "style"}:
            continue
        if sibling.name == "pre":
            parts.append(sibling.get_text())
            continue
        if sibling.name in {"p", "div", "blockquote"}:
            text = sibling.get_text("\n", strip=False)
            if text.strip():
                parts.append(text)
    return "\n".join(p.rstrip() for p in parts).strip()


def _parse_freelists_post(html: str, path: Path) -> dict | None:
    soup = BeautifulSoup(html, "lxml")
    # FreeLists uses either <h2> (older template) or <h1> (newer template) for
    # the post subject. In both cases the subject text starts with "[hashcash]".
    header = None
    for tag in soup.find_all(["h1", "h2"]):
        text = tag.get_text(strip=True)
        if text.startswith("[hashcash]"):
            header = tag
            break
    if header is None:
        return None
    subject = header.get_text(strip=True)
    if subject.startswith("[hashcash]"):
        subject = subject[len("[hashcash]") :].strip()

    meta_ul = header.find_next("ul")
    from_raw: str | None = None
    to_raw: str | None = None
    date_raw: str | None = None
    if meta_ul is not None:
        for li in meta_ul.find_all("li", recursive=False):
            label_el = li.find("em")
            if label_el is None:
                continue
            label = label_el.get_text(strip=True).lower()
            # Everything after the label text is the value.
            text = li.get_text(" ", strip=True)
            if ":" in text:
                value = text.split(":", 1)[1].strip()
            else:
                value = text.replace(label, "", 1).strip()
            if label == "from":
                from_raw = value
            elif label == "to":
                to_raw = value
            elif label == "date":
                date_raw = value

    body = _extract_freelists_body(header, meta_ul)
    if not body:
        return None
    from_name, from_email_addr = _from_fields(from_raw)
    return {
        "list_name": "hashcash",
        "source": path.name,
        "source_index": 0,
        "message_id": None,
        "in_reply_to": None,
        "refs": None,
        "from_raw": from_raw,
        "from_name": from_name,
        "from_email": from_email_addr,
        "to_raw": to_raw,
        "cc_raw": None,
        "subject": subject,
        "date_raw": date_raw,
        "date_utc": _parse_date(date_raw),
        "headers": "\n".join(
            f"{k}: {v}"
            for k, v in [
                ("From", from_raw),
                ("To", to_raw),
                ("Date", date_raw),
                ("Subject", subject),
            ]
            if v
        ),
        "body": body,
    }


def iter_freelists_posts(post_dir: Path) -> Iterator[dict]:
    for html_file in sorted(post_dir.glob("*.html")):
        html = html_file.read_text(encoding="utf-8", errors="replace")
        row = _parse_freelists_post(html, html_file)
        if row is not None:
            yield row


# -------- Driver --------


def load_mbox_directory(
    conn: sqlite3.Connection,
    dir_path: Path,
    list_name: str,
    *,
    pattern: str,
    desc: str,
) -> int:
    files = sorted(dir_path.glob(pattern))
    total = 0
    for path in tqdm(files, desc=desc, unit="file"):
        rows = list(iter_mbox_messages(path, list_name))
        if not rows:
            continue
        with conn:
            conn.executemany(INSERT_MESSAGE_SQL, rows)
        total += len(rows)
    return total


def load_freelists(conn: sqlite3.Connection, data_root: Path) -> int:
    post_dir = data_root / "hashcash" / "post"
    if not post_dir.is_dir():
        return 0
    rows: list[dict] = []
    for row in tqdm(
        iter_freelists_posts(post_dir),
        desc="hashcash parse",
        unit="msg",
    ):
        rows.append(row)
        if len(rows) >= 500:
            with conn:
                conn.executemany(INSERT_MESSAGE_SQL, rows)
            rows.clear()
    if rows:
        with conn:
            conn.executemany(INSERT_MESSAGE_SQL, rows)
    cur = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE list_name = 'hashcash'"
    )
    return cur.fetchone()[0]


def parse_all(
    conn: sqlite3.Connection,
    data_root: Path,
    *,
    lists: Iterable[str] = ("cryptography", "cypherpunks", "hashcash"),
) -> dict[str, int]:
    results: dict[str, int] = {}
    lists = list(lists)
    if "cryptography" in lists:
        results["cryptography"] = load_mbox_directory(
            conn,
            data_root / "cryptography",
            "cryptography",
            pattern="*.txt.gz",
            desc="cryptography parse",
        )
    if "cypherpunks" in lists:
        results["cypherpunks"] = load_mbox_directory(
            conn,
            data_root / "cypherpunks",
            "cypherpunks",
            pattern="cyp-*.txt",
            desc="cypherpunks parse",
        )
    if "hashcash" in lists:
        results["hashcash"] = load_freelists(conn, data_root)
    return results
