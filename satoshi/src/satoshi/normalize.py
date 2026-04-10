"""Text normalization for mailing-list messages prior to embedding.

Mailing list bodies carry a lot of boilerplate that would otherwise dominate
an embedding: PGP blocks, quoted reply chains, attribution headers, signature
blocks, and list footers. Stripping these before handing the text to an
embedding model meaningfully improves semantic similarity quality, especially
on the Cypherpunks list where many messages are >50% quoted text.

``normalize_email_text`` applies these transforms in order:

1. Remove encrypted ``BEGIN PGP MESSAGE`` / ``PGP PUBLIC KEY BLOCK`` blobs.
2. Remove ``BEGIN PGP SIGNATURE`` trailing blocks.
3. Strip the ``BEGIN PGP SIGNED MESSAGE`` header + ``Hash:`` line but keep the
   signed body.
4. Cut the standard ``\\n-- \\n`` signature delimiter and everything after it.
5. Cut the Cryptography and Cypherpunks mailing-list trailers (the "-----"
   separator line followed by an ``Unsubscribe by ...`` paragraph).
6. Drop common attribution lines (``On ..., X wrote:``, ``At HH:MM ..., wrote:``).
7. Drop quoted reply lines (``^>`` with any amount of leading ``>`` chars).
8. Collapse runs of 3+ blank lines to a single blank line.
"""

from __future__ import annotations

import re

_PGP_MESSAGE_RE = re.compile(
    r"-----BEGIN PGP (?:MESSAGE|PUBLIC KEY BLOCK)-----"
    r".*?-----END PGP (?:MESSAGE|PUBLIC KEY BLOCK)-----\s*",
    re.DOTALL,
)

_PGP_SIGNATURE_RE = re.compile(
    r"-----BEGIN PGP SIGNATURE-----.*?-----END PGP SIGNATURE-----\s*",
    re.DOTALL,
)

# "-----BEGIN PGP SIGNED MESSAGE-----\nHash: SHA1\n\n" — strip the header only.
_PGP_SIGNED_HEADER_RE = re.compile(
    r"-----BEGIN PGP SIGNED MESSAGE-----\s*\n(?:Hash:[^\n]*\n)*\s*\n",
)

# Standard signature delimiter. The canonical form is "\n-- \n" (trailing
# space!) but plenty of clients drop the trailing space, so allow both.
_SIG_DELIMITER_RE = re.compile(r"\n-- ?\n.*\Z", re.DOTALL)

# Mailing-list footer for Cryptography. Form:
#   ---------------------------------------------------------------------
#   The Cryptography Mailing List
#   Unsubscribe by sending "unsubscribe cryptography" to majordomo at ...
_LIST_FOOTER_RE = re.compile(
    r"\n-{20,}\s*\n"
    r"(?:The Cryptography Mailing List|The Cypherpunks Mailing List|"
    r"hashcash mailing list)\b.*\Z",
    re.DOTALL | re.IGNORECASE,
)

# Another common footer delimiter: 40+ underscores followed by list metadata.
_UNDERSCORE_FOOTER_RE = re.compile(r"\n_{20,}.*\Z", re.DOTALL)

# Attribution lines that introduce a quoted reply. Conservative — must end in
# "wrote:" or "writes:" or "said:" and be reasonably short.
_ATTRIBUTION_RE = re.compile(
    r"^[ \t]*"
    r"(?:On\s.{0,250}|At\s.{0,250}|In\b.{0,250}|[^\n]{1,250}?)"
    r"\s(?:wrote|writes|said):\s*$",
    re.MULTILINE,
)

# Quoted reply lines: any number of leading ``>`` characters (optionally with
# spaces between) at the start of a line, possibly preceded by whitespace.
_QUOTED_LINE_RE = re.compile(r"^[ \t]*(?:>+[ \t]?)+.*$", re.MULTILINE)

# Forwarded-message markers like "--- begin forwarded text" / "-----Original
# Message-----" / "--- Forwarded message ---".
_FORWARD_HEADER_RE = re.compile(
    r"^[ \t]*-{2,}\s*"
    r"(?:begin forwarded (?:text|message)|"
    r"forwarded message|original message|"
    r"begin pgp [a-z ]*block)"
    r"[^\n]*$",
    re.MULTILINE | re.IGNORECASE,
)

_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def normalize_email_text(text: str) -> str:
    """Clean a mailing-list message body for embedding."""
    if not text:
        return ""

    # 1. Encrypted payloads and public keys — zero useful signal.
    text = _PGP_MESSAGE_RE.sub("", text)
    # 2. Trailing signature blocks.
    text = _PGP_SIGNATURE_RE.sub("", text)
    # 3. Strip the PGP SIGNED MESSAGE header, keep the signed content.
    text = _PGP_SIGNED_HEADER_RE.sub("", text)
    # 4. Mail signature block (after "-- \n").
    text = _SIG_DELIMITER_RE.sub("", text)
    # 5. List footers.
    text = _LIST_FOOTER_RE.sub("", text)
    text = _UNDERSCORE_FOOTER_RE.sub("", text)
    # 6. Forwarded-message markers.
    text = _FORWARD_HEADER_RE.sub("", text)
    # 7. Attribution lines.
    text = _ATTRIBUTION_RE.sub("", text)
    # 8. Quoted lines.
    text = _QUOTED_LINE_RE.sub("", text)
    # 9. Tidy whitespace.
    text = _TRAILING_WS_RE.sub("", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def prepare_for_embedding(subject: str | None, body: str | None) -> str:
    """Concatenate ``subject`` and a normalized ``body`` into a single string.

    Subject lines carry a lot of semantic signal on technical mailing lists,
    so we prefix the cleaned body with ``Subject: ...``.
    """
    cleaned = normalize_email_text(body or "")
    subj = (subject or "").strip()
    if subj:
        return f"Subject: {subj}\n\n{cleaned}".strip()
    return cleaned
