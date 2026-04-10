# satoshi

Archive fetcher, parser, and analysis toolkit for three early cryptography
mailing lists that are frequently analyzed together:

| List          | Source                                                      | Format                         |
| ------------- | ----------------------------------------------------------- | ------------------------------ |
| Cryptography  | <https://www.metzdowd.com/pipermail/cryptography/>          | monthly `YYYY-Month.txt.gz` (pipermail / mbox) |
| Cypherpunks   | <https://cypherpunks.venona.com/raw/>                       | yearly `cyp-YYYY.txt` (MHonArc raw / mbox) |
| HashCash      | <https://www.freelists.org/archive/hashcash/> (hashcash.org is defunct) | per-post HTML on FreeLists   |

All three sources are normalized into a single SQLite database so they can be
queried together with pandas, plain SQL, or datasette.

## Setup

This project uses [`uv`](https://docs.astral.sh/uv/):

```sh
uv sync
```

## Build the database

The full pipeline (fetch + parse) is one command:

```sh
uv run python -m satoshi build
```

That will:

1. Download every monthly `.txt.gz` from the Cryptography pipermail archive into
   `data/raw/cryptography/`.
2. Download every yearly `cyp-YYYY.txt` from the venona Cypherpunks mirror into
   `data/raw/cypherpunks/`.
3. Scrape the FreeLists HashCash archive index, each month page, and every post
   page into `data/raw/hashcash/`.
4. Parse all cached files into `data/mailing_lists.sqlite` with the schema
   described below.

Subsequent runs are incremental: any archive already on disk is reused, so
`build` can be re-run cheaply. Use `--force` on the fetch/build commands to
re-download.

Individual steps:

```sh
uv run python -m satoshi fetch          # download only
uv run python -m satoshi parse          # parse already-cached files
uv run python -m satoshi stats          # row counts + date ranges
```

All commands take `--data-root`, `--db`, and `--lists` (comma-separated subset
of `cryptography,cypherpunks,hashcash`).

## Schema

```sql
CREATE TABLE messages (
    id              INTEGER PRIMARY KEY,
    list_name       TEXT NOT NULL,      -- 'cryptography' | 'cypherpunks' | 'hashcash'
    source          TEXT NOT NULL,      -- archive filename the message came from
    source_index    INTEGER NOT NULL,   -- position within that file (0-based)
    message_id      TEXT,
    in_reply_to     TEXT,
    refs            TEXT,               -- References header
    from_raw        TEXT,               -- unmodified From header
    from_name       TEXT,               -- parsed display name
    from_email      TEXT,               -- parsed, lower-cased address (de-obfuscated)
    to_raw          TEXT,
    cc_raw          TEXT,
    subject         TEXT,
    date_raw        TEXT,
    date_utc        TEXT,               -- ISO 8601 normalized
    headers         TEXT,               -- reconstructed header block
    body            TEXT                -- decoded text body
);
```

Parsing notes:

- Pipermail obfuscates email addresses by replacing `@` with ` at `. The
  parser stores the original obfuscated form in `from_raw` and writes a
  de-obfuscated address to `from_email`.
- Multipart messages are collapsed to concatenated text/plain parts.
- `date_utc` is best-effort; messages with unparseable dates have `NULL`.
- HashCash posts are scraped from FreeLists HTML (the hashcash.org site is
  defunct) and address values are stored as-is (FreeLists obfuscates them as
  `user@xxxxxxxxxxx`).

## Quick analysis examples

```python
import sqlite3, pandas as pd

conn = sqlite3.connect("data/mailing_lists.sqlite")

# Messages per year per list
pd.read_sql_query("""
    SELECT list_name, substr(date_utc, 1, 4) AS year, COUNT(*) AS n
    FROM messages
    WHERE date_utc IS NOT NULL
    GROUP BY list_name, year
    ORDER BY year, list_name
""", conn)

# Top 10 posters on the cryptography list
pd.read_sql_query("""
    SELECT from_email, COUNT(*) AS n
    FROM messages
    WHERE list_name = 'cryptography' AND from_email IS NOT NULL
    GROUP BY from_email
    ORDER BY n DESC
    LIMIT 10
""", conn)
```

## Layout

```
satoshi/
├── src/satoshi/
│   ├── __init__.py
│   ├── __main__.py   # argparse CLI (fetch / parse / build / stats)
│   ├── db.py         # SQLite schema + connect()
│   ├── fetch.py      # downloaders for all three sources
│   └── parse.py      # mbox + FreeLists HTML parsers
├── data/             # populated by `build` (gitignored)
│   ├── raw/
│   └── mailing_lists.sqlite
└── pyproject.toml
```
