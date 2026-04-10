"""SQLite schema and connection helpers for the mailing-list corpus."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY,
    list_name       TEXT NOT NULL,
    source          TEXT NOT NULL,
    source_index    INTEGER NOT NULL,
    message_id      TEXT,
    in_reply_to     TEXT,
    refs            TEXT,
    from_raw        TEXT,
    from_name       TEXT,
    from_email      TEXT,
    to_raw          TEXT,
    cc_raw          TEXT,
    subject         TEXT,
    date_raw        TEXT,
    date_utc        TEXT,
    headers         TEXT,
    body            TEXT,
    UNIQUE(list_name, source, source_index)
);

CREATE INDEX IF NOT EXISTS idx_messages_list      ON messages(list_name);
CREATE INDEX IF NOT EXISTS idx_messages_date      ON messages(date_utc);
CREATE INDEX IF NOT EXISTS idx_messages_from      ON messages(from_email);
CREATE INDEX IF NOT EXISTS idx_messages_msgid     ON messages(message_id);
CREATE INDEX IF NOT EXISTS idx_messages_inreplyto ON messages(in_reply_to);

CREATE TABLE IF NOT EXISTS fetch_log (
    id          INTEGER PRIMARY KEY,
    list_name   TEXT NOT NULL,
    url         TEXT NOT NULL,
    path        TEXT NOT NULL,
    bytes       INTEGER,
    fetched_at  TEXT NOT NULL,
    UNIQUE(list_name, url)
);
"""


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


INSERT_MESSAGE_SQL = """
INSERT OR IGNORE INTO messages (
    list_name, source, source_index, message_id, in_reply_to, refs,
    from_raw, from_name, from_email, to_raw, cc_raw,
    subject, date_raw, date_utc, headers, body
) VALUES (
    :list_name, :source, :source_index, :message_id, :in_reply_to, :refs,
    :from_raw, :from_name, :from_email, :to_raw, :cc_raw,
    :subject, :date_raw, :date_utc, :headers, :body
)
"""
