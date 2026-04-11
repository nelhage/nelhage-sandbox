"""Embed normalized mailing-list messages with the Voyage AI API.

Pipeline for a single run:

1. Iterate messages that don't yet have an embedding for the chosen model.
2. Normalize the body with :mod:`satoshi.normalize` and prepend the subject.
3. Truncate the cleaned text to ``MAX_CHARS_PER_DOC`` to stay comfortably
   inside the model's context window (Voyage also enforces ``truncation=True``
   server-side as a safety net).
4. Pack prepared documents into batches respecting Voyage's per-request limits
   (document count and rough token budget).
5. Call ``voyageai.Client.embed`` and insert the resulting vectors into
   ``message_embeddings`` as float32 ``BLOB``s.

The pipeline is resumable: re-running skips any ``(message_id, model)`` pair
already present in the table. Transient API failures are retried with
exponential backoff.

Set the ``VOYAGE_API_KEY`` environment variable before running.
"""

from __future__ import annotations

import datetime as dt
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
from tqdm import tqdm

from .normalize import prepare_for_embedding

# voyage-3-large has a 32k-token context window per document and a 120k-token
# limit per batch (`InvalidRequestError: max allowed tokens per submitted batch
# is 120000`). Char-based packing isn't safe: the chars/token ratio varies
# enough across messages (2.5–4.5+) that even with a pessimistic estimate a
# batch of 128 dense short messages can blow the 120k ceiling. So we tokenize
# every prepared document with Voyage's local tokenizer and pack by exact
# token counts.
MAX_TOKENS_PER_DOC = 30_000          # under 32k; Voyage truncation=True is a backstop
MAX_CHARS_PER_DOC = MAX_TOKENS_PER_DOC * 4  # 120k chars; first-pass cheap cap

MAX_BATCH_DOCS = 128
MAX_BATCH_TOKENS = 110_000           # 10k margin under the 120k API ceiling

DEFAULT_MODEL = "voyage-3-large"
DEFAULT_DIMENSION = 1024             # voyage-3-large supports 256/512/1024/2048


@dataclass
class PreparedDoc:
    message_id: int
    text: str
    char_count: int
    token_count: int
    truncated: bool


@dataclass
class EmbedStats:
    processed: int = 0
    skipped_empty: int = 0
    truncated: int = 0
    api_calls: int = 0
    total_tokens: int = 0


def _prepare(
    client,
    message_id: int,
    subject: str | None,
    body: str | None,
    *,
    model: str,
) -> PreparedDoc | None:
    text = prepare_for_embedding(subject, body)
    if not text.strip():
        return None
    truncated = False
    # Cheap first-pass char cap so we never tokenize an obviously oversized
    # blob (the 911k-char outlier and friends). The token cap below is the
    # one that matters for API correctness.
    if len(text) > MAX_CHARS_PER_DOC:
        text = text[:MAX_CHARS_PER_DOC]
        truncated = True
    token_count = client.count_tokens([text], model=model)
    # If the cheap char cap wasn't enough (rare: dense text under the char
    # limit but over the token limit), pessimistically retruncate based on
    # the observed chars/token ratio.
    if token_count > MAX_TOKENS_PER_DOC:
        ratio = len(text) / token_count
        new_len = int(MAX_TOKENS_PER_DOC * ratio * 0.95)  # 5% safety
        text = text[:new_len]
        truncated = True
        token_count = client.count_tokens([text], model=model)
    return PreparedDoc(
        message_id=message_id,
        text=text,
        char_count=len(text),
        token_count=token_count,
        truncated=truncated,
    )


def _iter_unembedded_rows(
    conn: sqlite3.Connection,
    model: str,
    *,
    where_sql: str = "",
    params: tuple = (),
) -> Iterator[tuple[int, str | None, str | None]]:
    """Stream (id, subject, body) for messages missing this model's embedding."""
    query = f"""
        SELECT m.id, m.subject, m.body
        FROM messages m
        LEFT JOIN message_embeddings e
               ON e.message_id = m.id AND e.model = ?
        WHERE e.message_id IS NULL
          AND m.body IS NOT NULL
          AND length(trim(m.body)) > 0
          {where_sql}
        ORDER BY m.id
    """
    cursor = conn.execute(query, (model, *params))
    for row in cursor:
        yield row


def _pack_batches(docs: Iterable[PreparedDoc]) -> Iterator[list[PreparedDoc]]:
    batch: list[PreparedDoc] = []
    batch_tokens = 0
    for doc in docs:
        # An oversized single document still gets its own batch so we never
        # drop work; per-doc truncation guarantees it's under MAX_TOKENS_PER_DOC
        # which is well under MAX_BATCH_TOKENS.
        if batch and (
            len(batch) >= MAX_BATCH_DOCS
            or batch_tokens + doc.token_count > MAX_BATCH_TOKENS
        ):
            yield batch
            batch, batch_tokens = [], 0
        batch.append(doc)
        batch_tokens += doc.token_count
    if batch:
        yield batch


def _call_voyage(client, texts: list[str], *, model: str, output_dimension: int):
    """Call ``client.embed`` with exponential-backoff retry on transient errors."""
    import voyageai.error as ve

    delay = 2.0
    for attempt in range(5):
        try:
            return client.embed(
                texts=texts,
                model=model,
                input_type="document",
                truncation=True,
                output_dtype="float",
                output_dimension=output_dimension,
            )
        except (ve.RateLimitError, ve.ServiceUnavailableError, ve.Timeout, ve.APIConnectionError) as exc:
            if attempt == 4:
                raise
            tqdm.write(f"  transient Voyage error ({type(exc).__name__}); retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2


def embed_messages(
    conn: sqlite3.Connection,
    *,
    model: str = DEFAULT_MODEL,
    output_dimension: int = DEFAULT_DIMENSION,
    limit: int | None = None,
    dry_run: bool = False,
) -> EmbedStats:
    """Embed all messages missing a vector for ``model``.

    ``dry_run=True`` runs the full normalize + batch pipeline but skips the
    API call and the database writes; useful for validating the pipeline
    without consuming Voyage credits.

    ``limit`` caps the number of messages processed in this run (also useful
    for smoke-testing).
    """
    import voyageai

    have_key = bool(os.environ.get("VOYAGE_API_KEY"))
    if not dry_run and not have_key:
        raise RuntimeError(
            "VOYAGE_API_KEY environment variable is not set; pass --dry-run to "
            "validate the pipeline without calling the API."
        )
    # We always need a client for local tokenization (count_tokens). The
    # tokenizer is bundled in the SDK so a dummy key is fine for dry-run.
    client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY") or "dry-run")
    stats = EmbedStats()

    total_remaining = conn.execute(
        """
        SELECT COUNT(*)
        FROM messages m
        LEFT JOIN message_embeddings e
               ON e.message_id = m.id AND e.model = ?
        WHERE e.message_id IS NULL
          AND m.body IS NOT NULL
          AND length(trim(m.body)) > 0
        """,
        (model,),
    ).fetchone()[0]
    total = min(total_remaining, limit) if limit is not None else total_remaining

    def prepared_stream() -> Iterator[PreparedDoc]:
        count = 0
        for mid, subj, body in _iter_unembedded_rows(conn, model):
            if limit is not None and count >= limit:
                return
            doc = _prepare(client, mid, subj, body, model=model)
            if doc is None:
                stats.skipped_empty += 1
                count += 1
                continue
            if doc.truncated:
                stats.truncated += 1
            count += 1
            yield doc

    pbar = tqdm(total=total, desc=f"embedding ({model})", unit="msg")
    for batch in _pack_batches(prepared_stream()):
        if dry_run:
            stats.processed += len(batch)
            stats.api_calls += 1
            pbar.update(len(batch))
            continue

        texts = [d.text for d in batch]
        result = _call_voyage(
            client, texts, model=model, output_dimension=output_dimension
        )
        stats.api_calls += 1
        stats.total_tokens += result.total_tokens

        now = dt.datetime.now(dt.timezone.utc).isoformat()
        rows = []
        for doc, vec in zip(batch, result.embeddings):
            arr = np.asarray(vec, dtype=np.float32)
            if arr.shape[0] != output_dimension:
                raise RuntimeError(
                    f"expected dim {output_dimension}, got {arr.shape[0]} for message {doc.message_id}"
                )
            rows.append(
                (
                    doc.message_id,
                    model,
                    output_dimension,
                    doc.char_count,
                    doc.token_count,
                    int(doc.truncated),
                    arr.tobytes(),
                    now,
                )
            )
        with conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO message_embeddings
                    (message_id, model, dim, input_chars, input_tokens, truncated,
                     vector, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        stats.processed += len(batch)
        pbar.update(len(batch))
    pbar.close()
    return stats


def load_embeddings(
    conn: sqlite3.Connection, model: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(message_ids, vectors)`` for all stored embeddings of ``model``."""
    rows = conn.execute(
        "SELECT message_id, dim, vector FROM message_embeddings WHERE model = ? ORDER BY message_id",
        (model,),
    ).fetchall()
    if not rows:
        return np.empty((0,), dtype=np.int64), np.empty((0, 0), dtype=np.float32)
    ids = np.asarray([r[0] for r in rows], dtype=np.int64)
    dim = rows[0][1]
    vectors = np.stack(
        [np.frombuffer(r[2], dtype=np.float32) for r in rows]
    )
    assert vectors.shape == (len(rows), dim)
    return ids, vectors
