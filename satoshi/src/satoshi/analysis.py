"""Poster-activity matrix and cosine similarity analysis.

For each poster (identified by the display-name portion of ``From``, since the
email local part is sometimes anonymized on FreeLists) we build a vector with
one element per month in the corpus. The value is the number of posts that
poster made in that month, aggregated across all three mailing lists.

The resulting matrix is small enough to keep dense: ~7.5k posters × ~350
months ≈ 20 MB at float32.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_name(raw: str | None) -> str | None:
    if not raw:
        return None
    name = _WHITESPACE_RE.sub(" ", raw).strip()
    # Some From headers come as '"Foo Bar"' — strip surrounding quotes.
    if len(name) >= 2 and name[0] == name[-1] == '"':
        name = name[1:-1].strip()
    return name.lower() or None


@dataclass
class PosterMatrix:
    posters: list[str]       # length N
    months: list[str]        # length M (YYYY-MM, sorted)
    counts: np.ndarray       # (N, M) float32 post counts
    total_posts: np.ndarray  # (N,) int64 — sum across months

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.counts, index=self.posters, columns=self.months)


def build_poster_month_matrix(
    conn: sqlite3.Connection,
    *,
    min_posts: int = 1,
) -> PosterMatrix:
    """Aggregate messages into a (poster × month) count matrix.

    ``min_posts`` filters out posters whose total across the whole corpus is
    below the threshold — useful because there is a long tail of one-off
    posters whose single-bucket vectors swamp the output of a cosine-similarity
    search.
    """
    df = pd.read_sql_query(
        """
        SELECT from_name, substr(date_utc, 1, 7) AS month
        FROM messages
        WHERE date_utc IS NOT NULL
          AND from_name IS NOT NULL
          AND trim(from_name) <> ''
        """,
        conn,
    )
    df["poster"] = df["from_name"].map(_normalize_name)
    df = df.dropna(subset=["poster"])

    counts = (
        df.groupby(["poster", "month"], sort=True).size().unstack(fill_value=0).sort_index()
    )
    # Ensure columns are sorted chronologically.
    counts = counts.reindex(columns=sorted(counts.columns))

    totals = counts.sum(axis=1)
    keep = totals >= min_posts
    counts = counts.loc[keep]
    totals = totals.loc[keep]

    return PosterMatrix(
        posters=counts.index.tolist(),
        months=counts.columns.tolist(),
        counts=counts.to_numpy(dtype=np.float32),
        total_posts=totals.to_numpy(dtype=np.int64),
    )


def cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    """Return the (N, N) row-wise cosine similarity matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    normalized = (matrix / safe).astype(np.float32, copy=False)
    return normalized @ normalized.T


def top_pairs(
    pm: PosterMatrix,
    sims: np.ndarray,
    *,
    k: int = 20,
    min_total_posts: int = 50,
) -> pd.DataFrame:
    """Return the top-k most similar poster pairs.

    Restricts to posters with at least ``min_total_posts`` across the whole
    corpus, so the results are not dominated by one-off posters whose
    single-bucket vectors are trivially parallel.
    """
    keep_mask = pm.total_posts >= min_total_posts
    keep_idx = np.where(keep_mask)[0]
    if len(keep_idx) < 2:
        return pd.DataFrame(columns=["poster_a", "poster_b", "cosine", "posts_a", "posts_b"])

    sub = sims[np.ix_(keep_idx, keep_idx)].copy()
    # Zero out the diagonal and lower triangle so we only get unique unordered
    # pairs.
    n = sub.shape[0]
    iu = np.triu_indices(n, k=1)
    flat = sub[iu]

    if k >= flat.size:
        order = np.argsort(-flat)
    else:
        part = np.argpartition(-flat, k)[:k]
        order = part[np.argsort(-flat[part])]

    rows = []
    posters = pm.posters
    totals = pm.total_posts
    for idx in order:
        i = keep_idx[iu[0][idx]]
        j = keep_idx[iu[1][idx]]
        rows.append(
            {
                "poster_a": posters[i],
                "poster_b": posters[j],
                "cosine": float(flat[idx]),
                "posts_a": int(totals[i]),
                "posts_b": int(totals[j]),
            }
        )
    return pd.DataFrame(rows)


def nearest_neighbors(
    pm: PosterMatrix,
    sims: np.ndarray,
    query: str,
    *,
    k: int = 10,
) -> pd.DataFrame:
    """Return the ``k`` most-similar posters to ``query`` (case-insensitive)."""
    q = _normalize_name(query)
    try:
        idx = pm.posters.index(q)
    except ValueError:
        raise KeyError(f"poster {query!r} not in matrix") from None
    row = sims[idx].copy()
    row[idx] = -1.0  # exclude self
    order = np.argsort(-row)[:k]
    return pd.DataFrame(
        {
            "poster": [pm.posters[i] for i in order],
            "cosine": row[order].astype(float),
            "total_posts": pm.total_posts[order].astype(int),
        }
    )


def save(
    pm: PosterMatrix,
    sims: np.ndarray,
    out_dir: Path,
) -> None:
    """Persist the matrix + similarities to ``out_dir`` as numpy + parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "poster_month_matrix.npz",
        counts=pm.counts,
        posters=np.asarray(pm.posters, dtype=object),
        months=np.asarray(pm.months, dtype=object),
        total_posts=pm.total_posts,
    )
    np.save(out_dir / "poster_cosine_similarity.npy", sims)
    pm.as_dataframe().to_parquet(out_dir / "poster_month_matrix.parquet")
