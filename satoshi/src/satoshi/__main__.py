"""Command-line interface for the satoshi archive pipeline.

Examples::

    uv run python -m satoshi fetch
    uv run python -m satoshi parse
    uv run python -m satoshi build           # fetch + parse
    uv run python -m satoshi stats
    uv run python -m satoshi similarities    # build poster-month matrix + cosines
    uv run python -m satoshi build --lists cryptography,cypherpunks
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .db import connect
from .fetch import fetch_all
from .parse import parse_all

DEFAULT_DATA_ROOT = Path("data/raw")
DEFAULT_DB_PATH = Path("data/mailing_lists.sqlite")
ALL_LISTS = ("cryptography", "cypherpunks", "hashcash")


def _parse_lists(value: str | None) -> tuple[str, ...]:
    if not value:
        return ALL_LISTS
    out = tuple(s.strip() for s in value.split(",") if s.strip())
    unknown = [s for s in out if s not in ALL_LISTS]
    if unknown:
        raise SystemExit(f"Unknown list(s): {unknown}. Known: {ALL_LISTS}")
    return out


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p.add_argument(
        "--lists",
        default=None,
        help=f"Comma-separated subset of {ALL_LISTS}. Default: all.",
    )


def _cmd_fetch(args: argparse.Namespace) -> None:
    conn = connect(args.db)
    fetch_all(args.data_root, conn, lists=_parse_lists(args.lists), force=args.force)


def _cmd_parse(args: argparse.Namespace) -> None:
    conn = connect(args.db)
    counts = parse_all(conn, args.data_root, lists=_parse_lists(args.lists))
    for name, n in counts.items():
        print(f"  {name:<14} {n:>8d} rows")


def _cmd_build(args: argparse.Namespace) -> None:
    conn = connect(args.db)
    lists = _parse_lists(args.lists)
    fetch_all(args.data_root, conn, lists=lists, force=args.force)
    counts = parse_all(conn, args.data_root, lists=lists)
    print()
    print("Parsed row counts:")
    for name, n in counts.items():
        print(f"  {name:<14} {n:>8d} rows")


def _cmd_stats(args: argparse.Namespace) -> None:
    conn = connect(args.db)
    print(f"Database: {args.db}")
    total = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    print(f"Total messages: {total}")
    for (list_name, n) in conn.execute(
        "SELECT list_name, COUNT(*) FROM messages GROUP BY list_name ORDER BY list_name"
    ):
        earliest, latest = conn.execute(
            "SELECT MIN(date_utc), MAX(date_utc) FROM messages WHERE list_name = ?",
            (list_name,),
        ).fetchone()
        print(f"  {list_name:<14} {n:>8d}  range: {earliest}  ..  {latest}")


def _cmd_similarities(args: argparse.Namespace) -> None:
    from . import analysis

    conn = connect(args.db)
    pm = analysis.build_poster_month_matrix(conn, min_posts=args.min_posts)
    print(
        f"Poster-month matrix: {pm.counts.shape}  "
        f"({pm.counts.nbytes / 1e6:.1f} MB)  "
        f"months {pm.months[0]}..{pm.months[-1]}"
    )
    sims = analysis.cosine_similarity(pm.counts)
    print(f"Cosine similarity:   {sims.shape}  ({sims.nbytes / 1e6:.1f} MB)")

    out_dir = args.out
    analysis.save(pm, sims, out_dir)
    print(f"Saved to {out_dir}/")
    print(f"  poster_month_matrix.npz     ({(out_dir / 'poster_month_matrix.npz').stat().st_size / 1e6:.1f} MB)")
    print(f"  poster_month_matrix.parquet ({(out_dir / 'poster_month_matrix.parquet').stat().st_size / 1e6:.1f} MB)")
    print(f"  poster_cosine_similarity.npy ({(out_dir / 'poster_cosine_similarity.npy').stat().st_size / 1e6:.1f} MB)")

    print()
    print(f"Top {args.top_pairs} most-similar pairs (>= {args.min_pair_posts} posts each):")
    pairs = analysis.top_pairs(
        pm, sims, k=args.top_pairs, min_total_posts=args.min_pair_posts
    )
    print(pairs.to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="satoshi", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fetch = sub.add_parser("fetch", help="Download archives to data/raw/")
    _add_common(p_fetch)
    p_fetch.add_argument("--force", action="store_true", help="Re-download cached files")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_parse = sub.add_parser("parse", help="Parse cached archives into the SQLite database")
    _add_common(p_parse)
    p_parse.set_defaults(func=_cmd_parse)

    p_build = sub.add_parser("build", help="Fetch then parse (the full pipeline)")
    _add_common(p_build)
    p_build.add_argument("--force", action="store_true", help="Re-download cached files")
    p_build.set_defaults(func=_cmd_build)

    p_stats = sub.add_parser("stats", help="Print row counts and date ranges")
    _add_common(p_stats)
    p_stats.set_defaults(func=_cmd_stats)

    p_sim = sub.add_parser(
        "similarities",
        help="Build the (poster × month) matrix and pairwise cosine similarities",
    )
    p_sim.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    p_sim.add_argument(
        "--out", type=Path, default=Path("data/analysis"), help="Output directory"
    )
    p_sim.add_argument(
        "--min-posts",
        type=int,
        default=1,
        help="Drop posters with fewer than this many total posts before building the matrix",
    )
    p_sim.add_argument(
        "--top-pairs", type=int, default=20, help="How many top similar pairs to print"
    )
    p_sim.add_argument(
        "--min-pair-posts",
        type=int,
        default=100,
        help="Restrict top-pairs output to posters with at least this many total posts",
    )
    p_sim.set_defaults(func=_cmd_similarities)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
