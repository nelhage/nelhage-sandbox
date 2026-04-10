"""Command-line interface for the satoshi archive pipeline.

Examples::

    uv run python -m satoshi fetch
    uv run python -m satoshi parse
    uv run python -m satoshi build           # fetch + parse
    uv run python -m satoshi stats
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

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
