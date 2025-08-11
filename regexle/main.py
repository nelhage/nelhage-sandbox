import datetime
import json
import string
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlencode

import cyclopts
import greenery
import httpx
import numpy as np
import z3
from cyclopts import Parameter

ALPHABET = string.ascii_uppercase


def to_fsm(re):
    return greenery.parse(re).to_fsm().reduce()


def flatten_fsm(fsm):
    nstate = len(fsm.states)
    nvocab = len(ALPHABET)

    assert fsm.states == set(range(nstate))
    state_map = np.full((nstate, nvocab), -1, dtype=np.int32)
    for st, map in fsm.map.items():
        flat = state_map[st]
        for i, c in enumerate(ALPHABET):
            for cc, dst in map.items():
                if cc.accepts(c):
                    flat[i] = dst
                    break
    return state_map


def fsm_to_z3func(solv, fsm, name):
    flat = flatten_fsm(fsm)
    nstate, nvocab = flat.shape

    state_func = z3.Function(
        name + "_xfer",
        z3.IntSort(),
        z3.IntSort(),
        z3.IntSort(),
    )

    it = np.nditer(flat, flags=["multi_index"])
    for next_state in it:
        state, char = it.multi_index
        solv.add(state_func(state, char) == next_state)

    s, c = z3.Ints("s c")
    solv.add(
        z3.ForAll(
            [s, c],
            (state_func(s, c) >= 0) & (state_func(s, c) < nstate),
        )
    )
    return state_func


def one_of(var, values):
    return z3.Or(*(var == v for v in values))


def dead_states(fsm):
    loop = {s for s, m in fsm.map.items() if set(m.values()) == {s}}
    return loop - fsm.finals


def assert_matches(solv, fsm, state_func, chars, name):
    flat = flatten_fsm(fsm)
    nstate, nvocab = flat.shape

    nchar = len(chars)
    dead = dead_states(fsm)

    states = z3.IntVector(name + "_state", 1 + nchar)
    for i, ch in enumerate(chars):
        solv.add(state_func(states[i], ch) == states[i + 1])

    dead_init = set()
    dead_all = set()

    for d in dead:
        dead_init |= set((flat[0] == d).nonzero()[0])
        dead_all |= set((flat == d).all(0).nonzero()[0])

    for ch in chars:
        solv.add(z3.Not(one_of(ch, dead_all)))
    solv.add(z3.Not(one_of(chars[0], dead_init - dead_all)))

    for st in states:
        solv.add(0 <= st)
        solv.add(st < nstate)
        solv.add(z3.Not(one_of(st, dead)))
    solv.add(states[0] == fsm.initial)
    solv.add(one_of(states[-1], fsm.finals))


DAY_EPOCH = datetime.date(2024, 5, 31)

Axis = Literal["x", "y", "z"]


@dataclass
class Clue:
    axis: Axis
    index: int

    pattern: str
    fsm: greenery.Fsm

    @property
    def name(self):
        return f"{self.axis}_{self.index}"

    @classmethod
    def from_pattern(cls, re: str, axis: Axis, index: int):
        fsm = to_fsm(re)
        return cls(axis=axis, index=index, pattern=re, fsm=fsm)

    def build_z3(self, solv: z3.Solver) -> z3.FuncDeclRef:
        return fsm_to_z3func(solv, self.fsm, self.name)


PUZZLE_CACHE = Path(__file__).parent / "puzzles"


def fetch_puzzle(day, side):
    print(f"Retrieving puzzle: {day=} {side=}. Interactive URL:")
    qstring = urlencode(dict(day=day, side=side))
    print("  https://regexle.com/?" + qstring)

    cached = PUZZLE_CACHE / f"puzzle_{day}_{side}.json"
    if not cached.is_file():
        resp = httpx.get("https://generator.regexle.com/api?" + qstring)
        cached.parent.mkdir(exist_ok=True)
        cached.write_bytes(resp.content)
    puzzle = json.loads(cached.read_text())

    assert puzzle["side"] == side
    return puzzle


@dataclass
class Stats:
    side: int
    day: int

    build_time: float
    solve_time: float
    z3_stats: dict[str, float]


@dataclass
class Options:
    verbose: bool = True
    threads: int | None = None

    def log(self, msg: str):
        if not self.verbose:
            return
        print(msg)


def solve_puzzle(puzzle, opts: Options) -> tuple[list[list[str]], Stats]:
    opts.log("Solving...")

    solv = z3.Solver()
    if opts.threads is not None:
        solv.set(threads=opts.threads)

    t_start = time.time()
    side = puzzle["side"]

    clues = {axis: [] for axis in "xyz"}
    for axis in "xyz":
        for i, re in enumerate(puzzle[axis]):
            clues[axis].append(
                Clue.from_pattern(
                    re,
                    axis=axis,
                    index=i,
                )
            )

    maxdim = (2 * side) - 1
    assert maxdim == puzzle["diameter"]

    grid = [z3.IntVector(f"x_{x}", maxdim) for x in range(maxdim)]

    for row in grid:
        for ch in row:
            solv.add(0 <= ch)
            solv.add(ch < len(ALPHABET))

    iota = np.arange(maxdim, dtype=int)

    word_coords = {}

    for x in range(maxdim):
        word_coords["x", x] = [(x, y) for y in reversed(iota) if abs(x - y) < side]
    for y in range(maxdim):
        word_coords["y", y] = [(x, y) for x in iota if abs(x - y) < side]
    for z in range(maxdim):
        word_coords["z", z] = [
            (x, y)
            for (x, y) in zip(iota, side - 1 - z + iota)
            if (0 <= x < maxdim) and (0 <= y < maxdim)
        ]

    for axis in clues:
        for clue in clues[axis]:
            coords = word_coords[clue.axis, clue.index]
            chars = [grid[x][y] for x, y in coords]
            z3fn = clue.build_z3(solv)

            assert_matches(
                solv,
                clue.fsm,
                z3fn,
                chars,
                clue.name,
            )

    opts.log(f"Querying z3...")
    t_check = time.time()

    if solv.check() != z3.sat:
        print("Failed to solve!", file=sys.stderr)
        print("Statistics:", file=sys.stderr)
        print(solv.statistics(), file=sys.stderr)
        raise AssertionError()

    # breakpoint()

    t_done = time.time()
    opts.log(f"check() took: {t_done - t_check:.1f}s")

    model = solv.model()
    solved = [[ALPHABET[model.eval(ch).as_long()] for ch in row] for row in grid]

    stats = solv.statistics()
    result = Stats(
        side=side,
        day=puzzle["day"],
        build_time=t_check - t_start,
        solve_time=t_done - t_check,
        z3_stats={k: stats.get_key_value(k) for k in stats.keys()},
    )

    return solved, result


app = cyclopts.App()


@app.default
def main(
    *,
    side: int = 3,
    day: int | None = None,
    date: datetime.date | str = datetime.date.today(),
    opts: Annotated[Options, Parameter(name="*")],
):
    if day is None:
        if isinstance(date, str):
            date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
        day = (date - DAY_EPOCH).days

    puzzle = fetch_puzzle(day, side)
    diameter = puzzle["diameter"]

    grid, _stats = solve_puzzle(puzzle, opts)

    for y in range(diameter):
        print(" " * (2 * (side - 1) - y), end="")
        for x in range(diameter):
            if abs(x - y) < side:
                char = grid[x][y]
            else:
                char = " "
            print(char, end=" ")
        print()


@app.command
def profile(*, size: int = 3, days: str = "400..410"):
    pass


if __name__ == "__main__":
    app()
