import datetime
import json
import string
import sys
import time
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path
from typing import Annotated, Iterator, Literal, Sequence, Type
from urllib.parse import urlencode

import cyclopts
import greenery
import httpx
import numpy as np
import pandas as pd
import z3
from cyclopts import Parameter
from tqdm import tqdm

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


def one_of(var, values):
    return z3.Or(*(var == v for v in values))


Axis = Literal["x", "y", "z"]


@dataclass
class Regex:
    pattern: str
    # vocabulary: str
    transition: np.ndarray  # (state, vocab)
    accept: np.ndarray  # (state,), bool

    @property
    def nstate(self) -> int:
        return self.transition.shape[0]

    @property
    def nvocab(self) -> int:
        return self.transition.shape[1]

    @classmethod
    def from_pattern(cls, pattern: str):
        fsm = to_fsm(pattern)
        transition = flatten_fsm(fsm)
        nstate = transition.shape[0]
        accept = np.zeros((nstate,), bool)
        for st in fsm.finals:
            accept[st] = True
        return cls(
            pattern=pattern,
            transition=transition,
            accept=accept,
        )

    def all_transitions(self) -> Iterator[tuple[tuple[int, int], int]]:
        it = np.nditer(self.transition, flags=["multi_index"])
        for next_state in it:
            state, char = it.multi_index
            yield (state, char), next_state.item()

    @cached_property
    def dead_states(self) -> set[int]:
        looped = self.transition == np.arange(self.nstate)[:, None]
        return set((looped.all(-1) & ~self.accept).nonzero()[0])

    @cached_property
    def dead_vocab(self) -> set[int]:
        dead = set()
        for d in self.dead_states:
            dead |= set((self.transition == d).all(0).nonzero()[0])
        return dead

    def dead_from(self, state: int) -> set[int]:
        dead = set()
        for d in self.dead_states:
            dead |= set((self.transition == d)[state].nonzero()[0])
        return dead


@dataclass
class Clue:
    axis: Axis
    index: int

    pattern: Regex

    @property
    def name(self):
        return f"{self.axis}{self.index}"

    @classmethod
    def from_pattern(cls, re: str, axis: Axis, index: int):
        fsm = to_fsm(re)
        return cls(axis=axis, index=index, pattern=Regex.from_pattern(re))


# Matchers encode a Regex into z3 constraints


class Matcher:
    def train(self, clues: list[Clue]):
        pass

    def assert_matches(self, solv: z3.Solver, clue: Clue, chars: list[z3.ArithRef]): ...


class IntFunc(Matcher):
    def build_func(self, solv: z3.Solver, clue: Clue):
        state_func = z3.Function(
            clue.name + "_xfer",
            z3.IntSort(),
            z3.IntSort(),
            z3.IntSort(),
        )
        pat = clue.pattern

        for (state, char), next_state in pat.all_transitions():
            solv.add(state_func(state, char) == next_state)

        s, c = z3.Ints("s c")
        solv.add(
            z3.ForAll(
                [s, c],
                (state_func(s, c) >= 0) & (state_func(s, c) < pat.nstate),
            )
        )
        return state_func

    def assert_matches(self, solv: z3.Solver, clue: Clue, chars: list[z3.ArithRef]):
        state_func = self.build_func(solv, clue)
        pat = clue.pattern

        nchar = len(chars)
        dead = pat.dead_states

        states = z3.IntVector(clue.name + "_state", 1 + nchar)
        for i, ch in enumerate(chars):
            solv.add(state_func(states[i], ch) == states[i + 1])

        dead_all = pat.dead_vocab
        dead_init = pat.dead_from(0)

        for ch in chars:
            solv.add(z3.Not(one_of(ch, dead_all)))
        solv.add(z3.Not(one_of(chars[0], dead_init - dead_all)))

        for st in states:
            solv.add(0 <= st)
            solv.add(st < pat.nstate)
            solv.add(z3.Not(one_of(st, dead)))
        solv.add(states[0] == 0)
        solv.add(one_of(states[-1], [i for i, v in enumerate(pat.accept) if v]))


class EnumFunc(Matcher):
    state_sort: z3.SortRef
    states: list[z3.ExprRef]

    def train(self, clues: list[Clue]):
        nstates = max(c.pattern.nstate for c in clues)
        self.state_sort, self.states = z3.EnumSort(
            "State", [f"s{i}" for i in range(nstates)]
        )

    def build_func(self, solv: z3.Solver, clue: Clue):
        state_func = z3.Function(
            clue.name + "_xfer",
            self.state_sort,
            z3.IntSort(),
            self.state_sort,
        )
        pat = clue.pattern

        for (state, char), next_state in pat.all_transitions():
            solv.add(state_func(self.states[state], char) == self.states[next_state])

        return state_func

    def assert_matches(self, solv: z3.Solver, clue: Clue, chars: list[z3.ArithRef]):
        state_func = self.build_func(solv, clue)
        pat = clue.pattern

        nchar = len(chars)
        dead = pat.dead_states

        states = [
            z3.Const(f"{clue.name}_state_{i}", self.state_sort)
            for i in range(nchar + 1)
        ]
        for i, ch in enumerate(chars):
            solv.add(state_func(states[i], ch) == states[i + 1])

        dead_all = pat.dead_vocab
        dead_init = pat.dead_from(0)

        for ch in chars:
            solv.add(z3.Not(one_of(ch, dead_all)))
        solv.add(z3.Not(one_of(chars[0], dead_init - dead_all)))

        for st in states:
            for d in dead:
                solv.add(st != self.states[d])

        solv.add(states[0] == self.states[0])
        solv.add(
            one_of(states[-1], [self.states[i] for i, v in enumerate(pat.accept) if v])
        )


STRATEGIES: dict[str, Type[Matcher]] = {
    "int_func": IntFunc,
    "enum_func": EnumFunc,
}

PUZZLE_CACHE = Path(__file__).parent / "puzzles"


def fetch_puzzle(opts, day, side):
    opts.log(f"Retrieving puzzle: {day=} {side=}. Interactive URL:")
    qstring = urlencode(dict(day=day, side=side))
    opts.log("  https://regexle.com/?" + qstring)

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
    strategy: str = "int_func"

    def log(self, msg: str):
        if not self.verbose:
            return
        print(msg)

    def get_matcher(self) -> Matcher:
        m = STRATEGIES.get(self.strategy, None)
        if m is None:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        return m()


def solve_puzzle(puzzle, opts: Options) -> tuple[list[list[str]], Stats]:
    opts.log("Solving...")

    solv = z3.Solver()
    if opts.threads is not None:
        solv.set(threads=opts.threads)

    t_start = time.time()
    side = puzzle["side"]

    matcher = opts.get_matcher()

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

    matcher.train([c for cs in clues.values() for c in cs])

    grid = [[z3.Int(f"grid_{x}_{y}") for y in range(maxdim)] for x in range(maxdim)]

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

            matcher.assert_matches(
                solv,
                clue,
                chars,
            )

    opts.log("Querying z3...")
    t_check = time.time()

    if solv.check() != z3.sat:
        print("Failed to solve!", file=sys.stderr)
        print("Statistics:", file=sys.stderr)
        print(solv.statistics(), file=sys.stderr)
        breakpoint()
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


def parse_date(type_, tokens: Sequence[cyclopts.Token]) -> datetime.date:
    date = tokens[0].value
    return datetime.datetime.strptime(date, "%Y-%m-%d").date()


def parse_range(type_, tokens: Sequence[cyclopts.Token]) -> list[int]:
    arg = tokens[0].value
    out = []
    for bit in arg.split(","):
        bit = bit.strip()
        if ".." in bit:
            start, end = (int(v) for v in bit.split(".."))
            out.extend(list(range(start, end)))
        else:
            out.append(int(bit))
    return sorted(out)


DAY_EPOCH = datetime.date(2024, 5, 31)


@app.default
def main(
    *,
    side: int = 3,
    day: int | None = None,
    date: Annotated[
        datetime.date, Parameter(converter=parse_date)
    ] = datetime.date.today(),
    opts: Annotated[Options, Parameter(name="*")] = Options(),
):
    if day is None:
        day = (date - DAY_EPOCH).days

    puzzle = fetch_puzzle(opts, day, side)
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
def profile(
    *,
    side: int = 3,
    days: Annotated[list[int], Parameter(converter=parse_range)] = list(
        range(400, 410)
    ),
    opts: Annotated[Options, Parameter(name="*")] = Options(verbose=False),
    out: Annotated[str, Parameter(alias="-o")] = "stats.json",
):
    all_stats = []
    for day in tqdm(days):
        puzzle = fetch_puzzle(opts, day=day, side=side)
        _, stats = solve_puzzle(puzzle, opts)
        all_stats.append(stats)

    df = pd.json_normalize([asdict(s) for s in all_stats])
    df.to_json(out)

    ts = df.solve_time
    print(f"z3 time:     {ts.mean():.2f}±{ts.std():.2f}s")
    print(f" (min, max): ({ts.min():.2f}, {ts.max():.2f})")


if __name__ == "__main__":
    app()
