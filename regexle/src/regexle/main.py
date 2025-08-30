import datetime
import json
import string
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property, partial
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Callable,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    Type,
)
from urllib.parse import urlencode

import cyclopts
import greenery
import httpx
import numpy as np
import z3
from cyclopts import Parameter
from tqdm import tqdm

from regexle import z3_re

if TYPE_CHECKING:
    import pandas as pd

ALPHABET = string.ascii_uppercase


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
    parsed: greenery.Pattern

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
        parsed = greenery.parse(pattern)
        fsm = parsed.to_fsm().reduce()

        transition = flatten_fsm(fsm)
        nstate = transition.shape[0]
        accept = np.zeros((nstate,), bool)
        for st in fsm.finals:
            accept[st] = True

        return cls(
            pattern=pattern,
            parsed=parsed,
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
        return cls(axis=axis, index=index, pattern=Regex.from_pattern(re))


# Matchers encode a Regex into z3 constraints


class Matcher:
    def __init__(self, config: dict[str, str] = {}):
        pass

    def train(self, solv: z3.Solver, clues: list[Clue]):
        pass

    def make_char(self, solv: z3.Solver, name: str) -> z3.ExprRef:
        ch = z3.Int(name, solv.ctx)
        solv.add(0 <= ch)
        solv.add(ch < len(ALPHABET))
        return ch

    def extract(self, model, ch) -> str:
        return ALPHABET[model.eval(ch).as_long()]

    def assert_matches(self, solv: z3.Solver, clue: Clue, chars: list[z3.ArithRef]): ...


def config_bool(config: dict[str, str], field: str, default: bool = False) -> bool:
    if field not in config:
        return default
    val = config[field]
    if val in ("1", "True"):
        return True
    if val in ("0", "False"):
        return False
    raise ValueError(f"Bad value for config option {field}: {val}!")


def config_literal(
    config: dict[str, str], field: str, options: Sequence[str], default: str
) -> str:
    if field not in config:
        return default
    val = config[field]
    if val in options:
        return val
    raise ValueError(
        f"Bad value for config option {field}: {val}. Expected one of: {', '.join(options)}!"
    )


class IntFunc(Matcher):
    def __init__(self, config: dict[str, str] = {}):
        self.prune = config_bool(config, "prune", True)

    def build_func(self, solv: z3.Solver, clue: Clue):
        ctx = solv.ctx
        state_func = z3.Function(
            clue.name + "_trans",
            z3.IntSort(ctx),
            z3.IntSort(ctx),
            z3.IntSort(ctx),
        )
        pat = clue.pattern

        for (state, char), next_state in pat.all_transitions():
            solv.add(state_func(state, char) == next_state)

        s, c = z3.Ints("s c", ctx)
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

        states = z3.IntVector(clue.name + "_state", 1 + nchar, ctx=solv.ctx)
        for i, ch in enumerate(chars):
            solv.add(state_func(states[i], ch) == states[i + 1])

        if self.prune:
            dead = pat.dead_states
            dead_all = pat.dead_vocab
            dead_init = pat.dead_from(0)
        else:
            dead = set()
            dead_all = set()
            dead_init = set()

        for ch in chars:
            for d in dead_all:
                solv.add(ch != d)
        for d in dead_init:
            solv.add(chars[0] != d)

        for st in states:
            solv.add(0 <= st)
            solv.add(st < pat.nstate)
            for d in dead:
                solv.add(st != d)
        solv.add(states[0] == 0)
        solv.add(one_of(states[-1], [i for i, v in enumerate(pat.accept) if v]))


class EnumFunc(Matcher):
    char_sort: z3.SortRef
    alphabet: list[z3.ExprRef]

    state_sort: z3.SortRef
    states: list[z3.ExprRef]

    def __init__(self, config: dict[str, str] = {}):
        self.prune = config_bool(config, "prune", True)
        self.func = config_literal(
            config, "func", ["z3", "lambda", "forall", "python", "array"], "z3"
        )

    def train(self, solv: z3.Solver, clues: list[Clue]):
        nstates = max(c.pattern.nstate for c in clues)
        self.state_sort, self.states = z3.EnumSort(
            "State",
            [f"s{i}" for i in range(nstates)],
            solv.ctx,
        )

        self.char_sort, self.alphabet = z3.EnumSort("Char", list(ALPHABET), solv.ctx)

    def make_char(self, solv: z3.Solver, name: str):
        return z3.Const(name, self.char_sort)

    def extract(self, model, ch) -> str:
        return str(model.eval(ch))

    def build_funcexpr(
        self, solv: z3.Solver, clue: Clue, st: z3.ExprRef, ch: z3.ExprRef
    ) -> z3.ExprRef:
        fn_expr = self.states[0]

        for state_i, state in enumerate(self.states[: clue.pattern.nstate]):
            expr = self.states[0]
            for i, next_state in enumerate(clue.pattern.transition[state_i]):
                expr = z3.If(ch == self.alphabet[i], self.states[next_state], expr)
            fn_expr = z3.If(st == state, expr, fn_expr)

        return fn_expr

    def build_lambda(self, solv: z3.Solver, clue: Clue):
        st = z3.Const("state", self.state_sort)
        ch = z3.Const("char", self.char_sort)

        lambda_ = z3.Lambda([st, ch], self.build_funcexpr(solv, clue, st, ch))

        def apply(st, ch):
            return lambda_[st, ch]

        return apply

    def build_pyfunc(self, solv: z3.Solver, clue: Clue):
        return partial(self.build_funcexpr, solv, clue)

    def build_forall(self, solv: z3.Solver, clue: Clue):
        state_func = z3.Function(
            clue.name + "_trans",
            self.state_sort,
            self.char_sort,
            self.state_sort,
        )

        st = z3.Const("state", self.state_sort)
        ch = z3.Const("char", self.char_sort)

        explicit = self.build_funcexpr(solv, clue, st, ch)
        solv.add(z3.ForAll([st, ch], state_func(st, ch) == explicit))
        return state_func

    def build_z3func(self, solv: z3.Solver, clue: Clue):
        state_func = z3.Function(
            clue.name + "_trans",
            self.state_sort,
            self.char_sort,
            self.state_sort,
        )
        pat = clue.pattern

        for (state, char), next_state in pat.all_transitions():
            solv.add(
                state_func(self.states[state], self.alphabet[char])
                == self.states[next_state]
            )

        return state_func

    def build_array(self, solv: z3.Solver, clue: Clue):
        state_func = z3.Array(
            clue.name + "_trans",
            self.state_sort,
            self.char_sort,
            self.state_sort,
        )
        pat = clue.pattern

        for (state, char), next_state in pat.all_transitions():
            state_func = z3.Update(
                state_func,
                self.states[state],
                self.alphabet[char],
                self.states[next_state],
            )

        def apply(st, ch):
            return state_func[st, ch]

        return apply

        return state_func

    def build_func(
        self, solv: z3.Solver, clue: Clue
    ) -> Callable[[z3.ExprRef, z3.ExprRef], z3.ExprRef]:
        match self.func:
            case "z3":
                return self.build_z3func(solv, clue)
            case "lambda":
                return self.build_lambda(solv, clue)
            case "forall":
                return self.build_forall(solv, clue)
            case "python":
                return self.build_pyfunc(solv, clue)
            case "array":
                return self.build_array(solv, clue)
            case _:
                raise AssertionError("unreachable")

    def assert_matches(self, solv: z3.Solver, clue: Clue, chars: list[z3.ArithRef]):
        state_func = self.build_func(solv, clue)
        pat = clue.pattern

        nchar = len(chars)

        states = [
            z3.Const(f"{clue.name}_state_{i}", self.state_sort)
            for i in range(nchar + 1)
        ]
        for i, ch in enumerate(chars):
            solv.add(state_func(states[i], ch) == states[i + 1])

        if self.prune:
            dead = pat.dead_states
            dead_all = pat.dead_vocab
            dead_init = pat.dead_from(0)
        else:
            dead = set()
            dead_all = set()
            dead_init = set()

        for ch in chars:
            for d in dead_all:
                solv.add(ch != self.alphabet[d])

        for d in dead_init:
            solv.add(chars[0] != self.alphabet[d])

        for st in states:
            for d in dead:
                solv.add(st != self.states[d])

        solv.add(states[0] == self.states[0])
        solv.add(
            one_of(states[-1], [self.states[i] for i, v in enumerate(pat.accept) if v])
        )


class EnumImplies(Matcher):
    char_sort: z3.SortRef
    alphabet: list[z3.ExprRef]

    state_sort: z3.SortRef
    states: list[z3.ExprRef]

    def train(self, solv: z3.Solver, clues: list[Clue]):
        nstates = max(c.pattern.nstate for c in clues)
        self.state_sort, self.states = z3.EnumSort(
            "State",
            [f"s{i}" for i in range(nstates)],
            solv.ctx,
        )

        self.char_sort, self.alphabet = z3.EnumSort("Char", list(ALPHABET), solv.ctx)

    def make_char(self, solv: z3.Solver, name: str):
        return z3.Const(name, self.char_sort)

    def extract(self, model, ch) -> str:
        return str(model.eval(ch))

    def map_state(self, solv: z3.Solver, clue: Clue, this_state, this_char, next_state):
        pat = clue.pattern

        for (state, char), out_state in pat.all_transitions():
            solv.add(
                z3.Implies(
                    (this_state == self.states[state])
                    & (this_char == self.alphabet[char]),
                    next_state == self.states[out_state],
                )
            )

    def assert_matches(self, solv: z3.Solver, clue: Clue, chars: list[z3.ArithRef]):
        pat = clue.pattern

        nchar = len(chars)
        dead = pat.dead_states

        states = [
            z3.Const(f"{clue.name}_state_{i}", self.state_sort)
            for i in range(nchar + 1)
        ]
        for i, ch in enumerate(chars):
            self.map_state(solv, clue, states[i], ch, states[i + 1])

        dead_all = pat.dead_vocab
        dead_init = pat.dead_from(0)

        for ch in chars:
            for d in dead_all:
                solv.add(ch != self.alphabet[d])

        for d in dead_init:
            solv.add(chars[0] != self.alphabet[d])

        for st in states:
            for d in dead:
                solv.add(st != self.states[d])

        solv.add(states[0] == self.states[0])
        solv.add(
            one_of(states[-1], [self.states[i] for i, v in enumerate(pat.accept) if v])
        )


class Z3RE(Matcher):
    def __init__(self, config: dict[str, str] = {}):
        self.simplify = config_bool(config, "simplify")
        self.char = config_literal(
            config, "char", ["len1", "index", "slice", "char_code"], "len1"
        )
        self.prune = config_bool(config, "prune", False)
        self.alphabet = None

    def make_char(self, solv: z3.Solver, name: str):
        match self.char:
            case "len1":
                c = z3.String(name, solv.ctx)
                solv.add(z3.Length(c) == 1)
                return c
            case "slice":
                c = z3.String(name, solv.ctx)
                return z3.SubString(c, 0, 1)
            case "index":
                if self.alphabet is None:
                    self.alphabet = z3.StringVal(ALPHABET, solv.ctx)
                idx = z3.Int(name + ".idx", solv.ctx)
                solv.add(idx >= 0)
                solv.add(idx < len(ALPHABET))
                return z3.SubString(self.alphabet, idx, 1)
            case "char_code":
                code = z3.Int(name + ".ord", solv.ctx)
                solv.add(code >= ord("A"))
                solv.add(code <= ord("Z"))
                return z3.StrFromCode(code)
            case _:
                raise AssertionError("unreachable")

    def extract(self, model, ch) -> str:
        return model.eval(ch).as_string()

    def build_re(self, solv: z3.Solver, clue: Clue):
        try:
            re = z3_re.z3_of_pat(clue.pattern.parsed, solv.ctx)
            if self.simplify:
                re = z3.simplify(re)
            return re
        except Exception:
            breakpoint()
            raise

    def assert_matches(self, solv: z3.Solver, clue: Clue, chars: list[z3.ArithRef]):
        re = self.build_re(solv, clue)

        string = z3.Concat(chars)
        solv.add(z3.InRe(string, re))

        if self.prune:
            pat = clue.pattern
            dead_all = pat.dead_vocab
            dead_init = pat.dead_from(0)

            valid_init = [c for i, c in enumerate(ALPHABET) if i not in dead_init]
            solv.add(z3.Or(chars[0] == c for c in valid_init))

            valid_rest = [c for i, c in enumerate(ALPHABET) if i not in dead_all]
            for ch in chars[1:]:
                solv.add(z3.Or(ch == c for c in valid_rest))


STRATEGIES: dict[str, Type[Matcher]] = {
    "int_func": IntFunc,
    "enum_func": EnumFunc,
    "enum_implies": EnumImplies,
    "z3_re": Z3RE,
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
    strategy_config: dict[str, str] = field(default_factory=dict)

    def log(self, msg: str):
        if not self.verbose:
            return
        print(msg)

    def get_matcher(self) -> Matcher:
        m = STRATEGIES.get(self.strategy, None)
        if m is None:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        return m(self.strategy_config)


def solve_puzzle(puzzle, opts: Options) -> tuple[list[list[str]], Stats]:
    opts.log("Solving...")

    ctx = z3.Context()
    solv = z3.Solver(ctx=ctx)
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

    matcher.train(solv, [c for cs in clues.values() for c in cs])

    grid = [
        [matcher.make_char(solv, f"grid_{x}_{y}") for y in range(maxdim)]
        for x in range(maxdim)
    ]

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

    t_done = time.time()
    opts.log(f"check() took: {t_done - t_check:.1f}s")

    model = solv.model()
    solved = [[matcher.extract(model, ch) for ch in row] for row in grid]

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


def run_scan(tests: Iterable[tuple[dict, Options]]) -> "pd.DataFrame":
    all_stats = []
    for puzzle, opts in tests:
        _, stats = solve_puzzle(puzzle, opts)
        all_stats.append(
            asdict(stats)
            | dict(
                strategy=opts.strategy,
                strategy_config=opts.strategy_config,
            )
        )

    import pandas as pd

    return pd.json_normalize(all_stats)


@app.command
def profile(
    *,
    side: int = 3,
    days: Annotated[list[int], Parameter(converter=parse_range)] = list(
        range(400, 410)
    ),
    opts: Annotated[Options, Parameter(name="*")] = Options(verbose=False),
    out: Annotated[str, Parameter(alias="-o")] = "stats/stats.json",
):
    tests = tqdm([(fetch_puzzle(opts, day=day, side=side), opts) for day in days])

    df = run_scan(tests)
    Path(out).parent.mkdir(exist_ok=True)
    df.to_json(out)

    ts = df.solve_time
    print(f"z3 time:     {ts.mean():.2f}±{ts.std():.2f}s")
    print(f" (min, max): ({ts.min():.2f}, {ts.max():.2f})")
    ts = df.build_time
    print(f"py time:     {ts.mean():.2f}±{ts.std():.2f}s")
    print(f" (min, max): ({ts.min():.2f}, {ts.max():.2f})")


@app.command
def matrix(
    *,
    out: Annotated[str, Parameter(alias="-o")] = "stats/matrix.json",
):
    tests = []

    opts = Options(verbose=False)
    for side in range(3, 9):
        for day in range(400, 450):
            puzzle = fetch_puzzle(opts, day, side)

            if side < 4:
                tests.append((puzzle, replace(opts, strategy="int_func")))
            tests.append((puzzle, replace(opts, strategy="enum_func")))
            tests.append((puzzle, replace(opts, strategy="z3_re")))

            if side < 6:
                tests.append(
                    (
                        puzzle,
                        replace(
                            opts,
                            strategy="enum_func",
                            strategy_config=dict(prune="0"),
                        ),
                    )
                )

    df = run_scan(tqdm(tests))
    Path(out).parent.mkdir(exist_ok=True)
    df.to_json(out)


if __name__ == "__main__":
    app()
