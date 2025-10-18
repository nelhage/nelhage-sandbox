from typing import (
    Generic,
    Protocol,
    Sequence,
    TypeVar,
)

import z3

from regexle import GoalLike

# Mappers map between Python objects and a Z3 representation

T = TypeVar("T", covariant=True)


class Mapper(Protocol, Generic[T]):
    def __init__(self, name: str, py_objs: list[T], ctx: z3.Context | None = None): ...

    @property
    def sort(self) -> z3.SortRef: ...

    def to_z3(self, idx: int) -> z3.AstRef: ...

    def from_z3(self, val: z3.AstRef) -> T | None: ...

    def make_const(self, solv: GoalLike, name: str) -> z3.AstRef: ...


class EnumMapper(Mapper[T]):
    def __init__(self, name: str, py_objs: Sequence[T], ctx=None):
        self._sort, self._vals = z3.EnumSort(
            name,
            [f"{name}__{v}" for v in py_objs],
            ctx,
        )
        self._from_decl = {v.decl(): o for v, o in zip(self._vals, py_objs)}

    @property
    def sort(self) -> z3.SortRef:
        return self._sort

    def to_z3(self, idx: int) -> z3.AstRef:
        return self._vals[idx]

    def from_z3(self, val: z3.AstRef) -> T | None:
        assert isinstance(val, z3.DatatypeRef), "z3 object must be datatype"
        return self._from_decl.get(val.decl())

    def make_const(self, solv: GoalLike, name: str) -> z3.AstRef:
        return z3.Const(name, self.sort)


class IntMapper(Mapper[T]):
    def __init__(self, name: str, py_objs: Sequence[T], ctx=None):
        self._sort = z3.IntSort(ctx)
        self._py_objs = py_objs

    @property
    def nval(self) -> int:
        return len(self._py_objs)

    @property
    def sort(self) -> z3.SortRef:
        return self._sort

    def to_z3(self, idx: int) -> z3.AstRef:
        return z3.IntVal(idx, self.sort.ctx)

    def from_z3(self, val: z3.AstRef) -> T:
        assert isinstance(val, z3.IntNumRef), "z3 object must be int"
        return self._py_objs[val.as_long()]

    def make_const(self, solv: GoalLike, name: str) -> z3.AstRef:
        val = z3.Int(name, self.sort.ctx)
        solv.add(val >= 0)
        solv.add(val < self.nval)
        return val


class StringMapper(Mapper[str]):
    def __init__(self, name: str, py_objs: Sequence[str], ctx=None):
        self._sort = z3.StringSort(ctx)
        self._py_objs = py_objs

    @property
    def nval(self) -> int:
        return len(self._py_objs)

    @property
    def sort(self) -> z3.SortRef:
        return self._sort

    def to_z3(self, idx: int) -> z3.AstRef:
        return z3.StringVal(self._py_objs[idx], ctx=self.sort.ctx)

    def from_z3(self, val: z3.AstRef) -> str:
        assert isinstance(val, z3.SeqRef), "z3 object must be string"
        return val.as_string()

    def make_const(self, solv: GoalLike, name: str) -> z3.AstRef:
        val = z3.String(name, self.sort.ctx)
        solv.add(z3.Length(val) == 1)
        return val
