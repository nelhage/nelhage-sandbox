"""Polymorphic summands: a base class plus three subclasses with different
`add` implementations. Summing over a random mix exercises a polymorphic
call site (the loop calls `s.add(total)` where `s` varies in type).
"""


class Summand:
    def __init__(self, value):
        self.value = value

    def add(self, total: int) -> int:
        raise NotImplementedError


class AddInt(Summand):
    # Add the stored int.
    def add(self, total: int) -> int:
        return total + self.value


class SquareInt(Summand):
    # Add the square of the stored int.
    def add(self, total: int) -> int:
        return total + self.value * self.value


class Identity(Summand):
    # Ignore the value; pass the running total through unchanged.
    def add(self, total: int) -> int:
        return total


KINDS = (AddInt, SquareInt, Identity)


def polysum(summands):
    total = 0
    for s in summands:
        total = s.add(total)
    return total
