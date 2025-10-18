from typing import Protocol

import z3


class GoalLike(Protocol):
    ctx: z3.Context

    def add(self, *args: z3.ExprRef | bool): ...
