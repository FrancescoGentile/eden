# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from torch import Tensor
from typing_extensions import override

from eden import utils

from ._problem import Problem


class RosenbrockFunction(Problem):
    r"""The Rosenbrock function optimization problem.

    The Rosenbrock function, also referred to as the Valley or Banana function, is a
    popular test problem for gradient-based optimization algorithms. The function is
    unimodal, and the global minimum lies in a narrow, parabolic valley. However, even
    though this valley is easy to find, convergence to the minimum is difficult.

    Given a solution $x = (x_1, x_2, \ldots, x_D)$, the fitness is computed as:

    $$
        f(x) = \sum_{i=1}^{D-1} [100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2]
    $$
    """

    def __init__(
        self,
        num_variables: int,
        range_: float | tuple[float, float] = (-5, 10),
    ) -> None:
        """Initializes the problem.

        Args:
            num_variables: The number of variables in the problem.
            range_: The range of values each variable of the problem can take. If a
                single value is provided, then the range is assumed to be symmetric
                around zero. If a tuple is provided, then the first element is the
                lower bound and the second element is the upper bound.
        """
        self.num_variables = num_variables

        if isinstance(range_, float | int):
            if range_ <= 0:
                msg = "The range must be a positive value."
                raise ValueError(msg)

            self.range = (-range_, range_)
        else:
            if range_[0] >= range_[1]:
                msg = "The lower bound must be less than the upper bound."
                raise ValueError(msg)

            self.range = range_

    @override
    def get_config(self) -> dict[str, Any]:
        return {"num_variables": self.num_variables, "range": self.range}

    @override
    def get_num_variables(self) -> int:
        return self.num_variables

    @override
    def map_variables(self, x: Tensor, x_range: tuple[float, float]) -> Tensor:
        return utils.change_range(x, x_range, self.range)

    @override
    def evaluate(self, x: Tensor) -> Tensor:
        first_factor = 100 * (x[1:] - x[:-1] ** 2) ** 2
        second_factor = (1 - x[:-1]) ** 2
        return (first_factor + second_factor).sum(dim=1)
