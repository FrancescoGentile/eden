# Copyright 2024 TED Team.
# SPDX-License-Identifier: Apache-2.0

from torch import Tensor

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

    def __init__(self, range: float | tuple[float, float] = (-5, 10)) -> None:  # noqa: A002
        """Initializes the problem.

        Args:
            range: The range of values each dimension of the solution can take. If a
                single value is provided, then the range is assumed to be symmetric
                around zero. If a tuple is provided, then the first element is the
                lower bound and the second element is the upper bound.
        """
        if isinstance(range, float | int):
            if range <= 0:
                msg = "The range must be a positive value."
                raise ValueError(msg)

            self.range = (-range, range)
        else:
            if range[0] >= range[1]:
                msg = "The lower bound must be less than the upper bound."
                raise ValueError(msg)

            self.range = range

    def map_to_problem_space(self, x: Tensor) -> Tensor:
        lower, upper = self.range
        return (x + 1) * (upper - lower) / 2 + lower

    def evaluate(self, x: Tensor) -> Tensor:
        first_factor = 100 * (x[1:] - x[:-1]) ** 2
        second_factor = (x[:-1] - 1) ** 2
        return (first_factor + second_factor).sum(dim=1)
