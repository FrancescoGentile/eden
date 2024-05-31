# Copyright 2024 TED Team.
# SPDX-License-Identifier: Apache-2.0

from torch import Tensor

from ._problem import Problem


class SphereFunction(Problem):
    r"""The Sphere function optimization problem.

    The Sphere function is a simple optimization problem with d local minima (where d is
    the dimensionality of the problem), but only one global minimum located at the
    origin. The function is continuous, convex, and unimodal, making it a popular test
    problem for optimization algorithms.

    Given a solution $x = (x_1, x_2, \ldots, x_D)$, the fitness is computed as:

    $$
        f(x) = \sum_{i=1}^{D} x_i^2
    $$
    """

    def __init__(self, range: float | tuple[float, float] = 5.12) -> None:  # noqa: A002
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
        return (x**2).sum(dim=1)

    def to_unnormalized_probability(self, fitness: Tensor) -> Tensor:
        return 1 / (fitness + 1)
