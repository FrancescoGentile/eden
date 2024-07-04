# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from torch import Tensor
from typing_extensions import override

from eden import utils

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

    def __init__(
        self,
        num_variables: int,
        range_: float | tuple[float, float] = 5.12,
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

        # compute the worst fitness value
        w0 = self.range[0] ** 2 * self.num_variables
        w1 = self.range[1] ** 2 * self.num_variables
        self.worst_fitness = max(w0, w1)

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
        return (x**2).sum(dim=1)

    @override
    def to_unnormalized_probability(self, fitness: Tensor) -> Tensor:
        return self.worst_fitness - fitness
