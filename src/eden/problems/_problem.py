# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

import abc
from typing import Any

from torch import Tensor


class Problem(abc.ABC):
    """Base class for optimization problems."""

    @abc.abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Returns the configuration of the problem."""

    @abc.abstractmethod
    def get_num_variables(self) -> int:
        """Returns the number of variables in the problem."""

    @abc.abstractmethod
    def map_variables(self, x: Tensor, x_range: tuple[float, float]) -> Tensor:
        """Maps the variables to the problem space.

        This method is used to map the variables from the space where the model operates
        to the space where the problem is defined. This is necessary because the model
        may operate in a different space than the problem.

        Args:
            x: The batch of solutions. This is a 2D tensor of shape `(N, D)` where N is
                the number of solutions and D is the dimensionality of each solution,
                which is equal to the number of variables in the problem.
            x_range: The range of values the variables can take in the space where the
                model operates. This is a tuple of two real numbers representing the
                lower and upper bounds of the range.

        Returns:
            The batch of solutions mapped to the problem space.
        """

    @abc.abstractmethod
    def evaluate(self, x: Tensor) -> Tensor:
        """Computes the fitness of the given solutions.

        !!! warning

            This method assumes that the solutions are already mapped to the problem
            space. If the solutions are not mapped, the results of this method may be
            incorrect.

        Args:
            x: The batch of solutions. This is a 2D tensor of shape `(N, D)` where N is
                the number of solutions and D is the dimensionality of each solution,
                which is equal to the number of variables in the problem.

        Returns:
            The tensor of fitness values. This is a 1D tensor of shape `(N,)`.
        """
        ...

    @abc.abstractmethod
    def to_unnormalized_probability(self, fitness: Tensor) -> Tensor:
        """Converts the fitness values to unnormalized probabilities.

        Args:
            fitness: The tensor of fitness values. This is a 1D tensor of shape `(N,)`.

        Returns:
            The tensor of unnormalized probabilities.
        """

    def map_and_evaluate(self, x: Tensor, x_range: tuple[float, float]) -> Tensor:
        """Maps the variables to the problem space and computes the fitness.

        This method combines the `map_variables` and `evaluate` methods to provide a
        single method that maps the variables to the problem space and computes the
        fitness.

        Args:
            x: The batch of solutions. This is a 2D tensor of shape `(N, D)` where N is
                the number of solutions and D is the dimensionality of each solution,
                which is equal to the number of variables in the problem.
            x_range: The range of values the variables can take in the space where the
                model operates. This is a tuple of two real numbers representing the
                lower and upper bounds of the range.

        Returns:
            The tensor of fitness values. This is a 1D tensor of shape `(N,)`.
        """
        x_mapped = self.map_variables(x, x_range)
        return self.evaluate(x_mapped)
