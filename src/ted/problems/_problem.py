# Copyright 2024 TED Team.
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol

from torch import Tensor


class Problem(Protocol):
    """Interface for all optimization problems."""

    def map_to_problem_space(self, x: Tensor) -> Tensor:
        """Maps the solutions to the problem space.

        This method is used to map the solution from the space where the optimization
        algorithm operates to the space where the problem is defined and where the
        fitness can be computed.

        Args:
            x: The tensor of solutions. This is a 2D tensor of shape (N, D) where N is
                the number of solutions and D is the dimensionality of each solution.
                Each element of the tensor is a real number in the [-1, 1] range.

        Returns:
            The tensor of solutions mapped to the problem space.
        """
        ...

    def evaluate(self, x: Tensor) -> Tensor:
        """Computes the fitness of the given solutions.

        !!! warning

            This method assumes that the solutions are already mapped to the problem
            space. If the solutions are not mapped, the results of this method may be
            incorrect.

        Args:
            x: The tensor of solutions. This is a 2D of shape (N, D) where N is the
                number of solutions and D is the dimensionality of each solution.

        Returns:
            The tensor of fitness values. This is a 1D tensor of shape (N,).
        """
        ...

    def to_unnormalized_probability(self, fitness: Tensor) -> Tensor:
        """Converts the fitness values to unnormalized probabilities.

        Args:
            fitness: The tensor of fitness values. This is a 1D tensor of shape (N,).

        Returns:
            The tensor of unnormalized probabilities. This is a 1D tensor of shape (N,).
        """
        ...
