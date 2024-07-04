# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

import abc

from torch import Tensor

from eden.nn import Module


class Model(Module, abc.ABC):
    """Base class for EDEN models."""

    @abc.abstractmethod
    def get_range(self) -> tuple[float, float]:
        """Returns the range of values each variable of the problem can take."""

    def freeze(self) -> None:
        """Freezes the model, preventing any parameters from being updated."""
        self.requires_grad_(requires_grad=False)

    def unfreeze(self) -> None:
        """Unfreezes the model, allowing parameters to be updated."""
        self.requires_grad_(requires_grad=True)

    @abc.abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        """Computes the energy of each solution in the batch.

        Args:
            x: The batch of solutions to evaluate. This is a 2D tensor of shape
                `(N, D)` where `N` is the number of solutions in the batch and
                `D` is the dimensionality of each solution (i.e. the number of
                genes in the genome).

        Returns:
            The energy of each solution in the batch. This is a 1D tensor of shape
            `(N,)`.
        """
