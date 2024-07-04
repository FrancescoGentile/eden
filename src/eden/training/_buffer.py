# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

import random
from collections import deque
from typing import Literal

import torch
from torch import Tensor

from ._config import BufferConfig


class Buffer:
    """A buffer of solutions."""

    def __init__(self, config: BufferConfig) -> None:
        self._xs = deque(maxlen=config.max_size)
        self._ys = deque(maxlen=config.max_size)

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def add(self, x: Tensor, y: Tensor) -> None:
        """Adds the given solutions and thier corresponding targets to the buffer.

        Args:
            x: The solutions to add. This can be both a single solution (1D tensor) or
                multiple solutions (2D tensor).
            y: The targets corresponding to the solutions. If `x` is a single solution,
                this should be a scalar tensor. If `x` is multiple solutions, this
                should be a 1D tensor with the same number of elements as the number of
                solutions in `x`.
        """
        if x.ndim == 1:
            if y.ndim != 0:
                msg = f"Expected `y` to be a scalar tensor, but got a {y.ndim}D tensor."
                raise RuntimeError(msg)

            self._xs.append(x)
            self._ys.append(y)
        elif x.ndim == 2:  # noqa: PLR2004
            if y.ndim != 1 or len(x) != len(y):
                msg = (
                    f"Expected `y` to be a tensor with shape ({len(x)},), but got a "
                    "tensor with shape {y.shape}."
                )
                raise RuntimeError(msg)

            for xi, yi in zip(x, y, strict=True):
                self._xs.append(xi)
                self._ys.append(yi)
        else:
            msg = f"Expected `x` to be a 1D or 2D tensor, but got a {x.ndim}D tensor."
            raise RuntimeError(msg)

    def clear(self) -> None:
        """Clears all elements from the buffer."""
        self._xs.clear()
        self._ys.clear()

    def is_empty(self) -> bool:
        """Returns whether the buffer is empty."""
        return len(self._xs) == 0

    def sample(
        self,
        n: int,
        *,
        strategy: Literal["uniform"] | None = None,  # noqa: ARG002
    ) -> tuple[Tensor, Tensor]:
        """Samples `n` solutions from the buffer.

        Args:
            n: The number of samples to generate.
            strategy: The strategy used to sample solutions from the buffer. You can
                use this argument to override the default strategy set in the
                configuration. If `None`, the default strategy is used.

        Returns:
            A tuple containing the sampled solutions and their corresponding targets.
            The solutions are returned as a 2D tensor with shape `(n, d)`, where `d` is
            the dimensionality of the solutions. The targets are returned as a 1D tensor
            with shape `(n,)`.
        """
        if not 0 < n <= len(self._xs):
            msg = f"The number of samples must be in the range (0, {len(self._xs)}]."
            raise ValueError(msg)

        indices = random.sample(range(len(self._xs)), n)
        x = torch.stack([self._xs[i] for i in indices])
        y = torch.stack([self._ys[i] for i in indices])

        return x, y

    # ----------------------------------------------------------------------- #
    # Magic Methods
    # ----------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._xs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._xs[index], self._ys[index]
