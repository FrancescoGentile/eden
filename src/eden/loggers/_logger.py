# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

from torch import Tensor
from typing_extensions import Protocol

from eden.training import State


class Logger(Protocol):
    """Interface for loggers."""

    def on_start(self, state: State) -> None:
        """Called at the start of training."""

    def on_iteration_start(self, state: State) -> None:
        """Called at the start of an iteration."""

    def on_sample(self, state: State, solutions: Tensor, fitnesses: Tensor) -> None:
        """Called after sampling solutions."""

    def on_loss(self, state: State, loss: Tensor) -> None:
        """Called after calculating the loss."""

    def on_iteration_end(self, state: State) -> None:
        """Called at the end of an iteration."""

    def on_end(self, state: State) -> None:
        """Called at the end of training."""
