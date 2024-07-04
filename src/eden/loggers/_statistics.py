# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING

from typing_extensions import override

from ._logger import Logger

if TYPE_CHECKING:
    from torch import Tensor

    from eden.training import State


class StatisticsLogger(Logger):
    """A logger that logs statistics."""

    def __init__(self, log_every_n_steps: int = 1) -> None:
        self._log_every_n_steps = log_every_n_steps

    @override
    def on_start(self, state: State) -> None:
        self._iter_stats = Statistics()
        self._train_stats = Statistics()

    @override
    def on_sample(self, state: State, solutions: Tensor, fitnesses: Tensor) -> None:
        self._iter_stats.update_fitness(fitnesses)
        self._train_stats.update_fitness(fitnesses)

    @override
    def on_loss(self, state: State, loss: Tensor) -> None:
        self._iter_stats.update_loss(loss, state.batch_size)
        self._train_stats.update_loss(loss, state.batch_size)

    @override
    def on_iteration_end(self, state: State) -> None:
        if (state.iteration_idx + 1) % self._log_every_n_steps != 0:
            return

        self.log_iter_statistics(state, self._iter_stats)
        self._iter_stats = Statistics()

    @override
    def on_end(self, state: State) -> None:
        self.log_train_statistics(state, self._train_stats)

    def log_iter_statistics(self, state: State, statistics: Statistics) -> None:
        """Logs the statistics computed from the last logging interval."""

    def log_train_statistics(self, state: State, statistics: Statistics) -> None:
        """Logs the statistics computed for the entire training process."""


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class Statistics:
    num_solutions: int = 0
    sum_losses: float = 0.0
    sum_fitnesses: float = 0.0
    min_fitness: float = math.inf
    max_fitness: float = -math.inf

    @property
    def avg_fitness(self) -> float:
        return self.sum_fitnesses / self.num_solutions

    @property
    def avg_loss(self) -> float:
        return self.sum_losses / self.num_solutions

    def update_fitness(self, fitnesses: Tensor) -> None:
        self.num_solutions += len(fitnesses)
        self.sum_fitnesses += fitnesses.sum().item()
        self.min_fitness = min(self.min_fitness, fitnesses.min().item())
        self.max_fitness = max(self.max_fitness, fitnesses.max().item())

    def update_loss(self, loss: Tensor, batch_size: int) -> None:
        self.sum_losses += loss.item() * batch_size
