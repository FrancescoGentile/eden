# Copyright 2024 TED Team.
# SPDX-License-Identifier: Apache-2.0

from tqdm import tqdm
from typing_extensions import override

from ted.training import State

from ._statistics import Statistics, StatisticsLogger


class ProgressBarLogger(StatisticsLogger):
    """A progress bar for training TED."""

    def __init__(self) -> None:
        super().__init__()

    @override
    def on_start(self, state: State) -> None:
        super().on_start(state)
        self._pbar = tqdm(total=state.max_evaluations, leave=True)

    @override
    def on_end(self, state: State) -> None:
        super().on_end(state)
        self._pbar.close()

    @override
    def log_iter_statistics(self, state: State, statistics: Statistics) -> None:
        self._pbar.update(state.num_evaluations - self._pbar.n)
        self._pbar.set_postfix(
            avg_fitness=statistics.avg_fitness,
            min_fitness=statistics.min_fitness,
            max_fitness=statistics.max_fitness,
            loss=statistics.avg_loss,
        )

    @override
    def log_train_statistics(self, state: State, statistics: Statistics) -> None:
        self._pbar.set_postfix(
            avg_fitness=statistics.avg_fitness,
            min_fitness=statistics.min_fitness,
            max_fitness=statistics.max_fitness,
            loss=statistics.avg_loss,
        )
