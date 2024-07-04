# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

import wandb
from typing_extensions import override

from eden.training import State

from ._statistics import Statistics, StatisticsLogger


class WandbLogger(StatisticsLogger):
    """A logger that logs data to Weights & Biases."""

    def __init__(self, project: str, log_every_n_steps: int = 1) -> None:
        super().__init__(log_every_n_steps)
        self._project = project

    @override
    def on_start(self, state: State) -> None:
        super().on_start(state)

        wandb.init(name=state.run_name, project=self._project)
        # define metrics
        wandb.define_metric("num_evaluations", summary="max")
        wandb.define_metric("avg_fitness", summary="mean")
        wandb.define_metric("min_fitness", summary="min")
        wandb.define_metric("max_fitness", summary="max")

    @override
    def on_end(self, state: State) -> None:
        super().on_end(state)

        wandb.finish()

    @override
    def log_iter_statistics(self, state: State, statistics: Statistics) -> None:
        wandb.log({
            "num_evaluations": state.num_evaluations,
            "avg_fitness": statistics.avg_fitness,
            "min_fitness": statistics.min_fitness,
            "max_fitness": statistics.max_fitness,
            "loss": statistics.avg_loss,
        })
