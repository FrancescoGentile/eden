# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path

from typing_extensions import override

from eden.training import State

from ._statistics import Statistics, StatisticsLogger


class FileLogger(StatisticsLogger):
    """A logger that logs data to a file."""

    def __init__(
        self,
        *,
        file: str = "output/{run_name}/training.log",
        log_every_n_steps: int = 1,
        error_if_exists: bool = True,
    ) -> None:
        super().__init__(log_every_n_steps)

        self._file = file
        self._error_if_exists = error_if_exists

        self._logger: logging.Logger

    @override
    def on_start(self, state: State) -> None:
        super().on_start(state)

        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s] - %(name)s - %(levelname)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        self._file = self._file.format(run_name=state.run_name)
        Path(self._file).parent.mkdir(parents=True, exist_ok=True)
        if Path(self._file).exists() and self._error_if_exists:
            msg = f"File {self._file} already exists."
            raise FileExistsError(msg)

        file_handler = logging.FileHandler(self._file)
        file_handler.setFormatter(formatter)

        self._logger.addHandler(file_handler)
        self._logger.info("Training started.")

    @override
    def on_end(self, state: State) -> None:
        super().on_end(state)

        for handler in self._logger.handlers:
            handler.close()
            self._logger.removeHandler(handler)

    @override
    def log_iter_statistics(self, state: State, statistics: Statistics) -> None:
        self._logger.info("Iteration %d", state.iteration_idx + 1)
        self._logger.info("\ttotal_solutions: %d", state.num_evaluations)
        self._logger.info("\tloss: %f", statistics.avg_loss)
        self._logger.info("\tavg_fitness: %f", statistics.avg_fitness)
        self._logger.info("\tmin_fitness: %f", statistics.min_fitness)
        self._logger.info("\tmax_fitness: %f", statistics.max_fitness)

    @override
    def log_train_statistics(self, state: State, statistics: Statistics) -> None:
        self._logger.info("Training completed.")
        self._logger.info("\ttotal_solutions: %d", state.num_evaluations)
        self._logger.info("\tloss: %f", statistics.avg_loss)
        self._logger.info("\tavg_fitness: %f", statistics.avg_fitness)
        self._logger.info("\tmin_fitness: %f", statistics.min_fitness)
        self._logger.info("\tmax_fitness: %f", statistics.max_fitness)
