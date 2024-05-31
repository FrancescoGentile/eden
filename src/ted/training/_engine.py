# Copyright 2024 TED Team.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import random
import timeit
from collections.abc import Sequence
from typing import TYPE_CHECKING

import coolname
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from ted.model import TED
from ted.problems import Problem

from ._buffer import Buffer
from ._config import Config
from ._sampler import Sampler

if TYPE_CHECKING:
    from ted.loggers import Logger


@dataclasses.dataclass
class State:
    """The state of the training engine."""

    run_name: str
    batch_size: int
    iteration_idx: int
    num_evaluations: int
    max_evaluations: int


class Engine:
    """The training engine for TED."""

    def __init__(
        self,
        config: Config,
        loggers: Sequence["Logger"] | None = None,
    ) -> None:
        self._config = config

        self._state: State
        self._buffer: Buffer = Buffer(config)
        self._sampler: Sampler = Sampler(config)
        self._loggers = loggers or []

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def run(
        self,
        model: TED,
        problem: Problem,
        optimizer: Optimizer,
        scheduler: LRScheduler | None = None,
    ) -> None:
        """Trains the model using the provided problem.

        Args:
            model: The model to train.
            problem: The problem to optimize the model for.
            optimizer: The optimizer to use to update the model's parameters.
            scheduler: The learning rate scheduler to use.
        """
        torch.set_grad_enabled(True)

        self._state = State(
            run_name=_generate_run_name(),
            batch_size=self._config.batch_size,
            iteration_idx=0,
            num_evaluations=0,
            max_evaluations=self._config.max_evaluations,
        )

        model = model.to(self._config.device)
        model.train()

        for logger in self._loggers:
            logger.on_start(self._state)

        while self._state.num_evaluations < self._config.max_evaluations:
            for logger in self._loggers:
                logger.on_iteration_start(self._state)

            # Forward pass
            samples, targets = self._sample_batch(model, problem)
            outputs = model(samples)
            loss = F.mse_loss(outputs, targets, reduction="mean")

            for logger in self._loggers:
                logger.on_loss(self._state, loss)

            # Backward pass
            loss.backward()

            if self._config.gradient_clip_norm is not None:
                params = model.parameters()
                nn.utils.clip_grad_norm_(params, self._config.gradient_clip_norm)

            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None:
                scheduler.step()

            for logger in self._loggers:
                logger.on_iteration_end(self._state)

            self._state.iteration_idx += 1

        for logger in self._loggers:
            logger.on_end(self._state)

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _sample_batch(self, model: TED, problem: Problem) -> tuple[Tensor, Tensor]:
        """Samples solutions from the model, updates the buffer and the statistics."""
        solutions = self._sampler.sample(model, self._buffer)
        fitnesses = problem.evaluate(solutions)
        self._state.num_evaluations += solutions.size(0)
        self._buffer.add(solutions.cpu(), fitnesses.cpu())

        for logger in self._loggers:
            logger.on_sample(self._state, solutions, fitnesses)

        targets = problem.to_unnormalized_probability(fitnesses)
        targets = -torch.log(targets)

        return solutions, targets


# ----------------------------------------------------------------------- #
# Private Functions
# ----------------------------------------------------------------------- #


def _generate_run_name() -> str:
    """Generates a unique run name."""
    # use the current time to seed the random number generator
    seed = int(timeit.default_timer() * 1e6)
    old_state = random.getstate()
    random.seed(seed)

    run_name = coolname.generate_slug(2)
    random.setstate(old_state)

    return run_name
