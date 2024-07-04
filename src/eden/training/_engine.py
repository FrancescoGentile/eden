# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import random
import timeit
from typing import Any

import coolname
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from eden.models import Model
from eden.problems import Problem

from ._buffer import Buffer
from ._config import Config
from ._sampler import Sampler


@dataclasses.dataclass
class State:
    """The state of the training engine."""

    run_name: str
    num_evaluations: int
    iteration_idx: int
    model: Model
    problem: Problem
    buffer: Buffer
    sampler: Sampler
    optimizer: Optimizer
    lr_scheduler: LRScheduler | None


class Engine:
    """The training engine."""

    def __init__(self, config: Config | None = None, **kwargs: Any) -> None:
        """Initializes the training engine.

        Args:
            config: The configuration of the training engine. If not provided, the
                configuration will be created from the provided keyword arguments.
            **kwargs: Additional arguments to pass to the training engine. These will be
                used to create the configuration if one is not provided. This is useful
                when instantiating the training engine from a configuration file.
        """
        config = config or Config.from_dict(kwargs)
        self._config = config

        self._state: State

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def run(
        self,
        model: Model,
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
        model = model.to(self._config.device)
        buffer = Buffer(self._config.buffer)
        sampler = Sampler(self._config.sampler, problem.get_num_variables())

        self._state = State(
            run_name=_generate_run_name(),
            num_evaluations=0,
            iteration_idx=0,
            model=model,
            problem=problem,
            buffer=buffer,
            sampler=sampler,
            optimizer=optimizer,
            lr_scheduler=scheduler,
        )

        self._populate_buffer()
        while True:
            self._update()

            if self._state.num_evaluations >= self._config.max_num_evaluations:
                break

            self._sample()

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _populate_buffer(self) -> None:
        """Fills the buffer with random solutions."""
        n, d = self._config.buffer.min_size, self._state.problem.get_num_variables()
        x = torch.rand(n, d)
        f = self._state.problem.map_and_evaluate(x, (0, 1))
        y = -torch.log(self._state.problem.to_unnormalized_probability(f))

        self._state.buffer.add(x, y)

    def _sample(self) -> None:
        """Performs the sampling step and updates the buffer."""
        model = self._state.model
        problem = self._state.problem
        buffer = self._state.buffer
        sampler = self._state.sampler

        solutions = sampler.sample(model, buffer)
        fitnesses = problem.map_and_evaluate(solutions, model.get_range())
        self._state.num_evaluations += solutions.size(0)

        # e^-E(x) = p_u(x) -> E(x) = -log(p_u(x))
        # where E(x) is the energy of the solution x estimated by the model
        # and p_u(x) is the unnormalized probability of x in the problem
        p_u = problem.to_unnormalized_probability(fitnesses)
        targets = -torch.log(p_u)

        # we store the solutions and targets on the CPU to save GPU memory
        # which is usually more scarce than RAM
        buffer.add(solutions.cpu(), targets.cpu())

    def _update(self) -> None:
        """Performs the update steps."""
        model = self._state.model
        optimizer = self._state.optimizer
        lr_scheduler = self._state.lr_scheduler

        model.train()
        optimizer.zero_grad()

        for _ in range(self._config.update.num_steps):
            x, y = self._state.buffer.sample(self._config.update.batch_size)
            x = x.to(self._config.device)
            y = y.to(self._config.device)

            outputs = model(x)
            loss = F.mse_loss(outputs, y)
            loss.backward()

            if self._config.update.gradient_clip_norm is not None:
                params = model.parameters()
                nn.utils.clip_grad_norm_(params, self._config.update.gradient_clip_norm)

            optimizer.step()
            optimizer.zero_grad()

            if lr_scheduler is not None:
                lr_scheduler.step()


# ----------------------------------------------------------------------- #
# Private Functions
# ----------------------------------------------------------------------- #


def _generate_run_name() -> str:
    """Generates a unique run name."""
    # use the current time to seed the random number generator
    # this is to avoid generating the same run name in case the user
    # specifies the same seed in the configuration
    seed = int(timeit.default_timer() * 1e6)
    old_state = random.getstate()
    random.seed(seed)

    run_name = coolname.generate_slug(2)
    random.setstate(old_state)

    return run_name
