# Copyright 2024 TED Team.
# SPDX-License-Identifier: Apache-2.0

import random

import numpy as np
import torch
from torch import Tensor, nn

from ted.model import TED

from ._buffer import Buffer
from ._config import Config


class Sampler:
    """Langevin dynamics based sampler for TED training."""

    def __init__(self, config: Config) -> None:
        self._batch_size = config.batch_size
        self._genome_size = config.genome_size
        self._num_steps = config.sampling_steps
        self._step_size = config.sampling_step_size
        self._noise_std = config.sampling_noise_std
        self._old_ratio = config.sampling_old_ratio
        self._device = config.device

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def sample(self, model: TED, buffer: Buffer) -> Tensor:
        """Samples a batch of solutions using Langevin dynamics.

        Args:
            model: The TED model to sample solutions for.
            buffer: The buffer of solutions to sample from.

        Returns:
            A batch of solutions sampled using Langevin dynamics.
        """
        is_training = model.training
        model.eval()
        model.requires_grad_(requires_grad=False)
        gradient_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        solutions = self._generate_initial_solutions(buffer)
        solutions = self._langevin_dynamics(model, solutions)

        torch.set_grad_enabled(gradient_enabled)
        model.requires_grad_(requires_grad=True)
        model.train(is_training)

        return solutions

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _generate_initial_solutions(self, buffer: Buffer) -> Tensor:
        """Generates initial solutions for Langevin dynamics."""
        n_old = np.random.binomial(self._batch_size, self._old_ratio)  # noqa: NPY002
        n_old = min(n_old, len(buffer))
        n_new = self._batch_size - n_old

        new_solutions = torch.rand(n_new, self._genome_size)
        new_solutions.mul_(2).sub_(1)

        if n_old == 0:
            return new_solutions.to(self._device)

        old_indices = random.sample(range(len(buffer)), n_old)
        old_solutions = torch.stack([buffer[i][0] for i in old_indices])

        if n_new == 0:
            return old_solutions.to(self._device)

        solutions = torch.cat([old_solutions, new_solutions], dim=0)
        solutions = solutions.to(self._device)

        return solutions

    def _langevin_dynamics(self, model: TED, solutions: Tensor) -> Tensor:
        """Performs Langevin dynamics on the solutions."""
        # allocate memory for the noise to avoid repeated allocation
        noise = torch.randn_like(solutions, device=self._device)

        for _ in range(self._num_steps):
            # add noise to the solutions
            noise.normal_(mean=0, std=self._noise_std)
            solutions.add_(noise)
            solutions.clamp_(min=-1, max=1)

            # compute the energy of the solutions
            solutions.requires_grad_(requires_grad=True)  # type: ignore
            energies = model(solutions)  # (B,)
            energies.sum().backward()

            # update the solutions
            grad: Tensor = solutions.grad  # type: ignore
            nn.utils.clip_grad_norm_(grad, 0.03)
            solutions.detach_()
            solutions.add_(grad, alpha=-self._step_size)
            solutions.clamp_(min=-1, max=1)

        return solutions.detach()
