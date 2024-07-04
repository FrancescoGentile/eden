# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from torch import Tensor, nn

from eden import utils
from eden.models import Model

from ._buffer import Buffer
from ._config import SamplerConfig


class Sampler:
    """Langevin dynamics based sampler."""

    def __init__(self, config: SamplerConfig, num_variables: int) -> None:
        self._config = config
        self._num_variables = num_variables

    # ----------------------------------------------------------------------- #
    # Public Methods
    # ----------------------------------------------------------------------- #

    def sample(self, model: Model, buffer: Buffer) -> Tensor:
        """Samples a batch of solutions using Langevin dynamics.

        Args:
            model: The model to sample solutions for.
            buffer: The buffer of solutions to sample from.

        Returns:
            A batch of solutions sampled using Langevin dynamics.
        """
        is_training = model.training
        model.eval()
        model.freeze()
        gradient_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        device = next(model.parameters()).device
        solutions = self._generate_initial_solutions(model, buffer, device)
        solutions = self._langevin_dynamics(model, solutions)

        torch.set_grad_enabled(gradient_enabled)
        model.unfreeze()
        model.train(is_training)

        return solutions

    # ----------------------------------------------------------------------- #
    # Private Methods
    # ----------------------------------------------------------------------- #

    def _generate_initial_solutions(
        self,
        model: Model,
        buffer: Buffer,
        device: torch.device,
    ) -> Tensor:
        """Generates initial solutions for Langevin dynamics."""
        n_old = np.random.binomial(self._config.num_samples, self._config.old_ratio)  # noqa: NPY002
        n_old = min(n_old, len(buffer))
        n_new = self._config.num_samples - n_old

        new_solutions = torch.rand(n_new, self._num_variables)
        new_solutions = utils.change_range(new_solutions, (0, 1), model.get_range())

        if n_old == 0:
            return new_solutions.to(device)

        # NOTE: is the uniform strategy the best choice here?
        old_solutions, _ = buffer.sample(n_old, strategy="uniform")

        if n_new == 0:
            return old_solutions.to(device)

        solutions = torch.cat([old_solutions, new_solutions], dim=0)
        solutions = solutions.to(device)

        return solutions

    def _langevin_dynamics(self, model: Model, solutions: Tensor) -> Tensor:
        """Performs Langevin dynamics on the solutions."""
        # allocate memory for the noise to avoid repeated allocation
        noise = torch.randn_like(solutions, device=solutions.device)
        low, high = model.get_range()

        for _ in range(self._config.num_steps):
            # add noise to the solutions
            noise.normal_(mean=0, std=self._config.noise_std)
            solutions.add_(noise)
            solutions.clamp_(min=low, max=high)

            # compute the energy of the solutions
            solutions.requires_grad_(requires_grad=True)  # type: ignore
            energies = model(solutions)  # (B,)
            energies.sum().backward()

            # update the solutions
            grad: Tensor = solutions.grad  # type: ignore
            if self._config.gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(grad, self._config.gradient_clip_norm)
            solutions.detach_()
            solutions.add_(grad, alpha=-self._config.step_size)
            solutions.clamp_(min=low, max=high)

        return solutions.detach()
