# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions."""

import random

import numpy as np
import torch
from torch import Tensor


def seed_all(seed: int) -> None:
    """Seeds all random number generators."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def change_range(
    x: Tensor,
    old_range: tuple[float, float],
    new_range: tuple[float, float],
) -> Tensor:
    """Changes the range of each element in the tensor.

    !!! note
        If the old range is equal to the new range, then the tensor is returned as is.

    Args:
        x: The tensor to change the range of.
        old_range: The old range of the tensor.
        new_range: The new range of the tensor.

    Returns:
        The tensor with the new range.
    """
    if old_range == new_range:
        return x

    old_min, old_max = old_range
    new_min, new_max = new_range

    return ((x - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min
