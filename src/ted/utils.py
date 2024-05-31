# Copyright 2024 TED Team.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions."""

import random

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seeds all random number generators."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
