# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

"""Module containing a collection of optimization problems."""

from ._problem import Problem
from ._rosenbrock import RosenbrockFunction
from ._sphere import SphereFunction

__all__ = [
    "Problem",
    "RosenbrockFunction",
    "SphereFunction",
]
