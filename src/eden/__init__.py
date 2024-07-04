# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

"""Estimation of Distribution using ENergy-based models."""

import argparse

from . import config, training, utils
from ._version import __version__


def train(args: argparse.Namespace) -> None:
    """Trains a model using the provided configuration."""
    cfg = config.read_config_file(args.config)

    if "seed" in cfg:
        utils.seed_all(cfg["seed"])

    problem = config.instantiate(cfg["problem"])
    model = config.instantiate(cfg["model"], num_variables=problem.get_num_variables())

    optimizer = config.instantiate(cfg["optimizer"], params=model.parameters())
    scheduler = None
    if "scheduler" in cfg:
        scheduler = config.instantiate(cfg["scheduler"], optimizer=optimizer)

    # callbacks = [config.instantiate(cb) for cb in cfg.get("callbacks", [])]
    engine = training.Engine(**cfg["engine"])

    engine.run(model=model, problem=problem, optimizer=optimizer, scheduler=scheduler)


def main() -> None:
    """Main entry point for the EDEN package."""
    description = "Command-line interface of the EDEN package."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--version", action="version", version=__version__)

    subparsers = parser.add_subparsers(required=True, title="subcommands")
    train_parser = subparsers.add_parser("train", help="Train a model.")
    train_parser.add_argument(
        "config",
        type=str,
        help="Path to the configuration file for training the model.",
    )
    train_parser.set_defaults(func=train)

    args = parser.parse_args()
    args.func(args)
