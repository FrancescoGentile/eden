# Copyright 2024 TED Team.
# SPDX-License-Identifier: Apache-2.0

"""The Topology-based Estimantion of Distribution (TED) algorithm."""

import argparse
from pathlib import Path

import hydra

from . import model, training, utils
from ._version import __version__


def train(args: argparse.Namespace) -> None:
    """Trains a model using the provided configuration."""
    config_file = Path(args.config).absolute()
    config_dir, config_name = config_file.parent, config_file.stem
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = hydra.compose(config_name=config_name)

    if "seed" in cfg:
        utils.seed_all(cfg.seed)

        cfg.model = cfg.model or {}
        ted_cfg = model.Config(**cfg.model)
        ted = model.TED(ted_cfg)

        optimizer = hydra.utils.instantiate(cfg.optimizer, params=ted.parameters())

        if "scheduler" in cfg:
            scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)
        else:
            scheduler = None

        problem = hydra.utils.instantiate(cfg.problem)

        cfg.engine = cfg.engine or {}
        engine_cfg = training.Config(**cfg.engine)
        loggers = [hydra.utils.instantiate(logger) for logger in cfg.get("loggers", [])]
        engine = training.Engine(engine_cfg, loggers=loggers)

        engine.run(model=ted, problem=problem, optimizer=optimizer, scheduler=scheduler)


def main() -> None:
    """Main entry point for the TED package."""
    description = "Command-line interface for the TED package."
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
