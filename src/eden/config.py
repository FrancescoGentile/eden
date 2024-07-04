# Copyright 2024 EDEN Team.
# SPDX-License-Identifier: Apache-2.0

"""Utility functions to handle configuration objects."""

import importlib
import json
from pathlib import Path
from typing import Any

import yaml


def read_config_file(path: Path | str) -> dict[str, Any]:
    """Reads a configuration file and returns its contents as a dictionary.

    Args:
        path: The path to the configuration file. Supported formats are JSON and YAML.

    Returns:
        The contents of the configuration file as a dictionary.
    """
    path = Path(path)
    if not path.is_file():
        msg = f"Configuration file '{path}' does not exist."
        raise FileNotFoundError(msg)

    match path.suffix.lower():
        case ".json":
            with path.open("r") as file:
                return json.load(file)
        case ".yaml" | ".yml":
            with path.open("r") as file:
                return yaml.safe_load(file)
        case _:
            msg = f"Unsupported configuration file format '{path.suffix}'."
            raise RuntimeError(msg)


def instantiate(
    config: dict[str, Any],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Instantiates an object or calls a function using the provided configuration.

    Args:
        config: The configuration dictionary.
        *args: Positional arguments to pass to the constructor. These will be appended
            to the arguments specified in the configuration dictionary under the key
            '_args_'.
        **kwargs: Additional keyword arguments to pass to the constructor. If a key in
            the configuration dictionary is the same as a key in the keyword arguments,
            the value in the keyword arguments will take precedence.

    Returns:
        The instantiated object or the result of the function call.
    """
    if "_target_" not in config:
        msg = "Missing '_target_' key in the configuration dictionary."
        raise RuntimeError(msg)

    module_name, target_name = config.pop("_target_").rsplit(".", 1)
    module = importlib.import_module(module_name)

    args = config.pop("_args_", []) + list(args)
    kwargs = {**config, **kwargs}

    target = getattr(module, target_name, None)
    if target is None:
        msg = f"Could not find target '{target_name}' in module '{module_name}'."
        raise RuntimeError(msg)

    return target(*args, **kwargs)
