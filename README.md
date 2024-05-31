<div align="center">

# TED

<h4>Topology-based Estimation of Distribution</h4>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://microsoft.github.io/pyright/)

[![Python](https://img.shields.io/badge/python-3.10_%7C_3.11_%7C_3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
![Static Badge](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)

</div>

## Installation

We currently do not provide a PyPI package for TED, but you can install it directly from GitHub using `pip`:

```bash
pip install "ted @ git+https://github.com/FrancescoGentile/TED.git"
```

If you instead need to work on the source code, you can clone the repository and install the package in editable mode:

```bash
git clone https://github.com/FrancescoGentile/TED.git
cd TED

# make sure that the active environment has python>=3.10
pip install -e .

# if you also need to install the development dependencies
pip install -r requirements-dev.lock
```

## Usage

At the moment, the command-line interface provides only the `train` command, which allows you to train a TED model using the provided configuration file. The configuration file is a YAML file that contains the configuration for the model architetcure, the parameters of the problem on which the model will be trained and the training parameters (e.g., the optimizer, the learning rate scheduler, the number of iterations, etc.). You can find an example configuration file in the `configs` directory.

To train a TED model, you can use the following command:

```bash
ted train /path/to/config.yaml
```
