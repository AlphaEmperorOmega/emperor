# Emperor

> **Work in Progress (Pre-release)** — This project is under active development and has not reached v1.0. Features, APIs, and configurations may change.

A PyTorch-based deep learning framework for experimenting with neural network architectures including Mixture of Experts (MoE), Vision Transformers (ViT), adaptive computation, and linear classifiers. Built on PyTorch Lightning for streamlined training across multiple image datasets.

## The Task

Emperor provides a modular system for building and evaluating neural network architectures on image classification tasks. Each experiment trains a model across standard benchmark datasets (MNIST, FashionMNIST, CIFAR-10, CIFAR-100) and logs metrics to TensorBoard.

The framework supports four model architectures:

- **linear** - Fully-connected classifier with configurable depth, activation, and dropout
- **linear_adaptive** - Fully-connected classifier where a generator network produces auxiliary parameters (diagonal weights, bias, memory) that are combined with the default layer parameters at each layer
- **experts** - Mixture of Experts architecture with learned routing between specialized sub-networks
- **vit** - Vision Transformer that splits images into patches and applies transformer encoder blocks

Each architecture defines preset configurations and a search space for hyperparameter exploration via grid search or random sampling.

## Environment Setup

**Requirements:** Python 3.11 - 3.13

**Note:** Windows users must use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) to run the setup scripts.

### Using env.sh (recommended)

```bash
source env.sh
```

This will:

1. Install `mise` if not already installed (for Python version management)
2. Install Python 3.13 via `mise`
3. Create a virtual environment at `./torchenv`
4. Install pip and all project dependencies from `pyproject.toml`
5. Activate the virtual environment

On subsequent runs, it simply activates the existing environment.

### Manual setup

```bash
curl https://mise.run | sh
export PATH="$HOME/.local/bin:$PATH"
mise install
python3 -m venv torchenv
source torchenv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Running Experiments

Use `run_experiment.sh` to train models. Each experiment runs across all four image datasets by default.

### List available experiments

```bash
source run_experiment.sh --list
```

```
Available experiments:
  experts
  linear_adaptive
  linear
  vit
```

### Run a specific configuration

```bash
source run_experiment.sh linear --name DEFAULT
source run_experiment.sh vit --name DEFAULT
source run_experiment.sh experts --name DEFAULT
```

The `--name` flag selects a preset configuration defined in the model file. Each model exposes its own set of named configurations (e.g. `DEFAULT`, `BASE`).

### Run grid search

```bash
source run_experiment.sh linear --run-all
```

This runs every combination in the model's search space across all datasets.

### Run random search

```bash
source run_experiment.sh linear --name BASE --num-samples 10
```

Randomly samples 10 configurations from the search space instead of exhaustively iterating all combinations. `--num-samples` can only be used with `--name`, not `--run-all`.

### Custom log folder

```bash
source run_experiment.sh linear --name DEFAULT --log-folder comparison_run
source run_experiment.sh vit --name DEFAULT --log-folder comparison_run
```

Use the same `--log-folder` across different models to group their TensorBoard logs together for comparison.

### Viewing results

Training metrics are logged to `logs/`. Launch TensorBoard to visualize them:

```bash
tensorboard --logdir logs/
```

Or scope to a specific experiment:

```bash
tensorboard --logdir logs/linear
```

## Project Structure

```
Emperor/              Main package
  attention/          Multi-head attention with masking, bias, and projection
  experts/            Mixture of Experts routing and expert selection
  adaptive/           Adaptive parameter generation and dynamic mixtures
  transformer/        Transformer encoder/decoder, positional embeddings, feed-forward
  linears/            Linear layer variants (diagonal, matrix, anti-diagonal)
  datasets/image/     Image dataset loaders (MNIST, CIFAR-10, CIFAR-100, FashionMNIST, SVHN)
  experiments/        Base experiment and classifier framework (PyTorch Lightning)
  halting/            Adaptive computation halting mechanisms
  behaviours/         Dynamic depth, bias, and diagonal options
  sampler/            Expert routing and sampling strategies
  neuron/             Biological neuron-inspired components
  embedding/          Positional embeddings (absolute, relative)
  base/               Core layer abstractions, config base classes, activations
  config.py           Global ModelConfig dataclass

models/               Experiment definitions
  linear.py           Linear classifier experiment
  linear_adaptive.py  Adaptive linear experiment
  experts.py          Mixture of Experts experiment
  vit.py              Vision Transformer experiment
  parser.py           CLI argument parser

docs/                 Unit tests (31 test files)
logs/                 TensorBoard logs (generated during training)
data/                 Cached datasets
```

## Quick Start

```bash
# 1. Set up the environment
source env.sh

# 2. Run a simple experiment
source run_experiment.sh linear --name DEFAULT

# 3. View training metrics
tensorboard --logdir logs/linear
```
