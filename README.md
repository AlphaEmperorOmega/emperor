# Emperor

> **Work in Progress (Pre-release)** — This project is under active development and has not reached v1.0. Features, APIs, and configurations may change.

## Vision

Emperor is a personal research framework for quickly building experimental neural network models from reusable components. It is designed for trying architecture ideas, combining them with existing building blocks, and testing them on preset datasets without rebuilding the same training, dataset, logging, and visualization plumbing every time.

The project exists to make model experimentation easier to repeat and easier to inspect. Models are built through hierarchical configuration, so pieces such as layer stacks, gates, recurrent wrappers, halting controllers, memory modules, and other experimental components can be reused and recombined across different ideas.

The main goal is understanding: build small models, compare variants, watch what happens during training, and use those observations to decide which ideas are worth exploring further.

---

## Quick Start

This path uses `linears/linear`, the currently stable end-to-end model. The
first training command is intentionally small: one preset, one dataset, and one
epoch.

```bash
# 1. Set up Python, install dependencies, activate ./torchenv, and start Viewer
source env.sh

# 2. Confirm the model catalog and available presets
source experiment.sh --list-models
source experiment.sh --model-type linears --model linear --list-presets

# 3. Inspect the baseline model without training
source experiment.sh --model-type linears --model linear --preset baseline --print-model

# 4. Run a smoke training job
source experiment.sh --model-type linears --model linear \
  --preset baseline \
  --datasets Mnist \
  --logdir quickstart \
  --config --num-epochs 1

# 5. Open the TensorBoard logs
tensorboard --logdir logs/quickstart
```

After `source env.sh`, the browser Viewer is available at
`http://localhost:9000`, backed by the local API on port `9999`.

The smoke run builds the `baseline` preset, trains it on `Mnist` for one epoch,
and writes TensorBoard events plus a `result.json`. The first run may also
download the dataset.

Logs for the smoke run are written under
`logs/quickstart/linears/linear/BASELINE/Mnist/<parameter-id>_<timestamp>/version_*/`.
Each completed run gets a `result.json`, and the best results summary is kept at
`logs/quickstart/linears/linear/best_results.json`.

Important scaling rules:

- Without `--datasets`, the command trains once for every dataset in
  `models/linears/linear/config.py`.
- Without `--config --num-epochs 1`, the command uses the model's configured
  epoch count.
- `--config` marks the following flags as model-config overrides.
- Runs multiply by selected presets, selected datasets, and search samples.

Once the smoke run works, compare small variants:

```bash
# Compare two presets on one dataset for one epoch each
source experiment.sh --model-type linears --model linear \
  --presets baseline gating \
  --datasets Mnist \
  --logdir comparison \
  --config --num-epochs 1

# Sample 10 random configs for one preset on one dataset
source experiment.sh --model-type linears --model linear \
  --preset baseline \
  --datasets Mnist \
  --random-search 10 \
  --logdir search_test \
  --config --num-epochs 1
```

For example, `--presets baseline gating --datasets Mnist Cifar10 --random-search 10`
creates `2 x 2 x 10` training runs.

## Environment Setup

**Requirements:** Python 3.13 (pinned in `mise.toml`). The package itself supports Python 3.11–3.13.

**Note:** Windows users must use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) to run the setup scripts.

### Using env.sh (recommended)

```bash
source env.sh
```

On first run, `env.sh` installs `mise` (if missing), uses it to provision the Python version pinned in `mise.toml`, creates a `./torchenv` virtualenv, installs the project from `pyproject.toml`, activates the environment, and starts the browser model viewer in the background. On subsequent runs it re-activates and reuses the already-running viewer servers when possible. The script is short and readable — open [`env.sh`](env.sh) if you want to see the exact sequence or adapt it.

Viewer logs and PID files are written under `viewer/.runtime/`. By default, the backend uses port `9999` and the frontend uses port `9000`. Stop or inspect the viewer with:

```bash
source env.sh --viewer-stop
source env.sh --viewer-status
```

Running `source env.sh` again is safe: it reuses live PID files and also checks whether ports `9999` or `9000` are already listening before starting another server.

## Running Experiments

Use `experiment.sh` to train a model. The general form is:

```bash
source experiment.sh --model-type <type> --model <name> [options]
```

Run `source experiment.sh` with no arguments to print the full list of flags.

### List available models

```bash
source experiment.sh --list-models
```

```
Available models:
  --model-type experts --model experts_linear
  --model-type experts --model experts_linear_adaptive
  --model-type linears --model linear
  --model-type linears --model linear_adaptive
  --model-type neuron --model neuron_linear
  --model-type parametric --model parametric_generator
  --model-type parametric --model parametric_matrix
  --model-type parametric --model parametric_vector
  --model-type transformer_encoder --model bert_linear
  --model-type transformer_encoder --model vit_linear
```

### List a model's presets

Each model exposes its own set of named presets. List them with `--list-presets`:

```bash
source experiment.sh --model-type linears --model linear --list-presets
```

```
Available presets for --model-type linears --model linear:
  baseline  --  Baseline linear stack preset; supports search-space flags.
  gating  --  Linear stack with a learned gate applied to hidden-layer outputs.
  halting  --  Linear stack with adaptive computation halting enabled.
  gating-halting  --  Linear stack with both learned gating and adaptive computation halting.
```

Every preset supports the search-space flags (`--grid-search`, `--random-search`).

### Run a preset

```bash
source experiment.sh --model-type linears --model linear --preset baseline
source experiment.sh --model-type linears --model linear --preset gating-halting
```

`--preset` selects which preset to run. Exactly one of `--preset`, `--presets`, `--all-presets`, or `--list-config` is required.

### Run selected presets

```bash
source experiment.sh --model-type linears --model linear --presets baseline gating
source experiment.sh --model-type linears --model linear --presets baseline gating --grid-search
```

`--presets` runs a selected subset of presets sequentially in one invocation. The first preset is the primary preset for compatibility with the existing `preset` field in viewer training jobs, and each selected preset is trained across the selected datasets before the command exits.

### Reference Architectures

- **linears/linear** — Fully-connected classifier with configurable depth, activation, and dropout
- **linears/linear_adaptive** — Fully-connected classifier where a generator network produces auxiliary parameters (diagonal weights, bias, memory) that are combined with the default layer parameters at each layer
- **experts/experts_linear** — Mixture of Experts architecture with learned routing between specialized linear sub-networks
- **experts/experts_linear_adaptive** — Mixture of Experts whose experts use adaptive (generator-produced) parameters
- **transformer_encoder/vit_linear** — Vision Transformer classifier with linear patch embeddings, a trainable class token, and configurable linear transformer sub-stacks
- **parametric/parametric_generator** — Adaptive parameter layer where expert weights are generated by a generator network
- **parametric/parametric_matrix** — Adaptive parameter layer where expert weights are mixed as full matrices via a shared router
- **parametric/parametric_vector** — Adaptive parameter layer where expert weights are mixed as vectors via independent routers with configurable layer stack options
- **neuron/neuron_linear** — Linear model wrapped with neuron hidden-block adaptation
- **transformer_encoder/bert_linear** — BERT pretraining model with MLM/NSP heads, token type embeddings, tied MLM decoder weights, and configurable linear transformer sub-stacks

> [!WARNING]
> The following reference architectures are currently broken on `main` and will fail to run until the next update: `linears/linear_adaptive`, `experts/experts_linear`, `experts/experts_linear_adaptive`, `transformer_encoder/vit_linear`, `parametric/parametric_generator`, `parametric/parametric_matrix`, `parametric/parametric_vector`, `transformer_encoder/bert_linear`. Only `linears/linear` is known to work end-to-end right now.

Run `source experiment.sh --model-type <type> --model <name> --list-presets` to discover any extra ablation presets a model exposes.

### Datasets

Each run trains one checkpoint per selected preset and per entry in the model's `DATASET_OPTIONS` list (in `config.py`), sequentially. A single `--preset` invocation produces `len(DATASET_OPTIONS)` checkpoints, `--presets p1 p2` produces `2 × len(DATASET_OPTIONS)` checkpoints, and a search flag multiplies that count. Add or remove dataset classes to change the targets; the full catalog lives under `emperor/datasets/`:

```python
DATASET_OPTIONS: list = [Mnist, FashionMNIST, Cifar10, Cifar100]
```

### Run all presets

```bash
source experiment.sh --model-type linears --model linear_adaptive --all-presets
```

Runs every preset in the model's `ExperimentOptions` sequentially across all datasets.

### Run grid search

```bash
source experiment.sh --model-type linears --model linear --preset baseline --grid-search
source experiment.sh --model-type linears --model linear_adaptive --all-presets --grid-search
```

Exhaustively runs every combination in the search space across all datasets.

The search space is defined in each model's `config.py` as variables prefixed with `SEARCH_SPACE_`:

```python
# models/linears/linear/config.py
SEARCH_SPACE_LEARNING_RATE = [1e-4, 1e-3, 1e-2]
SEARCH_SPACE_HIDDEN_DIM = [64, 128, 256]
SEARCH_SPACE_STACK_ACTIVATION = [RELU, SILU, GELU, MISH]
...
```

Restrict a sweep to specific axes with `--search-keys`, or supply values directly on the command line with `--search-set`:

```bash
source experiment.sh --model-type linears --model linear --preset baseline --grid-search --search-keys HIDDEN_DIM STACK_NUM_LAYERS
source experiment.sh --model-type linears --model linear --preset baseline --grid-search --search-set hidden_dim=64,128
```

### Run random search

```bash
source experiment.sh --model-type linears --model linear --preset baseline --random-search 10
source experiment.sh --model-type linears --model linear_adaptive --all-presets --random-search 10
```

Randomly samples 10 configurations from the search space. `--grid-search` and `--random-search` are mutually exclusive and can be combined with `--preset`, `--presets`, or `--all-presets`.

### Override config values

Use `--config` to override values from the model's `config.py` without editing the file (for example epochs or callbacks):

```bash
source experiment.sh --model-type linears --model linear --preset baseline --config --num-epochs 30 --callback-early-stopping-patience 0
```

Run `source experiment.sh --model-type linears --model linear --list-config` to see every overridable flag and its default.

### Inspect a model

Print the model structure instead of training it (requires `--preset`):

```bash
source experiment.sh --model-type linears --model linear --preset baseline --print-model
```

For browser-based preset inspection and graph visualization, see [`viewer/README.md`](viewer/README.md).

### Custom log folder

```bash
source experiment.sh --model-type linears --model linear --preset baseline --logdir comparison_run
source experiment.sh --model-type linears --model linear --preset gating --logdir comparison_run
```

Use the same `--logdir` across different models or presets to group their TensorBoard logs together for comparison.

### Viewing results

Training metrics are logged to `logs/`. Launch TensorBoard to visualize them:

```bash
tensorboard --logdir logs/
```

### Search results

When running `--grid-search` or `--random-search`, two additional files are written:

- **`result.json`** — saved inside each run's log directory with the exact parameters and final metrics for that run
- **`best_results.json`** — saved at `logs/{category}/{model}/best_results.json`, updated after every run that makes the top 5

`best_results.json` tracks the top 5 configurations per dataset ranked by `val_accuracy`:

```json
{
  "Mnist": [
    {"rank": 1, "option": "CONFIG", "params": {"learning_rate": "0.001", ...}, "metrics": {"val_accuracy": 0.984, "val_loss": 0.051, ...}},
    {"rank": 2, ...},
    ...
  ],
  "Cifar10": [...]
}
```

The file is updated in place as the search runs — if the process is interrupted, the top 5 found so far is preserved.

### License

[CC BY-NC 4.0](LICENSE) — free to use and modify for non-commercial purposes with attribution.
