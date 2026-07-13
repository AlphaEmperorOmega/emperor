# Emperor

**Current package version:** `0.1.0`

Emperor is a personal PyTorch research framework for building experimental neural
network models from reusable, inspectable components. The repository combines:

- `emperor/` - core neural modules, controller primitives, datasets, training
  helpers, and monitor callbacks.
- `models/` - reference experiment packages that compose the core modules into
  runnable architectures.
- `workbench/` - a local browser-based Model Visualizer for inspecting presets,
  editing config overrides, planning training runs, launching local jobs, and
  reviewing live or historical monitor data.

The main goal is repeatable model experimentation: build small models, compare
variants, watch what happens during training, and use the results to decide
which ideas are worth exploring next.

## Environment Setup

**Requirements:** the recommended setup uses `mise.toml`, which pins Python
3.13 and Node 24. You do not need to pre-install Python 3.13 yourself when using
`env.sh`; the script installs `mise` if needed and lets `mise` provision the
pinned Python and Node versions. `pyproject.toml` accepts manual Python installs
from 3.11 through 3.13 (`>=3.11,<3.14`), but the bootstrap/dev environment is
pinned to 3.13.

**Windows:** run the setup scripts from WSL.

Set up the repository:

```bash
source env.sh
```

On first run, `env.sh` installs `mise` if needed, provisions the pinned Python
and Node versions, creates `./torchenv`, installs the project from
`pyproject.toml`, installs Workbench frontend dependencies, activates the
virtualenv, and starts the Workbench backend and frontend in the background.

On later runs, it reuses the virtualenv and already-running Workbench servers when
possible. Runtime logs and PID files are written under `workbench/.runtime/`.

The Linux x86-64 Python 3.13 development/runtime environment is resolved through
`constraints/python-3.13-linux-x86_64.txt`. `env.sh` pins pip and supplies that
constraints file when installing `.[dev]`; changes to either `pyproject.toml` or
the constraints file invalidate its dependency marker. The equivalent manual
install is:

```bash
python -m pip install --upgrade pip==26.1.2
python -m pip install \
  --constraint constraints/python-3.13-linux-x86_64.txt \
  --build-constraint constraints/python-3.13-linux-x86_64.txt \
  -e ".[dev]"
```

The constraints snapshot records the complete environment used for the tested
migration. Regenerate it only after intentionally changing dependencies and
verifying the resulting clean resolution and full test suite.

After `source env.sh`, the Workbench is available at:

```text
http://localhost:9000
```

The local Workbench API defaults to:

```text
http://127.0.0.1:9999
```

Stop or inspect Workbench servers:

```bash
source env.sh --workbench-stop
source env.sh --workbench-status
```

Default ports can be overridden with:

```bash
export WORKBENCH_BACKEND_PORT=9999
export WORKBENCH_FRONTEND_PORT=9000
export NEXT_PUBLIC_WORKBENCH_API_URL=http://127.0.0.1:9999
```

## Current State

Use this command to validate the full local setup:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset baseline
```

`linears/linear` is the recommended smoke-test model. It currently exposes
baseline, gating, halting, memory, residual, post-norm, and recurrent preset
families. The other cataloged model packages are research/reference
implementations for adaptive parameters, experts, neuron clusters, and
transformer experiments. Before using one for a long run, verify the exact
preset with `--print-model` and a small one-epoch run.

The Workbench has grown into a local experiment workbench with three workspaces:

- **Model** - choose a model/preset/dataset, inspect module and operation
  graphs, review parameters, create config snapshots, and inspect historical
  training-run targets.
- **Training** - plan, start, and monitor local training jobs for the selected
  model configuration.
- **Logs** - browse historical TensorBoard runs, inspect saved graphs, and
  review metrics and monitor data from completed runs.

## Quick Start

After `source env.sh`, use this path to inspect the stable model and run one
short training job.

```bash
# 1. Confirm the model categories.
source experiment.sh --list-model-types

# 2. Confirm the available linear models.
source experiment.sh \
  --model-type linears \
  --list-models

# 3. Confirm the available linear presets.
source experiment.sh \
  --model-type linears \
  --model linear \
  --list-presets

# 4. Inspect the baseline model without training.
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset baseline \
  --print-model

# 5. Check optional monitor callbacks for training runs.
source experiment.sh \
  --model-type linears \
  --model linear \
  --list-monitors

# 6. Run a one-epoch smoke training job.
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --logdir quickstart \
  --config --num-epochs 1

# 7. Open TensorBoard logs if you want the raw event view.
tensorboard --logdir logs/quickstart
```

The smoke run builds the `baseline` preset, trains it on `mnist` for one epoch,
and may download the dataset on first use.

Each training run writes a Lightning/TensorBoard run directory under:

```text
logs/quickstart/linears/linear/BASELINE/Mnist/<parameter-id>_<timestamp>/version_*/
```

That `version_*` folder contains TensorBoard event files such as
`events.out.tfevents.*`, Lightning metadata such as `hparams.yaml`, and
Emperor's `result.json` with the final parameters and metrics. Checkpoint files
appear there only when checkpointing is enabled.

Important scaling rules:

- Without `--datasets`, a command trains once for every dataset listed by the
  model's `dataset_options.py`.
- Without `--config --num-epochs 1`, a command uses the model's configured epoch
  count.
- `--config` marks the following flags as model-config overrides.
- Runs multiply by selected presets, selected datasets, and search samples.

For example:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --presets baseline gating \
  --datasets mnist cifar10 \
  --random-search 10 \
  --logdir comparison \
  --config --num-epochs 1
```

That creates `2 x 2 x 10` planned training runs.

Before choosing overrides, print every overridable field and its default value
for the selected model:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --list-config
```

Common `config` override examples:

```bash
# Smaller/faster hidden stack.
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --config --hidden-dim 64 --stack-num-layers 2 --stack-dropout-probability 0.0

# Training hyperparameters.
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --config --batch-size 64 --learning-rate 0.0003 --trainer-gradient-clip-val 0.5
```

## Running Experiments

Use `experiment.sh` from the repository root:

```bash
source experiment.sh \
  --model-type <type> \
  --model <name> \
  [options]
```

Run with no arguments to print the full flag list:

```bash
source experiment.sh
```

List available model types:

```bash
source experiment.sh --list-model-types
```

List models within a type:

```bash
source experiment.sh \
  --model-type linears \
  --list-models
```

`--list-models` without `--model-type` still prints the full catalog.

List a model's presets:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --list-presets
```

List a model's monitor callbacks:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --list-monitors
```

Run one preset:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset baseline
```

Run selected presets sequentially:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --presets baseline gating memory \
  --datasets mnist \
  --config --num-epochs 1
```

Run every preset for a model:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --all-presets
```

Print a model structure instead of training:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset baseline \
  --print-model
```

Monitor callbacks apply to training runs, not `--print-model`.

Run one preset with selected monitors:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --monitors linear halting
```

Use a custom log folder:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset baseline \
  --logdir comparison_run
```

Use the same `--logdir` across runs to group TensorBoard logs for comparison.

### Datasets

Each model package defines a `DATASET_OPTIONS` list in `dataset_options.py`.
For `linears/linear`, the default image-classification datasets are:

```python
DATASET_OPTIONS: list = [Mnist, FashionMNIST, Cifar10, Cifar100]
```

List available datasets for a model:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --list-datasets
```

Restrict a run to one or more datasets:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist fashion-mnist
```

### Config Overrides

Use `--config` to override model config values without editing `config.py`:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset baseline \
  --config --num-epochs 30 --callback-early-stopping-patience 0
```

More examples:

```bash
# Change optimization and batch size.
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --config --batch-size 64 --learning-rate 0.0003

# Make the hidden stack smaller for fast iteration.
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --config --hidden-dim 64 --stack-num-layers 2

# Try a different activation and dropout value.
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --config --stack-activation RELU --stack-dropout-probability 0.1

# Override controller settings when the selected preset enables that controller.
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset recurrent-halting \
  --datasets mnist \
  --config --recurrent-max-steps 6 --recurrent-halting-threshold 0.95
```

List overridable fields:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --list-config
```

### Search

Run a grid search:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset baseline \
  --grid-search
```

Run a random search:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset baseline \
  --random-search 10
```

Restrict a sweep to selected axes:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset baseline \
  --grid-search \
  --search-keys HIDDEN_DIM STACK_NUM_LAYERS
```

Supply command-line search values:

```bash
source experiment.sh \
  --model-type linears \
  --model linear \
  --preset baseline \
  --grid-search \
  --search-set hidden_dim=64,128
```

`--grid-search` and `--random-search` are mutually exclusive and can be combined
with `--preset`, `--presets`, or `--all-presets`.

## Workbench

The Workbench is the local workbench for model experiments. Use it when you want to
pick a model, preset, and dataset visually; inspect what a preset builds before
training; adjust config overrides without assembling a long terminal command;
start or monitor local training jobs; and review historical runs after the
experiment finishes.

It is meant for the human loop around training: understanding the model shape,
checking planned runs, watching monitor signals while a job is active, and
reviewing logs once there are completed results. The terminal CLI remains the
simpler path for scripted or repeatable runs.

The Workbench should be treated as a vibecoded prototype: it is useful for local
research workflows today, but its implementation is expected to be refactored
as the experiment workflow and architecture settle.

See [`workbench/README.md`](workbench/README.md) for the focused Workbench guide.

Hosted or non-local deployments should configure explicit CORS origins in the
Workbench backend environment:

```bash
export WORKBENCH_API_CORS_ORIGINS='["https://workbench.example.com"]'
```

Hosted frontend builds should also set `NEXT_PUBLIC_WORKBENCH_API_URL` and, when
multiple API origins are allowed, `NEXT_PUBLIC_WORKBENCH_API_ALLOWED_ORIGINS` so
browser requests and bearer tokens cannot be redirected to arbitrary origins.

Bearer auth can be enabled with:

```bash
export WORKBENCH_API_AUTH_MODE=bearer
export WORKBENCH_API_TOKEN=<token>
```

Example hosted pairing:

```bash
# Frontend build env
export NEXT_PUBLIC_WORKBENCH_API_URL=https://api.example.com
export NEXT_PUBLIC_WORKBENCH_API_ALLOWED_ORIGINS='["https://api.example.com"]'

# Backend runtime env
export WORKBENCH_API_CORS_ORIGINS='["https://workbench.example.com"]'
export WORKBENCH_API_AUTH_MODE=bearer
export WORKBENCH_API_TOKEN='<replace-with-a-secret-token>'
```

The Workbench backend is local-file-backed. Its in-process locks protect shared
caches inside one backend worker; they are not cross-worker invalidation. Prefer
one backend worker for hosted Workbench deployments unless you add a shared
cache/invalidation design.

## Results and Logs

Training metrics are logged under `logs/`. Launch TensorBoard with:

```bash
tensorboard --logdir logs/
```

Run the download script from the project directory to archive the contents of
`./logs` directly from the filesystem:

```bash
bash download_logs.sh
bash download_logs.sh my_experiment
bash download_logs.sh logs/my_experiment
```

When a specific experiment folder is selected, the archive keeps that folder
prefix so importing it restores files under the same `logs/<experiment>/` path.

By default, the archive is written in the current directory. Pass an explicit
second argument to choose the zip path:

```bash
bash download_logs.sh logs emperor_logs.zip
bash download_logs.sh logs /tmp/emperor_logs.zip
```

To restore that archive into another local project, start the Workbench backend,
open the Workbench, choose **Import Logs** in the top navigation, and select the
produced `.zip` file. Local unauthenticated backends allow log imports by
default; hosted or read-only bearer-mode backends require
`WORKBENCH_API_ALLOW_LOG_IMPORTS=true`. The import extracts into that project's
server-side `logs/` directory and overwrites files that already exist at the
same archive paths. Compressed archive uploads and extracted archive contents
are uncapped by default. Set `WORKBENCH_API_MAX_UPLOAD_SIZE=<bytes>` or
`WORKBENCH_API_MAX_LOG_ARCHIVE_EXTRACTED_SIZE=<bytes>` only when a deployment
needs to reject large imports.

## Test and Quality Commands

Use `run_test.sh` from the repository root for the normal unit-test workflow.
It runs the docs unittest suite with fail-fast enabled, which is the quickest
way to see whether the core unit tests pass while fixing failures.

```bash
bash run_test.sh
```

Target a specific docs test module, class, or test method when iterating on a
failure:

```bash
bash run_test.sh layer
bash run_test.sh layer TestLayer
bash run_test.sh layer TestLayer test_forward_shape
```

The script maps the first argument to `docs/test_<name>.py` and then passes any
class or method name through to `python3 -m unittest -f`.

For Workbench-specific changes, run the relevant backend or frontend checks from
the package being changed after the unit suite is green.

## License

[CC BY-NC 4.0](LICENSE) - free to use and modify for non-commercial purposes
with attribution.
