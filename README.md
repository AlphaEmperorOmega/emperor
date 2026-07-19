# Emperor

**Current package version:** `0.1.0`

Emperor is a personal PyTorch research framework for building experimental neural
network models from reusable, inspectable components. The repository combines:

- `src/emperor/` - core neural modules, controller primitives, datasets, training
  helpers, and monitor callbacks.
- `src/models/` - reference experiment packages that compose the core modules into
  runnable architectures.
- `apps/workbench/` - a local browser-based Model Visualizer for inspecting presets,
  editing config overrides, planning training runs, launching local jobs, and
  reviewing live or historical monitor data.

The main goal is repeatable model experimentation: build small models, compare
variants, watch what happens during training, and use the results to decide
which ideas are worth exploring next.

## Environment Setup

**System requirements:** Git and [mise](https://mise.jdx.dev/). Mise provisions
the pinned Python 3.13 and Node 24 toolchains; Visual Studio, Xcode, and a
system Python or Node installation are not setup prerequisites.

The guaranteed CPU baseline covers Ubuntu 24.04 x64, WSL2 Ubuntu 24.04 x64,
Windows 11 x64 with PowerShell, and macOS 15+ Apple Silicon. Intel macOS is not
part of the baseline because PyTorch 2.12 has no Intel macOS wheel for the
required Python 3.13 runtime. MPS execution and CUDA on platforms other than
Linux x86-64 remain outside the guaranteed baseline.

Set up the repository and start the Workbench with the same commands everywhere:

```text
mise run setup --profile cpu
mise run dev
```

Setup creates `./torchenv` with the native `bin` or `Scripts` layout, installs
binary Python dependencies from the matching platform lock, runs `npm ci`, and
records a hash of every setup input. Repeating setup is idempotent. Switching
profiles recreates the launcher-owned virtualenv. CUDA profiles are available
only on Linux x86-64. The current profile uses PyTorch 2.12.0's CUDA 13.0
wheels:

```text
mise run setup --profile cuda
mise run dev --profile cuda
```

The `cuda-legacy` profile uses the official CUDA 12.6 wheels and guarantees the
GTX 1080 Ti (`sm_61`) target:

```text
mise run setup --profile cuda-legacy
mise run dev --profile cuda-legacy
```

Always pass the selected non-CPU profile to `dev`; its default remains `cpu`.
Other GPUs supported by the CUDA 12.6 wheel are best-effort. The wheels supply
the CUDA runtime, so a system CUDA toolkit is not required. The host NVIDIA
driver remains external: CUDA 12.x minor compatibility requires Linux driver
[525.60.13 or newer](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
while driver
[560.35.05 or newer](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-toolkit-release-notes/index.html)
avoids minor-compatibility feature limits for the pinned CUDA 12.6 Update 3
runtime. The CUDA 13 profile requires an R580-or-newer driver.

The native locks live under
`constraints/python-3.13-<platform>-<arch>-<profile>.txt`. The legacy lock is
standalone; it does not inherit any CUDA 13 pins.

### GTX 1080 Ti hardware gate

Setup verifies the Torch version, CUDA runtime, and compatible compiled Pascal
code without requiring an attached GPU. PyTorch 2.12's CUDA 12.6 Linux wheel
reports an `sm_60` cubin; NVIDIA's same-major forward binary compatibility
[makes that cubin valid](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html#binary-compatibility)
on the GTX 1080 Ti's `sm_61` capability. Run this manual gate on the target
machine before treating the hardware path as validated.

First confirm that the driver sees the card:

```bash
nvidia-smi
```

Then verify the exact build and physical device capability:

```bash
mise exec -- python tools/emperor_dev.py python -- -c \
  "import torch; assert torch.__version__ == '2.12.0+cu126'; assert torch.version.cuda == '12.6'; assert torch.cuda.get_device_capability(0) == (6, 1)"
```

Allocate and compute a CUDA tensor:

```bash
mise exec -- python tools/emperor_dev.py python -- -c \
  "import torch; x = torch.arange(1024, device='cuda'); print((x * x).sum().item())"
```

Finally run one explicit-GPU training epoch:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --config \
  --num-epochs 1 \
  --trainer-accelerator gpu \
  --trainer-devices 1
```

Emperor does not automatically select a GPU or change training precision for
this profile.

The Workbench is available at:

```text
http://localhost:9000
```

The local Workbench API defaults to:

```text
http://127.0.0.1:9999
```

Start, inspect, or stop validated Workbench process trees:

```text
mise run workbench:start
mise run workbench:status
mise run workbench:stop
```

Unix compatibility wrappers remain available as `source env.sh`,
`source experiment.sh`, `bash run_test.sh`, and `bash download_logs.sh`. Start
the Workbench with the CUDA 12.6 legacy profile using
`source env.sh --legacy-profile`.
PowerShell users can run `. .\env.ps1`, `. .\env.ps1 -WorkbenchStatus`, or
`. .\env.ps1 -WorkbenchStop` to activate the virtualenv and manage Workbench.

Default ports can be overridden with:

```bash
export WORKBENCH_BACKEND_PORT=9999
export WORKBENCH_FRONTEND_PORT=9000
export NEXT_PUBLIC_WORKBENCH_API_URL=http://127.0.0.1:9999
```

## Current State

Use this command to validate the full local setup:

```bash
mise run experiment -- \
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

After `mise run dev`, use this path to inspect the stable model and run one
short training job.

```bash
# 1. Confirm the model categories.
mise run experiment -- --list-model-types

# 2. Confirm the available linear models.
mise run experiment -- \
  --model-type linears \
  --list-models

# 3. Confirm the available linear presets.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --list-presets

# 4. Inspect the baseline model without training.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --print-model

# 5. Check optional monitor callbacks for training runs.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --list-monitors

# 6. Run a one-epoch smoke training job.
mise run experiment -- \
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

To train the decoder-only GPT baseline from scratch for one WikiText-2 epoch:

```bash
mise run experiment -- \
  --model-type gpt \
  --model linear \
  --preset baseline \
  --datasets wiki-text2 \
  --logdir gpt-wikitext2 \
  --config --num-epochs 1
```

The GPT packages use causal next-token loss and return token IDs from greedy
generation. `penn-treebank` is available as the alternative dataset name.

Each training run writes a Lightning/TensorBoard run directory under:

```text
logs/quickstart/linears/linear/BASELINE/Mnist/<parameter-id>_<timestamp>/version_*/
```

That `version_*` folder contains TensorBoard event files such as
`events.out.tfevents.*`, Lightning metadata such as `hparams.yaml`, and
Emperor's `result.json` with the final parameters and metrics. Checkpoint files
appear there only when checkpointing is enabled.

### Continuing from a checkpoint

Continue a single Run with `--resume-checkpoint`:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --resume-checkpoint logs/quickstart/linears/linear/BASELINE/Mnist/<run>/version_0/checkpoints/last.ckpt \
  --config --num-epochs 30
```

`NUM_EPOCHS` is the total target, not the number of additional epochs. A
checkpoint saved at epoch 25 has completed 26 epochs, so the new target must be
greater than 26. Continuation requires exactly one `--preset`, exactly one
explicitly selected dataset, and no `--presets`, `--all-presets`, grid search,
or random search.

Only continue from a trusted local Lightning checkpoint. Use the same Model
Package, preset, dataset, compatible Runtime Defaults, and code-compatible
model structure that created it. Emperor validates the complete checkpoint and
exact model state keys and tensor shapes; Lightning then restores optimizer,
scheduler, precision, loop, and compatible callback state.

When checkpointing is requested, `last.ckpt` records the most recently
completed epoch while the other retained checkpoint is the best monitored
epoch; these can be different. Continuation writes a new Run Artifact and
records only the source filename, epoch, and global step as lineage in progress
events and `result.json`. It never appends to or modifies the source Run
Artifact or checkpoint.

Important scaling rules:

- Without `--datasets`, a command trains once for every dataset listed by the
  model's `dataset_options.py`.
- Without `--config --num-epochs 1`, a command uses the model's configured epoch
  count.
- `--config` marks the following flags as model-config overrides.
- Runs multiply by selected presets, selected datasets, and search samples.

For example:

```bash
mise run experiment -- \
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
mise run experiment -- \
  --model-type linears \
  --model linear \
  --list-config
```

Common `config` override examples:

```bash
# Smaller/faster hidden stack.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --config --hidden-dim 64 --stack-num-layers 2 --stack-dropout-probability 0.0

# Training hyperparameters.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --config --batch-size 64 --learning-rate 0.0003 --trainer-gradient-clip-val 0.5
```

## Running Experiments

Use `experiment.sh` from the repository root:

```bash
mise run experiment -- \
  --model-type <type> \
  --model <name> \
  [options]
```

Run with no arguments to print the full flag list:

```bash
mise run experiment --
```

List available model types:

```bash
mise run experiment -- --list-model-types
```

List models within a type:

```bash
mise run experiment -- \
  --model-type linears \
  --list-models
```

`--list-models` without `--model-type` still prints the full catalog.

List a model's presets:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --list-presets
```

List a model's monitor callbacks:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --list-monitors
```

Run one preset:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline
```

Run selected presets sequentially:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --presets baseline gating memory \
  --datasets mnist \
  --config --num-epochs 1
```

Run every preset for a model:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --all-presets
```

Print a model structure instead of training:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --print-model
```

Monitor callbacks apply to training runs, not `--print-model`.

Run one preset with selected monitors:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --monitors linear halting
```

Use a custom log folder:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --logdir comparison_run
```

Use the same `--logdir` across runs to group TensorBoard logs for comparison.

### Datasets

Each Model Package groups its Dataset Metadata by Experiment Task in
`dataset_options.py`. For `linears/linear`, the image-classification Dataset
Metadata is:

```python
DATASET_OPTIONS_BY_TASK = {
    ExperimentTask.IMAGE_CLASSIFICATION: [
        Mnist,
        FashionMNIST,
        Cifar10,
        Cifar100,
    ],
}
```

BERT-pretraining Dataset Metadata uses the task-specific
`PennTreebankBertPretraining` and `WikiText2BertPretraining` adapters. Their raw
text source Interface accepts both TorchText's legacy `.splits(...)` datasets
and its later `root=..., split=...` factories, while always yielding one text
unit per source line for next-sentence construction. The verified constraints
currently select TorchText 0.6.0. Contract tests use deterministic offline
source fixtures and do not download corpora. They discover every declared
Dataset Metadata class and verify its task compatibility, one-batch shape,
dtype, collation semantics, and seed ownership through the public DataModule
Interface.

Real downloads remain explicit integration work performed by `prepare_data()`.
Run that network-dependent suite only when the required sources are available:

```bash
EMPEROR_RUN_DATASET_DOWNLOAD_TESTS=1 \
  PYTHONSAFEPATH=1 PYTHONPATH=tests \
  python -P -m unittest integration.datasets.test_dataset_downloads
```

List available datasets for a model:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --list-datasets
```

Restrict a run to one or more datasets:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist fashion-mnist
```

### Config Overrides

Use `--config` to override model config values without editing `config.py`:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --config --num-epochs 30 --callback-early-stopping-patience 0
```

More examples:

```bash
# Change optimization and batch size.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --config --batch-size 64 --learning-rate 0.0003

# Make the hidden stack smaller for fast iteration.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --config --hidden-dim 64 --stack-num-layers 2

# Try a different activation and dropout value.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --datasets mnist \
  --config --stack-activation RELU --stack-dropout-probability 0.1

# Override controller settings when the selected preset enables that controller.
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset recurrent-halting \
  --datasets mnist \
  --config --recurrent-max-steps 6 --recurrent-halting-threshold 0.95
```

List overridable fields:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --list-config
```

### Search

Run a grid search:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --grid-search
```

Run a random search:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --random-search 10
```

Restrict a sweep to selected axes:

```bash
mise run experiment -- \
  --model-type linears \
  --model linear \
  --preset baseline \
  --grid-search \
  --search-keys HIDDEN_DIM STACK_NUM_LAYERS
```

Supply command-line search values:

```bash
mise run experiment -- \
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

See [`apps/workbench/README.md`](apps/workbench/README.md) for the focused
Workbench guide.

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

Run the archive task from the project directory to archive the contents of
`./logs` directly from the filesystem:

```text
mise run logs:archive
mise run logs:archive -- my_experiment
mise run logs:archive -- logs/my_experiment
```

When a specific experiment folder is selected, the archive keeps that folder
prefix so importing it restores files under the same `logs/<experiment>/` path.

By default, the archive is written in the current directory. Pass an explicit
second argument to choose the zip path:

```text
mise run logs:archive -- logs emperor_logs.zip
mise run logs:archive -- logs path/to/emperor_logs.zip
```

To restore that archive into another local project, start the Workbench backend,
open the Workbench, choose **Import Logs** in the top navigation, and select the
produced `.zip` file. Backend defaults keep log imports disabled in every
authentication mode; enable them explicitly with
`WORKBENCH_API_ALLOW_LOG_IMPORTS=true`. The normal `mise run dev` launcher sets
that opt-in for its loopback development backend. The import extracts into that
project's server-side `logs/` directory and overwrites files that already exist
at the same archive paths. Enabled imports default to a 512 MiB compressed limit
and a 2 GiB extracted limit, at most 32,000 archive members, a 4 MiB cumulative
UTF-8 member-path budget, and one active archive import at a time. Override them
with
`WORKBENCH_API_MAX_UPLOAD_SIZE=<bytes>` or
`WORKBENCH_API_MAX_LOG_ARCHIVE_EXTRACTED_SIZE=<bytes>`,
`WORKBENCH_API_MAX_LOG_ARCHIVE_MEMBER_COUNT=<count>`,
`WORKBENCH_API_MAX_LOG_ARCHIVE_PATH_BYTES=<bytes>`, or
`WORKBENCH_API_LOG_ARCHIVE_UPLOAD_CONCURRENCY=<count>`. Size limits always have
finite defaults and cannot be disabled with null configuration values.

## Test and Quality Commands

Use the canonical task from the repository root for the normal unit-test workflow.
It discovers the external `tests/` tree with fail-fast enabled. The script adds
only that test tree to `PYTHONPATH`; Emperor and its first-party Model Packages
must come from the active environment's editable or regular installation. Run
`python -m pip install --no-deps -e .` first when using an environment that was
not created by `env.sh`.

```text
mise run test
```

Target a specific unit-test module, class, or test method when iterating on a
failure:

```text
mise run test -- layer
mise run test -- layer TestLayer
mise run test -- layer TestLayer test_forward_output_shape
```

The task maps the first argument to `tests/unit/test_<name>.py` and then passes
any class or method name through to `python -P -m unittest -f`. The historical
`bash run_test.sh` entry point delegates to the same implementation.

Verify the wheel/source-distribution manifests and compare clean editable and
regular installs from outside the checkout with:

```bash
python -P tools/verify_distribution.py
```

The verification uses the dependencies already available in the active
environment and never downloads or upgrades them.

Run strict Python type checking with:

```bash
pyright --project pyrightconfig.json
```

The strict Pyright include list covers stable capability areas and may only
grow. CI supplies the pinned Pyright executable; local Pyright use requires the
command to already be available.

Capture the informational PyTorch hot-path baseline with the platform-aware
project Python:

```text
mise exec -- python tools/emperor_dev.py python -- \
  -P tools/benchmark_pytorch_hotspots.py \
  --device all --warmup 5 --repetitions 30 --threads 1
```

The harness measures DataLoader behavior, placement and scalar synchronization
candidates, routing and halting presets, and MEMORY test-time-training inner
loops. It records timing variance, memory, environment, and exact model/input
identity. CUDA is synchronized for every sample when the installed PyTorch
build can execute on the available GPU; otherwise the hardware compatibility
reason is recorded as a skip. See
[`docs/architecture/pytorch-performance-baseline.md`](docs/architecture/pytorch-performance-baseline.md)
for the canonical conditions and interpretation. Runtime results remain
informational rather than hard CI thresholds.

Capture the production Workbench browser and long-session baseline after a
frontend build with:

```bash
cd apps/workbench/web
npm run build
npm run performance:budget
npm run performance:browser
```

The browser harness uses temporary backend data and a fake process only at the
Training Job runner boundary. It measures hydration, main-thread and React
work, repeated workspace/graph/chart cycles, API timings, heap growth, completed
Training and log-import flows, WebGL frames and disposal, and the existing
bundle budgets. See
[`docs/architecture/browser-performance-baseline.md`](docs/architecture/browser-performance-baseline.md)
for the canonical conditions and interpretation. Machine-sensitive results are
informational; stable workflow, disposal, error, and bundle checks fail the
harness.

For Workbench-specific changes, run the relevant backend or frontend checks from
the package being changed after the unit suite is green.

The live backend/frontend contract E2E uses two loopback FastAPI servers,
temporary filesystem roots, and fake Training Jobs. It requires the verified
Python environment and existing frontend dependencies, but downloads nothing:

```bash
cd apps/workbench/web
npm run test:contract:e2e
```

## License

[CC BY-NC 4.0](LICENSE) - free to use and modify for non-commercial purposes
with attribution.
