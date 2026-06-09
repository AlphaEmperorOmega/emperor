# Emperor Model Viewer

The viewer is a local developer tool for inspecting Emperor model presets in a browser. It builds a preset config, instantiates the model, serializes the `torch.nn.Module` tree into a graph, and can launch local subprocess-backed training jobs with optional monitor charts. It does not edit model config files.

The selected target preset remains the source of truth for preview, config schema, overrides, and search-space controls. In the training dock, `Run presets` can select one or more presets for a single sequential training job; the primary target preset is always included, and the backend validates overrides and search axes against every selected preset before starting.

## Architecture

- `viewer/backend` owns model discovery, config schema extraction, override parsing, graph serialization, subprocess training jobs, monitor data extraction, the FastAPI app, and the inspection CLI.
- `viewer/frontend` owns the browser UI, React Flow graph layout, config controls, training controls, monitor charts, and API error states.
- The viewer may import `emperor/` and `models/`.
- `emperor/` and `models/` must not import `viewer/`.

See [`FEATURE_ANALYSIS.md`](FEATURE_ANALYSIS.md) for the implemented feature matrix, endpoint contracts, workflow map, coverage map, and known limitations.

## Setup

```bash
source env.sh
```

Setup installs the Python and Node versions from `mise.toml`, installs the editable Python package, installs frontend dependencies, and starts both viewer servers in the background. It writes backend and frontend logs plus PID files under `viewer/.runtime/`. By default, the backend uses port `9999` and the frontend uses port `9000`.

Running `source env.sh` again is safe: it reuses live PID files and also checks whether ports `9999` or `9000` are already listening before starting another server.

Stop or inspect the viewer:

```bash
source env.sh --viewer-stop
source env.sh --viewer-status
```

## Run

Backend:

```bash
python -m uvicorn viewer.backend.api:app --reload --host 127.0.0.1 --port 9999
# or, from the repository root without activating the venv:
torchenv/bin/python -m uvicorn viewer.backend.api:app --reload --host 127.0.0.1 --port 9999
```

Frontend:

```bash
cd viewer/frontend
npm run dev
```

Open:

```text
http://localhost:9000
```

## Hosted Backend Settings

Hosted deployments should configure explicit frontend origins when the backend
is reachable outside loopback. `VIEWER_API_CORS_ORIGINS` uses JSON array
syntax:

```bash
export VIEWER_API_CORS_ORIGINS='["https://viewer.example.com","https://admin.example.com"]'
```

When bearer auth is enabled, keep origins specific to the deployed frontend
hosts. Do not use wildcard origins such as `["*"]` for authenticated hosted
deployments.

## Test

Backend:

```bash
python -m unittest discover -s viewer/backend/tests
```

Frontend:

```bash
cd viewer/frontend
npm test
npm run lint
npm run build
npm run typecheck
```

CLI inspection is still available through the root experiment script:

```bash
source experiment.sh linears/linear --preset baseline --print-model
python -m viewer.backend.cli --model linears/linear --preset baseline --format json
```

Training from the CLI can also run selected preset batches:

```bash
source experiment.sh linears/linear --presets baseline gating --grid-search
```

## Troubleshooting

- Backend unavailable: start `python -m uvicorn` on `127.0.0.1:9999`, or set `NEXT_PUBLIC_VIEWER_API_URL` before starting Next.js.
- Model import failed: check stale imports in that model package's `config.py`, `presets.py`, or `model.py`.
- Invalid override value: use enum names such as `GELU`, booleans such as `true`, and class names that are imported by the model config module.
- Empty graph: verify the selected preset can instantiate `Model(cfg)` without training.
