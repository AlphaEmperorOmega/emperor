# Viewer Implemented Feature Analysis

Date: 2026-06-02

This document describes the currently implemented viewer behavior in the dirty
workspace. It uses `viewer/` as the source of truth and covers inspection,
configuration, graph visualization, subprocess training, and monitor charts.

## Source Map

- Backend API: `viewer/backend/api.py`
- Discovery and schema: `viewer/backend/inspector/discovery.py`,
  `viewer/backend/inspector/schema.py`, `viewer/backend/inspector/overrides.py`
- Inspection and graph serialization: `viewer/backend/inspector/service.py`,
  `viewer/backend/inspector/graph.py`
- Training and monitor data: `viewer/backend/training_jobs.py`,
  `viewer/backend/training_worker.py`, `viewer/backend/training_events.py`,
  `viewer/backend/monitor_data.py`
- Frontend API client and state: `viewer/frontend/src/lib/api.ts`,
  `viewer/frontend/src/components/viewer/use-viewer-state.ts`,
  `viewer/frontend/src/components/viewer/state/`
- Frontend graph, config, training, and monitor UI:
  `viewer/frontend/src/components/viewer/`,
  `viewer/frontend/src/lib/graph/`
- CLI: `viewer/backend/cli.py`

## Feature Matrix

| Feature | Owner | Entrypoint | User workflow | Data flow | Contract | Error behavior | Evidence | Known limitation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| API health and CORS | Backend | `GET /health`, `create_app` | Frontend shows online/offline state and local dev frontends can call the API. | FastAPI returns a static health payload and CORS uses `ViewerApiSettings`. | `{ status: string }`; local origins on ports `9000`, `9001`, and `3000` allowed by default. | Health query failure renders "Backend unavailable"; CORS can be overridden through `VIEWER_API_CORS_ORIGINS`. | Backend tests: `test_api_health_and_inspect`, CORS tests. Frontend query gating tests. | No deeper dependency health is checked. |
| Model discovery | Backend | `GET /models`, `discover_models()` | User chooses a target model from the selector. | Scans `models/` packages with `__init__.py`, `config.py`, `presets.py`, and `model.py`. | `{ models: string[] }`, sorted package names. | Invalid or unknown model later returns `InspectorError` with HTTP 400. | Backend `test_model_discovery_lists_expected_packages`; frontend `ViewerApp` selector tests. | Only filesystem package shape is checked, not importability until model-scoped calls. |
| Preset discovery | Backend | `GET /models/{model}/presets`, `list_model_presets()` | Preset selector is populated after model selection. | Imports model parts, reads `ExperimentOptions`, converts enum names to CLI names. | `{ model, presets: [{ name, label, description }] }`. | Unknown or invalid model returns HTTP 400 through the shared inspector handler. | Backend `test_preset_discovery_for_linear`; frontend selector and preset-change tests. | Preset descriptions depend on enum values and can be empty. |
| Dataset discovery | Backend | `GET /models/{model}/datasets`, `list_model_datasets()` | Dataset checklist appears; first dataset is selected automatically. | Reads `DATASET_OPTIONS` and serializes class name, display label, flattened input dim, and class count. | `{ model, datasets: [{ name, label, inputDim, outputDim }] }`. | Missing `DATASET_OPTIONS` or unknown model returns HTTP 400. | Backend `test_dataset_discovery_for_linear`; frontend preview and training-post tests. | Dimensions default to `0` if dataset metadata is missing. |
| Monitor discovery | Backend and frontend state | `GET /models/{model}/monitors`, `list_model_monitors()` | User optionally checks monitors before starting training. | Reads `MONITOR_OPTIONS`, validates each `MonitorOption`, and returns labels, kinds, and defaults. | `{ model, monitors: [{ name, label, description, kinds, defaultEnabled }] }`. | Invalid, duplicate, or unknown monitor names become `InspectorError`; UI shows training-panel mutation errors. | Backend monitor discovery and unknown monitor tests; frontend selected monitor post test. | `defaultEnabled` is exposed but the frontend currently does not auto-select default monitors. |
| Config schema extraction | Backend | `GET /models/{model}/config-schema?preset=...`, `config_schema()` | User opens config summary or full config dialog. | Supported config keys are discovered, source comments provide sections, annotations/defaults provide field types and choices, preset locks are merged. | `{ model, fields: ConfigField[] }`; fields include `key`, `configKey`, `flag`, `label`, `section`, `type`, `default`, `nullable`, `choices`, `searchChoices`, `locked`, `lockedValue`, `lockedReason`. | Unknown preset/model returns HTTP 400; frontend shows "Config schema failed". | Backend schema type, abstract-choice, boundary-projector, and locked-field tests; frontend config dialog tests. | Section names are inferred from source comments and can drift if comments are not heading-like. |
| Override parsing and locked override rejection | Backend and frontend | `parse_override_mapping()`, `reject_locked_overrides()`, `useLockedOverrideSync()` | User edits config values, then updates preview or starts training. | Frontend stores string overrides by UI key; backend normalizes keys, parses values using model config types, maps to builder params, rejects abstract classes and locked preset-owned fields. | Overrides are request bodies as `Record<string, string>` in frontend; backend accepts `dict[str, Any]` and converts to config-builder parameters. | Unknown key, invalid value, abstract class, or locked field returns HTTP 400; locked fields are disabled and pruned in the frontend. | Backend override, abstract class, and locked preset tests; frontend locked-field and override-reset tests. | Frontend does not pre-validate all numeric/class parse errors before API submission. |
| Model inspection | Backend | `POST /inspect`, `inspect_model()` | Initial preview loads and user refreshes after config changes. | Backend loads model parts, resolves preset and dataset, builds config, instantiates `Model(cfg)`, serializes the module tree. | Request `{ model, preset, overrides, dataset? }`; response `{ model, preset, parameterCount, nodes, edges }`. | Build or instantiation failures become HTTP 400; frontend clears stale graph during refresh and shows preview error. | Backend API inspect and one-preset-per-model tests; frontend initial preview, stale response, update preview, and error panel tests. | Inspection instantiates the model but does not run a forward pass, so runtime tensor shape errors may remain hidden. |
| Backend graph serialization | Backend | `serialize_graph()` | Graph canvas, side panels, and charts use the serialized module tree. | Walks `named_children()`, emits root plus stable child paths and edges, attaches roles, parameter counts, direct shapes, config fields, dimensions, gates, halting, recurrent metadata, experts, attention, cluster coordinates, and terminal reach. | `GraphNode` has `id`, `label`, `typeName`, `path`, `graphRole`, `parameterCount`, `details`, and optional `config`; `GraphEdge` has `id`, `source`, `target`. | Lazy parameters are skipped; unexpected missing metadata simply omits details. | Backend graph serializer tests cover IDs, roles, counts, shapes, configs, experts, clusters, terminal reach, lazy params, and uniqueness. | Edge topology is module containment, not execution dataflow. |
| CLI inspection | Backend CLI | `python -m viewer.backend.cli --model ... --preset ... --format text|json` | Developer inspects preset structure in a terminal or script. | Reuses experiment parser and `inspect_model`; prints a text tree or JSON response. | CLI supports `--format text|json` plus experiment parser flags; rejects search modes. | `InspectorError` exits with the error text. | README command and backend inspection coverage through shared service tests. | Text mode currently prints one result per option but does not expose monitor/training behavior. |
| Frontend API client | Frontend | `viewer/frontend/src/lib/api.ts` | All UI requests go through typed helpers. | `fetch` calls API base URL, parses JSON with Zod schemas, and extracts JSON `detail` errors. | Typed helpers for health, models, presets, datasets, monitors, schema, inspect, training jobs, cancel, and monitor data. | Non-OK responses throw `Error(detail)`; schema mismatch throws Zod validation errors. | Frontend `api.test.ts` covers success, errors, URL encoding, POST bodies, cancel, job fetch, and monitor data query construction. | API base URL is configured only at Next.js build/runtime env level. |
| Frontend target and query state | Frontend | `useViewerState()`, `useViewerQueries()` | App auto-selects first model, preset, and dataset, then previews the graph. | React Query fetches health/models first; model-scoped queries are enabled after model selection; schema waits for model and preset. | Hook state includes selected model, preset, datasets, monitors, overrides, graph settings, graph data, and query objects. | Query errors render local error panels; stale preview responses are ignored by request id. | Frontend state hook tests and `ViewerApp` initial-selection tests. | Changing preset clears overrides but intentionally does not auto-refresh preview until requested. |
| Config summary and full config dialog | Frontend | `ConfigSummaryPanel`, `FullConfigDialog` | User scans config sections, opens the dialog, edits fields, resets fields or all overrides, updates preview, or opens a generated training command. | Schema fields are grouped by section; controls use field type, choices, nullable state, and locked state; overrides stay local until preview/training. | UI controls: bool switch, select for enum/class/search choices, number input, text input, nullable empty value as `None` for command display. | Loading, empty, disabled locked fields, reset disabled when no override; API failures surface on preview refresh. | Frontend config summary, section navigation, field editing, reset, and training-command tests. | No server-side draft persistence; closing the app loses unsent overrides. |
| Graph canvas, modes, scopes, and layout | Frontend | `PreviewToolbar`, `PreviewPanel`, `useGraphViewState()`, `layoutGraph()` | User switches simple/basic/full detail, opened/entire scope, expands nodes, collapses all, and selects nodes. | Backend graph is filtered by role and expansion; dagre lays out nodes; React Flow renders node cards and edges. | Detail modes: `simple`, `basic`, `full`; scopes: `opened`, `entire`. | Empty, loading, and preview-error states render in `PreviewPanel`; selection does not trigger relayout. | Frontend graph filtering, layout, selection, mode/scope, and viewer-app graph tests. | Layout is structural left-to-right containment and can become large in full/entire mode. |
| Graph card analysis views | Frontend | `GraphNodeView`, `lib/graph/*-diagrams.ts` | User reads child summaries, mechanism rows, stack/expert/cluster diagrams, parameter shape chips, and config/details accordions on graph cards. | Frontend derives summaries and diagrams from backend `details`, `config`, nodes, and edges. | Derived types: `ChildSummary`, `ExpertDiagram`, `StackDiagram`, `ClusterDiagram`. | Invalid or missing metadata falls back to child summaries or empty sections. | Frontend graph and graph-node-view tests cover summaries, mechanisms, experts, stacks, clusters, shapes, and sizing. | Diagrams are visualization summaries, not interactive simulation of execution. |
| Structure rail | Frontend | `ModelStructureCard`, `StructureBranch` | User navigates the model tree and reveals hidden nodes in the graph. | Builds hierarchy from current graph, auto-opens ancestors of selected node, and calls `revealGraphNode()`. | Tree rows use graph node IDs and paths. | No graph and no-root states render inline. | Frontend model-structure and viewer-app reveal tests. | The rail reflects the currently filtered detail graph, so hidden full-mode internals are absent in basic mode. |
| Location and terminal reach panels | Frontend | `GraphLocationsCard`, `SelectedNodeDetailsView` | User inspects neuron cluster coordinates and terminal reach, then reveals the owning graph node. | Frontend reads backend `details.cluster` and `details.terminalReach`, builds location summaries and reach grids. | Derived types: `GraphLocationSummary`, `TerminalReachGrid`. | Missing or invalid coordinate data is ignored or rendered as empty state. | Frontend graph location, terminal reach, and selected node tests; backend cluster/reach serialization tests. | Large coordinate sets are truncated for display. |
| Selected node details and monitor entry | Frontend | `SelectedNodeDetails`, `SelectedNodeDetailsView` | User selects a node, reads metadata, and opens monitor charts after a training job exists. | Selected node comes from graph state; active training job is propagated from `TrainingPanel` to the details panel. | Monitor button is enabled only when an active training job object exists. | No node selected state; monitor button disabled without a job. | Frontend selected node metadata and monitor chart opening tests. | The button is job-gated, not monitor-data-gated; opening can still show an empty/no-monitor state. |
| Training job creation, polling, and cancellation | Backend and frontend | `POST /training/jobs`, `GET /training/jobs/{id}`, `POST /training/jobs/{id}/cancel`, `TrainingPanel` | User chooses datasets/monitors, starts training, watches status/metrics/logs, and can cancel a running job. | Backend validates model, preset, datasets, monitors, and overrides, writes `payload.json`, launches `viewer.backend.training_worker`, stores job in memory, tails logs and progress events. Frontend posts the request, polls the job every second until terminal, and displays queue, metrics, log tail, and request summary. | `TrainingJob` response has `id`, `status`, `model`, `preset`, `datasets`, `overrides`, `monitors`, timestamps, `exitCode`, `pid`, progress fields, `metrics`, `logDir`, `events`, `logTail`, and `resultLinks`. | Empty dataset list, unknown preset/monitor, locked override, unknown job, and cancellation failures surface as HTTP 400 or mutation errors. | Backend fake-runner training tests; frontend training post, selected monitors, panel state, and cluster growth tests. | Job IDs and process handles are in-memory in the API process; after API restart, temp files remain but existing jobs are not recoverable through `/training/jobs/{id}`. |
| Training worker and progress events | Backend worker | `python -m viewer.backend.training_worker --payload ... --progress ...` | A started training job runs in a subprocess and writes progress for the UI. | Worker loads payload, resolves datasets and monitors, builds monitor callbacks, parses overrides, rejects locks, instantiates the experiment, and calls `train_model()` with progress, growth, and monitor callbacks. | Progress is JSON Lines under `/tmp/emperor-viewer-training/{job_id}/progress.jsonl`; logs go to `training.log`. | Worker writes a failed event and exits with code 1 on exception; manager marks failed from events or process exit code. | Backend fake-runner tests cover payload and command; cluster growth lib tests cover event summarization. | Worker behavior depends on each experiment's `train_model()` and callbacks emitting expected events. |
| TensorBoard monitor data | Backend and frontend | `GET /training/jobs/{id}/monitor-data?nodePath=...&dataset=...`, `MonitorChartsModal` | User opens monitor charts for a selected node and dataset. | Manager finds latest event `logDir` for the job/dataset, `TensorBoardMonitorReader` scans event files, filters tags by `nodePath/`, and returns scalars, latest histograms, and latest images. Frontend renders line charts, histogram bars, and image previews, polling while the job runs. | `MonitorData` has `jobId`, `nodePath`, `dataset`, `logDir`, `scalarSeries`, `histograms`, and `images`. | Unknown job or dataset returns HTTP 400; missing log dir, no event files, no matching tags, or no monitors renders empty data or modal empty states. | Backend TensorBoard filtering test; frontend API query construction and monitor modal opening tests. | Only matching TensorBoard tags with a node path prefix are shown; scalar points are capped at 500 and histogram buckets at 128. |
| UI primitives | Frontend | `components/ui/*` | Shared controls are used across selectors, buttons, badges, switches, checkboxes, segmented controls, and cards. | Components wrap native controls and class merging utilities. | Props are normal React component props plus local variants. | Native disabled and ARIA states are forwarded where applicable. | Frontend primitives tests. | They are local primitives, not a full design-system package. |

## Endpoint Contract Table

| Endpoint | Method | Request | Success response | Main implementation | Error behavior |
| --- | --- | --- | --- | --- | --- |
| `/health` | GET | None | `{ status: "ok" }` | `api.health()` | Fetch failure is client-side only. |
| `/models` | GET | None | `{ models: string[] }` | `discover_models()` | Import path failures become HTTP 400 if discovery cannot resolve `models`. |
| `/models/{model}/presets` | GET | Path `model` | `{ model, presets }` | `list_model_presets()` | Unknown or invalid model returns `{ detail }` with status 400. |
| `/models/{model}/datasets` | GET | Path `model` | `{ model, datasets }` | `list_model_datasets()` | Unknown model or missing dataset options returns 400. |
| `/models/{model}/monitors` | GET | Path `model` | `{ model, monitors }` | `list_model_monitors()` | Invalid or duplicate monitor options return 400. |
| `/models/{model}/config-schema` | GET | Path `model`, optional `preset` query | `{ model, fields }` | `config_schema()` | Unknown model or preset returns 400. |
| `/inspect` | POST | `{ model, preset, overrides, dataset? }` | `{ model, preset, parameterCount, nodes, edges }` | `inspect_model()` | Unknown override, locked override, build error, or instantiation error returns 400. |
| `/training/jobs` | POST | `{ model, preset, datasets, overrides, monitors }` | `TrainingJob` | `TrainingJobManager.create_job()` | Empty datasets, unknown preset/dataset/monitor, invalid override, or locked override returns 400. |
| `/training/jobs/{job_id}` | GET | Path `job_id` | `TrainingJob` | `TrainingJobManager.get_job()` | Unknown job returns 400. |
| `/training/jobs/{job_id}/monitor-data` | GET | Path `job_id`, query `nodePath`, optional `dataset` | `MonitorData` | `TrainingJobManager.get_monitor_data()` | Unknown job or dataset returns 400; missing log files return empty series. |
| `/training/jobs/{job_id}/cancel` | POST | Path `job_id` | `TrainingJob` | `TrainingJobManager.cancel_job()` | Unknown job returns 400; terminal jobs are serialized after status update. |

## UI Workflow Map

### Initial Preview

Happy path:

1. `ViewerApp` mounts and `useViewerQueries()` calls `/health` and `/models`.
2. The first model is selected, which enables `/presets`, `/datasets`, `/monitors`.
3. The first preset and first dataset are selected.
4. `requestPreview()` posts `/inspect`.
5. Backend builds config, instantiates the model, serializes graph nodes and edges.
6. Frontend filters graph to basic/opened view, lays it out, renders graph cards, structure rail, locations, and node details.

Failure path:

1. `/models` failure renders "Backend unavailable".
2. Model-scoped import failure renders "Model import failed", "Dataset discovery failed", or "Config schema failed".
3. `/inspect` failure clears the displayed graph while pending and then renders preview error state.
4. Stale successful inspect responses are discarded if a newer preview request has been issued.

### Config Override and Preview

Happy path:

1. User opens the full config dialog.
2. Schema sections and fields render using backend field metadata.
3. User changes a field; frontend stores a string override by schema `key`.
4. User clicks "Update Preview"; `/inspect` receives the selected model, preset, first selected dataset, and overrides.
5. Backend parses and applies overrides, then returns a refreshed graph.

Failure path:

1. Locked fields are disabled in the frontend and removed from local overrides when the schema changes.
2. If a locked or invalid value is still submitted, backend rejects it with `InspectorError`.
3. The UI surfaces the API error instead of updating the graph.

### Graph Exploration

Happy path:

1. User switches detail mode or graph scope.
2. Frontend filters backend nodes and edges by role and expansion state.
3. `layoutGraph()` recomputes positions only for structural changes.
4. Selecting a card updates selected state and the right panel without triggering relayout.
5. Structure and location rails can reveal nodes by opening ancestors in the graph.

Failure path:

1. Missing graph shows empty/no graph states.
2. Missing metadata simply removes badges, diagrams, location summaries, or terminal reach views.
3. Invalid location coordinates are ignored by the graph helper functions.

### Training Launch and Cancel

Happy path:

1. User selects datasets and optional monitors.
2. `TrainingPanel` posts `/training/jobs`.
3. Backend validates request, writes payload/progress/log files under `/tmp/emperor-viewer-training/{job_id}`, starts the worker subprocess, stores a `TrainingJob` in memory, and returns the serialized job.
4. Frontend stores the active job ID and polls `/training/jobs/{id}` every second until status is terminal.
5. Metrics, queue state, result links, cluster growth, and log tail render from job events.

Failure path:

1. No selected dataset prevents start in the UI and is rejected by the backend.
2. Unknown monitor, dataset, preset, invalid override, or locked override returns 400.
3. Worker exceptions write an error event and exit nonzero; manager reports `failed`.
4. Cancel calls `terminate()` for a running process and marks job `cancelled`.

### Monitor Charts

Happy path:

1. User selects a graph node and opens monitor charts after a training job exists.
2. Modal chooses current dataset or first job dataset.
3. Frontend calls `/training/jobs/{id}/monitor-data?nodePath={node.path}&dataset={dataset}`.
4. Backend locates the latest matching dataset `logDir` from progress events and reads TensorBoard event files.
5. Frontend renders scalar lines, latest histograms, and latest images; it polls while the job is running.

Failure path:

1. No active job disables the monitor button.
2. No selected monitors renders "No monitor selected".
3. Running job with no matching tags renders "No data yet".
4. Completed job with no matching tags renders "No tags for this node".
5. Unknown job or dataset returns 400.

## Backend Versus Frontend Graph Responsibilities

Backend produces:

- Stable node IDs and paths from module containment.
- Edge IDs, source IDs, and target IDs.
- `graphRole` classification: architecture, internal, or runtime.
- Parameter counts and direct weight/bias shape chips.
- Serializable config dataclass fields.
- Module metadata such as dims, activation, dropout, layer norm, recurrent/gate/halting, experts, causal attention, cluster coordinates, and terminal reach.

Frontend derives:

- Simple/basic/full filtering and opened/entire scope filtering.
- Hierarchy, ancestor lookup, and reveal behavior.
- Dagre layout and React Flow nodes/edges.
- Child summaries, gate/halting mechanism rows, long-stack overflow rows.
- Expert routing, stack, cluster, location, and terminal reach diagrams.
- Node card height calculations and selected-node decoration.

## Training State and Persistence

Persisted per job under `/tmp/emperor-viewer-training/{job_id}`:

- `payload.json`
- `progress.jsonl`
- `training.log`
- TensorBoard/event log directories referenced by progress events.

In-memory in the API process:

- Job ID registry.
- Process handle and PID.
- Current `TrainingJob` object and status updates.

Restart behavior:

- After an API process restart, existing temp files remain on disk.
- The new `TrainingJobManager` starts with an empty `jobs` map.
- Previous job IDs are therefore not retrievable through `/training/jobs/{id}`.
- Monitor data for previous jobs is not reachable through the API unless a future recovery mechanism recreates job registry entries.

## Test Coverage Map

| Area | Coverage |
| --- | --- |
| Backend discovery/schema/inspection | `viewer/backend/tests/test_inspector.py` covers model, preset, dataset, and monitor discovery; schema field types; abstract class filtering; preset locks; override parsing; dataset dimensions; API health/inspect; one inspectable preset per model. |
| Backend graph serialization | Same backend test file covers stable IDs, roles, parameter counts, direct shapes, config fields, expert metadata, neuron clusters, terminal reach, lazy params, and unique IDs/edges. |
| Backend training and monitor data | Same backend test file covers fake-process job creation, monitor validation, locked override rejection, TensorBoard tag filtering, CORS, shared error handling, and import regressions. |
| Frontend API client | `viewer/frontend/src/lib/api.test.ts` covers JSON content type, error detail extraction, schema validation failure, URL construction, inspect/training/cancel/fetch bodies, and monitor-data query construction. |
| Frontend state hooks | `viewer/frontend/src/components/viewer/state/*.test.ts` covers query enabling, target/override resets, stale preview response protection, and selection without relayout. |
| Frontend graph helpers | `viewer/frontend/tests/graph.test.ts` covers formatting, navigation, hierarchy, summaries, expert/stack/cluster/terminal/location helpers, detail/scope filtering, layout, and sizing. |
| Frontend graph components | `viewer/frontend/tests/graph-node-view.test.tsx`, `model-structure-card.test.tsx`, and `graph-locations-card.test.tsx` cover card interactions, diagrams, config/details accordions, structure reveal, selected styling, location render and reveal. |
| Frontend app workflows | `viewer/frontend/tests/viewer-app.test.tsx` covers selectors, initial preview, datasets, locked fields, training job posts, monitor posts, preview refresh, config dialog, command dialog/copy, graph modes/scopes/expansion, structure rail, locations, selected-node details, monitor modal opening, and API error rendering. |
| UI primitives and utilities | `viewer/frontend/src/components/ui/primitives.test.tsx`, `utils.test.ts`, `cluster-growth.test.ts`, and `training-command.test.ts` cover shared components, class merging, error messages, cluster-growth event summarization, and generated command formatting. |

Verification run for this analysis:

- `python -m unittest discover -s viewer/backend/tests`: 40 tests passed.
- `cd viewer/frontend && npm test`: 14 files passed, 201 tests passed.
- `cd viewer/frontend && npm run lint`: passed.
- `cd viewer/frontend && npm run build`: passed.
- `cd viewer/frontend && npm run typecheck`: passed after build regenerated `.next/types`.

Recommended verification after related edits:

- `cd viewer/frontend && npm test`
- `cd viewer/frontend && npm run lint`
- `cd viewer/frontend && npm run build`
- `cd viewer/frontend && npm run typecheck`

Note: standalone `npm run typecheck` depends on generated `.next/types`; run
`npm run build` first if `.next/types` is absent or stale.

## Manual Smoke Checklist

- Initial page loads, first model/preset/dataset auto-selects, graph appears.
- Edit a config override, click "Update Preview", and see graph refresh.
- Choose a preset with locked fields and verify disabled controls plus backend rejection if submitted manually.
- Switch graph modes: simple, basic, full.
- Switch graph scopes: opened, entire.
- Expand and collapse graph cards and reveal nodes from structure/location rails.
- Start training with at least one dataset and optional monitor, then verify polling, metrics/log tail, and queue display.
- Cancel a running job and verify status becomes `cancelled`.
- Open monitor charts for a selected node and switch datasets.

## Risk and Backlog List

| Priority | Item | Reason |
| --- | --- | --- |
| High | Add job recovery or explicit cleanup for `/tmp/emperor-viewer-training`. | Temp artifacts persist, but the API registry is in-memory and loses job IDs on restart. |
| High | Decide whether monitor `defaultEnabled` should auto-select in the frontend. | Backend exposes defaults, but current UI ignores them. |
| Medium | Add API-level tests for `/training/jobs/{id}/cancel` and `/training/jobs/{id}/monitor-data` through FastAPI. | Manager behavior is tested, but endpoint wiring has lighter direct coverage. |
| Medium | Add frontend coverage for monitor modal data rendering, not just opening. | Current tests cover API query construction and modal entry, while chart rendering has less workflow coverage. |
| Medium | Clarify typecheck/build ordering or adjust `tsconfig` include policy. | `.next/types` can make standalone `npm run typecheck` depend on a prior build. |
| Medium | Document exact progress event schema emitted by experiments. | Training UI depends on event keys such as `dataset`, `epoch`, `step`, `metrics`, and `logDir`. |
| Low | Add a model import health endpoint or richer discovery diagnostics. | `/models` checks package shape, while import failures appear later on scoped calls. |
| Low | Consider a graph execution-flow mode later. | Current graph is module containment, which is correct for inspection but not a tensor dataflow graph. |
