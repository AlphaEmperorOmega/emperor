# Emperor Model Workbench

The workbench is a local developer tool for inspecting Emperor model presets in a browser. It builds a preset config, instantiates the model, serializes the `torch.nn.Module` tree into a graph, and can launch local subprocess-backed training jobs with optional monitor charts. It does not edit model config files.

The Model workspace target remains the source of truth for Model preview. The
Training workspace owns an independent Training draft seeded from that target
only on first open. A draft can select multiple presets, or Config Snapshots
without a base preset, for one sequential Training Job. The backend Run plan is
authoritative for normal and search runs. Config Snapshot plans are the narrow
exception: the browser materializes their per-Run Snapshot identity because the
current run-plan HTTP request cannot represent it, and Training Job creation
then validates and canonicalizes the submitted plan.

## Architecture

- `src/models/catalog.py` owns concrete Model Package registration and the project
  Adapter composes that catalog with runtime services.
- `model_runtime.packages` owns the public Model Package Interface used by
  generic tooling.
- `model_runtime.inspection` owns transport-neutral Runtime Defaults schemas,
  override and preset-lock validation, selected-package construction, semantic
  graph facts, and Inspection errors.
- `model_runtime.runs` owns transport-neutral Run requests and plans, deterministic
  search expansion, synchronous execution, portable progress semantics,
  results, and relative artifact formats.
- `apps/workbench/api` owns camel-case HTTP serialization, historical Log Run
  and checkpoint enrichment, blocking and error mapping, subprocess Training
  Jobs, monitor data extraction, the FastAPI app, and project Adapter process
  policy.
- `apps/workbench/web` owns the browser UI, React Flow graph layout, config controls, training controls, monitor charts, and API error states.
- Workbench imports only public `model_runtime` Interfaces and communicates
  with the concrete model project through the versioned JSON project Adapter.
- Workbench production code imports neither `emperor` nor `models`; those
  packages do not import `emperor_workbench`.
- This dependency direction is enforced by
  `apps/workbench/api/tests/architecture/test_dependency_direction.py`.

`emperor_workbench.inspection` is the canonical semantic Inspection Interface.
It owns transport-neutral graph records, the in-process/subprocess executor
seam, worker protocol, and private historical checkpoint interpretation.
HTTP contracts and serialization remain under
`emperor_workbench.api.v1.inspection`; project inspection commands enter
through `emperor inspect` and `models.project_cli`.

`emperor_workbench.run_plans` is the canonical Workbench Run Plans Interface.
It owns preview materialization, exact-row and command projection, the
persistence codec, derived summaries and progress projection, and defensive
worker acceptance. Ordinary Training Job submission contains only exact row
identity, preset, dataset, and Runtime Defaults overrides. Snapshot Training
Job submission instead carries Snapshot identities plus backend-issued
semantic revisions; the backend re-resolves and materializes those Runs.
Status, command projections, epoch totals, metrics, errors, and summaries are
response-only state derived by Run Plans.
The worker payload embeds one canonical persisted Run Plan rather than repeating
its envelope alongside it. The worker revalidates exact rows through
the project Adapter's `accept_run_plan` operation and executes them through its
`execute_run_plan` operation; it never imports a concrete Model Package or
repeats grid or random Search Metadata selection.
`TrainingRunPlanView` remains the canonical frozen Run Plan value across
materialization, Training Job records, progress reduction, live projection, and
snapshot projection. Runtime Modules update exact typed Runs with immutable
copies and ask Run Plans to recompute the typed summary. Camel-case
dictionaries exist only while mapping HTTP, encoding or decoding the persisted
JSON document, and defensively accepting the worker payload; runtime, store,
projection, and lifecycle Implementations do not interpret those serialized
field names.

`emperor_workbench.training_jobs.TrainingJobService` is the sole public
Training Jobs Interface. The Run Plans Interface serves preview, Training Job
acceptance, persistence, live projection, and worker handoff. The private
Training Jobs package owns persisted job state, worker launch and
containment, lifecycle recovery, cancellation, progress replay/projection,
monitor lookup, and terminal cleanup. Raw progress JSONL remains authoritative,
while a bounded progress Module keeps a 100-event tail, sparse 256-event byte
offset checkpoints, compact terminal, event-count, and monitor-location
aggregates, and a caller-owned byte cursor for the one incremental reducer.
Summary readers do not consume that cursor; raw history pages seek from the
nearest checkpoint and stream from JSONL without materializing the complete
history.
Process-local progress and reducer caches are LRU-bounded, so released terminal
jobs reuse a bounded checkpoint instead of replaying their whole file on every
read. Blank lines, invalid JSON, and valid non-object JSON values are ignored;
an incomplete final line is retried after the writer completes it, while invalid
UTF-8 and filesystem read errors remain explicit failures. The app-scoped
`TrainingJobService` reports the same resolved cancellation capability used by
its launcher. `auto` resolves to cgroup v2 when writable on Linux, POSIX process
groups for the Linux fallback and macOS, and named Windows Job Objects on
Windows. Process-group mode never constructs or probes cgroup state.
Strict-cgroup probing is lazy and total for capability reads, but each launch
still requires a writable per-job cgroup immediately before starting the worker.
Cgroups and Windows Job Objects receive configured memory, CPU, and process
limits; the capabilities response reports whether those limits are enforced.
Training Job
admission is capped, worker environments are allowlisted, and private job state
is created with restrictive modes even under hostile umasks. Persisted strict
jobs recover their canonical cgroup through that same Adapter.
An unproven recovered job is `unknown` and remains an active Log Experiment
mutation blocker. Worker processes use only
`emperor_workbench.training_jobs.worker` and
`emperor_workbench.training_jobs.cgroup_worker`; persisted historical command
arrays remain inert observations and never authorize execution. Application
composition and ordinary create, read, cancel, restart, and recovery
verification cross `TrainingJobService`; private runtime access is reserved for
process containment, persistence, progress, and cache seams.

`emperor_workbench.run_history.RunHistoryService` is the sole backend Run
History Interface. It owns Log Run discovery and stable identities, pagination
and facets, TensorBoard queries, artifact/checkpoint lookup, historical
Inspection context, delete planning/execution, archive mutation, and completion
invalidation for one process-local cache graph. Scanner, query, deletion, ZIP,
and filesystem records are private capability Implementations. A generation-
aware catalog under the private state root serves repeated Run metadata, facets,
and pagination without recursive rescans. Mutations update or invalidate it,
while external additions are reconciled within 30 seconds. Config Snapshots use
the same persistent catalog policy. Scanner and query reuse one contained
`RunArtifactObservation` per Run for recursive freshness,
catalog counts, lazy result/hparams metadata, artifact detail, checkpoints, and
historical Inspection. The observation is capped at 10,000 filesystem entries,
16 directory levels, and 4 MiB per metadata file; artifact detail reports a
truncation reason when a cap is reached. Summary listings parse result metadata
only for returned Runs and do not project metrics, while full listings reuse
that same one-time parse. Its event-file member is the shared `EventFileIndex`,
so Run History does not implement a second TensorBoard containment policy.
The Interface returns frozen snake-case semantic pages, facets, checkpoints,
Run Artifact details, TensorBoard projections, archive-import results, and
deletion values. It never returns HTTP-shaped records or owns response
serialization. The Logs HTTP mapping Module beside the router and schemas owns
camel case, Model Package identity expansion, response row caps, truncation
metadata, and schema payload construction. The persistent catalog codec accepts
only its canonical schema version, authority root, entry keys, and exact Model
Package identities.
Historical Inspection receives only one frozen context containing canonical Run
identity, saved parameters, and contained checkpoint candidates; it does not
receive the Run History implementation. Every candidate freezes its resolved
path, byte size, and nanosecond modification time. The deep
`WorkbenchHistoricalInspection` Interface owns latest-candidate selection,
saved/checkpoint/request precedence, semantic retry, and graph annotation. It
opens at most 32 candidates, permits at most 256 MiB per file and 512 MiB in
aggregate, and verifies descriptor freshness before and after loading. The
generic shape interpreter handles the current checkpoint layout when a package
does not declare checkpoint metadata; package-specific state-dict knowledge is
supplied through the selected Model Package's explicit capability without
coupling Workbench to that package's Implementation.
Run History owns neither checkpoint ranking nor checkpoint shape
interpretation; it only freezes contained candidates and parses filename
epoch/step facts needed by its own Run Artifact metadata.

Request streaming, the bounded multipart spool, content metadata, admission,
authentication, mutation proof, and response validation remain in the HTTP
Adapter. The shared `emperor_workbench.log_experiments` package owns Log
Experiment identity and the one app-scoped mutation coordinator used by
Training Jobs and Run History. The shared `emperor_workbench.tensorboard`
package owns one contained event-file observation per read: trusted event
directories, fingerprint, total size, byte-budget decision, and accumulator
loading all come from the same `EventFileIndex`. Monitor, parameter-status, tag,
scalar, image, and text projections remain distinct, while their root-keyed LRU
caches share one generation-checked publication and invalidation implementation.
Training Jobs and Run History both consume this Module; neither capability
imports the other's private package.

The frontend Training Module owns the Training draft, backend Run-plan
coordination, confirmation and mutation lifecycle, active Training Job
projection, polling, and Training-triggered Logs cache refresh. Rendering uses
`useTrainingWorkspace()` through its `draft`, `plan`, `job`, `dialogs`, and
semantic `actions` groups. Full Config uses the narrower
`useTrainingConfiguration()` projection. Logs, graph, and header consumers use
the read-only `useActiveTrainingJob()` projection; connection/auth composition
clears Training through one internal command rather than manipulating Training
state directly.

The frontend Model Package Inspection Module owns one complete preset, Config
Snapshot, or historical Run target; Model Package/preset selection; Experiment
Task-compatible Dataset Metadata; Runtime Defaults; version-1 browser
restoration; and Inspection request identity. Configuration Source tabs are
browser presentation state, so browsing Presets, Snapshots, or Runs does not
replace the last complete Inspection. Callers consume the grouped
`useModelPackageInspection()` Interface (`target`, `browser`, `options`,
`runtimeDefaults`, `status`, and semantic `actions`) plus the focused
`useModelPackageCatalog()` projection. The private preview Implementation owns
cache reuse, forced refresh, cancellation, and stale-response suppression;
graph state observes semantic transition revisions and owns selection and
expansion resets.

One pure Inspection target lifecycle reducer owns Model Package identity, the
active preset, Config Snapshot, or historical Run, Experiment Task-compatible
datasets, active Runtime Defaults, restoration phase, connection generation,
and semantic transition revision and cause. Model Package, target, metadata,
Runtime Defaults, restoration, missing-Snapshot, and connection-reset changes
enter through explicit events. Effects are limited to delivering query results,
publishing version-1 storage, coordinating private historical browsing, and
executing the single request derived from the complete lifecycle; they do not
repair separate target fields.

The frontend Model Package Metadata Module accepts one structured Model Package,
preset, and Search Metadata selection. It privately owns TanStack query keys,
protected-read gating, cancellation, retry and stale-time policy, and caching.
Inspection, Training, and the Config Snapshot editor keep independent
selections and consume only focused Model Package, preset, Dataset Metadata,
default Experiment Task, Monitor Metadata, Runtime Defaults schema, and Search
Metadata projections with stable readiness facts; raw query results do not
cross the Module Interface.

Target-scoped historical browsing is a private collaborator of that frontend
Inspection owner. It owns Run queries, Experiment/Dataset/preset filter
cascades, tag reconciliation, selected-Run validity, and the transition from a
complete filter choice to one historical Inspection target. Rendering receives
the focused `useHistoricalRuns()` browsing projection, while graph orchestration
receives a separate read-only facts projection. General Run browsing and
deletion remain owned by Logs Workspace.

The frontend Experiment Monitor parameter-activity Module owns source-aware
loading for active Training Jobs, one historical Run, and historical Run groups.
It selects query identity, applies protected-access and monitor-readiness gates,
polls active Training Jobs, reuses historical status, and derives loading
activity and path mismatch through the shared selector. The graph and
parameter-activity minimap remain separate visual Adapters over that lifecycle.

The frontend Graph Display Module owns semantic detail and scope filtering,
full and visible navigation, child-summary and diagram projection, activity
decoration, card data, and numeric card geometry. Graph state consumes one
coherent projection; the dynamically loaded Dagre leaf only positions its cards
and edges. Selection is a cheap decoration pass that preserves every unaffected
node and edge reference. Graph renderers read the same grouped geometry used by
height projection, while the parameter-activity minimap remains a distinct
visual Adapter with its own shared renderer-and-layout geometry.

The frontend Workbench Layout Module owns the workspace frame, narrow stacking,
wide sidebar/primary/details occupancy, responsive borders and overflow,
full-width workspace and lazy-fallback spans, and Training's horizontally
scrollable three-region grid. Model, Logs, and Training remain distinct
rendering Adapters that provide semantic region slots. Active-workspace and
mount lifecycle remain in the Workbench shell. Training's wide-layout
Implementation is a private deferred leaf so layout ownership does not pull it
into the initial browser bundle.

The frontend Logs Workspace Module owns paginated browsing across the full Run
catalog through explicit experiment, Dataset, Model Package, preset, and tag
filters; progressive tag and chart-query coordination; selected Run details;
and the capability-guarded deletion lifecycle. No experiments are selected on
first activation, so Run metadata is available for browsing while tag and chart
reads stay cold until an experiment is chosen. Rendering consumes four focused
projections: `useLogsBrowser()`, `useLogsCharts()`, `useLogRunDetail()`, and
`useLogsDeletion()`. Nullable selections, query keys and chunking, stale-data
reconciliation, delete filters, mutations, and cache invalidation remain private
Implementation. Browsing state stays mounted after the first Logs activation;
queries are disabled while hidden, render-session chart settings reset on
unmount, and an already-issued deletion may finish and reconcile through the
next authoritative refresh.

An active `LogsChartsProvider` mounts only while Logs is visible, owns the chart
lifecycle, and publishes the completed `useLogsCharts()` projection. Chart
planning receives semantic Run/tag/loading facts and semantic commands; raw
TanStack results, state setters, and low-level toggles do not cross its seam.
The Logs Workspace Implementation also owns selected Run Artifact loading, so
`useLogRunDetail()` is a read-only context projection with no query work.

The Logs Charts Implementation owns visibility activation, progressive
ten-Run/six-tag scalar planning, tag-refresh gating, compatible stale-series
retention, per-group and per-tag status, best-Run reuse, checkpoints, media,
confusion matrices, and refresh commands behind `useLogsCharts()`. Near-visible
charts do not read scalars until their visibility command arrives. During a Run
or tag replacement, the projection retains only series compatible with the new
Run and tag scope until authoritative results settle. The recorded batching
benchmark remains evidence rather than a reason for speculative policy changes.

One private Logs Scalar Card Module owns near-viewport entry, loading and error
frames, chronological summary, checkpoint projection, chart options, header and
information dialog, bounded legend, Run selection, and linked hover/focus
highlighting. The ordinary scalar Adapter supplies one line and dot legend per
Run. The train-validation Adapter supplies phase-labelled solid and dashed
lines. Both retain the shared deep scalar-option builder for axes, smoothing,
zoom, checkpoint markers, and line emphasis.

The Logs transport Implementation privately owns bounded request scheduling
beside its chunk sizes and aggregation. Log Run tags are sequential, media reads
permit two active chunks, and disk-heavy scalar reads are globally serialized
across callers. Queued scalar aborts are removed before transport, failures stop
new chunks, and concurrent responses are aggregated in request order. There is
no exposed generic scheduler without a second Adapter.

Config Snapshot records and the modal editor have lifecycles separate from the
Inspection target. `useConfigSnapshotRecords()` exposes the current Model
Package records and target-selection/mutation commands, while
`useConfigSnapshotEditor()` owns an explicit draft or edit session initialized
from either Model or Training. Editing a snapshot never mutates preset Runtime
Defaults merely to open the dialog. Backend Config Snapshot create and update
operations are atomic through the app-scoped persistence Adapter, and stored
records are immutable values with Model Package/preset-scoped name and Runtime
Defaults uniqueness.

The frontend Workbench Connection Module owns the normalized API base URL,
hosted-origin enforcement, browser-session bearer identity, public capability
checks, the protected `GET /models` authentication probe, and the ordering of
connection-wide invalidation. Domain modules receive one semantic protected-
access gate and expose only their own reset commands; rendering does not access
storage, query keys, or credentials. The shared HTTP client attaches the current
credential and mutation proof at request time and rejects responses whose
private connection revision is obsolete.

The frontend browser scenario harness exposes six capability Interfaces:
Config, graph, Logs, Experiment Monitors, overview, and Training. Each scenario
Module imports one family harness with narrowed setup, fixture, driver, and
observation groups. Stateful route dispatch, mutation observations, app reset,
and raw fixture records remain private to the shared Implementation. The React
Flow test Adapter supplies only the third-party canvas seam and renders the
production graph `nodeTypes`; it does not reproduce graph cards or formatting.


## Backend API Compatibility

The public ASGI import target is stable at `emperor_workbench.api:app`. Code that
needs a configured app should import `create_app` from `emperor_workbench.api`
and `WorkbenchApiSettings` from `emperor_workbench.settings`. The settings
Module is the canonical Interface; the former `emperor_workbench.core.config`
path is not supported.

HTTP routes are mounted at root paths such as `/health`, `/models`, and
`/training/jobs`. The `emperor_workbench.api.v1` package is an internal FastAPI
organization namespace, not a public `/v1` URL prefix. Do not add `/v1` routes
without updating the frontend contract and backend API contract tests in the
same change.

## Setup

```text
mise run setup --profile cpu
mise run dev
```

Mise provisions Python 3.13 and Node 24. Setup creates the native virtualenv,
installs the Emperor model distribution and the separately declared
`emperor-workbench` application, then installs the frontend dependencies. `dev`
starts both Workbench services without injecting the checkout into
`PYTHONPATH`. Logs and validated JSON process metadata live under
`.runtime/workbench/`. Backend state and config snapshots live under its
`state/` and `snapshots/` directories. Metadata includes PID, process creation
time, command identity, argv, and port, so stale or reused PIDs are never trusted.
Startup waits for the backend health endpoint and a frontend HTTP response.
By default, the backend uses port `9999` and the frontend uses port `9000`.

Running setup or `dev` again is safe: setup hashes its inputs and service reuse
requires every persisted identity field plus a healthy HTTP response.

Stop or inspect the workbench:

```text
mise run workbench:stop
mise run workbench:status
```

Native PowerShell users can use `. .\env.ps1`, `-WorkbenchStatus`, and
`-WorkbenchStop`.

## Run

Backend:

```bash
python -m emperor_workbench --reload --host 127.0.0.1 --port 9999
```

An installed environment may use the equivalent `emperor-workbench` command.
Set `EMPEROR_PROJECT_ADAPTER_COMMAND` when the model project's
`emperor-project-adapter` executable is not on `PATH`.

Frontend:

```bash
cd apps/workbench/web
npm run dev
```

Open:

```text
http://localhost:9000
```

## Hosted Backend Settings

Hosted deployments must configure both explicit frontend origins and API host
authorities when the backend is reachable outside loopback. Both settings use
JSON array syntax:

```bash
export WORKBENCH_API_CORS_ORIGINS='["https://workbench.example.com","https://admin.example.com"]'
export WORKBENCH_API_TRUSTED_HOSTS='["api.example.com"]'
```

When bearer auth is enabled, keep origins specific to the deployed frontend
hosts. Do not use wildcard origins such as `["*"]` for authenticated hosted
deployments.

Hosted frontend builds should also lock which API origins browser requests may
target. If `NEXT_PUBLIC_WORKBENCH_API_URL` points at a non-local API, the frontend
locks requests to that configured origin by default. Set
`NEXT_PUBLIC_WORKBENCH_API_ALLOWED_ORIGINS` to make the allowlist explicit or to
allow multiple API origins:

```bash
export NEXT_PUBLIC_WORKBENCH_API_URL=https://api.example.com/workbench
export NEXT_PUBLIC_WORKBENCH_API_ALLOWED_ORIGINS='["https://api.example.com"]'
```

For a bearer-protected hosted deployment, set the frontend and backend values as
a matched pair before building or starting the services:

```bash
# Frontend build environment. These values are public and are bundled into
# browser JavaScript by Next.js.
export NEXT_PUBLIC_WORKBENCH_API_URL=https://api.example.com
export NEXT_PUBLIC_WORKBENCH_API_ALLOWED_ORIGINS='["https://api.example.com"]'

# Backend runtime environment. Keep the token secret out of frontend env files.
export WORKBENCH_API_CORS_ORIGINS='["https://workbench.example.com"]'
export WORKBENCH_API_TRUSTED_HOSTS='["api.example.com"]'
export WORKBENCH_API_AUTH_MODE=bearer
export WORKBENCH_API_TOKEN='<replace-with-a-secret-token>'
```

In a clean browser session, open **Connection**, enter the API operator's
bearer token under **Session bearer token**, and choose **Sign in**. The token is
stored only in browser session storage; it is not bundled into the frontend and
must never be placed in a `NEXT_PUBLIC_*` variable. The same control replaces a
token or logs out. Sign-in, replacement, and logout clear protected browser
caches and mutations before active API reads are retried, and rejected
credentials are shown as an authentication state in the Workbench. Protected
callers remain inactive until `GET /models` verifies the current bearer
revision; `/health` and `/capabilities` remain public. One session token is used
for every API origin explicitly permitted by the frontend allowlist, so changing
the base URL retains the token while forcing a new probe.

The in-app base URL is normalized and stored in browser local storage only after
the browser verifies the write. Invalid or disallowed stored values are removed,
fall back to `NEXT_PUBLIC_WORKBENCH_API_URL`, and produce an actionable status.
Storage write/removal failure leaves the previous identity and caches intact.
A base-URL change cancels current work, clears all connection-scoped query and
mutation data, resets protected Training, Logs, graph, historical, editor, and
Inspection state through their semantic commands, then reloads capabilities and
authentication. The complete Inspection target is retained and rebuilt only
after the new backend's metadata is ready. Token replacement and logout follow
the same sequence for protected state while retaining public connection checks;
late responses from any previous revision are ignored.

The development launcher starts one Uvicorn process for the Workbench backend. The
backend uses in-process locks around Training Job, Run History, TensorBoard, and
progress caches so concurrent requests in that process do not corrupt shared
state. Run History uses generation-checked cache publication so a read that
started before an import or deletion cannot repopulate invalidated state after
that mutation completes. Those locks are not cross-worker invalidation. If a hosted deployment
uses multiple Uvicorn or Gunicorn workers, each worker has its own process-local
caches and may observe filesystem changes after its own cache TTL or explicit
cache clear path. Use one worker for the local-file-backed Workbench unless you add
a shared cache/invalidation design.

For manual or hosted read-only deployments, leave local mutation endpoints
disabled. Backend mutation actions such as training, log deletion, and config
snapshot creation, rename, and deletion require explicit backend-side unsafe
local mutation opt-in only when you intentionally allow the API to mutate local
files or processes:

```bash
export WORKBENCH_API_ALLOW_UNSAFE_LOCAL_MUTATIONS=true
```

Log archive imports are narrower than the broad local mutation switch and are
disabled by default in every authentication mode. Enable them explicitly only
when that filesystem mutation is intended:

```bash
export WORKBENCH_API_ALLOW_LOG_IMPORTS=true
```

The normal `mise run dev` launcher explicitly enables both local mutations and
log imports for its loopback development backend. A manually started backend
retains the read-only defaults.

Unauthenticated browser mutations must also send
`X-Workbench-Mutation: true`; the included frontend adds this header. The
backend rejects mutation requests with an untrusted `Origin` or
`Sec-Fetch-Site: cross-site` before reading the request body. CORS origins still
need to list every trusted, separately hosted frontend.

Every mutation must also send an `Idempotency-Key` of 1–128 printable ASCII
characters. The frontend creates a UUID once per mutation command and reuses it
for retries. The backend journals request hashes and serialized results under
`WORKBENCH_API_STATE_ROOT`; an identical retry is replayed, a conflicting retry
returns `409`, and a missing key returns `428`. Keep this state root private,
writable, and durable across backend restarts. It also holds generation-aware
Run and Config Snapshot catalogs.

The default aggregate budgets are a 1 MiB JSON request body, 64 MiB of
TensorBoard event work per request, a 128 MiB byte-weighted TensorBoard cache,
and 1 MiB per progress JSONL record. Inspection defaults to 4 GiB, 4 CPUs, and
60 seconds. Training defaults to two active jobs and, when its containment mode
enforces limits, 16 GiB, 8 CPUs, and 512 processes per job. Override these
through the matching
`WORKBENCH_API_MAX_JSON_BODY_BYTES`, `WORKBENCH_API_TENSORBOARD_*`,
`WORKBENCH_API_MAX_PROGRESS_RECORD_BYTES`, `WORKBENCH_API_INSPECTION_*`, and
`WORKBENCH_API_MAX_ACTIVE_TRAINING_JOBS`/`WORKBENCH_API_TRAINING_JOB_*`
settings only after sizing the host.

`WORKBENCH_API_TRAINING_CANCELLATION_MODE` defaults to `auto`. Its accepted
values are `auto`, `strict-cgroup`, `process-group`, and
`windows-job-object`. Use an explicit concrete value only to require that
mechanism; unsupported combinations fail with an actionable error. The public
capabilities response exposes the resolved concrete mode as
`trainingCancellationMode` and reports strict resource-limit enforcement as
`trainingResourceLimitsEnforced`.

Backend contributors must classify every non-safe HTTP operation beside its
route registration with `@declare_http_operation(...)` from
`emperor_workbench.api._mutations`. Use `READ_ONLY` for POST queries and
planning, `LOCAL_MUTATION` for the broad local file/process opt-in, and
`LOG_IMPORT` for the narrower archive-import opt-in. Application construction
derives an immutable catalog from the final mounted `APIRoute`s and fails when
a declaration is missing, unknown, duplicated, conflicting, or not mounted.
That catalog drives early origin/proof enforcement through Starlette's native
route matcher, while the same declaration performs the route-local operational
check. Do not add a separate method/path mutation list.

## Import Logs

Use `mise run logs:archive` from a project directory to create a log archive,
then open the Workbench and choose **Import Logs** in the top navigation. The backend
validates archive metadata, decompresses each member once into a private staged
directory, then commits through descriptor-relative no-follow operations. It
overwrites only an unchanged regular file at the same archive path and fails
closed if an ancestor or target changes during import.

Importing logs is a local file mutation scoped to the backend logs folder. It
remains disabled unless `WORKBENCH_API_ALLOW_LOG_IMPORTS=true` is set. The
backend advertises upload support and the compressed upload-size cap through
`/capabilities`. Enabled imports default to a 512 MiB compressed limit and a
2 GiB extracted limit, at most 32,000 archive members, a 4 MiB cumulative UTF-8
member-path budget, and one active archive import at a time. Override them with
`WORKBENCH_API_MAX_UPLOAD_SIZE=<bytes>` or
`WORKBENCH_API_MAX_LOG_ARCHIVE_EXTRACTED_SIZE=<bytes>`,
`WORKBENCH_API_MAX_LOG_ARCHIVE_MEMBER_COUNT=<count>`,
`WORKBENCH_API_MAX_LOG_ARCHIVE_PATH_BYTES=<bytes>`, or
`WORKBENCH_API_LOG_ARCHIVE_UPLOAD_CONCURRENCY=<count>`. Size limits always have
finite defaults; setting their optional configuration values to null does not
disable them.

Training Job start, log deletion, and archive replacement are serialized per
top-level Log Experiment inside one backend process. An import or deletion that
follows a started Training Job is rejected while that job can still write to
the same experiment; when the filesystem mutation wins the ordering, the job
starts only after it finishes. Archives spanning multiple experiments acquire
those experiment scopes in a stable order.

Archive validation, member/path/extracted-size budgets, CRC validation, and
single-pass staging complete before destination writes. Each staged file is
atomically renamed relative to validated directory descriptors, but the whole
archive is deliberately not transactional: if a later operating-system commit
fails, earlier replacements remain. Run History removes the private staging
tree and invalidates affected caches even on that partial failure. Every archive
member must be rooted directly at an experiment path; a top-level `logs/`
wrapper is rejected.

Read paths ignore event directories containing a TensorBoard event file that
resolves outside the requested Run root; mutation paths reject unsafe targets.
Hosted archive imports require Linux/POSIX operations equivalent to
`openat`/`renameat` with no-follow directory traversal and fail closed when they
are unavailable. The local-only same-user limitation remains: another process
running as the backend's Unix identity can interfere with any local
filesystem-backed service, and the coordinator does not coordinate separate
backend workers.

## Test

Backend:

```bash
cd apps/workbench/api
python -m pytest
python -m ruff check .
```

Frontend:

```bash
cd apps/workbench/web
npm ci
npm run lint
npm run typecheck
npm test
npm run test:contract:e2e
npm run build
npm run performance:budget
npm run performance:browser
```

The frontend supports Node `>=20.9.0`; repository development and CI use Node
24. Next.js 16 uses Turbopack for both normal development and production builds,
and lint remains an explicit gate rather than part of `next build`.

The harness starts temporary loopback backend and production frontend servers,
drives headless Chromium through repeated Model, Training, Logs, chart-dialog,
Training Job, import, and 3D graph workflows, and removes its temporary state.
It never uses the configured user logs or snapshots and never starts a real
training process. Bundle limits and stable functional checks are enforced;
runtime, heap, API, and software-rendered WebGL measurements remain
informational.

`test:contract:e2e` starts two real bearer-protected backend apps on temporary
loopback ports and drives them through the typed frontend API client. Their
logs, snapshots, and fake Training Job state live under temporary directories
and are removed after the test. The contract covers capability loading,
authentication, protected mutations, API-origin switching, normalized errors,
and logout without starting a real training process.

CLI inspection is available through the canonical Emperor task:

```bash
mise run experiment -- --model-type linears --model linear --preset baseline --print-model
mise run experiment -- --model-type linears --model linear --preset baseline --print-model --format json
```

Monitor discovery is available for terminal training runs:

```bash
mise run experiment -- --model-type linears --model linear --list-monitors
mise run experiment -- --model-type linears --model linear --preset baseline --datasets mnist --monitors linear halting
```

`--monitors` applies to training runs and cannot be combined with `--print-model`.

Every Run Plan exposes canonical `commandArgv` plus `commands.posix` and
`commands.powershell`. The historical POSIX `command` field remains for
compatibility. The command dialog suggests PowerShell in Windows browsers and
POSIX elsewhere; its shell selector persists an explicit override for WSL and
remote-browser use.

Training from the CLI can also run selected preset batches:

```bash
mise run experiment -- --model-type linears --model linear --presets baseline gating --grid-search
```

## Troubleshooting

- Backend unavailable: start `python -m emperor_workbench --host 127.0.0.1 --port 9999`, or set `NEXT_PUBLIC_WORKBENCH_API_URL` before starting Next.js.
- Model import failed: check stale imports in that model package's `config.py`, `presets.py`, or `model.py`.
- Invalid override value: use enum names such as `GELU`, booleans such as `true`, and class names that are imported by the model config module.
- Empty graph: verify the selected preset can instantiate `Model(cfg)` without training.
