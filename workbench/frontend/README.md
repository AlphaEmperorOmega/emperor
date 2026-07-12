# Emperor Workbench Frontend

Next.js App Router frontend for the Emperor Model Workbench. The app renders the
local workbench workspace at `/` and talks to the Emperor workbench API from the
browser.

## Requirements

- Node.js `^18.18.0 || ^19.8.0 || >=20.0.0`
- npm
- A running Emperor workbench API. Local development defaults to
  `http://127.0.0.1:9999`.

## Setup

Install dependencies:

```bash
npm install
```

If the API is not running at the default local URL, create a local environment
file and set the public API base URL:

```bash
NEXT_PUBLIC_WORKBENCH_API_URL=http://127.0.0.1:9999
```

`NEXT_PUBLIC_WORKBENCH_API_URL` is intentionally public because browser-side code
uses it to call the API. Do not put secrets in `NEXT_PUBLIC_*` variables.

Local development keeps the in-app API base URL switcher unlocked so you can
point the browser at different loopback backends. Hosted builds should also set
`NEXT_PUBLIC_WORKBENCH_API_ALLOWED_ORIGINS` to the API origins that may receive
browser requests and bearer tokens:

```bash
NEXT_PUBLIC_WORKBENCH_API_ALLOWED_ORIGINS=https://api.example.com
```

Use a comma-separated list or JSON array when a hosted frontend may talk to more
than one API origin.

## Development

Start the development server:

```bash
npm run dev
```

The dev server listens on port `9000` by default. Override it with `PORT`:

```bash
PORT=3000 npm run dev
```

Open `http://localhost:9000` unless you selected a different port.

## Available Commands

- `npm run dev`: start Next.js locally on `${PORT:-9000}`.
- `npm run build`: create a production Next.js build.
- `npm run lint`: run ESLint with zero warnings allowed.
- `npm run typecheck`: generate Next.js route/app types, then run TypeScript
  without emitting files.
- `npm run test`: run the Vitest unit and component test suite.
- `npm run test:contract:e2e`: start two temporary bearer-protected backend
  processes and run the live typed-client contract.
- `npm run performance:browser`: run the production-build browser and
  long-session performance harness.

`npm run typecheck` writes generated Next.js types under `.next/types`. Run it
sequentially with `npm run build`; running both against the same `.next`
directory at the same time can race while those generated files are refreshed.

## Testing

Unit and component tests live under `tests/` and alongside source files with
`.test.ts` or `.test.tsx` names. Run them with:

```bash
npm run test
```

The live backend/frontend contract uses the repository Python environment when
`torchenv/bin/python` exists, or `WORKBENCH_E2E_PYTHON`/`python` otherwise. It
requires the Workbench backend dependencies, permission to bind temporary
loopback ports, and permission to create temporary directories. Run it from
this package with:

```bash
npm run test:contract:e2e
```

The runner starts two bearer-protected backend apps with isolated temporary
logs, snapshots, and fake Training Job state. It covers capability loading,
authentication, protected mutations, API-origin switching, normalized errors,
and logout without starting a real training process, then removes its temporary
state.

After a successful production build, the browser performance harness can be run
with:

```bash
npm run performance:browser
```

It starts temporary loopback services and headless Chromium; it does not use the
configured user logs or snapshots.

## Browser Scenario Harness

Workbench browser scenarios consume one of six capability Interfaces: Config,
graph, Logs, Experiment Monitors, overview, or Training. Each harness exposes a
narrow setup contract plus semantic fixture, driver, and observation groups.
The shared routing Implementation, mutation records, app reset, and raw fixture
declarations stay private, so a scenario cannot couple itself to another
capability's setup surface.

The private React Flow test Adapter verifies the immutable-canvas options,
preserves projected node and edge facts, and invokes the production graph
`nodeTypes`. Graph cards, accessibility text, formatting, expansion, detail
mode, and activity presentation therefore remain owned by production graph
Modules rather than a parallel test presenter.

## Connection and Browser Session

The Workbench Connection module owns the active API base URL, hosted-origin
policy, bearer session, capability and authentication status, request identity,
and protected-state invalidation. Rendering consumes derived states and complete
actions; it does not read browser storage, coordinate query keys, or interpret
the Model Package catalog as authentication state.

API base URLs must be absolute `http://` or `https://` URLs without user
information, a query string, or a fragment. A successful runtime change is
normalized and stored in local storage. Invalid or disallowed stored values are
removed and fall back to the configured default with an actionable status. If
local storage is unavailable or does not retain a write/removal, the requested
change fails atomically and the previous connection and caches remain active.

Bearer tokens are stored only in browser session storage and attached at request
time. They never enter public connection state, query keys, persistent query
data, snapshots, or normalized error text. One session token is shared across
the origins explicitly permitted by the hosted allowlist; changing the API base
URL does not erase it. If session storage cannot retain or remove a token,
sign-in, replacement, or logout fails without publishing a partial identity.

`/health` and `/capabilities` are public connection checks. When bearer auth is
enabled, `GET /models` verifies the credential for the current connection
revision. All other protected reads, polling, pagination, previews, uploads, and
mutations remain inactive until that probe succeeds; when backend authentication
is disabled they become active after capabilities resolve. Rejected credentials
remain replaceable or removable from **API Connection**.

Base-URL changes cancel current requests, advance the private request revision,
clear all query and mutation data, invoke each domain module's semantic reset,
then re-check capabilities and authentication. A complete Inspection target is
retained and rebuilt against the new backend only after its metadata is ready.
Token replacement and logout use the same ordering for protected state while
retaining public health/capability data. Late responses cannot repopulate the
new identity even when a transport ignores `AbortSignal`; active imports are
also aborted and their late completion is ignored.

## Config Snapshot Lifecycles

The Config Snapshot records Module is the authoritative persistence Interface.
It serializes create, rename, update, and removal commands; publishes pending,
success, and actionable failure state; retains the exact failed command for
retry; and completes only after persistence and record invalidation settle.
Connection changes clear that lifecycle generation so a late result cannot
publish status or invalidate records for the new connection.

The editor session, Model Package Inspection target, and Training draft remain
separate lifecycle owners at this seam. The editor owns its isolated Runtime
Defaults draft and keeps save confirmation open until the records Module
reports authoritative success. Inspection changes an active Config Snapshot
target back to its preset only after removal succeeds. Training removes a
Config Snapshot from its Run Plan selection only after the same authoritative
outcome. Persistence and retry retain Locality in the records Module while
those post-success transitions stay with the owner whose state they change.

## Training Run Plans

The private Training Run Plan Module owns effective Runtime Defaults, Search
Metadata lock and conflict rules, request identity, backend planning, exact
retry, random resampling identity, start readiness, and large-grid
confirmation. Rendering consumes its grouped search and plan projections and
does not recalculate combinations, conflicts, unlocked axes, or estimated Runs.

Backend Run Plans remain authoritative for normal, grid, and random Runs. The
only browser-materialized exception is a Run Plan containing selected Config
Snapshots, because those Workbench-owned records do not exist in generic Runs.
That exception is normalized and summarized in the same private Module before
it becomes an immutable launch request.

The Training Job Module receives that exact launch request and owns only process
lifecycle: launch, observation, cancellation, terminal reset, and mutation
failure. It does not repeat Search Metadata selection or reconstruct Run Plan
identity.

## Training Job Lifecycle

One app-scoped Training Job Implementation owns active identity, launch and
cancellation mutations, polling, terminal-state protection, Logs refresh,
started-folder notification, semantic reset, and connection-generation
quarantine. It stays mounted when the Training workspace is hidden, so polling
and terminal refresh do not depend on rendering.

Header, graph, and Logs callers receive only the read-only active Training Job
projection. Training composition receives semantic launch, cancel, and reset
commands; no raw identity setter or polling callback crosses the ownership seam.
Training draft and Run Plan reset remain separate and register directly with the
Workbench Connection transition registry.

Training draft/configuration state and Training Job lifecycle remain in the
app-scoped provider graph because Model, Full Config, Logs, and header consumers
use their focused projections. Log-folder selection, Run Plan generation, and
the Training workspace projection live behind a sticky dynamic execution
provider. That provider loads on the first Training visit and stays mounted
after workspace switches, preserving execution state without charging its
implementation to the initial route bundle.

## Logs Workspace State

The Logs Workspace Module owns experiment, Run, and scalar-tag loading;
selection normalization; started-Run seeding; detail choice; pagination;
deletion tombstones; and authoritative-refresh reconciliation in one private
Implementation. Its public seam exposes four focused projections: browser,
chart source, detail, and deletion. The provider wires those projections and
private semantic reset commands without receiving the raw internal state shape.

Experiment changes retain a manual scalar-tag selection while Run facets and
tag options are still refreshing. Once fresh tags are authoritative, the owner
preserves a matching selection, replaces a stale selection with preferred
defaults, or falls back to available non-standard tags. The deletion
Implementation derives and freezes its own subset filters, while the workspace
owner reconciles deleted Runs, detail choice, and started experiments only
after the relevant refresh succeeds.

The Logs Charts owner keeps viewport activation, ten-Run/six-tag progressive
planning, tag-refresh gating, compatible stale-series retention, group status,
and refresh behavior behind the chart projection. Collapsed or not-yet-visible
charts stay cold; newly visible charts request only their selected scalar tags.
Replacement windows preserve series that still match the current Run and tag
scope and remove incompatible series immediately.

Scalar chart rendering uses one private card Implementation for lazy entry,
loading and failure, summary, checkpoints, options, information, legend,
selection, and linked highlighting. Ordinary metrics adapt one line per Run;
train-validation comparisons adapt phase-labelled solid and dashed lines. The
shared ECharts option builder remains the single owner of axes, smoothing,
zoom, checkpoint markers, and emphasis.

Scalar, multi-Run scalar, and histogram charts retain concise image labels and
visible summaries. Their adjacent Chart Data action opens the shared,
keyboard-contained table dialog with 100-row pagination, source-series identity,
and an explicit warning whenever API metadata says the returned dataset is
incomplete.

## Full Config Sessions

The internal Full Config coordination Module adapts the active shell token into
exactly one editing session: Model Runtime Defaults, Training Runtime Defaults,
Config Snapshot draft, or Config Snapshot edit. It selects identity, schema,
Runtime Defaults, loading state, record library, mutation lifecycle, command
projection, and semantic edit/reset/close/save actions before rendering.

Model Package Inspection, Training, Config Snapshot records, and the Config
Snapshot editor retain separate state ownership. The coordination Module only
arbitrates which Interface is active. A Model Full Config snapshot save opens a
transient editor session and clears it after save or cancellation; snapshot
draft and edit sessions remain open until their owning dialog closes them.

## Runtime Defaults Schema Presentation

One pure schema-presentation Module derives the visible Runtime Defaults tree
from schema fields, current overrides, and search selection. Its Interface
contains contextual field labels and values, section nesting and identifiers,
direct and descendant metrics, controlled disablement, inheritance hints,
grouped model fields, navigation defaults, and searchable field projections.

Full Config navigation, section, field, and search Adapters consume that shared
presentation. They retain only interaction and visual state; they do not
reconstruct schema relationships or edit semantics. Runtime Defaults commands
remain owned by the active Full Config session, so presentation derivation is a
behavior-preserving read model rather than another editing owner.

## Deployment

Set `NEXT_PUBLIC_WORKBENCH_API_URL` in the deployment environment when the browser
must call an API endpoint other than `http://127.0.0.1:9999`, then build with:

```bash
npm run build
```

When `NEXT_PUBLIC_WORKBENCH_API_URL` points at a non-local API, the client locks API
requests to that configured origin by default. Set
`NEXT_PUBLIC_WORKBENCH_API_ALLOWED_ORIGINS` to make the allowlist explicit or to
permit additional API origins. Values are baked into the browser bundle at build
time.

For a hosted bearer-auth setup, build the frontend with public API routing values
only:

```bash
NEXT_PUBLIC_WORKBENCH_API_URL=https://api.example.com
NEXT_PUBLIC_WORKBENCH_API_ALLOWED_ORIGINS='["https://api.example.com"]'
```

Set backend-only values such as `WORKBENCH_API_AUTH_MODE=bearer` and
`WORKBENCH_API_TOKEN=<secret>` in the backend runtime environment, not in this
frontend package.

Deploy the resulting Next.js app with the platform or runtime used by the
project. Keep backend-only secrets out of this frontend package and out of
`NEXT_PUBLIC_*` variables.
