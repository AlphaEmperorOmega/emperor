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
