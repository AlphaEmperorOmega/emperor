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

`npm run typecheck` writes generated Next.js types under `.next/types`. Run it
sequentially with `npm run build`; running both against the same `.next`
directory at the same time can race while those generated files are refreshed.

## Testing

Unit and component tests live under `tests/` and alongside source files with
`.test.ts` or `.test.tsx` names. Run them with:

```bash
npm run test
```

There is no end-to-end test runner configured in this package today. Add the
runner, scripts, and docs in the same change if browser-level coverage is
introduced later.

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
