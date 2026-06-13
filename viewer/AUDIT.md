# Viewer Audit — 2026-06-12

Full audit of `viewer/` (FastAPI backend + Next.js frontend): project structure, API implementation, performance, security, robustness. Evidence is a mix of static review, live runtime probes against the running backend (`127.0.0.1:9999`), and full test/build suite runs.

## Verdict summary

| Area | Verdict |
|------|---------|
| Next.js structure | **Proper.** App-router conventions followed; no structural changes needed. |
| Frontend implementation | **High quality.** 577/577 tests, lint/typecheck/build green, 290 kB first-load JS. |
| Backend API design | **Good layering and contracts**, but **async handlers run blocking work on the event loop** — the single biggest implementation flaw. |
| Security | Solid for a local tool (constant-time bearer auth, path-traversal hardening), one CORS bug breaks a feature. |
| Robustness | Subprocess lifecycle has gaps (no reaping, cancel doesn't wait). |
| Current runtime state | **Broken end-to-end** — not a viewer defect; mid-refactor drift between `emperor/` and `models/` (see Blocker). |

## Empirical evidence (runtime probes + suites)

- `/health` 0.6 ms, `/models` 1.1 ms, `/models/{m}/config-schema` 2.6 ms / 19 kB — cheap endpoints are fast.
- `POST /inspect` → **HTTP 400 for every model/preset**: `LayerConfig.__init__() got an unexpected keyword argument 'shared_halting_flag'`.
- `POST /logs/tags` (1 run): **~0.30 s on every call** — no caching (cold == warm).
- `POST /logs/scalars` (1 run × 10 tags): **3.54 s, 243 kB**. Downsampling to 500 points/series works.
- **Event-loop blocking proven**: `/health` latency inflates 0.6 ms → **267 ms** during a concurrent `/logs/tags`, and → **3,075 ms** during `/logs/scalars`. One logs fetch freezes every other request, including the 1 s training-job poll and 1.5 s monitor-chart polls.
- CORS preflight: `OPTIONS … Access-Control-Request-Method: PATCH` → **400 Disallowed**; `GET` → 200.
- Backend suite: 300 tests, **1 failure + 35 errors — all 36 from the `shared_halting_flag` drift** (68 occurrences in the log). Ruff clean. Viewer's own logic is green.
- Frontend suite: **45 files / 577 tests pass**, eslint `--max-warnings 0` clean, `tsc --noEmit` clean, `next build` green. Bundle: `/` route 179 kB, 290 kB first-load JS (good for React Flow + ECharts + dagre; dialogs/panels are `next/dynamic`).

## Blocker (repo drift, not a viewer defect)

**B0 — `shared_halting_flag` removed from emperor, still passed by every model package.**
The in-progress refactor (commit `20a1861` plus uncommitted edits to `emperor/base/layer/config.py`) dropped the kwarg, but all 10 `models/*/config_builder.py` (and `parametric/*/presets.py`) still pass `shared_halting_flag=False`. Every preset build fails → `/inspect` is dead, 36 backend tests error. The viewer surfaced this correctly; fix belongs in `models/` as part of the refactor, not in `viewer/`.

## Findings

### High

| # | Finding | Evidence |
|---|---------|----------|
| H1 | **Blocking sync work inside `async def` handlers.** Torch model instantiation + graph serialization (`inspector/service.py:88` via `api/v1/routers/inspection.py:26`), TensorBoard `EventAccumulator.Reload()` (`tensorboard_reader.py` via logs/training routers), and per-poll file reads all execute on the event loop. | `/health` 0.6 ms → 3,075 ms during one scalars fetch. Fix: offload via `asyncio.to_thread` at the router/service boundary. |
| H2 | **CORS `allow_methods` missing `PATCH`** (`middleware.py:16`) while `PATCH /config-snapshots/{id}` exists (`api/v1/routers/config_snapshots.py:61`) and the UI calls it (`full-config-dialog.tsx` → `use-config-snapshots.ts` → `renameConfigSnapshot`). | Preflight probe returns 400. Snapshot rename is broken from any browser. |
| H3 | **`cancel_job` sends `terminate()` without `wait()`** (`training_jobs.py:418-427`). Child never reaped (zombie until manager exit); if the worker ignores SIGTERM, the job is marked `cancelled` while still running. | Code read. Fix: `terminate()` → `wait(timeout)` → `kill()` escalation. |

### Medium

| # | Finding | Evidence |
|---|---------|----------|
| M1 | **No TensorBoard read caching.** Every tags/scalars/monitor-data call rebuilds `EventAccumulator` from disk (~0.3 s). Monitor charts poll per node at 1.5 s; several open nodes plus the 1 s job poll can saturate the (currently blocking) loop. | Cold == warm at 0.30 s; `use-monitor-chart-queries.ts:86`. |
| M2 | **No concurrency limit on training jobs** — every `POST /training/jobs` spawns a subprocess immediately; N requests = N concurrent trainings. | `TrainingJobManager` has no queue/semaphore. |
| M3 | **Worker processes orphaned across API restart.** In-memory `_processes` is lost; no PID-based reattach or reaping. Mitigation already present: `_refresh` flips handle-less live jobs to `"unknown"` (`training_jobs.py:557-559`) — recon claims of jobs stuck "running" forever were wrong. | Code read. |
| M4 | **`viewer/README.md:15` links `FEATURE_ANALYSIS.md`, which does not exist.** | File absent. |
| M5 | **No request body size limit** on POST endpoints. Low practical risk for a loopback dev tool; matters if hosted with bearer auth. | `middleware.py` adds only CORS. |
| M6 | **Dead API surface:** `GET /training/jobs/{id}/events` + `fetchTrainingJobEvents` (`src/lib/api/training-jobs.ts:375`) have zero non-test callers. | grep. Keep-or-remove decision; contract tests still maintain it. |

### Low

- L1 — `training-panel.tsx:568`: list key suffixes index over a `slice(-12)` window, so keys shift as additions append → needless re-renders of tiny chips. Cosmetic.
- L2 — `training-search-axis-list.tsx` sits at components root; belongs in `training/`.
- L3 — `viewer/backend/routes/*` deprecated shims still present (README documents removal criteria — fine, just tracked debt).
- L4 — Contract is hand-maintained zod ↔ pydantic. The 1,619-line `test_api_contract.py` (30 endpoints, required-field parity) makes drift unlikely but every endpoint change must touch both sides; OpenAPI-driven codegen is a future option, not a need.

## What is done well

- **Next.js structure**: server `layout.tsx`/`page.tsx` with `"use client"` only at `providers.tsx`/`error.tsx`; `next/font`; `reactStrictMode`; `optimizePackageImports`; route typegen in CI script.
- **API discipline**: `api-boundary.test.ts` AST-scans the whole source tree to forbid `fetch`/URL/env literals outside `src/lib/api/*`; every response zod-validated with detailed issue formatting.
- **Polling done right (frontend)**: refetch intervals terminate at terminal job status; queries gated by `enabled`; query keys normalized/sorted.
- **Bundle hygiene**: ECharts registered per-component from `echarts/core`; heavy dialogs via `next/dynamic`; React Flow in read-only mode; 290 kB first load.
- **Backend layering**: app factory + DI container, thin deprecated shims clearly marked, settings centralized (`core/config.py`) with env-prefix and auth-mode validation.
- **Security basics**: `secrets.compare_digest` bearer auth, `safe_child_path` traversal rejection with dedicated tests, symlink rejection on deletes, subprocess launched as arg-list (no shell).
- **Test depth**: 300 backend + 577 frontend tests; dependency-direction test enforcing `emperor`/`models` must not import `viewer`; the suite caught the live drift exactly as designed.

## Prioritized recommendations

Fixed in this pass:
- **H2** — `54e907b`: PATCH added to CORS `allow_methods`; preflight test parameterized over PATCH. Verified live (preflight 400 → 200).
- **H1 (inspect)** — `a076913`: inspect handler converted to sync so FastAPI dispatches it to the worker threadpool; guard test (`EXPECTED_THREADPOOL_ROUTE_PAIRS` in `test_app_factory.py`) fails if a listed route becomes `async def` again.
- **H1 (logs reads)** — `e1740cc`: the six read-only logs handlers (runs, experiments, tags, scalars, parameter-status, monitor-data) converted to threadpool dispatch. Verified live: `/health` during a 3.3 s scalars fetch dropped from 3,075 ms to 5–26 ms. Logs **delete** endpoints and **training** routes deliberately stay async: they share mutable `TrainingJobManager` state, so threadpool conversion needs the manager lock (backlog item below) first.
- **H3** — implemented and tested (terminate → `wait(timeout=5s)` → `kill()` escalation in `cancel_job`, `ProcessHandle` protocol extended with `wait`/`kill`, reap + kill-escalation tests added). **Left uncommitted**: `training_jobs.py`/`test_training_jobs.py` carry unrelated uncommitted cluster-growth WIP, so the fix cannot be committed without sweeping that in.
- **M4** — `f2d2c20`: dead README reference removed.

Deferred — recommended backlog order:
1. M1: cache `EventAccumulator` per run dir keyed by latest event-file mtime/size; optionally a batched monitor-data endpoint to collapse per-node polling.
2. Add a `threading.RLock` to `TrainingJobManager`, then convert the training routes and logs delete routes to threadpool dispatch too (finishes H1).
3. M3: persist worker PIDs and reattach/reap on startup.
4. M2: cap concurrent jobs (simple semaphore + `queued` status).
4. M6: remove or wire the events endpoint; adjust contract tests.
5. M5: body-size guard middleware if the backend is ever hosted non-loopback.
6. L1/L2: trivial frontend cleanups; L4: consider OpenAPI codegen only if endpoint churn grows.
7. B0 (outside viewer): update `models/*/config_builder.py` and `parametric/*/presets.py` to stop passing `shared_halting_flag` once the emperor refactor lands.
