# Frontend Chart Migration — Hand-rolled SVG → ECharts

Living migration doc. Tracks the replacement of the viewer's hand-rolled SVG plots with ECharts.

## Context

The viewer frontend rendered all data plots as **hand-rolled SVG** with a custom linear-scale
helper, across two independent chart subsystems:

- **Logs workspace** — `log-scalar-chart.tsx` draws a multi-run scalar overlay per tag.
- **Monitor modal** — `monitor-charts.tsx` draws `ScalarChart`, `MultiRunScalarChart`, and a
  bar-based `HistogramChart` (plus `MonitorImage`, which is an image, not a chart).

These gave static lines with no hover tooltips, no zoom/pan, no smoothing, no synced crosshair,
no log scale. This migration replaces the SVG renderers with **ECharts** (`echarts` +
`echarts-for-react`) to get that interactivity, while keeping the data layer, query hooks,
grouping logic, and design system untouched.

**Hard data constraint:** the backend serves only **scalars**, **per-step histogram buckets**,
and **images**. There is no hparams, embedding, or distribution-over-time data. TensorBoard's
parallel-coordinates, 3D embedding projector, and ridgeline distribution views are therefore
**out of scope** — they require new backend endpoints (see "Future / backend-gated").

## Decisions

- **Scope:** migrate Logs + Monitor (all current plots).
- **Integration:** `echarts-for-react` (via its tree-shakeable core entrypoint).
- **Interactions:** tooltip + synced crosshair, zoom/pan + reset, EMA smoothing, log-scale Y +
  step/wall-time X axis.

## Architecture

One shared chart primitive plus pure, testable option-builders. Component files map our data
shapes to an ECharts `option`; the primitive renders it. This mirrors the old `chart-scale.ts`
math vs `*-chart.tsx` rendering split and keeps logic unit-testable without a canvas.

### New files

| File | Purpose |
|---|---|
| `src/components/features/viewer/charts/echart.tsx` | `"use client"` wrapper around `ReactEChartsCore` (from `echarts-for-react/lib/core`). Props: `option`, `group?`, `className`, `style?`, `onEvents?`. Handles resize (ResizeObserver), `notMerge`, dispose-on-unmount, and `echarts.connect(group)` for synced crosshair. |
| `src/lib/echarts/register.ts` | Registers only the used modules (`LineChart`, `BarChart`, `CustomChart`, `GridComponent`, `TooltipComponent`, `AxisPointerComponent`, `DataZoomComponent`, `LegendComponent`, `MarkLineComponent`, `CanvasRenderer`). |
| `src/lib/echarts/theme.ts` | `emperorDarkTheme` + `registerEmperorTheme()` matching design tokens; mono font resolved at runtime from `--font-mono`. |
| `src/lib/echarts/scalar-options.ts` | `buildScalarLineOption(lines, opts)` — step/wall X, log/linear Y, dataZoom, raw+smoothed pairs, de-duped axis tooltip. |
| `src/lib/echarts/smoothing.ts` | `applyEmaSmoothing(points, weight)` — TensorBoard EMA with debias. |
| `src/lib/echarts/histogram-options.ts` | `buildHistogramBarOption(histogram, { maxCount })` — true histogram via `custom` series (respects bucket widths). |
| `*.test.ts` (smoothing, scalar-options, histogram-options) | 20 unit tests on the pure builders. Replaces `chart-scale.test.ts`. |

### Reused (not recreated)

- `ChartFrame`, `ErrorPanel`, `ViewModeButton`, `SegmentedControl`, `Badge`, `Button`.
- Palettes: `scalarSeriesColors` + `multiRunLineColors` (`lib/charts.ts`).
- Formatters: `formatNumber`, `formatRunLabel`, `formatRunDisplayName`.
- Data layer untouched: `api.ts`, `use-log-queries.ts`, `monitor/use-monitor-chart-queries.ts`,
  `monitor/grouping.ts`, `types/monitor.ts`, `use-logs-workspace-state.ts`.

## File-by-file changes

### Logs subsystem

- `logs/log-scalar-chart.tsx` — renders `<EChart>` via `buildScalarLineOption`; keeps the section
  header, value badge, and clickable run legend, plus `role="img"` label for tests.
- `logs/logs-chart-panel.tsx` — adds smoothing slider, step·time and lin·log toggles; all charts
  share group `logs-scalars` for synced crosshair + dataZoom.
- `logs-workspace.tsx` — threads smoothing/yScale/xMode state into the panel.

### Monitor subsystem

- `monitor/monitor-charts.tsx` — `ScalarChart`, `MultiRunScalarChart`, `HistogramChart` render
  `<EChart>` inside the existing `ChartFrame` (titles, badges, footers, legends, aria-labels
  preserved). Scalar charts share group `monitor-scalars` (histograms excluded — value x-axis).

### Cleanup

- Removed `lib/chart-scale.ts` + `lib/chart-scale.test.ts` (only its own test imported it).

## Theme

`emperorDarkTheme` maps to `app/globals.css`: ink `#ecebf5` / dim `#9c9cb6` / faint `#62627c`,
axes `rgba(255,255,255,0.07)` / `0.045`, tooltip panel `#0c0c15`, violet accent `#a78bfa`, mono
font, palettes `scalarSeriesColors` / `multiRunLineColors`.

## Interactions

- Tooltip + synced crosshair: `tooltip.trigger:'axis'`, `axisPointer.type:'cross'`, group connect.
- Zoom/pan: `dataZoom` inside + slider (logs panel).
- EMA smoothing: raw line faint behind bold smoothed line.
- Log/linear Y, step/wall-time X toggles.

## Testing

- `vitest.setup.ts` stubs `HTMLCanvasElement.prototype.getContext` (jsdom has no canvas).
- Pure builders covered by unit tests; integration tests mount the real `<EChart>` harmlessly.

## Todo

- [x] Deps: add `echarts` + `echarts-for-react`; create `lib/echarts/register.ts`
- [x] Theme: add `lib/echarts/theme.ts` + `registerEmperorTheme`
- [x] Primitive: add `charts/echart.tsx` wrapper (resize, dispose, `connect` group)
- [x] Test setup: add canvas `getContext` stub to `vitest.setup.ts`
- [x] Builders: add `scalar-options.ts` + `smoothing.ts` + unit tests; export
  `scalarSeriesColors` from `lib/charts.ts`
- [x] Logs charts: migrate `log-scalar-chart.tsx` to `<EChart>`
- [x] Logs panel: smoothing / log-scale / x-mode controls + synced group `logs-scalars`
- [x] Logs state: thread view-option state through `logs-workspace.tsx`
- [x] Histogram builder: `histogram-options.ts` + tests (custom series; `CustomChart` registered)
- [x] Monitor charts: migrate scalar/multi-run/histogram in `monitor-charts.tsx`
- [x] Monitor modal: synced group `monitor-scalars` (scalars only)
- [x] Cleanup: remove `chart-scale.ts` + test

## Status: complete

All tasks done and verified:

- `tsc --noEmit` clean; `eslint . --max-warnings 0` clean
- vitest: 437 passed incl. 20 new builder tests; the 30 `viewer-app.test.tsx` failures are
  pre-existing and unrelated (confirmed by stashing the change)
- `next build` succeeds — client-only ECharts mounts via `dynamic(ssr:false)`, static prerender
  passes

## Future / backend-gated (NOT in this migration)

Each needs new backend data first:

- **Distribution ridgeline** — per-tag histogram series across steps; offset areas / `themeRiver`.
- **HParams** parallel-coordinates + scatter matrix — needs an hparams endpoint.
- **Embedding projector** (3D scatter via `echarts-gl`) — needs an embeddings endpoint + PCA/UMAP.
