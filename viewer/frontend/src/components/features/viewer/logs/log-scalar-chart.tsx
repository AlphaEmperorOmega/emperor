import { Badge } from "@/components/ui/badge";
import { type LogRun, type LogScalarSeries } from "@/lib/api";
import { buildChartPath, buildLinearScale, formatChartDomain } from "@/lib/chart-scale";
import { formatNumber, formatRunLabel } from "@/lib/logs/helpers";

const SERIES_COLORS = [
  "#7c6dff",
  "#22d3ee",
  "#f59e0b",
  "#34d399",
  "#f472b6",
  "#a78bfa",
  "#fb7185",
  "#60a5fa",
  "#facc15",
  "#2dd4bf",
];

export function LogScalarChart({
  tag,
  series,
  runsById,
  runOrder,
  onSelectRun,
}: {
  tag: string;
  series: LogScalarSeries[];
  runsById: Map<string, LogRun>;
  runOrder: string[];
  onSelectRun: (runId: string) => void;
}) {
  const width = 760;
  const height = 188;
  const paddingX = 34;
  const paddingY = 22;
  const allPoints = series.flatMap((entry) => entry.points);
  const scale = buildLinearScale(allPoints, {
    width,
    height,
    padding: { x: paddingX, y: paddingY },
    pathKind: "path",
  });
  const { minStep, maxStep } = scale.domain;
  const { minLabel, maxLabel } = formatChartDomain(scale.domain);

  return (
    <section className="edge grid gap-3 rounded-card p-4">
      <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="truncate text-sm font-bold text-ink">{tag}</h2>
          <div className="mt-0.5 font-mono text-xs text-ink-faint">
            {series.length} lines · step {minStep} to {maxStep}
          </div>
        </div>
        <Badge>{minLabel} to {maxLabel}</Badge>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="h-52 w-full overflow-visible text-violet"
        role="img"
        aria-label={`${tag} scalar chart`}
      >
        <line
          x1={paddingX}
          y1={height - paddingY}
          x2={width - paddingX}
          y2={height - paddingY}
          stroke="rgba(255,255,255,0.12)"
          strokeWidth="1"
        />
        <line
          x1={paddingX}
          y1={paddingY}
          x2={paddingX}
          y2={height - paddingY}
          stroke="rgba(255,255,255,0.12)"
          strokeWidth="1"
        />
        <text x={paddingX} y={height - 4} fill="rgba(230,232,255,0.45)" fontSize="10">
          {minStep}
        </text>
        <text
          x={width - paddingX}
          y={height - 4}
          fill="rgba(230,232,255,0.45)"
          fontSize="10"
          textAnchor="end"
        >
          {maxStep}
        </text>
        <text x="4" y={paddingY + 4} fill="rgba(230,232,255,0.45)" fontSize="10">
          {maxLabel}
        </text>
        <text x="4" y={height - paddingY} fill="rgba(230,232,255,0.45)" fontSize="10">
          {minLabel}
        </text>
        {series.map((entry) => {
          const color =
            SERIES_COLORS[Math.max(runOrder.indexOf(entry.runId), 0) % SERIES_COLORS.length];
          if (entry.points.length === 1) {
            const point = entry.points[0];
            const { x, y } = scale.coordinate(point);
            return (
              <circle
                key={entry.runId}
                cx={x}
                cy={y}
                r="3.5"
                fill={color}
                aria-label={runsById.get(entry.runId)?.runName}
              />
            );
          }
          return (
            <path
              key={entry.runId}
              d={buildChartPath(entry.points, scale)}
              fill="none"
              stroke={color}
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
              opacity="0.9"
            />
          );
        })}
      </svg>

      <div className="grid gap-1.5 sm:grid-cols-2 xl:grid-cols-3">
        {series.map((entry) => {
          const run = runsById.get(entry.runId);
          if (!run) {
            return null;
          }
          const color =
            SERIES_COLORS[Math.max(runOrder.indexOf(entry.runId), 0) % SERIES_COLORS.length];
          const latest = entry.points.at(-1);
          return (
            <button
              key={entry.runId}
              type="button"
              className="grid min-w-0 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 rounded-[9px] border border-line-soft bg-black/20 px-2 py-1.5 text-left text-xs transition hover:border-line hover:bg-white/[0.035] focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              onClick={() => onSelectRun(entry.runId)}
            >
              <span
                className="h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: color }}
                aria-hidden
              />
              <span className="truncate text-ink-dim">{formatRunLabel(run)}</span>
              {latest && (
                <span className="font-mono text-ink-faint">{formatNumber(latest.value)}</span>
              )}
            </button>
          );
        })}
      </div>
    </section>
  );
}
