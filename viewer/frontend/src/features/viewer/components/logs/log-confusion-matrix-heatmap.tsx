import { Fragment } from "react";
import { formatNumber } from "@/features/viewer/state/logs/logs-selectors";
import { type ConfusionMatrixHeatmap } from "@/features/viewer/state/logs/log-diagnostics";
import { cn } from "@/lib/utils";

function heatmapColor(value: number, trueClass: number, predictedClass: number) {
  const alpha = Math.max(0.08, Math.min(0.92, value));
  if (trueClass === predictedClass) {
    return `rgba(64, 201, 127, ${alpha})`;
  }
  return `rgba(239, 86, 86, ${alpha})`;
}

function matrixValueMap(cells: ConfusionMatrixHeatmap["cells"]) {
  return new Map(
    cells.map((cell) => [`${cell.trueClass}:${cell.predictedClass}`, cell.value]),
  );
}

export function LogConfusionMatrixHeatmaps({
  heatmaps,
}: {
  heatmaps: ConfusionMatrixHeatmap[];
}) {
  if (heatmaps.length === 0) {
    return null;
  }

  return (
    <section className="grid gap-3">
      <div className="flex min-w-0 items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="text-sm font-bold text-ink">Confusion Matrix</div>
          <div className="text-xs text-ink-faint">
            Latest row-normalized rates from selected TensorBoard runs
          </div>
        </div>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        {heatmaps.map((heatmap) => {
          const values = matrixValueMap(heatmap.cells);
          const classes = Array.from(
            { length: heatmap.classCount },
            (_, index) => index,
          );

          return (
            <div
              key={heatmap.key}
              className="grid min-w-0 gap-3 rounded-[8px] border border-line bg-white/[0.018] p-3"
            >
              <div className="min-w-0">
                <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
                  {heatmap.split}
                </div>
                <div className="truncate text-sm font-semibold text-ink">
                  {heatmap.runLabel}
                </div>
              </div>
              <div className="overflow-x-auto">
                <div
                  className="grid min-w-max gap-1"
                  style={{
                    gridTemplateColumns: `30px repeat(${heatmap.classCount}, 30px)`,
                  }}
                >
                  <div aria-hidden />
                  {classes.map((predictedClass) => (
                    <div
                      key={`predicted-${predictedClass}`}
                      className="grid h-7 place-items-center font-mono text-[10px] text-ink-faint"
                    >
                      {predictedClass}
                    </div>
                  ))}
                  {classes.map((trueClass) => (
                    <Fragment key={`row-${trueClass}`}>
                      <div
                        key={`true-${trueClass}`}
                        className="grid h-7 place-items-center font-mono text-[10px] text-ink-faint"
                      >
                        {trueClass}
                      </div>
                      {classes.map((predictedClass) => {
                        const value = values.get(`${trueClass}:${predictedClass}`) ?? 0;
                        return (
                          <div
                            key={`${trueClass}-${predictedClass}`}
                            className={cn(
                              "grid h-7 w-7 place-items-center rounded-[4px] border",
                              "border-white/10 font-mono text-[9px] text-white shadow-inner",
                            )}
                            style={{
                              backgroundColor: heatmapColor(
                                value,
                                trueClass,
                                predictedClass,
                              ),
                            }}
                            title={[
                              `true ${trueClass}`,
                              `predicted ${predictedClass}`,
                              formatNumber(value),
                            ].join(", ")}
                          >
                            {value >= 0.01 ? Math.round(value * 100) : ""}
                          </div>
                        );
                      })}
                    </Fragment>
                  ))}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
