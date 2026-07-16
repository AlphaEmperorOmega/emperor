import { GraphChip } from "@/features/workbench/components/graph/graph-chip";
import { graphParameterActivityStatusClassNames } from "@/features/workbench/components/graph/graph-parameter-indicators";
import {
  type GraphParameterActivity,
  type GraphParameterActivityChannel,
} from "@/lib/graph";
import { graphCardGeometry } from "@/lib/graph/constants";
import { cn } from "@/lib/utils";

type ParameterShapeEntry = {
  key: string;
  label: string;
  shape: string;
};

function parameterShapeChannel(
  key: string,
  parameterActivity: GraphParameterActivity | undefined,
): {
  channel?: GraphParameterActivityChannel;
  activityLabel?: "weights" | "bias";
} {
  if (!parameterActivity) {
    return {};
  }
  if (key === "weightShape") {
    return { channel: parameterActivity.weights, activityLabel: "weights" };
  }
  if (key === "biasShape" && parameterActivity.bias) {
    return { channel: parameterActivity.bias, activityLabel: "bias" };
  }
  return {};
}

function shapeDisplayLabel(entry: ParameterShapeEntry) {
  return entry.key === "biasShape" ? "B" : entry.label.toUpperCase();
}

export function GraphNodeParameterShapes({
  nodeId,
  entries,
  parameterActivity,
  dimsText,
}: {
  nodeId: string;
  entries: ParameterShapeEntry[];
  parameterActivity?: GraphParameterActivity;
  dimsText?: string;
}) {
  return (
    <div
      className={cn(
        "grid shrink-0 items-center",
        dimsText && "grid-cols-[auto_minmax(0,1fr)_minmax(0,1fr)]",
        !dimsText && "grid-cols-2",
      )}
      style={{
        columnGap: graphCardGeometry.parameterShapes.rowGap,
        marginTop: graphCardGeometry.parameterShapes.marginBlockStart,
        rowGap: graphCardGeometry.parameterShapes.rowGap,
      }}
      data-testid={`parameter-shapes-${nodeId}`}
    >
      {dimsText && (
        <GraphChip
          aria-label={`input/output: ${dimsText}`}
          data-testid={`parameter-shape-dims-${nodeId}`}
          title={`input/output: ${dimsText}`}
          tone="default"
          className="inline-flex w-fit max-w-full items-center px-2 type-label"
          style={{ height: graphCardGeometry.parameterShapes.rowHeight }}
        >
          <span className="truncate font-mono text-ink">{dimsText}</span>
        </GraphChip>
      )}
      {entries.map((entry) => {
        const label = shapeDisplayLabel(entry);
        const { channel, activityLabel } = parameterShapeChannel(
          entry.key,
          parameterActivity,
        );
        const activityText =
          channel && activityLabel
            ? `${activityLabel} activity ${channel.status}`
            : undefined;
        const accessibleLabel = activityText
          ? `${label} shape ${entry.shape}, ${activityText}`
          : `${label} shape ${entry.shape}`;

        return (
          <GraphChip
            key={entry.key}
            aria-label={accessibleLabel}
            data-testid={`parameter-shape-${nodeId}-${entry.key}`}
            title={
              activityText
                ? `${label} shape: ${entry.shape} (${activityText})`
                : `${label} shape: ${entry.shape}`
            }
            tone={channel ? "none" : "violet"}
            className={cn(
              "grid min-w-0 grid-cols-[18px_minmax(0,1fr)] items-center gap-1.5 px-2 type-label",
              channel
                ? graphParameterActivityStatusClassNames[channel.status]
                : "border-violet/25 bg-violet/15",
            )}
            style={{ height: graphCardGeometry.parameterShapes.rowHeight }}
          >
            <span
              className={cn(
                "truncate font-semibold uppercase",
                channel ? "text-current" : "text-violet-muted",
              )}
            >
              {label}
            </span>
            <span
              className={cn(
                "truncate font-mono",
                channel ? "text-current" : "text-ink",
              )}
            >
              {entry.shape}
            </span>
          </GraphChip>
        );
      })}
    </div>
  );
}
