import { type GraphDetailMode } from "@/lib/graph";
import {
  GraphNodeSimpleBadges,
} from "@/features/workbench/components/graph/graph-node-badges";
import { graphCardGeometry } from "@/lib/graph/constants";

export function GraphNodeHeader({
  nodeId,
  label,
  subtitle,
  graphDetailMode,
  parameterCount,
  parameterSizeBytes,
  simpleDimsText,
}: {
  nodeId: string;
  label: string;
  subtitle: string;
  graphDetailMode: GraphDetailMode;
  parameterCount: number;
  parameterSizeBytes: number;
  simpleDimsText?: string;
}) {
  const isSimpleMode = graphDetailMode === "simple";
  const hasSimpleDimensions = Boolean(simpleDimsText);

  if (isSimpleMode) {
    return (
      <div className="shrink-0">
        <div
          className="min-w-0 truncate type-heading font-bold leading-6 text-ink"
          data-testid={`graph-node-title-row-${nodeId}`}
        >
          {label}
        </div>
        {hasSimpleDimensions && (
          <div
            className="mt-1 flex h-5 min-w-0 items-center gap-1.5 overflow-hidden"
            data-testid={`graph-node-simple-metrics-${nodeId}`}
          >
            <GraphNodeSimpleBadges
              parameterCount={parameterCount}
              parameterText={undefined}
              parameterSizeBytes={parameterSizeBytes}
              modelSizeText={undefined}
              dimsText={simpleDimsText}
            />
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="shrink-0">
      <div
        className="flex min-w-0 items-start gap-2"
        data-testid={`graph-node-title-row-${nodeId}`}
      >
        <div
          className="min-w-0 flex-1 truncate type-heading font-bold text-ink"
          style={{ lineHeight: `${graphCardGeometry.titleLineHeight}px` }}
        >
          {label}
        </div>
      </div>
      <div
        className="truncate font-mono type-compact text-ink-dim"
        style={{
          height: graphCardGeometry.subtitle.height,
          lineHeight: `${graphCardGeometry.subtitle.height}px`,
          marginTop: graphCardGeometry.subtitle.marginBlockStart,
        }}
      >
        {subtitle}
      </div>
    </div>
  );
}
