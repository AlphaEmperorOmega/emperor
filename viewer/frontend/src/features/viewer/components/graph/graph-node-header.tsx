import { type GraphDetailMode } from "@/lib/graph";
import {
  GraphNodeSimpleBadges,
} from "@/features/viewer/components/graph/graph-node-badges";

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
  const hasSimpleMetrics = Boolean(simpleDimsText);

  if (isSimpleMode) {
    return (
      <div className="shrink-0">
        <div
          className="min-w-0 truncate text-[18px] font-bold leading-6 text-ink"
          data-testid={`graph-node-title-row-${nodeId}`}
        >
          {label}
        </div>
        {hasSimpleMetrics && (
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
        <div className="min-w-0 flex-1 truncate text-[18px] font-bold leading-6 text-ink">
          {label}
        </div>
      </div>
      <div className="mt-1.5 truncate font-mono text-[13px] leading-5 text-ink-dim">
        {subtitle}
      </div>
    </div>
  );
}
