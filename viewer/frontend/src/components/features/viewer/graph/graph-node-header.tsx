import { type ReactNode } from "react";
import { type GraphDetailMode } from "@/lib/graph";
import {
  GraphNodeBadgeRow,
  GraphNodeInlineBadges,
  GraphNodeSimpleBadges,
} from "@/components/features/viewer/graph/graph-node-badges";

export function GraphNodeHeader({
  nodeId,
  label,
  subtitle,
  graphDetailMode,
  parameterCount,
  childCount,
  simpleParameterText,
  simpleDimsText,
  expansionButton,
  parameterIndicators,
  monitorButton,
}: {
  nodeId: string;
  label: string;
  subtitle: string;
  graphDetailMode: GraphDetailMode;
  parameterCount: number;
  childCount: number;
  simpleParameterText?: string;
  simpleDimsText?: string;
  expansionButton: ReactNode;
  parameterIndicators: ReactNode;
  monitorButton: ReactNode;
}) {
  const isSimpleMode = graphDetailMode === "simple";
  const isBasicMode = graphDetailMode === "basic";
  const hasGraphBadges = parameterCount > 0 || childCount > 0;

  if (isSimpleMode) {
    return (
      <div className="flex shrink-0 items-center gap-2">
        {expansionButton}
        <div className="min-w-0 flex-1">
          <div className="flex min-w-0 flex-nowrap items-center gap-2">
            <div className="min-w-0 flex-1 truncate text-[18px] font-bold leading-6 text-ink">
              {label}
            </div>
            <GraphNodeSimpleBadges
              parameterCount={parameterCount}
              parameterText={simpleParameterText}
              dimsText={simpleDimsText}
            />
          </div>
        </div>
        {parameterIndicators}
        {monitorButton}
      </div>
    );
  }

  return (
    <div className="shrink-0">
      <div
        className="flex min-w-0 items-start gap-2"
        data-testid={`graph-node-title-row-${nodeId}`}
      >
        {expansionButton}
        <div className="flex min-w-0 flex-1 flex-nowrap items-center gap-1.5">
          <div className="min-w-0 flex-1 truncate text-[18px] font-bold leading-6 text-ink">
            {label}
          </div>
          {isBasicMode && (
            <GraphNodeInlineBadges
              parameterCount={parameterCount}
              childCount={childCount}
            />
          )}
        </div>
        {parameterIndicators}
        {monitorButton}
      </div>
      {!isBasicMode && hasGraphBadges && (
        <GraphNodeBadgeRow
          nodeId={nodeId}
          parameterCount={parameterCount}
          childCount={childCount}
        />
      )}
      <div className="mt-1.5 truncate font-mono text-[13px] leading-5 text-ink-dim">
        {subtitle}
      </div>
    </div>
  );
}
