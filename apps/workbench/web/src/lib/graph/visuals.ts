import { MarkerType } from "@xyflow/react";
import { workbenchVisualTokens } from "@/lib/visual-tokens";

/** Creates independent React Flow style records from shared visual facts. */
export function workbenchGraphEdgeVisual() {
  return {
    markerEnd: {
      type: MarkerType.ArrowClosed,
      color: workbenchVisualTokens.gradientMiddle,
    },
    style: { stroke: workbenchVisualTokens.violetDeep, strokeWidth: 2 },
  } as const;
}
