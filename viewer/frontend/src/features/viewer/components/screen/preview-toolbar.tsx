import { RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { ViewModeButton } from "@/features/viewer/components/view-mode-button";
import { useGraphView } from "@/features/viewer/providers/viewer-providers";

export function PreviewToolbar() {
  const {
    graph,
    graphKind,
    setGraphKind: onGraphKindChange,
    operationGraph,
    previewVisualizationMode,
    graphDetailMode,
    graphScope,
    operationGraphScope,
    expandedGraphNodeIds,
    expandedOperationGroupIds,
    setPreviewVisualizationMode: onPreviewVisualizationModeChange,
    setGraphDetailMode: onGraphDetailModeChange,
    setGraphScope: onGraphScopeChange,
    setOperationGraphScope: onOperationGraphScopeChange,
    collapseGraphNodes: onCollapseGraphNodes,
    collapseOperationGraphNodes: onCollapseOperationGraphNodes,
  } = useGraphView();
  const isGraphMode = previewVisualizationMode === "graph";
  const isOperationGraph = isGraphMode && graphKind === "operation";
  const activeGraphScope = isOperationGraph ? operationGraphScope : graphScope;
  const activeExpandedCount = isOperationGraph
    ? expandedOperationGroupIds.size
    : expandedGraphNodeIds.size;
  const activeGraphAvailable = isOperationGraph
    ? operationGraph?.status === "ok"
    : Boolean(graph);
  const activeGraphLabel =
    isOperationGraph && operationGraph
      ? `${operationGraph.model} / ${operationGraph.preset}`
      : graph
        ? `${graph.model} / ${graph.preset}`
        : "Waiting for preview data";
  const setActiveGraphScope = isOperationGraph
    ? onOperationGraphScopeChange
    : onGraphScopeChange;
  const collapseActiveGraphNodes = isOperationGraph
    ? onCollapseOperationGraphNodes
    : onCollapseGraphNodes;
  return (
    <div className="flex items-center justify-between gap-3 border-b border-line bg-[rgba(8,8,14,0.35)] px-[22px] backdrop-blur">
      <div className="hidden min-w-0 sm:block">
        <div className="text-xs font-bold uppercase tracking-[0.1em] text-ink-dim">
          Preview
        </div>
        <div className="mt-0.5 truncate font-mono text-xs text-ink-dim">
          {activeGraphLabel}
        </div>
      </div>
      <div className="flex min-w-0 flex-nowrap items-center justify-end gap-2 overflow-x-auto">
        <SegmentedControl aria-label="Preview visualization">
          <ViewModeButton
            active={previewVisualizationMode === "graph"}
            onClick={() => onPreviewVisualizationModeChange("graph")}
          >
            Graph
          </ViewModeButton>
          <ViewModeButton
            active={previewVisualizationMode === "parameters"}
            onClick={() => onPreviewVisualizationModeChange("parameters")}
          >
            Parameters
          </ViewModeButton>
        </SegmentedControl>
        {!isOperationGraph && (
          <SegmentedControl aria-label="Graph detail">
            <ViewModeButton
              active={graphDetailMode === "simple"}
              onClick={() => onGraphDetailModeChange("simple")}
            >
              Simple
            </ViewModeButton>
            <ViewModeButton
              active={graphDetailMode === "basic"}
              onClick={() => onGraphDetailModeChange("basic")}
            >
              Basic
            </ViewModeButton>
            <ViewModeButton
              active={graphDetailMode === "full"}
              onClick={() => onGraphDetailModeChange("full")}
            >
              Full
            </ViewModeButton>
          </SegmentedControl>
        )}
        {isGraphMode && (
          <>
            <SegmentedControl aria-label="Graph kind">
              <ViewModeButton
                active={graphKind === "module"}
                onClick={() => onGraphKindChange("module")}
              >
                Module
              </ViewModeButton>
              <ViewModeButton
                active={graphKind === "operation"}
                onClick={() => onGraphKindChange("operation")}
              >
                Operations
              </ViewModeButton>
            </SegmentedControl>
            <SegmentedControl aria-label="Graph scope">
              <ViewModeButton
                active={activeGraphScope === "opened"}
                onClick={() => setActiveGraphScope("opened")}
              >
                Opened
              </ViewModeButton>
              <ViewModeButton
                active={activeGraphScope === "entire"}
                onClick={() => setActiveGraphScope("entire")}
              >
                Entire
              </ViewModeButton>
            </SegmentedControl>
            <Button
              variant="secondary"
              aria-label={
                isOperationGraph
                  ? "Collapse all operation groups"
                  : "Collapse all graph nodes"
              }
              title={
                isOperationGraph
                  ? "Collapse all operation groups"
                  : "Collapse all graph nodes"
              }
              onClick={collapseActiveGraphNodes}
              disabled={!activeGraphAvailable || activeExpandedCount === 0}
              className="h-[34px] w-[34px] shrink-0 px-0 text-[13px] sm:w-auto sm:px-3"
            >
              <RotateCcw className="h-3.5 w-3.5" aria-hidden />
              <span className="hidden sm:inline">Collapse All</span>
            </Button>
          </>
        )}
      </div>
    </div>
  );
}
