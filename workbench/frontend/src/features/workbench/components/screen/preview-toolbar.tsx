import { RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { WorkbenchPrimaryToolbar } from "@/features/workbench/components/shared/workbench-primary-toolbar";
import { ViewModeButton } from "@/features/workbench/components/view-mode-button";
import { useGraphView } from "@/features/workbench/providers/workbench-providers";

export function PreviewToolbar() {
  const {
    graph,
    graphDetailMode,
    graphScope,
    expandedGraphNodeIds,
    setGraphDetailMode: onGraphDetailModeChange,
    setGraphScope: onGraphScopeChange,
    collapseGraphNodes: onCollapseGraphNodes,
  } = useGraphView();
  const activeGraphAvailable = Boolean(graph);
  const activeGraphLabel =
    graph ? `${graph.model} / ${graph.preset}` : "Waiting for preview data";
  return (
    <WorkbenchPrimaryToolbar title="Preview" detail={activeGraphLabel}>
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
      <SegmentedControl aria-label="Graph scope">
        <ViewModeButton
          active={graphScope === "opened"}
          onClick={() => onGraphScopeChange("opened")}
        >
          Opened
        </ViewModeButton>
        <ViewModeButton
          active={graphScope === "entire"}
          onClick={() => onGraphScopeChange("entire")}
        >
          Entire
        </ViewModeButton>
      </SegmentedControl>
      <Button
        variant="secondary"
        aria-label="Collapse all graph nodes"
        title="Collapse all graph nodes"
        onClick={onCollapseGraphNodes}
        disabled={!activeGraphAvailable || expandedGraphNodeIds.size === 0}
        className="h-touch w-touch shrink-0 px-0 text-xs sm:w-auto sm:px-3 md:h-control"
      >
        <RotateCcw className="h-3.5 w-3.5" aria-hidden />
        <span className="hidden sm:inline">Collapse All</span>
      </Button>
    </WorkbenchPrimaryToolbar>
  );
}
