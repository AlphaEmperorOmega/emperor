import dynamic from "next/dynamic";
import { Activity, Clock3, GitBranch, RefreshCw } from "lucide-react";
import { EmptyState } from "@/features/workbench/components/empty-state";
import { ErrorPanel } from "@/features/workbench/components/error-panel";
import { workbenchStatusCopy } from "@/features/workbench/components/shared/status-copy";
import { StatusPill } from "@/features/workbench/components/status-pill";
import {
  useGraphView,
  useHistoricalRuns,
  useModelPackageInspection,
  useWorkbenchConnection,
} from "@/features/workbench/providers/workbench-providers";
import { isWorkbenchProtectedAccessReady } from "@/features/workbench/providers/workbench-connection-provider";
import { useParameterActivityMinimapState } from "@/features/workbench/state/graph-monitor/use-parameter-activity-minimap-state";
import { errorMessage } from "@/lib/utils";

// The ReactFlow canvas (@xyflow/react + graph node views) is the largest piece
// of client JS in this view, but it only renders after a model is inspected.
// Loading it dynamically keeps it out of the initial bundle; the panel shell and
// empty state below stay static so the default view paints immediately.
const GraphPreviewPanel = dynamic(
  () =>
    import("@/features/workbench/components/screen/graph-canvas").then(
      (module) => module.GraphPreviewPanel,
    ),
  { ssr: false },
);

function isNeuronModelType(modelType: string) {
  return modelType === "neuron" || modelType === "neurons";
}

export function PreviewPanel() {
  const workbenchConnection = useWorkbenchConnection();
  const { target, browser } = useModelPackageInspection();
  const selectedModelType = target.modelPackage.modelType;
  const selectedTargetMode =
    target.kind === "historical-run" ? "experiment" : target.kind;
  const selectedExperimentRunId =
    target.kind === "historical-run" ? target.run.runId : "";
  const {
    selectedLogRunId,
    selectedLogRun,
    selectedLogRunMonitorEligibility,
  } = useHistoricalRuns();
  const {
    graph,
    graphForDetail,
    nodes,
    edges,
    selectedNodeId,
    previewInspection,
    isParameterStatusLoading,
    isParameterStatusPathMismatch,
    setSelectedNodeId: onSelectNode,
    revealGraphNode: onRevealNode,
    openCluster3d,
  } = useGraphView();
  const isPreviewBuilding = previewInspection.isBuilding;
  const isPreviewError = previewInspection.isError;
  const previewError = previewInspection.error;
  const hasSelectedClusterNode = Boolean(
    selectedNodeId &&
      graphForDetail?.nodes.some(
        (node) => node.id === selectedNodeId && node.typeName === "NeuronCluster",
      ),
  );
  const cluster3dNodeId = isNeuronModelType(selectedModelType)
    ? graph?.nodes.find((node) => node.typeName === "NeuronCluster")?.id ?? null
    : null;
  const hasPreviewGraph = Boolean(graph);
  const isHistoricalRunSelectionPending =
    browser.mode === "experiment" && !selectedLogRunId;
  const isMonitorEligibilityChecking =
    selectedTargetMode === "experiment" &&
    Boolean(selectedExperimentRunId) &&
    selectedLogRunMonitorEligibility === "checking";
  const isMonitorIneligible =
    selectedTargetMode === "experiment" &&
    Boolean(selectedExperimentRunId) &&
    selectedLogRunMonitorEligibility === "ineligible";
  const showActivityLoading = hasPreviewGraph && isParameterStatusLoading;
  const parameterActivityMinimap = useParameterActivityMinimapState({
    graph,
    selectedTargetMode,
    selectedExperimentRunId,
    selectedLogRun,
    selectedLogRunMonitorEligibility,
    protectedReadsEnabled: isWorkbenchProtectedAccessReady(workbenchConnection),
  });

  return (
    <div className="relative h-full min-h-0 overflow-hidden bg-transparent">
      {isPreviewError && (
        <div className="absolute left-4 right-4 top-4 z-10">
          <ErrorPanel title="Preview failed" message={errorMessage(previewError)} />
        </div>
      )}
      {(isPreviewBuilding ||
        isHistoricalRunSelectionPending ||
        isMonitorEligibilityChecking ||
        isMonitorIneligible ||
        showActivityLoading ||
        isParameterStatusPathMismatch) && (
        <div
          role="status"
          aria-live="polite"
          className="absolute left-4 top-4 z-10 flex max-w-[calc(100%-2rem)] flex-wrap gap-2"
        >
          {isPreviewBuilding && (
            <StatusPill
              icon={<RefreshCw className="h-4 w-4 motion-safe:animate-spin" aria-hidden />}
              label="preview"
              value="building"
              tone="warn"
            />
          )}
          {isHistoricalRunSelectionPending && (
            <StatusPill
              icon={<Clock3 className="h-4 w-4" aria-hidden />}
              label="experiment"
              value="pending"
              tone="warn"
            />
          )}
          {isMonitorEligibilityChecking && (
            <StatusPill
              icon={<Activity className="h-4 w-4 motion-safe:animate-pulse" aria-hidden />}
              label="monitor"
              value="checking"
              tone="warn"
            />
          )}
          {isMonitorIneligible && (
            <StatusPill
              icon={<Activity className="h-4 w-4" aria-hidden />}
              label="monitor"
              value="No monitor data for graph activity"
              tone="warn"
              className="h-auto min-h-[34px]"
            />
          )}
          {showActivityLoading && (
            <StatusPill
              icon={<Activity className="h-4 w-4 motion-safe:animate-pulse" aria-hidden />}
              label="activity"
              value="loading"
              tone="warn"
            />
          )}
          {isParameterStatusPathMismatch && (
            <StatusPill
              icon={<Activity className="h-4 w-4" aria-hidden />}
              label="monitor"
              value="path mismatch"
              tone="warn"
            />
          )}
        </div>
      )}
      <div className="relative h-full min-h-0 overflow-hidden">
        {hasPreviewGraph && (
          <GraphPreviewPanel
            graph={graphForDetail}
            minimapGraph={graph}
            nodes={nodes}
            edges={edges}
            selectedNodeId={selectedNodeId}
            hasSelectedClusterNode={hasSelectedClusterNode}
            cluster3dNodeId={cluster3dNodeId}
            parameterActivityMinimap={parameterActivityMinimap}
            onSelectNode={onSelectNode}
            onRevealNode={onRevealNode}
            onOpenCluster3d={openCluster3d}
          />
        )}
        {!hasPreviewGraph && !isPreviewBuilding && !isPreviewError && (
          <EmptyState
            title={workbenchStatusCopy.empty.graph}
            detail={workbenchStatusCopy.empty.graphDetail}
            icon={<GitBranch className="h-4 w-4" aria-hidden />}
          />
        )}
      </div>
    </div>
  );
}
