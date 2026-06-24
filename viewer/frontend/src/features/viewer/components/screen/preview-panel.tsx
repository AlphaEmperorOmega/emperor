import dynamic from "next/dynamic";
import { GitBranch, RefreshCw } from "lucide-react";
import { EmptyState } from "@/features/viewer/components/empty-state";
import { ErrorPanel } from "@/features/viewer/components/error-panel";
import { viewerStatusCopy } from "@/features/viewer/components/shared/status-copy";
import { StatusPill } from "@/features/viewer/components/status-pill";
import {
  useGraphView,
  useTargetConfig,
} from "@/features/viewer/providers/viewer-providers";
import { errorMessage } from "@/lib/utils";

// The ReactFlow canvas (@xyflow/react + graph node views) is the largest piece
// of client JS in this view, but it only renders after a model is inspected.
// Loading it dynamically keeps it out of the initial bundle; the panel shell and
// empty state below stay static so the default view paints immediately.
const GraphPreviewPanel = dynamic(
  () =>
    import("@/features/viewer/components/screen/graph-canvas").then(
      (module) => module.GraphPreviewPanel,
    ),
  { ssr: false },
);

function isNeuronModelType(modelType: string) {
  return modelType === "neuron" || modelType === "neurons";
}

export function PreviewPanel() {
  const { selectedModelType } = useTargetConfig();
  const {
    graph,
    graphForDetail,
    nodes,
    edges,
    selectedNodeId,
    previewInspection,
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

  return (
    <div className="relative h-full min-h-0 overflow-hidden bg-transparent">
      {isPreviewError && (
        <div className="absolute left-4 right-4 top-4 z-10">
          <ErrorPanel title="Preview failed" message={errorMessage(previewError)} />
        </div>
      )}
      {isPreviewBuilding && (
        <div className="absolute left-4 top-4 z-10">
          <StatusPill
            icon={<RefreshCw className="h-4 w-4 motion-safe:animate-spin" />}
            label="preview"
            value="building"
            tone="warn"
          />
        </div>
      )}
      <div className="relative h-full min-h-0 overflow-hidden">
        <GraphPreviewPanel
          graph={graphForDetail}
          nodes={nodes}
          edges={edges}
          selectedNodeId={selectedNodeId}
          hasSelectedClusterNode={hasSelectedClusterNode}
          cluster3dNodeId={cluster3dNodeId}
          onSelectNode={onSelectNode}
          onRevealNode={onRevealNode}
          onOpenCluster3d={openCluster3d}
        />
        {!hasPreviewGraph && !isPreviewBuilding && !isPreviewError && (
          <EmptyState
            title={viewerStatusCopy.empty.graph}
            detail={viewerStatusCopy.empty.graphDetail}
            icon={<GitBranch className="h-4 w-4" aria-hidden />}
          />
        )}
      </div>
    </div>
  );
}
