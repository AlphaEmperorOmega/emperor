import { GitBranch, RefreshCw } from "lucide-react";
import { Background, Controls, Panel, ReactFlow } from "@xyflow/react";
import { EmptyState } from "@/components/features/viewer/empty-state";
import { ErrorPanel } from "@/components/features/viewer/error-panel";
import { GraphLocationsCard } from "@/components/features/viewer/graph/graph-locations-card";
import { nodeTypes } from "@/components/features/viewer/graph/graph-node-view";
import { StatusPill } from "@/components/features/viewer/status-pill";
import { useGraphView } from "@/components/features/viewer/providers/viewer-providers";
import { errorMessage } from "@/lib/utils";

export function PreviewPanel() {
  const {
    graph,
    graphForDetail,
    nodes,
    edges,
    selectedNodeId,
    previewInspection,
    setSelectedNodeId: onSelectNode,
    revealGraphNode: onRevealNode,
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
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          minZoom={0.5}
          maxZoom={1.5}
          fitViewOptions={{ padding: 0.12, minZoom: 1, maxZoom: 1 }}
          onNodeClick={(_, node) => onSelectNode(node.id)}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          nodesFocusable={false}
          nodeClickDistance={4}
        >
          <Background gap={26} color="rgba(255,255,255,0.05)" />
          <Controls showInteractive={false} position="bottom-left" />
          {hasSelectedClusterNode && (
            <Panel
              position="bottom-right"
              className="nodrag nopan hidden xl:block"
              style={{ right: 28, bottom: 24 }}
            >
              <GraphLocationsCard
                key={selectedNodeId}
                graph={graphForDetail}
                selectedNodeId={selectedNodeId}
                onRevealNode={onRevealNode}
                className="max-h-[42vh]"
              />
            </Panel>
          )}
        </ReactFlow>
        {!graph && !isPreviewBuilding && !isPreviewError && (
          <EmptyState
            title="No graph loaded"
            detail="Preview data has not returned yet."
            icon={<GitBranch className="h-4 w-4" aria-hidden />}
          />
        )}
      </div>
    </div>
  );
}
