import { useState } from "react";
import { Activity, Box, GitBranch, RefreshCw } from "lucide-react";
import { Background, Controls, Panel, ReactFlow } from "@xyflow/react";
import { IconButton } from "@/components/ui/icon-button";
import { EmptyState } from "@/features/viewer/components/empty-state";
import { ErrorPanel } from "@/features/viewer/components/error-panel";
import { GraphLocationsCard } from "@/features/viewer/components/graph/graph-locations-card";
import {
  GraphNodeRenderModeProvider,
  nodeTypes,
} from "@/features/viewer/components/graph/graph-node-view";
import { GraphStructurePanel } from "@/features/viewer/components/screen/graph-structure-panel";
import { ParameterBrowserPanel } from "@/features/viewer/components/screen/parameter-browser-panel";
import { viewerStatusCopy } from "@/features/viewer/components/shared/status-copy";
import { StatusPill } from "@/features/viewer/components/status-pill";
import {
  useGraphView,
  useTargetConfig,
} from "@/features/viewer/providers/viewer-providers";
import { type InspectResponse } from "@/lib/api";
import { cn, errorMessage } from "@/lib/utils";

function isNeuronModelType(modelType: string) {
  return modelType === "neuron" || modelType === "neurons";
}

export function PreviewPanel() {
  const { selectedModelType } = useTargetConfig();
  const {
    graph,
    graphKind,
    operationGraph,
    graphForDetail,
    previewVisualizationMode,
    nodes,
    edges,
    operationNodes,
    operationEdges,
    selectedNodeId,
    selectedOperationNodeId,
    parameterFocusNodeId,
    previewInspection,
    operationInspection,
    setParameterFocusNodeId,
    setSelectedNodeId: onSelectNode,
    setSelectedOperationNodeId: onSelectOperationNode,
    revealGraphNode: onRevealNode,
    openCluster3d,
  } = useGraphView();
  const isPreviewBuilding = previewInspection.isBuilding;
  const isPreviewError = previewInspection.isError;
  const previewError = previewInspection.error;
  const isOperationBuilding = operationInspection.isBuilding;
  const isOperationError = operationInspection.isError;
  const operationError = operationInspection.error;
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
  const isOperationMode =
    previewVisualizationMode === "graph" && graphKind === "operation";

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
        {hasPreviewGraph && previewVisualizationMode === "parameters" ? (
          <ParameterBrowserPanel
            graph={graphForDetail}
            selectedNodeId={selectedNodeId}
            focusNodeId={parameterFocusNodeId}
            onFocusNode={setParameterFocusNodeId}
            onRevealNode={onRevealNode}
          />
        ) : isOperationMode ? (
          <OperationPreviewPanel
            operationGraph={operationGraph}
            nodes={operationNodes}
            edges={operationEdges}
            selectedNodeId={selectedOperationNodeId}
            isLoading={isOperationBuilding}
            isError={isOperationError}
            error={operationError}
            onSelectNode={onSelectOperationNode}
          />
        ) : (
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
        )}
        {!hasPreviewGraph && !isPreviewBuilding && !isPreviewError && !isOperationMode && (
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

function OperationPreviewPanel({
  operationGraph,
  nodes,
  edges,
  selectedNodeId,
  isLoading,
  isError,
  error,
  onSelectNode,
}: {
  operationGraph: ReturnType<typeof useGraphView>["operationGraph"];
  nodes: ReturnType<typeof useGraphView>["operationNodes"];
  edges: ReturnType<typeof useGraphView>["operationEdges"];
  selectedNodeId: string | null;
  isLoading: boolean;
  isError: boolean;
  error: unknown;
  onSelectNode: (nodeId: string | null) => void;
}) {
  if (isError) {
    return (
      <EmptyState
        title="Operations failed"
        detail={errorMessage(error)}
        icon={<Activity className="h-4 w-4" aria-hidden />}
      />
    );
  }

  if (!operationGraph) {
    return (
      <EmptyState
        title={isLoading ? "Operations are loading" : "Operations are not loaded"}
        detail={
          isLoading
            ? "The operation graph is being traced with torch.export."
            : "The operation graph is traced when this mode is opened."
        }
        icon={<Activity className="h-4 w-4" aria-hidden />}
      />
    );
  }

  if (operationGraph.status === "unsupported") {
    return (
      <EmptyState
        title="Operations unavailable"
        detail={
          operationGraph.warnings[0] ??
          "This model cannot currently be traced with torch.export."
        }
        icon={<Activity className="h-4 w-4" aria-hidden />}
      />
    );
  }

  if (operationGraph.nodes.length === 0) {
    return (
      <EmptyState
        title="No operations"
        detail="torch.export returned an empty graph for this target."
        icon={<Activity className="h-4 w-4" aria-hidden />}
      />
    );
  }

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      fitView
      minZoom={0.45}
      maxZoom={1.6}
      fitViewOptions={{ padding: 0.14, minZoom: 0.85, maxZoom: 1 }}
      onNodeClick={(_, node) => onSelectNode(node.id)}
      nodesDraggable={false}
      nodesConnectable={false}
      elementsSelectable={false}
      nodesFocusable={false}
      onlyRenderVisibleElements
      nodeClickDistance={4}
    >
      <Background gap={26} color="rgba(255,255,255,0.05)" />
      <Controls showInteractive={false} position="bottom-left" />
      {selectedNodeId && (
        <Panel
          position="top-right"
          className="nodrag nopan"
          style={{ right: 18, top: 18 }}
        >
          <div className="rounded-[8px] border border-line bg-[rgba(8,8,14,0.82)] px-3 py-2 font-mono text-[11px] text-ink-dim shadow-[0_14px_34px_rgba(0,0,0,0.32)] backdrop-blur">
            {selectedNodeId}
          </div>
        </Panel>
      )}
    </ReactFlow>
  );
}

function GraphPreviewPanel({
  graph,
  nodes,
  edges,
  selectedNodeId,
  hasSelectedClusterNode,
  cluster3dNodeId,
  onSelectNode,
  onRevealNode,
  onOpenCluster3d,
}: {
  graph: InspectResponse | undefined;
  nodes: ReturnType<typeof useGraphView>["nodes"];
  edges: ReturnType<typeof useGraphView>["edges"];
  selectedNodeId: string | null;
  hasSelectedClusterNode: boolean;
  cluster3dNodeId: string | null;
  onSelectNode: (nodeId: string | null) => void;
  onRevealNode: (nodeId: string) => void;
  onOpenCluster3d: (nodeId?: string) => void;
}) {
  const [isStructureOpen, setIsStructureOpen] = useState(false);
  const [isViewportMoving, setIsViewportMoving] = useState(false);
  const structurePanelId = "graph-structure-panel";

  return (
    <GraphNodeRenderModeProvider isViewportMoving={isViewportMoving}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        fitView
        minZoom={0.5}
        maxZoom={1.5}
        fitViewOptions={{ padding: 0.12, minZoom: 1, maxZoom: 1 }}
        onMoveStart={() => setIsViewportMoving(true)}
        onMoveEnd={() => setIsViewportMoving(false)}
        onNodeClick={(_, node) => onSelectNode(node.id)}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        nodesFocusable={false}
        onlyRenderVisibleElements
        nodeClickDistance={4}
      >
        <Background gap={26} color="rgba(255,255,255,0.05)" />
        <Controls showInteractive={false} position="bottom-left" />
        {graph && (
          <Panel
            position="top-right"
            className="nodrag nopan"
            style={{ right: 18, top: 18 }}
          >
            <div className="flex items-end gap-2">
              <IconButton
                label={`${isStructureOpen ? "Close" : "Open"} graph structure`}
                title={`${isStructureOpen ? "Close" : "Open"} graph structure`}
                icon={<GitBranch className="h-4 w-4" aria-hidden />}
                variant="edge"
                aria-expanded={isStructureOpen}
                aria-controls={isStructureOpen ? structurePanelId : undefined}
                onClick={(event) => {
                  event.stopPropagation();
                  setIsStructureOpen((current) => !current);
                }}
                className={cn(
                  "shadow-[0_14px_34px_rgba(0,0,0,0.32)]",
                  isStructureOpen && "border-cyan-200/45 bg-cyan-300/[0.11] text-ink",
                )}
              />
              {isStructureOpen && (
                <GraphStructurePanel
                  panelId={structurePanelId}
                  graph={graph}
                  selectedNodeId={selectedNodeId}
                  onRevealNode={onRevealNode}
                  onClose={() => setIsStructureOpen(false)}
                />
              )}
            </div>
          </Panel>
        )}
        {(cluster3dNodeId || hasSelectedClusterNode) && (
          <Panel
            position="bottom-right"
            className="nodrag nopan hidden xl:block"
            style={{ right: 28, bottom: 24 }}
          >
            <div className="flex flex-col items-end gap-2">
              <div className="flex items-end gap-2">
                {cluster3dNodeId && (
                  <button
                    type="button"
                    aria-label="Open 3D cluster view"
                    title="Open 3D cluster view"
                    onClick={(event) => {
                      event.stopPropagation();
                      onOpenCluster3d(cluster3dNodeId);
                    }}
                    className="grid h-10 w-10 place-items-center rounded-[10px] border border-cyan-200/35 bg-black/45 text-cyan-100 shadow-[0_18px_40px_-24px_rgba(0,0,0,0.95)] backdrop-blur-md transition hover:border-cyan-200/55 hover:bg-cyan-300/[0.13] hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                  >
                    <Box className="h-[17px] w-[17px]" aria-hidden />
                  </button>
                )}
                {hasSelectedClusterNode && (
                  <GraphLocationsCard
                    key={selectedNodeId}
                    graph={graph}
                    selectedNodeId={selectedNodeId}
                    onRevealNode={onRevealNode}
                    className="max-h-[42vh]"
                  />
                )}
              </div>
            </div>
          </Panel>
        )}
      </ReactFlow>
    </GraphNodeRenderModeProvider>
  );
}
