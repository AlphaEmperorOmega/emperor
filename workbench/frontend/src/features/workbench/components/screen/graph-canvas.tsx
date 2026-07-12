import { useState } from "react";
import { Box, GitBranch, Map as MapIcon } from "lucide-react";
import { Background, Controls, Panel, ReactFlow } from "@xyflow/react";
import { IconButton } from "@/components/ui/icon-button";
import { GraphLocationsCard } from "@/features/workbench/components/graph/graph-locations-card";
import { ParameterActivityMinimapDialog } from "@/features/workbench/components/graph/parameter-activity-minimap-dialog";
import {
  GraphNodeRenderModeProvider,
  nodeTypes,
} from "@/features/workbench/components/graph/graph-node-view";
import { GraphStructurePanel } from "@/features/workbench/components/screen/graph-structure-panel";
import { type ParameterActivityMinimapState } from "@/features/workbench/state/graph-monitor/use-parameter-activity-minimap-state";
import { useGraphView } from "@/features/workbench/providers/workbench-providers";
import { type InspectResponse } from "@/lib/api";
import { cn } from "@/lib/utils";
import { workbenchVisualTokens } from "@/lib/visual-tokens";

export function GraphPreviewPanel({
  graph,
  minimapGraph,
  nodes,
  edges,
  selectedNodeId,
  hasSelectedClusterNode,
  cluster3dNodeId,
  parameterActivityMinimap,
  onSelectNode,
  onRevealNode,
  onOpenCluster3d,
}: {
  graph: InspectResponse | undefined;
  minimapGraph?: InspectResponse;
  nodes: ReturnType<typeof useGraphView>["nodes"];
  edges: ReturnType<typeof useGraphView>["edges"];
  selectedNodeId: string | null;
  hasSelectedClusterNode: boolean;
  cluster3dNodeId: string | null;
  parameterActivityMinimap?: ParameterActivityMinimapState;
  onSelectNode: (nodeId: string | null) => void;
  onRevealNode: (nodeId: string) => void;
  onOpenCluster3d: (nodeId?: string) => void;
}) {
  const [isStructureOpen, setIsStructureOpen] = useState(false);
  const [isMinimapOpen, setIsMinimapOpen] = useState(false);
  const [isViewportMoving, setIsViewportMoving] = useState(false);
  const structurePanelId = "graph-structure-panel";
  const minimapButtonTitle = parameterActivityMinimap?.canOpen
    ? `${isMinimapOpen ? "Close" : "Open"} parameter activity minimap`
    : (parameterActivityMinimap?.disabledReason ??
      "Parameter activity minimap is unavailable");

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
        <Background gap={26} color={workbenchVisualTokens.graphGrid} />
        <Controls showInteractive={false} position="bottom-left" />
        {graph && (
          <Panel
            position="top-right"
            className="nodrag nopan"
            style={{ right: 18, top: 18 }}
          >
            <div className="flex items-end gap-2">
              {parameterActivityMinimap?.shouldRenderButton && (
                <IconButton
                  label={minimapButtonTitle}
                  title={minimapButtonTitle}
                  icon={<MapIcon className="h-4 w-4" aria-hidden />}
                  variant="edge"
                  disabled={!parameterActivityMinimap.canOpen}
                  aria-expanded={isMinimapOpen}
                  onClick={(event) => {
                    event.stopPropagation();
                    if (!parameterActivityMinimap.canOpen) {
                      return;
                    }
                    setIsMinimapOpen((current) => !current);
                  }}
                  className={cn(
                    "shadow-floating",
                    isMinimapOpen &&
                      "border-cyan/45 bg-cyan/[0.11] text-ink",
                  )}
                />
              )}
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
                  "shadow-floating",
                  isStructureOpen && "border-cyan/45 bg-cyan/[0.11] text-ink",
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
        {isMinimapOpen &&
          minimapGraph &&
          parameterActivityMinimap?.canOpen &&
          parameterActivityMinimap?.selectedRunSource && (
            <ParameterActivityMinimapDialog
              graph={minimapGraph}
              activityByNodePath={parameterActivityMinimap.activityByNodePath}
              source={parameterActivityMinimap.selectedRunSource}
              onClose={() => setIsMinimapOpen(false)}
            />
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
                    className="grid h-control-lg w-control-lg place-items-center rounded-control border border-cyan/35 bg-panel/90 text-cyan shadow-panel backdrop-blur-md transition-[color,background-color,border-color,box-shadow,transform] duration-150 ease-out hover:border-cyan/60 hover:bg-cyan/10 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-bg"
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
