import { useState } from "react";
import { Box, GitBranch } from "lucide-react";
import { Background, Controls, Panel, ReactFlow } from "@xyflow/react";
import { IconButton } from "@/components/ui/icon-button";
import { GraphLocationsCard } from "@/features/viewer/components/graph/graph-locations-card";
import {
  GraphNodeRenderModeProvider,
  nodeTypes,
} from "@/features/viewer/components/graph/graph-node-view";
import { GraphStructurePanel } from "@/features/viewer/components/screen/graph-structure-panel";
import { useGraphView } from "@/features/viewer/providers/viewer-providers";
import { type InspectResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

export function GraphPreviewPanel({
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
