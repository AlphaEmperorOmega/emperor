import {
  useCallback,
  useEffect,
  useMemo,
  useState,
  type MouseEvent,
} from "react";
import { createPortal } from "react-dom";
import dagre from "dagre";
import {
  ChevronRight,
  LineChart,
  Maximize2,
  Minimize2,
  X,
} from "lucide-react";
import {
  Background,
  Controls,
  Handle,
  Position,
  ReactFlow,
  ReactFlowProvider,
  type Edge,
  type Node,
  type NodeProps,
} from "@xyflow/react";
import { IconButton } from "@/components/ui/icon-button";
import {
  graphParameterActivityStatusClassNames,
  parameterActivityLabel,
} from "@/features/workbench/components/graph/graph-parameter-indicators";
import { LazyMonitorChartsModal } from "@/features/workbench/components/monitor/lazy-monitor-charts-modal";
import { DialogShell } from "@/features/workbench/components/shared/dialog-shell";
import {
  buildLinearMonitorComparisonCandidateGroups,
  collapseParameterActivityMinimapNodes,
  deriveParameterActivityMinimapModel,
  expandAllParameterActivityMinimapNodes,
  filterParameterActivityMinimapGraphByExpansion,
  type GraphParameterActivity,
  type ParameterActivityMinimapModel,
} from "@/lib/graph";
import { parameterActivityMinimapGeometry } from "@/lib/graph/constants";
import { workbenchGraphEdgeVisual } from "@/lib/graph/visuals";
import { type GraphNode, type InspectResponse } from "@/lib/api";
import { workbenchVisualTokens } from "@/lib/visual-tokens";
import { cn } from "@/lib/utils";
import { type MonitorChartsSource } from "@/types/monitor";

export type ParameterActivityMinimapNodeData = {
  nodeId: string;
  path: string;
  activity?: GraphParameterActivity;
  isParameterBearing: boolean;
  canToggleExpansion: boolean;
  isExpanded: boolean;
  onToggleExpansion: () => void;
  onOpenMonitor?: () => void;
};

type ParameterActivityMinimapDialogProps = {
  graph: InspectResponse;
  activityByNodePath?: Map<string, GraphParameterActivity>;
  source: MonitorChartsSource;
  onClose: () => void;
};

function stopPropagation(event: MouseEvent<HTMLElement>) {
  event.stopPropagation();
}

function activityChannels(activity: GraphParameterActivity) {
  return [
    {
      key: "weights",
      label: "W",
      title: `Weights ${activity.weights.status}`,
      channel: activity.weights,
    },
    ...(activity.bias
      ? [
          {
            key: "bias",
            label: "b",
            title: `Bias ${activity.bias.status}`,
            channel: activity.bias,
          },
        ]
      : []),
  ];
}

function minimapNodeWidth(activity: GraphParameterActivity | undefined) {
  return activity
    ? parameterActivityMinimapGeometry.activityNodeWidth
    : parameterActivityMinimapGeometry.branchNodeWidth;
}

function ParameterActivityMinimapNode({
  data,
}: NodeProps<Node<ParameterActivityMinimapNodeData>>) {
  const nodeLabel = data.isParameterBearing
    ? `Parameter activity for ${data.path}: ${
        data.activity ? parameterActivityLabel(data.activity) : "unknown"
      }`
    : `Parameter activity branch ${data.path}`;

  return (
    <div
      aria-label={nodeLabel}
      data-testid={`parameter-activity-minimap-node-${data.nodeId}`}
      className={cn(
        "nodrag nopan edge flex items-center justify-center overflow-hidden rounded-card px-1.5 shadow-panel transition hover:brightness-110",
        data.activity ? "gap-1" : undefined,
      )}
      style={{
        height: parameterActivityMinimapGeometry.nodeHeight,
        width: minimapNodeWidth(data.activity),
      }}
    >
      <Handle
        type="target"
        position={Position.Left}
        className="!h-0 !w-0 !border-0 !bg-transparent"
      />
      {data.canToggleExpansion && (
        <button
          type="button"
          aria-label={`${data.isExpanded ? "Collapse" : "Expand"} ${data.path}`}
          title={`${data.isExpanded ? "Collapse" : "Expand"} ${data.path}`}
          onClick={(event) => {
            event.stopPropagation();
            data.onToggleExpansion();
          }}
          onMouseDown={stopPropagation}
          className="nodrag nopan relative z-10 grid h-touch w-touch shrink-0 place-items-center rounded-control-md border border-line bg-control text-ink-dim transition-colors hover:border-line-hover hover:bg-control-hover hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:h-control-sm md:w-control-sm"
        >
          <ChevronRight
            className={cn(
              "h-3.5 w-3.5 transition-transform",
              data.isExpanded && "rotate-90",
            )}
            aria-hidden
          />
        </button>
      )}
      {data.activity && (
        <span className="relative z-10 flex min-w-0 items-center gap-1">
          {activityChannels(data.activity).map((channel) => (
            <span
              key={channel.key}
              title={channel.title}
              aria-label={channel.title}
              className={cn(
                "inline-flex h-6 min-w-6 shrink-0 items-center justify-center rounded-chip border px-1.5 font-mono type-meta font-bold leading-none",
                graphParameterActivityStatusClassNames[channel.channel.status],
              )}
            >
              {channel.label}
            </span>
          ))}
        </span>
      )}
      {data.onOpenMonitor && (
        <button
          type="button"
          aria-label={`Open monitor charts for ${data.path}`}
          title={`Open monitor charts for ${data.path}`}
          onClick={(event) => {
            event.stopPropagation();
            data.onOpenMonitor?.();
          }}
          onMouseDown={stopPropagation}
          className="nodrag nopan relative z-10 grid h-touch w-touch shrink-0 place-items-center rounded-control-sm border border-violet/30 bg-accent-soft text-violet-muted transition-colors hover:border-violet/50 hover:bg-violet/20 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:h-control-sm md:w-control-sm"
        >
          <LineChart className="h-3.5 w-3.5" aria-hidden />
        </button>
      )}
      <Handle
        type="source"
        position={Position.Right}
        className="!h-0 !w-0 !border-0 !bg-transparent"
      />
    </div>
  );
}

const minimapNodeTypes = {
  parameterActivityMinimapNode: ParameterActivityMinimapNode,
};

function layoutMinimapNodes({
  graph,
  model,
  expandedNodeIds,
  activityByNodePath,
  onToggleExpansion,
  onOpenMonitor,
}: {
  graph: InspectResponse | undefined;
  model: ParameterActivityMinimapModel;
  expandedNodeIds: Set<string>;
  activityByNodePath?: Map<string, GraphParameterActivity>;
  onToggleExpansion: (nodeId: string) => void;
  onOpenMonitor: (node: GraphNode) => void;
}): { nodes: Array<Node<ParameterActivityMinimapNodeData>>; edges: Edge[] } {
  if (!graph) {
    return { nodes: [], edges: [] };
  }

  const dagreGraph = new dagre.graphlib.Graph();
  dagreGraph.setDefaultEdgeLabel(() => ({}));
  dagreGraph.setGraph({
    rankdir: "LR",
    nodesep: parameterActivityMinimapGeometry.nodeGap,
    ranksep: parameterActivityMinimapGeometry.rankGap,
  });
  graph.nodes.forEach((node) => {
    const activity = activityByNodePath?.get(node.path);
    dagreGraph.setNode(node.id, {
      width: minimapNodeWidth(activity),
      height: parameterActivityMinimapGeometry.nodeHeight,
    });
  });
  graph.edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });
  dagre.layout(dagreGraph);

  const nodes: Array<Node<ParameterActivityMinimapNodeData>> = graph.nodes.map(
    (node) => {
      const activity = activityByNodePath?.get(node.path);
      const isParameterBearing = model.parameterNodeIds.has(node.id);
      const canToggleExpansion = model.expandableNodeIds.has(node.id);
      const position = dagreGraph.node(node.id);
      const width = minimapNodeWidth(activity);

      return {
        id: node.id,
        type: "parameterActivityMinimapNode",
        position: {
          x: (position?.x ?? width / 2) - width / 2,
          y:
            (position?.y ?? parameterActivityMinimapGeometry.nodeHeight / 2) -
            parameterActivityMinimapGeometry.nodeHeight / 2,
        },
        style: {
          width,
          height: parameterActivityMinimapGeometry.nodeHeight,
        },
        data: {
          nodeId: node.id,
          path: node.path,
          activity,
          isParameterBearing,
          canToggleExpansion,
          isExpanded: expandedNodeIds.has(node.id),
          onToggleExpansion: () => onToggleExpansion(node.id),
          onOpenMonitor: isParameterBearing ? () => onOpenMonitor(node) : undefined,
        },
      };
    },
  );
  const edges: Edge[] = graph.edges.map((edge) => ({
    id: edge.id,
    source: edge.source,
    target: edge.target,
    ...workbenchGraphEdgeVisual(),
  }));

  return { nodes, edges };
}

function ParameterActivityMinimapDialogContent({
  graph,
  activityByNodePath,
  source,
  onClose,
}: ParameterActivityMinimapDialogProps) {
  const model = useMemo(
    () => deriveParameterActivityMinimapModel({ graph, activityByNodePath }),
    [activityByNodePath, graph],
  );
  const [expandedNodeIds, setExpandedNodeIds] = useState(
    () => expandAllParameterActivityMinimapNodes(model),
  );
  const [monitorNode, setMonitorNode] = useState<GraphNode | undefined>();

  useEffect(() => {
    setExpandedNodeIds(expandAllParameterActivityMinimapNodes(model));
    setMonitorNode(undefined);
  }, [model]);

  const toggleExpansion = useCallback((nodeId: string) => {
    setExpandedNodeIds((current) => {
      const next = new Set(current);
      if (next.has(nodeId)) {
        next.delete(nodeId);
      } else {
        next.add(nodeId);
      }
      return next;
    });
  }, []);
  const visibleGraph = useMemo(
    () => filterParameterActivityMinimapGraphByExpansion(model, expandedNodeIds),
    [expandedNodeIds, model],
  );
  const flow = useMemo(
    () =>
      layoutMinimapNodes({
        graph: visibleGraph,
        model,
        expandedNodeIds,
        activityByNodePath,
        onToggleExpansion: toggleExpansion,
        onOpenMonitor: setMonitorNode,
      }),
    [activityByNodePath, expandedNodeIds, model, toggleExpansion, visibleGraph],
  );
  const monitorComparisonCandidateGroups = useMemo(
    () => buildLinearMonitorComparisonCandidateGroups(graph, monitorNode),
    [graph, monitorNode],
  );

  return (
    <>
      <DialogShell
        titleId="parameter-activity-minimap-title"
        size="fullscreen"
        onClose={onClose}
        className="grid bg-black/[0.72] p-3 sm:p-6"
        panelClassName="grid max-h-full min-h-0 w-[min(1200px,calc(100vw-1.5rem))] max-w-none grid-rows-[auto_minmax(0,1fr)] justify-self-center sm:w-[min(1200px,calc(100vw-3rem))]"
      >
        <header className="flex min-w-0 items-center justify-between gap-3 border-b border-line bg-panel/90 p-3 backdrop-blur sm:p-4">
          <h2
            id="parameter-activity-minimap-title"
            className="min-w-0 truncate text-base font-semibold text-ink"
          >
            Parameter activity
          </h2>
          <div className="flex shrink-0 items-center gap-2">
            <IconButton
              label="Collapse all"
              title="Collapse all"
              icon={<Minimize2 className="h-4 w-4" aria-hidden />}
              variant="edge"
              onClick={() =>
                setExpandedNodeIds(collapseParameterActivityMinimapNodes(model))
              }
            />
            <IconButton
              label="Expand all"
              title="Expand all"
              icon={<Maximize2 className="h-4 w-4" aria-hidden />}
              variant="edge"
              onClick={() =>
                setExpandedNodeIds(expandAllParameterActivityMinimapNodes(model))
              }
            />
            <IconButton
              label="Close"
              title="Close"
              icon={<X className="h-4 w-4" aria-hidden />}
              variant="edge"
              onClick={onClose}
            />
          </div>
        </header>
        <div className="min-h-[60vh] min-w-0 bg-minimap">
          <ReactFlowProvider
            initialNodes={flow.nodes}
            initialEdges={flow.edges}
            initialMinZoom={0.35}
            initialMaxZoom={2}
            fitView
            initialFitViewOptions={{ padding: 0.22 }}
          >
            <ReactFlow
              nodes={flow.nodes}
              edges={flow.edges}
              nodeTypes={minimapNodeTypes}
              fitView
              minZoom={0.35}
              maxZoom={2}
              fitViewOptions={{ padding: 0.22 }}
              nodesDraggable={false}
              nodesConnectable={false}
              elementsSelectable={false}
              nodesFocusable={false}
              onlyRenderVisibleElements
              nodeClickDistance={4}
            >
              <Background gap={22} color={workbenchVisualTokens.lineSoft} />
              <Controls showInteractive={false} position="bottom-left" />
            </ReactFlow>
          </ReactFlowProvider>
        </div>
      </DialogShell>
      {monitorNode && (
        <LazyMonitorChartsModal
          node={monitorNode}
          source={source}
          comparisonCandidateGroups={monitorComparisonCandidateGroups}
          onClose={() => setMonitorNode(undefined)}
        />
      )}
    </>
  );
}

export function ParameterActivityMinimapDialog(
  props: ParameterActivityMinimapDialogProps,
) {
  const dialog = <ParameterActivityMinimapDialogContent {...props} />;

  return typeof document === "undefined"
    ? dialog
    : createPortal(dialog, document.body);
}
