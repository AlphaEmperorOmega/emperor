import {
  createContext,
  memo,
  useContext,
  useState,
  type KeyboardEvent,
  type MouseEvent,
  type ReactNode,
} from "react";
import { ChevronRight, Info, LineChart } from "lucide-react";
import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { ComponentInfoDialog } from "@/features/workbench/components/graph/component-info-dialog";
import { GraphIconButton } from "@/features/workbench/components/graph/graph-icon-button";
import {
  GraphNodeDetails,
  GraphNodeDetailsToggle,
} from "@/features/workbench/components/graph/graph-node-details";
import { GraphNodeHeader } from "@/features/workbench/components/graph/graph-node-header";
import { GraphNodeFooterStats } from "@/features/workbench/components/graph/graph-node-badges";
import { GraphNodeChildSummaries } from "@/features/workbench/components/graph/graph-node-child-summaries";
import {
  ClusterDiagramView,
  ExpertDiagramView,
  StackDiagramView,
} from "@/features/workbench/components/graph/graph-node-diagrams";
import { GraphNodeParameterShapes } from "@/features/workbench/components/graph/graph-node-parameter-shapes";
import {
  type WorkbenchNodeData,
  formatModelSize,
  nodeDimsText,
  nodeDetailEntries,
  parameterShapeEntries,
  parameterShapeDimsText,
  simpleGraphParamText,
} from "@/lib/graph";
import { graphCardGeometry } from "@/lib/graph/constants";
import { cn } from "@/lib/utils";

const GraphNodeViewportMovingContext = createContext(false);

export function GraphNodeRenderModeProvider({
  children,
  isViewportMoving,
}: {
  children: ReactNode;
  isViewportMoving: boolean;
}) {
  return (
    <GraphNodeViewportMovingContext.Provider value={isViewportMoving}>
      {children}
    </GraphNodeViewportMovingContext.Provider>
  );
}

const GraphNodeView = memo(function GraphNodeView({
  data,
  selected,
}: NodeProps<Node<WorkbenchNodeData>>) {
  const isViewportMoving = useContext(GraphNodeViewportMovingContext);
  const [isComponentInfoOpen, setIsComponentInfoOpen] = useState(false);
  const isSimpleMode = data.graphDetailMode === "simple";
  const simpleParamText = isSimpleMode ? simpleGraphParamText(data.parameterCount) : undefined;
  const modelSizeText = data.isRootNode ? formatModelSize(data.parameterSizeBytes) : undefined;
  const simpleDimsText = isSimpleMode
    ? nodeDimsText(data.details, data.config) ?? data.stackDiagram?.dims
    : undefined;
  const activateLabel = data.canToggleExpansion
    ? `Select and ${data.isExpanded ? "collapse" : "expand"} ${data.path}`
    : `Select ${data.path}`;
  const handleActivate = () => {
    data.onActivateNode();
  };
  const handleSelectClick = (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    handleActivate();
  };
  const handleSelectKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    event.stopPropagation();
  };
  const componentInfoDialog = isComponentInfoOpen ? (
    <ComponentInfoDialog
      node={{
        typeName: data.typeName,
        description: data.description,
        path: data.path,
        config: data.config,
      }}
      onClose={() => setIsComponentInfoOpen(false)}
    />
  ) : null;

  if (isViewportMoving) {
    return (
      <>
        <MovingGraphNodeShell
          data={data}
          selected={selected}
          activateLabel={activateLabel}
          onActivate={handleActivate}
          modelSizeText={modelSizeText}
          simpleParamText={simpleParamText}
          simpleDimsText={simpleDimsText}
        />
        {componentInfoDialog}
      </>
    );
  }

  const entries = nodeDetailEntries(data.details, data.config);
  const parameterShapes = parameterShapeEntries(data.details);
  const hasMetadata = entries.length > 0;
  const detailToggleLabel = data.config ? "Config options" : "Details";
  const detailsId = `graph-node-details-${data.nodeId.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
  const rendersChildSummaries =
    !isSimpleMode && !data.clusterDiagram && !data.stackDiagram && !data.expertDiagram;
  const shouldRenderChildSummaries =
    rendersChildSummaries && data.childSummaries.length > 0;
  const hasChildSummaryParameterActivity = data.childSummaries.some(
    (summary) => summary.parameterActivity,
  );
  const visibleDimsTexts = [
    ...(shouldRenderChildSummaries
      ? data.childSummaries.map((summary) => summary.dims)
      : []),
    ...(data.stackDiagram
      ? [
          data.stackDiagram.dims,
          ...data.stackDiagram.cells.map((cell) => cell.dims),
        ]
      : []),
  ];
  const shapeDimsText =
    !isSimpleMode && parameterShapes.length > 0
      ? parameterShapeDimsText(data.details, data.config, visibleDimsTexts)
      : undefined;
  const expansionButton = data.canToggleExpansion ? (
    <GraphIconButton
      className="flex h-touch w-touch shrink-0 items-center justify-center rounded-control-md border border-line bg-panel-2/90 text-ink-dim transition-[color,background-color,border-color] duration-150 hover:border-line-hover hover:bg-control-hover hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:h-control-sm md:w-control-sm"
      label={`${data.isExpanded ? "Collapse" : "Expand"} tree ${data.path}`}
      onClick={data.onToggleExpansion}
      icon={
        <ChevronRight
          className={cn("h-3.5 w-3.5 transition-transform", data.isExpanded && "rotate-90")}
          aria-hidden
        />
      }
    />
  ) : null;
  const componentInfoButton = (
    <GraphIconButton
      className="flex h-touch w-touch shrink-0 items-center justify-center rounded-control-md border border-line bg-panel-2/90 text-ink-dim transition-[color,background-color,border-color] duration-150 hover:border-line-hover hover:bg-control-hover hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:h-control-sm md:w-control-sm"
      label={`Open component info for ${data.path}`}
      onClick={() => setIsComponentInfoOpen(true)}
      icon={<Info className="h-3.5 w-3.5" aria-hidden />}
    />
  );
  const monitorLabel = `Open monitor charts for ${data.path}`;
  const handleMonitorClick = (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
    data.onOpenMonitor?.();
  };
  const handleMonitorKeyDown = (event: KeyboardEvent<HTMLButtonElement>) => {
    event.stopPropagation();
  };
  const renderMonitorButton = (compact: boolean) => {
    if (!data.canOpenMonitor || !data.onOpenMonitor) {
      return null;
    }

    if (compact) {
      return (
        <button
          type="button"
          title={monitorLabel}
          aria-label={monitorLabel}
          className="nodrag nopan flex h-touch w-touch shrink-0 items-center justify-center rounded-control-sm border border-violet/35 bg-accent-soft text-violet-muted transition-[color,background-color,border-color] duration-150 hover:border-violet/60 hover:bg-violet/20 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:h-control-sm md:w-control-sm"
          onClick={handleMonitorClick}
          onKeyDown={handleMonitorKeyDown}
        >
          <LineChart className="h-3.5 w-3.5" aria-hidden />
        </button>
      );
    }

    return (
      <GraphIconButton
        className="flex h-touch w-touch shrink-0 items-center justify-center rounded-control-md border border-violet/30 bg-accent-soft text-violet-muted transition-[color,background-color,border-color] duration-150 hover:border-violet/55 hover:bg-violet/20 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:h-control-sm md:w-control-sm"
        label={monitorLabel}
        onClick={() => data.onOpenMonitor?.()}
        icon={<LineChart className="h-3.5 w-3.5" aria-hidden />}
      />
    );
  };
  const footerStats = (
    <GraphNodeFooterStats
      nodeId={data.nodeId}
      parameterCount={data.parameterCount}
      parameterSizeBytes={data.parameterSizeBytes}
      modelSizeText={modelSizeText}
      childCount={data.childCount}
    />
  );
  const detailsToggle =
    !isSimpleMode && hasMetadata ? (
      <GraphNodeDetailsToggle
        detailsId={detailsId}
        toggleLabel={detailToggleLabel}
        path={data.path}
        hasConfig={Boolean(data.config)}
        isExpanded={data.isDetailsExpanded}
        onToggleDetails={data.onToggleDetails}
      />
    ) : null;
  const inlineMonitorButton =
    shouldRenderChildSummaries && hasChildSummaryParameterActivity
      ? renderMonitorButton(true)
      : null;
  const footerMonitorButton = inlineMonitorButton ? null : renderMonitorButton(false);

  return (
    <>
      <div
        role="group"
        aria-label={data.path}
        data-testid={`graph-node-card-${data.nodeId}`}
        className={cn(
          "nodrag nopan edge relative flex w-full flex-col overflow-hidden rounded-card px-region transition-[background-color,border-color,box-shadow] duration-150 ease-out",
          selected ? "edge-sel" : "hover:border-line-hover",
        )}
        style={{
          height: data.height,
          paddingBottom: graphCardGeometry.paddingBlock,
          paddingTop: graphCardGeometry.paddingBlock,
        }}
      >
        <button
          type="button"
          aria-label={activateLabel}
          aria-expanded={data.canToggleExpansion ? data.isExpanded : undefined}
          onClick={handleSelectClick}
          onKeyDown={handleSelectKeyDown}
          className="absolute inset-0 z-0 rounded-card focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus"
        />
        <Handle type="target" position={Position.Left} />
        <div className="pointer-events-none relative z-10 flex h-full w-full flex-col [&_button]:pointer-events-auto [&_[role=button]]:pointer-events-auto">
          <GraphNodeHeader
            nodeId={data.nodeId}
            label={data.label}
            subtitle={data.subtitle}
            graphDetailMode={data.graphDetailMode}
            parameterCount={data.parameterCount}
            parameterSizeBytes={data.parameterSizeBytes}
            simpleDimsText={simpleDimsText}
          />
          {!isSimpleMode && parameterShapes.length > 0 && (
            <GraphNodeParameterShapes
              nodeId={data.nodeId}
              entries={parameterShapes}
              parameterActivity={data.parameterActivity}
              dimsText={shapeDimsText}
            />
          )}
          {!isSimpleMode && data.clusterDiagram ? (
            <ClusterDiagramView diagram={data.clusterDiagram} nodeId={data.nodeId} />
          ) : !isSimpleMode && data.stackDiagram ? (
            <StackDiagramView diagram={data.stackDiagram} nodeId={data.nodeId} />
          ) : !isSimpleMode && data.expertDiagram ? (
            <ExpertDiagramView diagram={data.expertDiagram} nodeId={data.nodeId} />
          ) : shouldRenderChildSummaries ? (
            <GraphNodeChildSummaries
              nodeId={data.nodeId}
              summaries={data.childSummaries}
              monitorButton={inlineMonitorButton}
            />
          ) : null}
          {!isSimpleMode && hasMetadata && (
            <GraphNodeDetails
              detailsId={detailsId}
              entries={entries}
              isExpanded={data.isDetailsExpanded}
            />
          )}
          <GraphNodeActionBar
            nodeId={data.nodeId}
            detailsToggle={detailsToggle}
            expansionButton={expansionButton}
            componentInfoButton={componentInfoButton}
            footerStats={footerStats}
            monitorButton={footerMonitorButton}
          />
        </div>
        <Handle type="source" position={Position.Right} />
      </div>
      {componentInfoDialog}
    </>
  );
});

function GraphNodeActionBar({
  nodeId,
  detailsToggle,
  expansionButton,
  componentInfoButton,
  footerStats,
  monitorButton,
}: {
  nodeId: string;
  detailsToggle: ReactNode;
  expansionButton: ReactNode;
  componentInfoButton: ReactNode;
  footerStats: ReactNode;
  monitorButton: ReactNode;
}) {
  return (
    <div
      className="flex shrink-0 items-center justify-between gap-2"
      style={{
        height: graphCardGeometry.actionBar.height,
        marginTop: graphCardGeometry.actionBar.marginBlockStart,
      }}
      data-testid={`graph-node-action-bar-${nodeId}`}
    >
      <div className="flex min-w-0 items-center gap-1.5">
        {detailsToggle}
        {expansionButton}
        {componentInfoButton}
      </div>
      <div className="flex min-w-0 flex-1 items-center justify-end gap-2">
        {footerStats}
        <div className="flex shrink-0 items-center gap-1">{monitorButton}</div>
      </div>
    </div>
  );
}

function MovingGraphNodeShell({
  data,
  selected,
  activateLabel,
  onActivate,
  modelSizeText,
  simpleParamText,
  simpleDimsText,
}: {
  data: WorkbenchNodeData;
  selected: boolean;
  activateLabel: string;
  onActivate: () => void;
  modelSizeText?: string;
  simpleParamText?: string;
  simpleDimsText?: string;
}) {
  const parameterText = simpleParamText ?? simpleGraphParamText(data.parameterCount);

  return (
    <div
      role="group"
      aria-label={data.path}
      data-testid={`graph-node-moving-${data.nodeId}`}
      className={cn(
        "nodrag nopan edge relative flex w-full flex-col justify-center overflow-hidden rounded-card px-shell py-region transition-none",
        selected && "edge-sel",
      )}
      style={{ height: data.height }}
    >
      <button
        type="button"
        aria-label={activateLabel}
        aria-expanded={data.canToggleExpansion ? data.isExpanded : undefined}
        onClick={onActivate}
        className="absolute inset-0 z-0 rounded-card focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus"
      />
      <Handle type="target" position={Position.Left} />
      <div className="pointer-events-none relative z-10 min-w-0">
        <div className="min-w-0 truncate type-title font-bold leading-5 text-ink">
          {data.label}
        </div>
        <div className="mt-1 min-w-0 truncate font-mono type-meta leading-4 text-ink-dim">
          {data.path}
        </div>
        <div className="mt-2 flex min-w-0 flex-wrap items-center gap-1.5 overflow-hidden font-mono type-caption font-semibold text-ink-faint">
          {parameterText && <span className="shrink-0">{parameterText}</span>}
          {modelSizeText && <span className="shrink-0">{modelSizeText}</span>}
          {simpleDimsText && <span className="shrink-0">{simpleDimsText}</span>}
          {data.childCount > 0 && (
            <span className="shrink-0">
              {data.childCount} {data.childCount === 1 ? "child" : "children"}
            </span>
          )}
        </div>
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

export const nodeTypes = {
  workbenchNode: GraphNodeView,
};
