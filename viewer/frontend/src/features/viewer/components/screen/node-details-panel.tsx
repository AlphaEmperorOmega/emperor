import { SelectedNodeDetails } from "@/features/viewer/components/graph/selected-node-details";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { MetricCard } from "@/features/viewer/components/shared/metric-card";
import { SidePanel } from "@/features/viewer/components/shared/side-panel";
import {
  useActiveTrainingJob,
  useGraphView,
  useHistoricalRuns,
} from "@/features/viewer/providers/viewer-providers";
import { formatCompactCount, formatExactCount } from "@/lib/graph";

export function NodeDetailsPanel() {
  const {
    graph,
    graphKind,
    operationGraph,
    previewVisualizationMode,
    selectedOperationNode,
    selectedNode,
    selectedMonitorNode,
    selectedMonitorComparisonCandidateGroups,
  } = useGraphView();
  const { activeTrainingJob } = useActiveTrainingJob();
  const {
    historicalMonitorRuns: historicalRuns,
    selectedHistoricalExperiment: historicalExperiment,
    selectedHistoricalDataset: historicalDataset,
    selectedHistoricalRunPreset: historicalPreset,
    selectedLogRunHasMonitorTags: historicalRunHasMonitorTags,
    logRunTagsLoading: historicalRunTagsLoading,
  } = useHistoricalRuns();

  if (previewVisualizationMode === "graph" && graphKind === "operation") {
    return (
      <SidePanel
        title="Operation Details"
        subtitle={
          operationGraph ? (
            <div className="grid gap-[9px]">
              <MetricCard
                label="Ops"
                value={formatCompactCount(operationGraph.nodes.length)}
                valueTitle={`${formatExactCount(operationGraph.nodes.length)} operations`}
                valueClassName="gradient-text mt-1.5 truncate font-mono text-[22px] font-extrabold leading-tight tracking-[-0.02em]"
              />
            </div>
          ) : undefined
        }
      >
        <OperationNodeDetails node={selectedOperationNode} />
      </SidePanel>
    );
  }

  return (
    <SidePanel
      title="Node Details"
      subtitle={
        graph ? (
          <div className="grid gap-[9px]">
            <MetricCard
              label="Params"
              value={formatCompactCount(graph.parameterCount)}
              valueTitle={`${formatExactCount(graph.parameterCount)} parameters`}
              valueClassName="gradient-text mt-1.5 truncate font-mono text-[22px] font-extrabold leading-tight tracking-[-0.02em]"
            />
          </div>
        ) : undefined
      }
    >
      <SelectedNodeDetails
        node={selectedNode}
        monitorNode={selectedMonitorNode}
        comparisonCandidateGroups={selectedMonitorComparisonCandidateGroups}
        activeTrainingJob={activeTrainingJob}
        historicalRuns={historicalRuns}
        historicalExperiment={historicalExperiment}
        historicalDataset={historicalDataset}
        historicalPreset={historicalPreset}
        historicalRunHasMonitorTags={historicalRunHasMonitorTags}
        historicalRunTagsLoading={historicalRunTagsLoading}
      />
    </SidePanel>
  );
}

function OperationNodeDetails({
  node,
}: {
  node: ReturnType<typeof useGraphView>["selectedOperationNode"];
}) {
  if (!node) {
    return (
      <InlineStatus className="rounded-[16px]">
        No operation selected
      </InlineStatus>
    );
  }

  const rawRows: Array<[string, unknown]> = [
    ["Kind", node.opKind],
    ["Target", node.target],
    ["Module", node.modulePath ?? "None"],
    ["Group", node.groupId ?? "None"],
    [
      "Shape",
      Array.isArray(node.details.shape)
        ? node.details.shape.join(" x ")
        : node.details.shape,
    ],
    ["Dtype", node.details.dtype],
    ["Input", node.details.inputKind],
  ];
  const rows = rawRows.filter(
    ([, value]) => value !== undefined && value !== null && value !== "",
  );

  return (
    <div className="grid gap-3">
      <div>
        <div className="font-mono text-xs uppercase tracking-[0.1em] text-ink-faint">
          {node.id}
        </div>
        <h3 className="mt-1 break-words text-lg font-extrabold text-ink">
          {node.label}
        </h3>
      </div>
      <div className="grid gap-2">
        {rows.map(([label, value]) => (
          <div
            key={label}
            className="rounded-[8px] border border-line bg-white/[0.025] p-3"
          >
            <div className="text-[11px] font-bold uppercase tracking-[0.09em] text-ink-faint">
              {label}
            </div>
            <div className="mt-1 break-words font-mono text-[12px] text-ink">
              {String(value)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
