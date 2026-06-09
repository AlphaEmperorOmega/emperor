import { SelectedNodeDetails } from "@/components/features/viewer/graph/selected-node-details";
import { MetricCard } from "@/components/features/viewer/shared/metric-card";
import { SidePanel } from "@/components/features/viewer/shared/side-panel";
import {
  useGraphView,
  useHistoricalRuns,
  useTraining,
} from "@/components/features/viewer/providers/viewer-providers";
import { formatCompactCount, formatExactCount } from "@/lib/graph";

export function NodeDetailsPanel() {
  const { graph, selectedNode, selectedMonitorNode, selectedMonitorComparisonCandidateGroups } =
    useGraphView();
  const { activeTrainingJob } = useTraining();
  const {
    historicalMonitorRuns: historicalRuns,
    selectedHistoricalExperiment: historicalExperiment,
    selectedHistoricalDataset: historicalDataset,
    selectedHistoricalRunPreset: historicalPreset,
    selectedLogRunHasMonitorTags: historicalRunHasMonitorTags,
    logRunTagsQuery,
  } = useHistoricalRuns();
  const historicalRunTagsLoading = logRunTagsQuery.isLoading;
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
