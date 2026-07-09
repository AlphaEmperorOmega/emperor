import { Hash } from "lucide-react";
import { SelectedNodeDetails } from "@/features/workbench/components/graph/selected-node-details";
import { MetricCard } from "@/features/workbench/components/shared/metric-card";
import { SidePanel } from "@/features/workbench/components/shared/side-panel";
import {
  useActiveTrainingJob,
  useGraphView,
  useHistoricalRuns,
} from "@/features/workbench/providers/workbench-providers";
import { formatCompactCount, formatExactCount } from "@/lib/graph";

export function NodeDetailsPanel() {
  const {
    graph,
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

  return (
    <SidePanel
      title="Node Details"
      subtitle={
        graph ? (
          <div className="grid gap-[9px]">
            <MetricCard
              icon={<Hash className="h-[15px] w-[15px] text-violet" aria-hidden />}
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
