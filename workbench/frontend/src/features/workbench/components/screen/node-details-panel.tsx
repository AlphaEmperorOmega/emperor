import dynamic from "next/dynamic";
import { Hash } from "lucide-react";
import { MetricCard } from "@/features/workbench/components/shared/metric-card";
import { InlineStatus } from "@/features/workbench/components/shared/inline-status";
import { SidePanel } from "@/features/workbench/components/shared/side-panel";
import {
  useGraphView,
  useHistoricalRuns,
} from "@/features/workbench/providers/workbench-providers";
import { useActiveTrainingJob } from "@/features/workbench/providers/training-provider";
import { formatCompactCount, formatExactCount } from "@/lib/graph";

const SelectedNodeDetails = dynamic(
  () =>
    import(
      "@/features/workbench/components/graph/selected-node-details"
    ).then((module) => module.SelectedNodeDetails),
  {
    ssr: false,
    loading: () => (
      <InlineStatus busy className="rounded-card">Loading node details…</InlineStatus>
    ),
  },
);

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
          <div className="grid gap-2">
            <MetricCard
              icon={<Hash className="h-[15px] w-[15px] text-violet" aria-hidden />}
              label="Params"
              value={formatCompactCount(graph.parameterCount)}
              valueTitle={`${formatExactCount(graph.parameterCount)} parameters`}
              valueClassName="gradient-text mt-1.5 truncate font-mono type-display font-extrabold leading-tight tracking-display"
            />
          </div>
        ) : undefined
      }
    >
      {selectedNode ? (
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
      ) : (
        <InlineStatus className="rounded-dialog">No node selected</InlineStatus>
      )}
    </SidePanel>
  );
}
