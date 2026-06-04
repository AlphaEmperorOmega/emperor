import { EdgeCard } from "@/components/ui/edge-card";
import { SelectedNodeDetails } from "@/components/features/viewer/selected-node-details";
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
    selectedLogRunHasMonitorTags: historicalRunHasMonitorTags,
    logRunTagsQuery,
  } = useHistoricalRuns();
  const historicalRunTagsLoading = logRunTagsQuery.isLoading;
  return (
    <aside className="min-h-0 overflow-y-auto border-t border-line bg-[linear-gradient(180deg,rgba(13,12,22,0.6),rgba(8,8,13,0.4))] px-[18px] pb-8 pt-5 backdrop-blur lg:border-l lg:border-t-0">
      <div className="mb-4 grid gap-3">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-base font-bold text-ink">Node Details</h2>
        </div>
        {graph && (
          <div className="grid gap-[9px]">
            <EdgeCard className="rounded-[12px] px-3 py-3">
              <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
                Params
              </div>
              <div
                className="gradient-text mt-1.5 truncate font-mono text-[22px] font-extrabold leading-tight tracking-[-0.02em]"
                title={`${formatExactCount(graph.parameterCount)} parameters`}
              >
                {formatCompactCount(graph.parameterCount)}
              </div>
            </EdgeCard>
          </div>
        )}
      </div>
      <SelectedNodeDetails
        node={selectedNode}
        monitorNode={selectedMonitorNode}
        comparisonCandidateGroups={selectedMonitorComparisonCandidateGroups}
        activeTrainingJob={activeTrainingJob}
        historicalRuns={historicalRuns}
        historicalExperiment={historicalExperiment}
        historicalDataset={historicalDataset}
        historicalRunHasMonitorTags={historicalRunHasMonitorTags}
        historicalRunTagsLoading={historicalRunTagsLoading}
      />
    </aside>
  );
}
