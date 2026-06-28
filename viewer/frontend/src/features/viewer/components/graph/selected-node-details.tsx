import { useState } from "react";
import { SelectedNodeDetailsView } from "@/features/viewer/components/graph/selected-node-details-view";
import { LazyMonitorChartsModal } from "@/features/viewer/components/monitor/lazy-monitor-charts-modal";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { type GraphNode, type LogRun, type TrainingJob } from "@/lib/api";
import { type LinearMonitorComparisonCandidateGroups } from "@/lib/graph/monitor-targets";
import { type MonitorChartsSource } from "@/types/monitor";

export function SelectedNodeDetails({
  node,
  monitorNode,
  comparisonCandidateGroups,
  activeTrainingJob,
  historicalRuns = [],
  historicalExperiment = "",
  historicalDataset = "",
  historicalPreset = "",
  historicalRunHasMonitorTags = false,
  historicalRunTagsLoading = false,
}: {
  node: GraphNode | undefined;
  monitorNode?: GraphNode | undefined;
  comparisonCandidateGroups?: LinearMonitorComparisonCandidateGroups;
  activeTrainingJob: TrainingJob | undefined;
  historicalRuns?: LogRun[];
  historicalExperiment?: string;
  historicalDataset?: string;
  historicalPreset?: string;
  historicalRunHasMonitorTags?: boolean;
  historicalRunTagsLoading?: boolean;
}) {
  const [isMonitorOpen, setIsMonitorOpen] = useState(false);

  if (!node) {
    return (
      <InlineStatus className="rounded-[16px]">
        No node selected
      </InlineStatus>
    );
  }

  const activeLinearTrainingJob = activeTrainingJob?.monitors.includes("linear")
    ? activeTrainingJob
    : undefined;
  const monitorSource: MonitorChartsSource | undefined = monitorNode && activeLinearTrainingJob
    ? { kind: "active-job", job: activeLinearTrainingJob }
    : monitorNode && historicalRuns.length > 0 && historicalRunHasMonitorTags
      ? {
          kind: "historical-run-group",
          runs: historicalRuns,
          experiment: historicalExperiment,
          dataset: historicalDataset,
          preset: historicalPreset,
        }
      : undefined;
  const monitorButtonTitle = !monitorNode
    ? "Monitor charts are available for LinearLayer nodes"
    : activeLinearTrainingJob
      ? "Open monitor charts for the active training job"
      : historicalRunTagsLoading
        ? "Checking historical TensorBoard tags for this node"
        : historicalRuns.length > 0
          ? historicalRunHasMonitorTags
            ? "Open monitor charts for the filtered historical runs"
            : "No TensorBoard monitor tags matched this node in the filtered historical runs"
          : "Start a viewer training job or select a historical run with monitor tags";

  return (
    <>
      <SelectedNodeDetailsView
        node={node}
        canOpenMonitors={Boolean(monitorSource)}
        monitorButtonTitle={monitorButtonTitle}
        onOpenMonitors={() => {
          if (monitorSource) {
            setIsMonitorOpen(true);
          }
        }}
      />
      {isMonitorOpen && monitorSource && monitorNode && (
        <LazyMonitorChartsModal
          node={monitorNode}
          source={monitorSource}
          comparisonCandidateGroups={comparisonCandidateGroups}
          onClose={() => setIsMonitorOpen(false)}
        />
      )}
    </>
  );
}
