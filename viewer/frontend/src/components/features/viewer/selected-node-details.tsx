import { useState } from "react";
import dynamic from "next/dynamic";
import { SelectedNodeDetailsView } from "@/components/features/viewer/selected-node-details-view";
import { InlineStatus } from "@/components/features/viewer/shared/inline-status";
import { type GraphNode, type LogRun, type TrainingJob } from "@/lib/api";
import { type LinearMonitorComparisonCandidateGroups } from "@/lib/graph/monitor-targets";
import { type MonitorChartsSource } from "@/types/monitor";

// Lazy-loaded so the monitor charts modal and its inline SVG charts ship in a
// separate chunk that only downloads when a node's charts are opened.
const MonitorChartsModal = dynamic(
  () =>
    import("@/components/features/viewer/monitor-charts-modal").then(
      (module) => module.MonitorChartsModal,
    ),
  { ssr: false },
);

export function SelectedNodeDetails({
  node,
  monitorNode,
  comparisonCandidateGroups,
  activeTrainingJob,
  historicalRuns = [],
  historicalExperiment = "",
  historicalDataset = "",
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
        <MonitorChartsModal
          node={monitorNode}
          source={monitorSource}
          comparisonCandidateGroups={comparisonCandidateGroups}
          onClose={() => setIsMonitorOpen(false)}
        />
      )}
    </>
  );
}
