import dynamic from "next/dynamic";
import { useGraphMonitor } from "@/features/viewer/providers/viewer-providers";

const MonitorChartsModal = dynamic(
  () =>
    import("@/features/viewer/components/monitor/monitor-charts-modal").then(
      (module) => module.MonitorChartsModal,
    ),
  { ssr: false },
);

export function ConnectedMonitorChartsModal() {
  const {
    graphMonitorNode,
    graphMonitorSource,
    graphMonitorComparisonCandidateGroups,
    closeGraphNodeMonitor,
  } = useGraphMonitor();

  if (!graphMonitorNode || !graphMonitorSource) {
    return null;
  }

  return (
    <MonitorChartsModal
      node={graphMonitorNode}
      source={graphMonitorSource}
      comparisonCandidateGroups={graphMonitorComparisonCandidateGroups}
      onClose={closeGraphNodeMonitor}
    />
  );
}
