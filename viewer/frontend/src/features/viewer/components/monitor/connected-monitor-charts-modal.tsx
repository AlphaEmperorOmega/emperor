import { MonitorChartsModal } from "@/features/viewer/components/monitor/monitor-charts-modal";
import { useGraphMonitor } from "@/features/viewer/providers/viewer-providers";

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
