import { LazyMonitorChartsModal } from "@/features/viewer/components/monitor/lazy-monitor-charts-modal";
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
    <LazyMonitorChartsModal
      node={graphMonitorNode}
      source={graphMonitorSource}
      comparisonCandidateGroups={graphMonitorComparisonCandidateGroups}
      onClose={closeGraphNodeMonitor}
    />
  );
}
