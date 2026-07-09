import { LazyMonitorChartsModal } from "@/features/workbench/components/monitor/lazy-monitor-charts-modal";
import { useGraphMonitor } from "@/features/workbench/providers/workbench-providers";

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
