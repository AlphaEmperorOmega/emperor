import { LogsChartPanel } from "@/features/workbench/components/logs/logs-chart-panel";
import { LogsSidebar } from "@/features/workbench/components/logs/logs-sidebar";
import {
  useLogsBrowser,
  useLogsCharts,
  useLogsDeletion,
} from "@/features/workbench/providers/logs-workspace-provider";

export function ConnectedLogsSidebarPanel() {
  const browser = useLogsBrowser();
  const deletion = useLogsDeletion();
  return <LogsSidebar browser={browser} deletion={deletion} />;
}

export function ConnectedLogsGraphPreviewPanel() {
  const charts = useLogsCharts();
  return <LogsChartPanel charts={charts} />;
}
