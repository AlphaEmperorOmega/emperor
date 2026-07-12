import dynamic from "next/dynamic";
import { NodeDetailsPanel } from "@/features/workbench/components/screen/node-details-panel";
import { PreviewPanel } from "@/features/workbench/components/screen/preview-panel";
import { PreviewToolbar } from "@/features/workbench/components/screen/preview-toolbar";
import { WorkbenchModelSidebar } from "@/features/workbench/components/workbench-model-sidebar";
import {
  WorkbenchThreeRegionLayout,
  WorkbenchWideWorkspaceRegion,
  WorkbenchWorkspaceLoadingStatus,
} from "@/features/workbench/components/workbench-workspace-layout";
import {
  useGraphMonitor,
  useGraphView,
} from "@/features/workbench/providers/workbench-providers";
import {
  type FullConfigDialogControls,
  type WorkbenchDialogControls,
} from "@/features/workbench/state/use-workbench-workspace-shell";
import { type WorkbenchWorkspace } from "@/types/workbench";

function TrainingWorkspaceLoadingFallback() {
  return <WorkbenchWorkspaceLoadingStatus label="Loading training workspace…" />;
}

const TrainingPanel = dynamic(
  () =>
    import("@/features/workbench/components/training-panel").then(
      (module) => module.TrainingPanel,
    ),
  {
    ssr: false,
    loading: TrainingWorkspaceLoadingFallback,
  },
);
const FullConfigDialog = dynamic(
  () =>
    import("@/features/workbench/components/config/full-config-dialog").then(
      (module) => module.FullConfigDialog,
    ),
  { ssr: false },
);
const ApiConnectionDialog = dynamic(
  () =>
    import(
      "@/features/workbench/components/screen/api-connection-dialog"
    ).then((module) => module.ApiConnectionDialog),
  { ssr: false },
);
const ImportLogsDialog = dynamic(
  () =>
    import("@/features/workbench/components/screen/import-logs-dialog").then(
      (module) => module.ImportLogsDialog,
    ),
  { ssr: false },
);

const FeatureListDialog = dynamic(
  () =>
    import("@/features/workbench/components/feature-list-dialog").then(
      (module) => module.FeatureListDialog,
    ),
  { ssr: false },
);
const ConnectedLogsSidebarPanel = dynamic(
  () =>
    import("@/features/workbench/components/logs/logs-workspace").then(
      (module) => module.ConnectedLogsSidebarPanel,
    ),
  { ssr: false },
);
const ConnectedLogsGraphPreviewPanel = dynamic(
  () =>
    import("@/features/workbench/components/logs/logs-workspace").then(
      (module) => module.ConnectedLogsGraphPreviewPanel,
    ),
  { ssr: false },
);
const ConnectedLogRunDetailsPanel = dynamic(
  () =>
    import("@/features/workbench/components/logs/log-run-details-panel").then(
      (module) => module.ConnectedLogRunDetailsPanel,
    ),
  { ssr: false },
);
const ConnectedMonitorChartsModal = dynamic(
  () =>
    import(
      "@/features/workbench/components/monitor/connected-monitor-charts-modal"
    ).then((module) => module.ConnectedMonitorChartsModal),
  { ssr: false },
);
const ConnectedNeuronCluster3DPopup = dynamic(
  () =>
    import("@/features/workbench/components/graph/neuron-cluster-3d-popup").then(
      (module) => module.ConnectedNeuronCluster3DPopup,
    ),
  { ssr: false },
);

export function WorkbenchWorkspaceRegions({
  activeWorkspace,
  onOpenFullConfig,
}: {
  activeWorkspace: WorkbenchWorkspace;
  onOpenFullConfig: FullConfigDialogControls["open"];
}) {
  if (activeWorkspace === "model") {
    return (
      <WorkbenchThreeRegionLayout
        sidebar={
          <WorkbenchModelSidebar onOpenFullConfig={onOpenFullConfig} />
        }
        primary={
          <div className="grid h-full min-h-0 grid-rows-[56px_minmax(0,1fr)] bg-transparent">
            <PreviewToolbar />
            <PreviewPanel />
          </div>
        }
        details={<NodeDetailsPanel />}
      />
    );
  }

  if (activeWorkspace === "logs") {
    return (
      <WorkbenchThreeRegionLayout
        sidebar={<ConnectedLogsSidebarPanel />}
        primary={<ConnectedLogsGraphPreviewPanel />}
        details={<ConnectedLogRunDetailsPanel />}
      />
    );
  }

  if (activeWorkspace === "training") {
    return (
      <WorkbenchWideWorkspaceRegion>
        <TrainingPanel />
      </WorkbenchWideWorkspaceRegion>
    );
  }

  return null;
}

export function WorkbenchWorkspaceOverlays({
  activeWorkspace,
  fullConfigDialog,
  featureListDialog,
  apiConnectionDialog,
  importLogsDialog,
}: {
  activeWorkspace: WorkbenchWorkspace;
  fullConfigDialog: FullConfigDialogControls;
  featureListDialog: WorkbenchDialogControls;
  apiConnectionDialog: WorkbenchDialogControls;
  importLogsDialog: WorkbenchDialogControls;
}) {
  const isModelWorkspace = activeWorkspace === "model";
  const { cluster3dNodeId } = useGraphView();
  const { graphMonitorNode, graphMonitorSource } = useGraphMonitor();

  return (
    <>
      {fullConfigDialog.isOpen && (
        <FullConfigDialog
          mode={fullConfigDialog.mode}
          scope={fullConfigDialog.scope}
          onClose={fullConfigDialog.close}
        />
      )}
      {featureListDialog.isOpen && (
        <FeatureListDialog onClose={featureListDialog.close} />
      )}
      {apiConnectionDialog.isOpen && (
        <ApiConnectionDialog onClose={apiConnectionDialog.close} />
      )}
      {importLogsDialog.isOpen && (
        <ImportLogsDialog onClose={importLogsDialog.close} />
      )}
      {isModelWorkspace && graphMonitorNode && graphMonitorSource && (
        <ConnectedMonitorChartsModal />
      )}
      {isModelWorkspace && cluster3dNodeId && <ConnectedNeuronCluster3DPopup />}
    </>
  );
}
