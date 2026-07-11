import dynamic from "next/dynamic";
import { NodeDetailsPanel } from "@/features/workbench/components/screen/node-details-panel";
import { PreviewPanel } from "@/features/workbench/components/screen/preview-panel";
import { PreviewToolbar } from "@/features/workbench/components/screen/preview-toolbar";
import { WorkbenchModelSidebar } from "@/features/workbench/components/workbench-model-sidebar";
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
  return (
    <div
      className="grid h-full min-h-[560px] place-items-center lg:min-h-0"
      role="status"
      aria-label="Loading training workspace"
    >
      <span className="text-xs text-ink-faint">Loading training workspace</span>
    </div>
  );
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

export function WorkbenchWorkspaceSidebar({
  activeWorkspace,
  onOpenFullConfig,
}: {
  activeWorkspace: WorkbenchWorkspace;
  onOpenFullConfig: FullConfigDialogControls["open"];
}) {
  const isModelWorkspace = activeWorkspace === "model";
  const isLogsWorkspace = activeWorkspace === "logs";

  if (activeWorkspace === "training") {
    return null;
  }

  return (
    <aside className="min-h-0 overflow-y-auto border-b border-line bg-[linear-gradient(180deg,rgba(13,12,22,0.6),rgba(8,8,13,0.4))] px-4 pb-7 pt-[18px] backdrop-blur lg:border-b-0 lg:border-r">
      <div className="grid gap-[22px]">
        {isModelWorkspace ? (
          <WorkbenchModelSidebar onOpenFullConfig={onOpenFullConfig} />
        ) : isLogsWorkspace ? (
          <ConnectedLogsSidebarPanel />
        ) : null}
      </div>
    </aside>
  );
}

export function WorkbenchWorkspaceMain({
  activeWorkspace,
}: {
  activeWorkspace: WorkbenchWorkspace;
}) {
  if (activeWorkspace === "model") {
    return (
      <>
        <div className="grid min-h-[560px] grid-rows-[56px_minmax(0,1fr)] bg-transparent lg:min-h-0">
          <PreviewToolbar />
          <PreviewPanel />
        </div>

        <NodeDetailsPanel />
      </>
    );
  }

  if (activeWorkspace === "logs") {
    return (
      <>
        <ConnectedLogsGraphPreviewPanel />
        <ConnectedLogRunDetailsPanel />
      </>
    );
  }

  if (activeWorkspace === "training") {
    return (
      <div className="grid h-full min-h-[560px] min-w-0 grid-rows-[minmax(0,1fr)] overflow-hidden lg:col-span-3 lg:min-h-0">
        <TrainingPanel />
      </div>
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
