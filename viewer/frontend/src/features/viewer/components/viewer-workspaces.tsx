import dynamic from "next/dynamic";
import { FullConfigDialog } from "@/features/viewer/components/config/full-config-dialog";
import { ConnectedMonitorChartsModal } from "@/features/viewer/components/monitor/connected-monitor-charts-modal";
import { ConnectedTrainingWorkspace } from "@/features/viewer/components/connected-training-panel";
import { ApiConnectionDialog } from "@/features/viewer/components/screen/api-connection-dialog";
import { ImportLogsDialog } from "@/features/viewer/components/screen/import-logs-dialog";
import { NodeDetailsPanel } from "@/features/viewer/components/screen/node-details-panel";
import { PreviewPanel } from "@/features/viewer/components/screen/preview-panel";
import { PreviewToolbar } from "@/features/viewer/components/screen/preview-toolbar";
import { ViewerModelSidebar } from "@/features/viewer/components/viewer-model-sidebar";
import { useGraphView } from "@/features/viewer/providers/viewer-providers";
import {
  type FullConfigDialogControls,
  type ViewerDialogControls,
} from "@/features/viewer/state/use-viewer-workspace-shell";
import { type ViewerWorkspace } from "@/types/viewer";

const FeatureListDialog = dynamic(
  () =>
    import("@/features/viewer/components/feature-list-dialog").then(
      (module) => module.FeatureListDialog,
    ),
  { ssr: false },
);
const ConnectedLogsSidebarPanel = dynamic(
  () =>
    import("@/features/viewer/components/logs/logs-workspace").then(
      (module) => module.ConnectedLogsSidebarPanel,
    ),
  { ssr: false },
);
const ConnectedLogsGraphPreviewPanel = dynamic(
  () =>
    import("@/features/viewer/components/logs/logs-workspace").then(
      (module) => module.ConnectedLogsGraphPreviewPanel,
    ),
  { ssr: false },
);
const ConnectedLogRunDetailsPanel = dynamic(
  () =>
    import("@/features/viewer/components/logs/log-run-details-panel").then(
      (module) => module.ConnectedLogRunDetailsPanel,
    ),
  { ssr: false },
);
const ConnectedNeuronCluster3DPopup = dynamic(
  () =>
    import("@/features/viewer/components/graph/neuron-cluster-3d-popup").then(
      (module) => module.ConnectedNeuronCluster3DPopup,
    ),
  { ssr: false },
);

export function ViewerWorkspaceSidebar({
  activeWorkspace,
  onOpenFullConfig,
}: {
  activeWorkspace: ViewerWorkspace;
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
          <ViewerModelSidebar onOpenFullConfig={onOpenFullConfig} />
        ) : isLogsWorkspace ? (
          <ConnectedLogsSidebarPanel />
        ) : null}
      </div>
    </aside>
  );
}

export function ViewerWorkspaceMain({
  activeWorkspace,
  onOpenFullConfig,
}: {
  activeWorkspace: ViewerWorkspace;
  onOpenFullConfig: FullConfigDialogControls["open"];
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
      <div className="h-full min-h-[560px] min-w-0 overflow-hidden lg:col-span-3 lg:min-h-0">
        <ConnectedTrainingWorkspace onOpenFullConfig={onOpenFullConfig} />
      </div>
    );
  }

  return null;
}

export function ViewerWorkspaceOverlays({
  activeWorkspace,
  fullConfigDialog,
  featureListDialog,
  apiConnectionDialog,
  importLogsDialog,
}: {
  activeWorkspace: ViewerWorkspace;
  fullConfigDialog: FullConfigDialogControls;
  featureListDialog: ViewerDialogControls;
  apiConnectionDialog: ViewerDialogControls;
  importLogsDialog: ViewerDialogControls;
}) {
  const isModelWorkspace = activeWorkspace === "model";
  const { cluster3dNodeId } = useGraphView();

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
      {isModelWorkspace && <ConnectedMonitorChartsModal />}
      {isModelWorkspace && cluster3dNodeId && <ConnectedNeuronCluster3DPopup />}
    </>
  );
}
