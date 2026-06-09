import dynamic from "next/dynamic";
import { CompareWorkspace } from "@/features/viewer/components/compare-workspace";
import { ConnectedMonitorChartsModal } from "@/features/viewer/components/monitor/connected-monitor-charts-modal";
import { ConnectedTrainingPanel } from "@/features/viewer/components/connected-training-panel";
import { NodeDetailsPanel } from "@/features/viewer/components/screen/node-details-panel";
import { PreviewPanel } from "@/features/viewer/components/screen/preview-panel";
import { PreviewToolbar } from "@/features/viewer/components/screen/preview-toolbar";
import { ViewerModelSidebar } from "@/features/viewer/components/viewer-model-sidebar";
import { ViewerWorkspaceNav } from "@/features/viewer/components/viewer-workspace-nav";
import { type ViewerDialogControls } from "@/features/viewer/state/use-viewer-workspace-shell";
import { type ViewerWorkspace } from "@/types/viewer";

const FullConfigDialog = dynamic(
  () =>
    import("@/features/viewer/components/config/full-config-dialog").then(
      (module) => module.FullConfigDialog,
    ),
  { ssr: false },
);
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

export function ViewerWorkspaceSidebar({
  activeWorkspace,
  onChangeWorkspace,
  onOpenFullConfig,
}: {
  activeWorkspace: ViewerWorkspace;
  onChangeWorkspace: (workspace: ViewerWorkspace) => void;
  onOpenFullConfig: () => void;
}) {
  const isModelWorkspace = activeWorkspace === "model";
  const isLogsWorkspace = activeWorkspace === "logs";

  return (
    <aside className="min-h-0 overflow-y-auto border-b border-line bg-[linear-gradient(180deg,rgba(13,12,22,0.6),rgba(8,8,13,0.4))] px-4 pb-7 pt-[18px] backdrop-blur lg:border-b-0 lg:border-r">
      <div className="grid gap-[22px]">
        <ViewerWorkspaceNav
          activeWorkspace={activeWorkspace}
          onChange={onChangeWorkspace}
        />

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
  onChangeWorkspace,
}: {
  activeWorkspace: ViewerWorkspace;
  onChangeWorkspace: (workspace: ViewerWorkspace) => void;
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

  return (
    <div className="h-full min-h-[560px] min-w-0 overflow-hidden lg:col-span-2 lg:min-h-0">
      <CompareWorkspace onUseTarget={() => onChangeWorkspace("model")} />
    </div>
  );
}

export function ViewerWorkspaceOverlays({
  activeWorkspace,
  fullConfigDialog,
  featureListDialog,
}: {
  activeWorkspace: ViewerWorkspace;
  fullConfigDialog: ViewerDialogControls;
  featureListDialog: ViewerDialogControls;
}) {
  const isModelWorkspace = activeWorkspace === "model";

  return (
    <>
      {isModelWorkspace && fullConfigDialog.isOpen && (
        <FullConfigDialog onClose={fullConfigDialog.close} />
      )}
      {featureListDialog.isOpen && (
        <FeatureListDialog onClose={featureListDialog.close} />
      )}
      {isModelWorkspace && <ConnectedMonitorChartsModal />}
      {isModelWorkspace && (
        <ConnectedTrainingPanel onOpenFullConfig={fullConfigDialog.open} />
      )}
    </>
  );
}
