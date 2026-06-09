import dynamic from "next/dynamic";
import { CompareWorkspace } from "@/components/features/viewer/compare-workspace";
import { ConnectedMonitorChartsModal } from "@/components/features/viewer/connected-monitor-charts-modal";
import { ConnectedTrainingPanel } from "@/components/features/viewer/connected-training-panel";
import { AppHeader } from "@/components/features/viewer/screen/app-header";
import { NodeDetailsPanel } from "@/components/features/viewer/screen/node-details-panel";
import { PreviewPanel } from "@/components/features/viewer/screen/preview-panel";
import { PreviewToolbar } from "@/components/features/viewer/screen/preview-toolbar";
import { type LogsWorkspaceState } from "@/components/features/viewer/state/use-logs-workspace-state";
import { ViewerModelSidebar } from "@/components/features/viewer/viewer-model-sidebar";
import { ViewerWorkspaceNav } from "@/components/features/viewer/viewer-workspace-nav";
import { type ViewerWorkspace } from "@/types/viewer";

const FullConfigDialog = dynamic(
  () =>
    import("@/components/features/viewer/config/full-config-dialog").then(
      (module) => module.FullConfigDialog,
    ),
  { ssr: false },
);
const FeatureListDialog = dynamic(
  () =>
    import("@/components/features/viewer/feature-list-dialog").then(
      (module) => module.FeatureListDialog,
    ),
  { ssr: false },
);
const LogsSidebarPanel = dynamic(
  () =>
    import("@/components/features/viewer/logs-workspace").then(
      (module) => module.LogsSidebarPanel,
    ),
  { ssr: false },
);
const LogsGraphPreviewPanel = dynamic(
  () =>
    import("@/components/features/viewer/logs-workspace").then(
      (module) => module.LogsGraphPreviewPanel,
    ),
  { ssr: false },
);
const LogRunDetailsPanel = dynamic(
  () =>
    import("@/components/features/viewer/logs/log-run-details-panel").then(
      (module) => module.LogRunDetailsPanel,
    ),
  { ssr: false },
);

type ViewerScreenProps = {
  activeWorkspace: ViewerWorkspace;
  onChangeWorkspace: (workspace: ViewerWorkspace) => void;
  isFullConfigOpen: boolean;
  onOpenFullConfig: () => void;
  onCloseFullConfig: () => void;
  isFeatureListOpen: boolean;
  onOpenFeatureList: () => void;
  onCloseFeatureList: () => void;
  logsState: LogsWorkspaceState;
};

export function ViewerScreen({
  activeWorkspace,
  onChangeWorkspace,
  isFullConfigOpen,
  onOpenFullConfig,
  onCloseFullConfig,
  isFeatureListOpen,
  onOpenFeatureList,
  onCloseFeatureList,
  logsState,
}: ViewerScreenProps) {
  const isModelWorkspace = activeWorkspace === "model";
  const isLogsWorkspace = activeWorkspace === "logs";

  return (
    <main className="grid h-dvh min-h-0 grid-rows-[60px_minmax(0,1fr)_auto] overflow-hidden bg-bg text-ink">
      <AppHeader activeWorkspace={activeWorkspace} onOpenFeatureList={onOpenFeatureList} />

      <section className="grid min-h-0 grid-cols-1 overflow-auto lg:grid-cols-[344px_minmax(0,1fr)_332px] lg:overflow-hidden">
        <aside className="min-h-0 overflow-y-auto border-b border-line bg-[linear-gradient(180deg,rgba(13,12,22,0.6),rgba(8,8,13,0.4))] px-4 pb-7 pt-[18px] backdrop-blur lg:border-b-0 lg:border-r">
          <div className="grid gap-[22px]">
            <ViewerWorkspaceNav
              activeWorkspace={activeWorkspace}
              onChange={onChangeWorkspace}
            />

            {isModelWorkspace ? (
              <ViewerModelSidebar onOpenFullConfig={onOpenFullConfig} />
            ) : isLogsWorkspace ? (
              <LogsSidebarPanel state={logsState} />
            ) : null}
          </div>
        </aside>

        {isModelWorkspace ? (
          <>
            <div className="grid min-h-[560px] grid-rows-[56px_minmax(0,1fr)] bg-transparent lg:min-h-0">
              <PreviewToolbar />
              <PreviewPanel />
            </div>

            <NodeDetailsPanel />
          </>
        ) : isLogsWorkspace ? (
          <>
            <LogsGraphPreviewPanel state={logsState} />
            <LogRunDetailsPanel selectedRun={logsState.selectedRun} />
          </>
        ) : (
          <div className="h-full min-h-[560px] min-w-0 overflow-hidden lg:col-span-2 lg:min-h-0">
            <CompareWorkspace onUseTarget={() => onChangeWorkspace("model")} />
          </div>
        )}
      </section>

      {isModelWorkspace && isFullConfigOpen && (
        <FullConfigDialog onClose={onCloseFullConfig} />
      )}
      {isFeatureListOpen && <FeatureListDialog onClose={onCloseFeatureList} />}
      {isModelWorkspace && <ConnectedMonitorChartsModal />}
      {isModelWorkspace && (
        <ConnectedTrainingPanel onOpenFullConfig={onOpenFullConfig} />
      )}
    </main>
  );
}
