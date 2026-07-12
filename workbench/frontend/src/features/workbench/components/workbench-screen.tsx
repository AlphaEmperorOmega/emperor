import { AppHeader } from "@/features/workbench/components/screen/app-header";
import {
  WorkbenchWorkspaceRegions,
  WorkbenchWorkspaceOverlays,
} from "@/features/workbench/components/workbench-workspaces";
import { WorkbenchWorkspaceFrame } from "@/features/workbench/components/workbench-workspace-layout";
import { type WorkbenchScreenShell } from "@/features/workbench/state/use-workbench-workspace-shell";
import { type ReactNode } from "react";

export function WorkbenchScreen({
  shell,
  workspaceBoundary,
}: {
  shell: WorkbenchScreenShell;
  workspaceBoundary?: (content: ReactNode) => ReactNode;
}) {
  const {
    activeWorkspace,
    onChangeWorkspace,
    fullConfigDialog,
    featureListDialog,
    apiConnectionDialog,
    importLogsDialog,
  } = shell;
  return (
    <main className="grid h-dvh min-h-0 grid-rows-[calc(60px+env(safe-area-inset-top))_minmax(0,1fr)] overflow-hidden bg-bg text-ink">
      <a
        href="#workbench-workspace-content"
        className="sr-only focus-visible:not-sr-only focus-visible:absolute focus-visible:left-3 focus-visible:top-3 focus-visible:z-[100] focus-visible:rounded-control focus-visible:border focus-visible:border-violet/70 focus-visible:bg-panel focus-visible:px-3 focus-visible:py-2 focus-visible:text-sm focus-visible:font-semibold focus-visible:text-ink focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
      >
        Skip to workspace
      </a>
      <AppHeader
        activeWorkspace={activeWorkspace}
        onChangeWorkspace={onChangeWorkspace}
        onOpenFeatureList={featureListDialog.open}
        onOpenApiConnection={apiConnectionDialog.open}
        onOpenImportLogs={importLogsDialog.open}
      />

      <WorkbenchWorkspaceFrame workspaceBoundary={workspaceBoundary}>
        <WorkbenchWorkspaceRegions
          activeWorkspace={activeWorkspace}
          onOpenFullConfig={fullConfigDialog.open}
        />
      </WorkbenchWorkspaceFrame>

      <WorkbenchWorkspaceOverlays
        activeWorkspace={activeWorkspace}
        fullConfigDialog={fullConfigDialog}
        featureListDialog={featureListDialog}
        apiConnectionDialog={apiConnectionDialog}
        importLogsDialog={importLogsDialog}
      />
    </main>
  );
}
