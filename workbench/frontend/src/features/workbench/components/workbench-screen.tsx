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
    <main className="grid h-dvh min-h-0 grid-rows-[60px_minmax(0,1fr)] overflow-hidden bg-bg text-ink">
      <a
        href="#workbench-workspace-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-[100] focus:rounded-[9px] focus:border focus:border-focus focus:bg-panel focus:px-3 focus:py-2 focus:text-sm focus:font-semibold focus:text-ink focus:outline-none"
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
