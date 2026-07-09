import { AppHeader } from "@/features/workbench/components/screen/app-header";
import {
  WorkbenchWorkspaceMain,
  WorkbenchWorkspaceOverlays,
  WorkbenchWorkspaceSidebar,
} from "@/features/workbench/components/workbench-workspaces";
import { type WorkbenchScreenShell } from "@/features/workbench/state/use-workbench-workspace-shell";

export function WorkbenchScreen({ shell }: { shell: WorkbenchScreenShell }) {
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

      <section
        id="workbench-workspace-content"
        tabIndex={-1}
        className="grid min-h-0 grid-cols-1 overflow-auto lg:grid-cols-[344px_minmax(0,1fr)_332px] lg:overflow-hidden"
      >
        <WorkbenchWorkspaceSidebar
          activeWorkspace={activeWorkspace}
          onOpenFullConfig={fullConfigDialog.open}
        />
        <WorkbenchWorkspaceMain
          activeWorkspace={activeWorkspace}
          onOpenFullConfig={fullConfigDialog.open}
        />
      </section>

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
