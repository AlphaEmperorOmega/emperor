import { AppHeader } from "@/features/viewer/components/screen/app-header";
import {
  ViewerWorkspaceMain,
  ViewerWorkspaceOverlays,
  ViewerWorkspaceSidebar,
} from "@/features/viewer/components/viewer-workspaces";
import { type ViewerScreenShell } from "@/features/viewer/state/use-viewer-workspace-shell";

export function ViewerScreen({ shell }: { shell: ViewerScreenShell }) {
  const {
    activeWorkspace,
    onChangeWorkspace,
    fullConfigDialog,
    featureListDialog,
    apiConnectionDialog,
  } = shell;

  return (
    <main className="grid h-dvh min-h-0 grid-rows-[60px_minmax(0,1fr)_auto] overflow-hidden bg-bg text-ink">
      <a
        href="#viewer-workspace-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-3 focus:top-3 focus:z-[100] focus:rounded-[9px] focus:border focus:border-focus focus:bg-panel focus:px-3 focus:py-2 focus:text-sm focus:font-semibold focus:text-ink focus:outline-none"
      >
        Skip to workspace
      </a>
      <AppHeader
        activeWorkspace={activeWorkspace}
        onOpenFeatureList={featureListDialog.open}
        onOpenApiConnection={apiConnectionDialog.open}
      />

      <section
        id="viewer-workspace-content"
        tabIndex={-1}
        className="grid min-h-0 grid-cols-1 overflow-auto lg:grid-cols-[344px_minmax(0,1fr)_332px] lg:overflow-hidden"
      >
        <ViewerWorkspaceSidebar
          activeWorkspace={activeWorkspace}
          onChangeWorkspace={onChangeWorkspace}
          onOpenFullConfig={fullConfigDialog.open}
        />
        <ViewerWorkspaceMain
          activeWorkspace={activeWorkspace}
          onChangeWorkspace={onChangeWorkspace}
        />
      </section>

      <ViewerWorkspaceOverlays
        activeWorkspace={activeWorkspace}
        fullConfigDialog={fullConfigDialog}
        featureListDialog={featureListDialog}
        apiConnectionDialog={apiConnectionDialog}
      />
    </main>
  );
}
