import { AppHeader } from "@/features/viewer/components/screen/app-header";
import {
  ViewerWorkspaceMain,
  ViewerWorkspaceOverlays,
  ViewerWorkspaceSidebar,
} from "@/features/viewer/components/viewer-workspaces";
import { type ViewerScreenShell } from "@/features/viewer/state/use-viewer-workspace-shell";

export function ViewerScreen({ shell }: { shell: ViewerScreenShell }) {
  const { activeWorkspace, onChangeWorkspace, fullConfigDialog, featureListDialog } = shell;

  return (
    <main className="grid h-dvh min-h-0 grid-rows-[60px_minmax(0,1fr)_auto] overflow-hidden bg-bg text-ink">
      <AppHeader activeWorkspace={activeWorkspace} onOpenFeatureList={featureListDialog.open} />

      <section className="grid min-h-0 grid-cols-1 overflow-auto lg:grid-cols-[344px_minmax(0,1fr)_332px] lg:overflow-hidden">
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
      />
    </main>
  );
}
