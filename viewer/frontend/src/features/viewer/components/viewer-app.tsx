"use client";
// Client boundary: owns Viewer workspace/dialog state and viewer providers.

import { ViewerProviders } from "@/features/viewer/providers/viewer-providers";
import { LogsWorkspaceProvider } from "@/features/viewer/providers/logs-workspace-provider";
import { ViewerScreen } from "@/features/viewer/components/viewer-screen";
import { useViewerWorkspaceShell } from "@/features/viewer/state/use-viewer-workspace-shell";

export function ViewerApp() {
  const workspaceShell = useViewerWorkspaceShell();

  return (
    <ViewerProviders onJobStarted={workspaceShell.onJobStarted}>
      <LogsWorkspaceProvider state={workspaceShell.logsWorkspaceState}>
        <ViewerScreen shell={workspaceShell.screen} />
      </LogsWorkspaceProvider>
    </ViewerProviders>
  );
}
