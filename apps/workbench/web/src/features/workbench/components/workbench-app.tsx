"use client";
// Client boundary: owns Workbench workspace/dialog state and workbench providers.

import { useCallback, useState } from "react";
import { WorkbenchProviders } from "@/features/workbench/providers/workbench-providers";
import { WorkbenchScreen } from "@/features/workbench/components/workbench-screen";
import { useWorkbenchWorkspaceShell } from "@/features/workbench/state/use-workbench-workspace-shell";
import { type WorkbenchWorkspace } from "@/types/workbench";

export function WorkbenchApp({
  initialWorkspace = "model",
}: {
  initialWorkspace?: WorkbenchWorkspace;
}) {
  const workspaceShell = useWorkbenchWorkspaceShell(initialWorkspace);
  const [startedLogFolders, setStartedLogFolders] = useState<readonly string[]>(
    [],
  );
  const rememberStartedLogFolder = useCallback((logFolder: string) => {
    setStartedLogFolders((current) =>
      current.includes(logFolder) ? current : [...current, logFolder],
    );
  }, []);
  const clearStartedLogFoldersForConnectionChange = useCallback(() => {
    setStartedLogFolders([]);
  }, []);
  const trainingRuntimeActivated =
    workspaceShell.trainingWorkspaceActivated ||
    workspaceShell.screen.fullConfigDialog.isOpen;
  const screen = (
    <WorkbenchScreen
      deferredWorkspaceOrder={workspaceShell.deferredWorkspaceOrder}
      shell={workspaceShell.screen}
      startedLogFolders={startedLogFolders}
      trainingRuntimeActivated={trainingRuntimeActivated}
    />
  );

  return (
    <WorkbenchProviders
      activeWorkspace={workspaceShell.screen.activeWorkspace}
      onJobStarted={rememberStartedLogFolder}
      clearShellForConnectionChange={clearStartedLogFoldersForConnectionChange}
    >
      {screen}
    </WorkbenchProviders>
  );
}
