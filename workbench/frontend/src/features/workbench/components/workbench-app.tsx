"use client";
// Client boundary: owns Workbench workspace/dialog state and workbench providers.

import { WorkbenchProviders } from "@/features/workbench/providers/workbench-providers";
import { LogsWorkspaceProvider } from "@/features/workbench/providers/logs-workspace-provider";
import { WorkbenchScreen } from "@/features/workbench/components/workbench-screen";
import { useWorkbenchWorkspaceShell } from "@/features/workbench/state/use-workbench-workspace-shell";

export function WorkbenchApp() {
  const workspaceShell = useWorkbenchWorkspaceShell();

  return (
    <WorkbenchProviders activeWorkspace={workspaceShell.screen.activeWorkspace}>
      <LogsWorkspaceProvider enabled={workspaceShell.screen.activeWorkspace === "logs"}>
        <WorkbenchScreen shell={workspaceShell.screen} />
      </LogsWorkspaceProvider>
    </WorkbenchProviders>
  );
}
