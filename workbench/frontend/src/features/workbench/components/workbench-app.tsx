"use client";
// Client boundary: owns Workbench workspace/dialog state and workbench providers.

import dynamic from "next/dynamic";
import { type ReactNode, useCallback, useState } from "react";
import { WorkbenchProviders } from "@/features/workbench/providers/workbench-providers";
import { WorkbenchScreen } from "@/features/workbench/components/workbench-screen";
import {
  WorkbenchWideWorkspaceRegion,
  WorkbenchWorkspaceLoadingStatus,
} from "@/features/workbench/components/workbench-workspace-layout";
import { useWorkbenchWorkspaceShell } from "@/features/workbench/state/use-workbench-workspace-shell";

function LogsWorkspaceProviderLoadingFallback() {
  return (
    <WorkbenchWideWorkspaceRegion>
      <WorkbenchWorkspaceLoadingStatus label="Loading logs workspace…" />
    </WorkbenchWideWorkspaceRegion>
  );
}

const DeferredLogsWorkspaceProvider = dynamic(
  () =>
    import("@/features/workbench/providers/logs-workspace-provider").then(
      (module) => module.LogsWorkspaceProvider,
    ),
  {
    ssr: false,
    loading: LogsWorkspaceProviderLoadingFallback,
  },
);

export function WorkbenchApp() {
  const workspaceShell = useWorkbenchWorkspaceShell();
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
  const workspaceBoundary = workspaceShell.logsWorkspaceActivated
    ? (content: ReactNode) => (
        <DeferredLogsWorkspaceProvider
          enabled={workspaceShell.screen.activeWorkspace === "logs"}
          startedExperiments={startedLogFolders}
        >
          {content}
        </DeferredLogsWorkspaceProvider>
      )
    : undefined;

  return (
    <WorkbenchProviders
      activeWorkspace={workspaceShell.screen.activeWorkspace}
      onJobStarted={rememberStartedLogFolder}
      onOpenFullConfig={workspaceShell.screen.fullConfigDialog.open}
      clearShellForConnectionChange={clearStartedLogFoldersForConnectionChange}
    >
      <WorkbenchScreen
        shell={workspaceShell.screen}
        workspaceBoundary={workspaceBoundary}
      />
    </WorkbenchProviders>
  );
}
