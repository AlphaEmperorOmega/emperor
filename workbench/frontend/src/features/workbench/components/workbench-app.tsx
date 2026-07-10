"use client";
// Client boundary: owns Workbench workspace/dialog state and workbench providers.

import dynamic from "next/dynamic";
import { type ReactNode, useCallback, useState } from "react";
import { WorkbenchProviders } from "@/features/workbench/providers/workbench-providers";
import { WorkbenchScreen } from "@/features/workbench/components/workbench-screen";
import { useWorkbenchWorkspaceShell } from "@/features/workbench/state/use-workbench-workspace-shell";

function LogsWorkspaceProviderLoadingFallback() {
  return (
    <div
      className="grid h-full min-h-[560px] place-items-center lg:col-span-3 lg:min-h-0"
      role="status"
      aria-label="Loading logs workspace"
    >
      <span className="text-xs text-ink-faint">Loading logs workspace</span>
    </div>
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
      snapshotLibraryEnabled={workspaceShell.screen.fullConfigDialog.isOpen}
      onJobStarted={rememberStartedLogFolder}
    >
      <WorkbenchScreen
        shell={workspaceShell.screen}
        workspaceBoundary={workspaceBoundary}
      />
    </WorkbenchProviders>
  );
}
