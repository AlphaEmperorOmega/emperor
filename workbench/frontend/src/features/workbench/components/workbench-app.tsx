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

function TrainingExecutionProviderLoadingFallback() {
  return (
    <WorkbenchWideWorkspaceRegion>
      <WorkbenchWorkspaceLoadingStatus label="Loading training workspace…" />
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

const DeferredTrainingExecutionProvider = dynamic(
  () =>
    import(
      "@/features/workbench/providers/training-execution-provider"
    ).then((module) => module.TrainingExecutionProvider),
  {
    ssr: false,
    loading: TrainingExecutionProviderLoadingFallback,
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
  const hasDeferredWorkspace = workspaceShell.deferredWorkspaceOrder.length > 0;
  const workspaceBoundary = hasDeferredWorkspace
    ? (content: ReactNode) => {
        let boundedContent = content;
        for (
          let index = workspaceShell.deferredWorkspaceOrder.length - 1;
          index >= 0;
          index -= 1
        ) {
          const workspace = workspaceShell.deferredWorkspaceOrder[index];
          if (workspace === "logs") {
            boundedContent = (
              <DeferredLogsWorkspaceProvider
                enabled={workspaceShell.screen.activeWorkspace === "logs"}
                startedExperiments={startedLogFolders}
              >
                {boundedContent}
              </DeferredLogsWorkspaceProvider>
            );
          } else {
            boundedContent = (
              <DeferredTrainingExecutionProvider
                activeWorkspace={workspaceShell.screen.activeWorkspace}
                onOpenFullConfig={workspaceShell.screen.fullConfigDialog.open}
              >
                {boundedContent}
              </DeferredTrainingExecutionProvider>
            );
          }
        }
        return boundedContent;
      }
    : undefined;

  return (
    <WorkbenchProviders
      activeWorkspace={workspaceShell.screen.activeWorkspace}
      onJobStarted={rememberStartedLogFolder}
      clearShellForConnectionChange={clearStartedLogFoldersForConnectionChange}
    >
      <WorkbenchScreen
        shell={workspaceShell.screen}
        workspaceBoundary={workspaceBoundary}
      />
    </WorkbenchProviders>
  );
}
