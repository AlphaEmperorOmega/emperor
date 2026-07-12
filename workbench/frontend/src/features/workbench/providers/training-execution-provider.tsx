"use client";

import { useCallback, type ReactNode } from "react";
import {
  TrainingWorkspaceContextProvider,
  useTrainingDraft,
  useTrainingPolling,
} from "@/features/workbench/providers/training-provider";
import {
  isWorkbenchProtectedAccessReady,
  useRegisterWorkbenchConnectionReset,
  useWorkbenchCapabilities,
  useWorkbenchConnection,
} from "@/features/workbench/providers/workbench-connection-provider";
import { useConfigSnapshotEditor } from "@/features/workbench/providers/workbench-providers";
import { useTrainingWorkspaceState } from "@/features/workbench/state/training/use-training-workspace-state";
import { useTrainingJobExecution } from "@/features/workbench/state/training/use-training-job-execution";
import { type TrainingDraftSeed } from "@/features/workbench/state/training/use-training-draft-state";
import { type FullConfigDialogControls } from "@/features/workbench/state/use-workbench-workspace-shell";
import { type ConfigSnapshot } from "@/lib/config-snapshots";
import { type WorkbenchWorkspace } from "@/types/workbench";

export function TrainingExecutionProvider({
  activeWorkspace,
  onOpenFullConfig,
  children,
}: {
  activeWorkspace: WorkbenchWorkspace;
  onOpenFullConfig: FullConfigDialogControls["open"];
  children: ReactNode;
}) {
  const configuration = useTrainingDraft();
  const polling = useTrainingPolling();
  const snapshotEditor = useConfigSnapshotEditor();
  const { capabilities } = useWorkbenchCapabilities();
  const workbenchConnection = useWorkbenchConnection();
  const protectedReadsEnabled =
    activeWorkspace === "training" &&
    isWorkbenchProtectedAccessReady(workbenchConnection);
  const lifecycle = useTrainingJobExecution({
    enabled: isWorkbenchProtectedAccessReady(workbenchConnection),
    polling,
  });
  const openTrainingConfig = useCallback(
    () => onOpenFullConfig("default", "training"),
    [onOpenFullConfig],
  );
  const createPresetSnapshot = useCallback(
    (target: TrainingDraftSeed) => {
      if (snapshotEditor.actions.beginDraft(target)) {
        onOpenFullConfig("snapshotDraft");
      }
    },
    [onOpenFullConfig, snapshotEditor.actions],
  );
  const editConfigSnapshot = useCallback(
    (snapshot: ConfigSnapshot) => {
      if (snapshotEditor.actions.beginEdit(snapshot)) {
        onOpenFullConfig("snapshotEdit");
      }
    },
    [onOpenFullConfig, snapshotEditor.actions],
  );
  const duplicateConfigSnapshot = useCallback(
    (snapshot: ConfigSnapshot) => {
      if (snapshotEditor.actions.beginDuplicate(snapshot)) {
        onOpenFullConfig("snapshotDraft");
      }
    },
    [onOpenFullConfig, snapshotEditor.actions],
  );
  const training = useTrainingWorkspaceState({
    configuration,
    trainingEnabled: capabilities.trainingEnabled && protectedReadsEnabled,
    protectedReadsEnabled,
    onOpenFullConfig: openTrainingConfig,
    onCreatePresetSnapshot: createPresetSnapshot,
    onEditConfigSnapshot: editConfigSnapshot,
    onDuplicateConfigSnapshot: duplicateConfigSnapshot,
    trainingJob: lifecycle,
  });
  useRegisterWorkbenchConnectionReset(training.clearForConnectionChange);
  useRegisterWorkbenchConnectionReset(lifecycle.clearForConnectionChange);

  return (
    <TrainingWorkspaceContextProvider value={training.workspace}>
      {children}
    </TrainingWorkspaceContextProvider>
  );
}
