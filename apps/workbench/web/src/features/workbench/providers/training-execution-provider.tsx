"use client";

import { useCallback, type ReactNode } from "react";
import {
  TrainingConfigurationContextProvider,
  TrainingDraftContextProvider,
  TrainingWorkspaceContextProvider,
  useTrainingDraft,
} from "@/features/workbench/providers/training-execution-context";
import { useTrainingPolling } from "@/features/workbench/providers/training-provider";
import {
  isWorkbenchProtectedAccessReady,
  useRegisterWorkbenchConnectionReset,
  useWorkbenchCapabilities,
  useWorkbenchConnection,
} from "@/features/workbench/providers/workbench-connection-provider";
import {
  useConfigSnapshotEditor,
  useModelPackageCatalog,
  useModelPackageInspection,
} from "@/features/workbench/providers/workbench-providers";
import { useTrainingConfigurationState } from "@/features/workbench/state/training/use-training-configuration-state";
import { useTrainingWorkspaceState } from "@/features/workbench/state/training/use-training-workspace-state";
import { useTrainingJobExecution } from "@/features/workbench/state/training/use-training-job-execution";
import { type TrainingDraftSeed } from "@/features/workbench/state/training/use-training-draft-state";
import { type FullConfigDialogControls } from "@/features/workbench/state/use-workbench-workspace-shell";
import { type ConfigSnapshot } from "@/lib/config-snapshots";
import { type ModelIdentity } from "@/lib/api/model-catalog";
import { type WorkbenchWorkspace } from "@/types/workbench";

const emptyTrainingDraftSeed: TrainingDraftSeed = {
  modelType: "",
  model: "",
  preset: "",
};

function TrainingExecutionController({
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

function TrainingConfigurationRuntime({
  activeWorkspace,
  models,
  onOpenFullConfig,
  protectedReadsEnabled,
  seed,
  children,
}: {
  activeWorkspace: WorkbenchWorkspace;
  models: ModelIdentity[];
  onOpenFullConfig: FullConfigDialogControls["open"];
  protectedReadsEnabled: boolean;
  seed: TrainingDraftSeed;
  children: ReactNode;
}) {
  const training = useTrainingConfigurationState({
    activeWorkspace,
    models,
    seed,
    protectedReadsEnabled,
  });
  useRegisterWorkbenchConnectionReset(training.clearForConnectionChange);

  return (
    <TrainingConfigurationContextProvider value={training.configuration}>
      <TrainingDraftContextProvider value={training.draft}>
        <TrainingExecutionController
          activeWorkspace={activeWorkspace}
          onOpenFullConfig={onOpenFullConfig}
        >
          {children}
        </TrainingExecutionController>
      </TrainingDraftContextProvider>
    </TrainingConfigurationContextProvider>
  );
}

export function TrainingExecutionProvider({
  activated = true,
  activeWorkspace,
  onOpenFullConfig,
  children,
}: {
  activated?: boolean;
  activeWorkspace: WorkbenchWorkspace;
  onOpenFullConfig: FullConfigDialogControls["open"];
  children: ReactNode;
}) {
  const catalog = useModelPackageCatalog();
  const { capabilities } = useWorkbenchCapabilities();
  const workbenchConnection = useWorkbenchConnection();
  const modelTarget = useModelPackageInspection();
  const protectedReadsEnabled =
    activeWorkspace === "training" &&
    isWorkbenchProtectedAccessReady(workbenchConnection);
  const seed = {
    modelType: modelTarget.browser.selectedModelType,
    model: modelTarget.browser.selectedModel,
    preset: modelTarget.browser.selectedPreset,
  };
  const seedReady = Boolean(activated && seed.modelType && seed.model);

  if (activated && capabilities.trainingEnabled && !seedReady) {
    return null;
  }

  return (
    <TrainingConfigurationRuntime
      key={seedReady ? "captured-training-seed" : "pending-training-seed"}
      activeWorkspace={activeWorkspace}
      models={catalog.modelPackages.records}
      seed={seedReady ? seed : emptyTrainingDraftSeed}
      protectedReadsEnabled={protectedReadsEnabled}
      onOpenFullConfig={onOpenFullConfig}
    >
      {children}
    </TrainingConfigurationRuntime>
  );
}
