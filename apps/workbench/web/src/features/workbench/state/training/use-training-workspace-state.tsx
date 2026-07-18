import { useCallback, useMemo } from "react";
import { buildClusterGrowth } from "@/lib/cluster-growth";
import { type ConfigSnapshot } from "@/lib/config-snapshots";
import { type TrainingDraftState } from "@/features/workbench/state/training/use-training-configuration-state";
import { type TrainingDraftSeed } from "@/features/workbench/state/training/use-training-draft-state";
import { type TrainingJobExecution } from "@/features/workbench/state/training/use-training-job-execution";
import { useTrainingLogFolderState } from "@/features/workbench/state/training/use-training-log-folder-state";
import { useTrainingPlanState } from "@/features/workbench/state/training/use-training-plan-state";

export type TrainingWorkspaceStateInput = {
  configuration: TrainingDraftState;
  trainingEnabled: boolean;
  protectedReadsEnabled: boolean;
  onOpenFullConfig: () => void;
  onCreatePresetSnapshot: (target: TrainingDraftSeed) => void;
  onEditConfigSnapshot: (snapshot: ConfigSnapshot) => void;
  onDuplicateConfigSnapshot: (snapshot: ConfigSnapshot) => void;
  trainingJob: TrainingJobExecution;
};

export function useTrainingWorkspaceState({
  configuration,
  trainingEnabled,
  protectedReadsEnabled,
  onOpenFullConfig,
  onCreatePresetSnapshot,
  onEditConfigSnapshot,
  onDuplicateConfigSnapshot,
  trainingJob,
}: TrainingWorkspaceStateInput) {
  const logFolder = useTrainingLogFolderState({
    enabled: protectedReadsEnabled,
  });
  const modelSetup = configuration.setup.model;
  const variantSetup = configuration.setup.variants;
  const experimentTaskSetup = configuration.setup.experimentTask;
  const datasetSetup = configuration.setup.datasets;
  const monitorSetup = configuration.setup.monitors;
  const runtimeDefaults = configuration.runtimeDefaults;
  const searchMetadata = configuration.searchMetadata;
  const selectedSnapshotIdSet = useMemo(
    () => new Set(variantSetup.selectedSnapshotIds),
    [variantSetup.selectedSnapshotIds],
  );
  const selectedSnapshots = useMemo(
    () =>
      variantSetup.snapshots.filter((snapshot) =>
        selectedSnapshotIdSet.has(snapshot.id),
      ),
    [selectedSnapshotIdSet, variantSetup.snapshots],
  );
  const lifecycle = trainingJob;
  const planState = useTrainingPlanState({
    draft: {
      modelPackage: {
        modelType: modelSetup.selectedType,
        model: modelSetup.selected,
        primaryPreset: variantSetup.primaryPreset,
        selectedPresets: variantSetup.selectedPresets,
        selectedSnapshots,
      },
      experiment: {
        task: experimentTaskSetup.selected,
        datasets: datasetSetup.selected,
        monitors: monitorSetup.selected,
        logFolder: logFolder.state.value,
        hasValidLogFolder: logFolder.state.isValid,
      },
      runtimeDefaults: {
        sections: runtimeDefaults.sections,
        overrides: runtimeDefaults.active,
      },
      searchMetadata,
    },
    availability: {
      trainingEnabled,
      protectedReadsEnabled,
    },
    execution: {
      activeRunPlan: lifecycle.job?.runPlan,
      isJobRunning: lifecycle.isRunning,
      canLaunch: lifecycle.canLaunchRunPlan,
      launch: lifecycle.launchRunPlan,
    },
  });
  const clearLogFolderForConnectionChange =
    logFolder.clearForConnectionChange;
  const clearPlanForConnectionChange = planState.clearForConnectionChange;

  const editConfigSnapshot = useCallback(
    (snapshotId: string) => {
      const snapshot = variantSetup.snapshots.find(
        (candidate) => candidate.id === snapshotId,
      );
      if (snapshot) {
        onEditConfigSnapshot(snapshot);
      }
    },
    [onEditConfigSnapshot, variantSetup.snapshots],
  );
  const createPresetSnapshot = useCallback(
    (preset: string) =>
      onCreatePresetSnapshot({
        modelType: modelSetup.selectedType,
        model: modelSetup.selected,
        preset,
      }),
    [
      modelSetup.selected,
      modelSetup.selectedType,
      onCreatePresetSnapshot,
    ],
  );
  const duplicateConfigSnapshot = useCallback(
    (snapshotId: string) => {
      const snapshot = variantSetup.snapshots.find(
        (candidate) => candidate.id === snapshotId,
      );
      if (snapshot) {
        onDuplicateConfigSnapshot(snapshot);
      }
    },
    [onDuplicateConfigSnapshot, variantSetup.snapshots],
  );
  const clearForConnectionChange = useCallback(() => {
    clearLogFolderForConnectionChange();
    clearPlanForConnectionChange();
  }, [clearLogFolderForConnectionChange, clearPlanForConnectionChange]);

  const presetCountLabel = planState.presetCountLabel;
  const datasetCountLabel = planState.datasetCountLabel;
  const setup = useMemo(
    () => ({
      ...configuration.setup,
      variants: {
        ...variantSetup,
        createPresetSnapshot,
        editSnapshot: editConfigSnapshot,
        duplicateSnapshot: duplicateConfigSnapshot,
      },
    }),
    [
      configuration.setup,
      createPresetSnapshot,
      duplicateConfigSnapshot,
      editConfigSnapshot,
      variantSetup,
    ],
  );
  const draft = useMemo(
    () => ({
      setup,
      logFolder,
      runtimeDefaults,
      searchMetadata,
      status: {
        ...configuration.status,
        trainingEnabled,
        canOpenFullConfig: Boolean(
          modelSetup.selected &&
            variantSetup.primaryPreset &&
            configuration.status.isSchemaReady,
        ),
        activeConfigSnapshotCount: planState.activeConfigSnapshotCount,
        selectedPresetCount: planState.selectedPresetCount,
      },
    }),
    [
      configuration.status,
      logFolder,
      modelSetup.selected,
      planState,
      runtimeDefaults,
      searchMetadata,
      setup,
      trainingEnabled,
      variantSetup.primaryPreset,
    ],
  );
  const plan = useMemo(
    () => ({
      display: planState.displayRunPlan,
      displayedRunCount: planState.displayedRunCount,
      isPlanning: planState.isDisplayPlanning,
      error: planState.displayPlanError,
      canStart: planState.canStart,
      canResample: planState.canResample,
      canRetry: planState.canRetry,
      isResampling: planState.isResampling,
      presetCountLabel,
      datasetCountLabel,
      selectedPresetCount: planState.selectedPresetCount,
      datasetCount: planState.datasetCount,
      search: planState.search,
    }),
    [datasetCountLabel, planState, presetCountLabel],
  );
  const job = useMemo(
    () => ({
      value: lifecycle.job,
      status: lifecycle.job?.status ?? "idle",
      isRunning: lifecycle.isRunning,
      canReset: lifecycle.canResetTraining,
      isStarting: lifecycle.isStarting,
      isCancelling: lifecycle.isCancelling,
      error: lifecycle.trainingError,
      clusterGrowth: buildClusterGrowth(lifecycle.job),
    }),
    [lifecycle],
  );
  const dialogs = useMemo(
    () => ({
      largeGridConfirmation: {
        isOpen: planState.confirmation.isOpen,
        isRequired: planState.confirmation.isRequired,
      },
    }),
    [planState.confirmation.isOpen, planState.confirmation.isRequired],
  );
  const actions = useMemo(
    () => ({
      openFullConfig: onOpenFullConfig,
      startJob: planState.actions.start,
      confirmLargeGridSearch: planState.actions.confirmLargeGridSearch,
      cancelLargeGridSearch: planState.actions.cancelLargeGridSearch,
      cancelJob: lifecycle.cancelTraining,
      resetJob: lifecycle.resetTraining,
      resamplePlan: planState.actions.resample,
      retryPlan: planState.actions.retry,
    }),
    [lifecycle, onOpenFullConfig, planState.actions],
  );
  const workspace = useMemo(
    () => ({ draft, plan, job, dialogs, actions }),
    [actions, dialogs, draft, job, plan],
  );
  return {
    workspace,
    clearForConnectionChange,
  };
}

export type TrainingWorkspace = ReturnType<
  typeof useTrainingWorkspaceState
>["workspace"];
