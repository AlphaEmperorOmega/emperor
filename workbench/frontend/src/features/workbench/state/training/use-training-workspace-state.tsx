import { useCallback, useMemo } from "react";
import { buildClusterGrowth } from "@/lib/cluster-growth";
import { type ConfigSnapshot } from "@/lib/config-snapshots";
import { type ModelIdentity, type TrainingJob } from "@/lib/api";
import {
  modelNameForId,
  modelsForType,
  modelTypeOptions as createModelTypeOptions,
} from "@/lib/selection";
import { type WorkbenchWorkspace } from "@/types/workbench";
import { useTrainingDraftState } from "@/features/workbench/state/training/use-training-draft-state";
import { useTrainingJobController } from "@/features/workbench/state/training/use-training-job-controller";
import { useTrainingLogFolderState } from "@/features/workbench/state/training/use-training-log-folder-state";
import { useTrainingPlanState } from "@/features/workbench/state/training/use-training-plan-state";

type TrainingSeed = {
  modelType: string;
  model: string;
  preset: string;
};

export type TrainingWorkspaceStateInput = {
  activeWorkspace: WorkbenchWorkspace;
  models: ModelIdentity[];
  seed: TrainingSeed;
  trainingEnabled: boolean;
  protectedReadsEnabled: boolean;
  onOpenFullConfig: () => void;
  onCreatePresetSnapshot: (target: TrainingSeed) => void;
  onEditConfigSnapshot: (snapshot: ConfigSnapshot) => void;
  onDuplicateConfigSnapshot: (snapshot: ConfigSnapshot) => void;
  activeTrainingJob: TrainingJob | undefined;
  progressError: string;
  onActiveJobIdChange: (jobId: string | null) => void;
  onJobChange: (job: TrainingJob | undefined) => void;
};

export function useTrainingWorkspaceState({
  activeWorkspace,
  models,
  seed,
  trainingEnabled,
  protectedReadsEnabled,
  onOpenFullConfig,
  onCreatePresetSnapshot,
  onEditConfigSnapshot,
  onDuplicateConfigSnapshot,
  activeTrainingJob,
  progressError,
  onActiveJobIdChange,
  onJobChange,
}: TrainingWorkspaceStateInput) {
  const configuration = useTrainingDraftState({
    activeWorkspace,
    models,
    seed,
    protectedReadsEnabled,
  });
  const logFolder = useTrainingLogFolderState({
    enabled: activeWorkspace === "training" && protectedReadsEnabled,
  });
  const modelTypeOptions = useMemo(
    () => createModelTypeOptions(configuration.models),
    [configuration.models],
  );
  const modelOptions = useMemo(
    () =>
      modelsForType(
        configuration.models,
        configuration.selectedModelType,
      ).map((model) => ({
        value: model.model,
        label: modelNameForId(model),
      })),
    [configuration.models, configuration.selectedModelType],
  );
  const presetOptions = useMemo(
    () =>
      configuration.presets.map((preset) => ({
        value: preset.name,
        label: preset.name,
      })),
    [configuration.presets],
  );
  const selectedSnapshotIdSet = useMemo(
    () => new Set(configuration.selectedSnapshotIds),
    [configuration.selectedSnapshotIds],
  );
  const selectedSnapshots = useMemo(
    () =>
      configuration.configSnapshots.filter((snapshot) =>
        selectedSnapshotIdSet.has(snapshot.id),
      ),
    [configuration.configSnapshots, selectedSnapshotIdSet],
  );
  const planState = useTrainingPlanState({
    configSections: configuration.configSections,
    overrides: configuration.bulkOverrides,
    selectedTrainingSnapshots: selectedSnapshots,
    selectedModelType: configuration.selectedModelType,
    selectedModel: configuration.selectedModel,
    selectedPreset: configuration.selectedPrimaryPreset,
    selectedTrainingPresets: configuration.selectedPresets,
    selectedExperimentTask: configuration.selectedExperimentTask,
    selectedDatasets: configuration.selectedDatasets,
    trainingSearch: configuration.search,
    searchAxes: configuration.searchAxes,
    searchLoading: configuration.searchLoading,
    trainingEnabled,
    logFolder: logFolder.value,
  });
  const lifecycle = useTrainingJobController({
    selectedModelType: configuration.selectedModelType,
    selectedModel: configuration.selectedModel,
    selectedPreset: configuration.selectedPrimaryPreset,
    selectedTrainingPresets: configuration.selectedPresets,
    selectedExperimentTask: configuration.selectedExperimentTask,
    selectedDatasets: configuration.selectedDatasets,
    effectiveOverrides: planState.effectiveOverrides,
    logFolder: logFolder.value,
    selectedMonitors: configuration.selectedMonitors,
    trainingSearch: planState.effectiveTrainingSearch,
    searchPayload: planState.searchPayload,
    submittedRunPlan: planState.snapshotRunPlan,
    canPlan: planState.canPlan,
    protectedReadsEnabled,
    hasValidLogFolder: logFolder.isValid,
    plannedRunCount: planState.plannedRunCount,
    activeTrainingJob,
    progressError,
    onActiveJobIdChange,
    onJobChange,
  });
  const setLogFolderMode = logFolder.setMode;
  const setExistingLogFolder = logFolder.setExistingValue;
  const setNewLogFolder = logFolder.setNewValue;
  const clearDraftForConnectionChange =
    configuration.clearForConnectionChange;
  const clearLogFolderForConnectionChange =
    logFolder.clearForConnectionChange;
  const clearLifecycleForConnectionChange =
    lifecycle.clearForConnectionChange;

  const editConfigSnapshot = useCallback(
    (snapshotId: string) => {
      const snapshot = configuration.configSnapshots.find(
        (candidate) => candidate.id === snapshotId,
      );
      if (snapshot) {
        onEditConfigSnapshot(snapshot);
      }
    },
    [configuration.configSnapshots, onEditConfigSnapshot],
  );
  const createPresetSnapshot = useCallback(
    (preset: string) =>
      onCreatePresetSnapshot({
        modelType: configuration.selectedModelType,
        model: configuration.selectedModel,
        preset,
      }),
    [
      configuration.selectedModel,
      configuration.selectedModelType,
      onCreatePresetSnapshot,
    ],
  );
  const duplicateConfigSnapshot = useCallback(
    (snapshotId: string) => {
      const snapshot = configuration.configSnapshots.find(
        (candidate) => candidate.id === snapshotId,
      );
      if (snapshot) {
        onDuplicateConfigSnapshot(snapshot);
      }
    },
    [configuration.configSnapshots, onDuplicateConfigSnapshot],
  );
  const selectLogFolderMode = useCallback(
    (mode: "existing" | "new") => setLogFolderMode(mode),
    [setLogFolderMode],
  );
  const selectExistingLogFolder = useCallback(
    (folder: string) => setExistingLogFolder(folder),
    [setExistingLogFolder],
  );
  const nameNewLogFolder = useCallback(
    (folder: string) => setNewLogFolder(folder),
    [setNewLogFolder],
  );
  const clearForConnectionChange = useCallback(() => {
    clearDraftForConnectionChange();
    clearLogFolderForConnectionChange();
    clearLifecycleForConnectionChange();
  }, [
    clearDraftForConnectionChange,
    clearLifecycleForConnectionChange,
    clearLogFolderForConnectionChange,
  ]);

  const presetCountLabel = `${planState.selectedTrainingPresetCount} preset${
    planState.selectedTrainingPresetCount === 1 ? "" : "s"
  }`;
  const datasetCountLabel = `${configuration.selectedDatasets.length} dataset${
    configuration.selectedDatasets.length === 1 ? "" : "s"
  }`;
  const draft = useMemo(
    () => ({
      datasetOptions: configuration.datasets,
      experimentTaskOptions: configuration.experimentTaskOptions,
      selectedModelType: configuration.selectedModelType,
      selectedModel: configuration.selectedModel,
      selectedPrimaryPreset: configuration.selectedPrimaryPreset,
      selectedPresets: configuration.selectedPresets,
      selectedExperimentTask: configuration.selectedExperimentTask,
      selectedSnapshotIds: configuration.selectedSnapshotIds,
      selectedDatasets: configuration.selectedDatasets,
      bulkOverrides: configuration.bulkOverrides,
      configSnapshots: configuration.configSnapshots,
      monitorOptions: configuration.monitors,
      snapshotOverrideWarning: configuration.snapshotOverrideWarning,
      selectedMonitors: configuration.selectedMonitors,
      monitorsLoading: configuration.monitorsLoading,
      searchAxes: configuration.searchAxes,
      searchLoading: configuration.searchLoading,
      trainingEnabled,
      canOpenFullConfig: Boolean(
        configuration.selectedModel &&
          configuration.selectedPrimaryPreset &&
          configuration.isSchemaReady,
      ),
      modelTypeOptions,
      modelOptions,
      presetOptions,
      logFolder: {
        mode: logFolder.mode,
        existingValue: logFolder.existingValue,
        newValue: logFolder.newValue,
        options: logFolder.options,
        isLoading: logFolder.isLoading,
        existingHelp: logFolder.existingHelp,
        newValid: logFolder.newValid,
        newError: logFolder.newError,
      },
      activeConfigSnapshotCount: planState.activeConfigSnapshotCount,
      selectedPresetCount: planState.selectedTrainingPresetCount,
    }),
    [
      configuration,
      logFolder,
      modelOptions,
      modelTypeOptions,
      presetOptions,
      planState,
      trainingEnabled,
    ],
  );
  const plan = useMemo(
    () => ({
      display: lifecycle.progressRunPlan,
      displayedRunCount: lifecycle.displayedRunCount,
      isPlanning: lifecycle.isProgressPlanning,
      error: lifecycle.progressPlanError,
      canStart: lifecycle.canStart,
      canResample: lifecycle.canResampleRunPlan,
      canRetry: lifecycle.canRetryRunPlan,
      isResampling: lifecycle.isResampling,
      presetCountLabel,
      datasetCountLabel,
      search: {
        effective: planState.effectiveTrainingSearch,
        conflictKeys: planState.searchConflictKeys,
        validation: planState.trainingSearchValidation,
        lockSummary: planState.searchLockSummary,
        modeLabel: planState.searchModeLabel,
        activeAxisCount: planState.activeSearchAxisCount,
      },
    }),
    [
      datasetCountLabel,
      lifecycle,
      presetCountLabel,
      planState.activeSearchAxisCount,
      planState.effectiveTrainingSearch,
      planState.searchConflictKeys,
      planState.searchLockSummary,
      planState.searchModeLabel,
      planState.trainingSearchValidation,
    ],
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
        isOpen: lifecycle.showLargeGridConfirmation,
        isRequired: lifecycle.requiresLargeGridConfirmation,
      },
    }),
    [
      lifecycle.requiresLargeGridConfirmation,
      lifecycle.showLargeGridConfirmation,
    ],
  );
  const actions = useMemo(
    () => ({
      openFullConfig: onOpenFullConfig,
      selectModelType: configuration.selectModelType,
      selectModel: configuration.selectModel,
      selectPrimaryPreset: configuration.selectPrimaryPreset,
      selectPresets: configuration.setPresetSelection,
      togglePreset: configuration.togglePreset,
      excludePreset: configuration.excludeDraftPreset,
      makePresetPrimary: configuration.makePresetPrimary,
      selectAllPresets: configuration.selectAllPresets,
      selectOnlyPrimaryPreset: configuration.selectOnlyPrimaryPreset,
      selectSnapshots: configuration.setSnapshotSelection,
      removeSnapshot: configuration.removeSnapshot,
      excludeSnapshot: configuration.excludeSnapshot,
      selectExperimentTask: configuration.selectExperimentTask,
      selectDatasets: configuration.setDatasetSelection,
      toggleDataset: configuration.toggleDataset,
      selectAllDatasets: configuration.selectAllDatasets,
      selectFirstDataset: configuration.selectFirstDataset,
      selectMonitors: configuration.setMonitorSelection,
      selectAllMonitors: configuration.selectAllMonitors,
      clearMonitors: configuration.clearMonitors,
      createPresetSnapshot,
      editConfigSnapshot,
      duplicateConfigSnapshot,
      updateSearch: configuration.updateSearch,
      selectLogFolderMode,
      selectExistingLogFolder,
      nameNewLogFolder,
      startJob: lifecycle.startTraining,
      confirmLargeGridSearch: lifecycle.confirmLargeGridSearch,
      cancelLargeGridSearch: lifecycle.cancelLargeGridSearch,
      cancelJob: lifecycle.cancelTraining,
      resetJob: lifecycle.resetTraining,
      resamplePlan: lifecycle.resampleRunPlan,
      retryPlan: lifecycle.retryRunPlan,
    }),
    [
      configuration,
      createPresetSnapshot,
      duplicateConfigSnapshot,
      editConfigSnapshot,
      lifecycle,
      nameNewLogFolder,
      onOpenFullConfig,
      selectExistingLogFolder,
      selectLogFolderMode,
    ],
  );
  const workspace = useMemo(
    () => ({ draft, plan, job, dialogs, actions }),
    [actions, dialogs, draft, job, plan],
  );
  const configurationInterface = useMemo(
    () => ({
      selectedModelType: configuration.selectedModelType,
      selectedModel: configuration.selectedModel,
      selectedPrimaryPreset: configuration.selectedPrimaryPreset,
      selectedSnapshotIds: configuration.selectedSnapshotIds,
      selectedMonitors: configuration.selectedMonitors,
      configSections: configuration.configSections,
      fieldCount: configuration.fieldCount,
      bulkOverrides: configuration.bulkOverrides,
      inactiveLockedOverrideCount: configuration.inactiveLockedOverrideCount,
      schemaLoading: configuration.schemaLoading,
      includeSnapshot: configuration.includeSnapshot,
      excludeSnapshot: configuration.excludeSnapshot,
      updateOverride: configuration.updateOverride,
      clearOverride: configuration.clearOverride,
      resetOverrides: configuration.resetOverrides,
    }),
    [configuration],
  );

  return {
    configuration: configurationInterface,
    workspace,
    clearForConnectionChange,
  };
}

export type TrainingWorkspace = ReturnType<
  typeof useTrainingWorkspaceState
>["workspace"];
export type TrainingConfiguration = ReturnType<
  typeof useTrainingWorkspaceState
>["configuration"];
