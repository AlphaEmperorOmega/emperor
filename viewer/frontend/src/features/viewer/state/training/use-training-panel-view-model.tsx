import { type Dispatch, type SetStateAction, useMemo } from "react";
import {
  type Dataset,
  type ModelIdentity,
  type MonitorOption,
  type Preset,
  type SearchAxis,
  type TrainingJob,
} from "@/lib/api";
import { type ConfigSection, type OverrideValues } from "@/lib/config";
import { type ConfigSnapshot } from "@/lib/config-snapshots";
import { buildClusterGrowth } from "@/lib/cluster-growth";
import { type TrainingSearchState } from "@/lib/training-search";
import { metricLabel } from "@/lib/training/summary";
import { useTrainingJobController } from "@/features/viewer/state/training/use-training-job-controller";
import { useTrainingLogFolderState } from "@/features/viewer/state/training/use-training-log-folder-state";
import { useTrainingPanelOptions } from "@/features/viewer/state/training/training-panel-options";
import { useTrainingRequestState } from "@/features/viewer/state/training/use-training-request-state";

export type TrainingPanelViewModelInput = {
  models: ModelIdentity[];
  presets: Preset[];
  datasetOptions: Dataset[];
  experimentTaskOptions?: Array<{ value: string; label: string }>;
  configSections: ConfigSection[];
  selectedModelType: string;
  selectedModel: string;
  selectedPreset: string;
  selectedTrainingPresets: string[];
  selectedExperimentTask?: string;
  selectedDatasets: string[];
  overrides: OverrideValues;
  allConfigSnapshots: ConfigSnapshot[];
  configSnapshotCount: number;
  selectedTrainingSnapshotIds: string[];
  monitorOptions: MonitorOption[];
  snapshotOverrideWarning: string;
  selectedMonitors: string[];
  monitorsLoading: boolean;
  searchAxes: SearchAxis[];
  searchLoading: boolean;
  trainingSearch: TrainingSearchState;
  trainingEnabled: boolean;
  canOpenFullConfig: boolean;
  onOpenFullConfig: () => void;
  onSetMonitors: (monitors: string[]) => void;
  onSelectAllMonitors: () => void;
  onClearMonitors: () => void;
  onSelectModelType: (modelType: string) => void;
  onSelectModel: (model: string) => void;
  onSelectPreset: (preset: string) => void;
  onSetTrainingPresets: (presets: string[]) => void;
  onSetTrainingSnapshotSelection: (snapshotIds: string[]) => void;
  onToggleTrainingPreset: (preset: string) => void;
  onToggleDraftTrainingPreset: (preset: string) => void;
  onExcludeDraftTrainingPreset: (preset: string) => void;
  onMakeTrainingPresetPrimary: (preset: string) => void;
  onSelectAllTrainingPresets: () => void;
  onSelectPrimaryTrainingPreset: () => void;
  onSelectExperimentTask?: (experimentTask: string) => void;
  onSetDatasets: (datasets: string[]) => void;
  onToggleDataset: (dataset: string) => void;
  onSelectAllDatasets: () => void;
  onSelectFirstDataset: () => void;
  onRemoveConfigSnapshot: (snapshotId: string) => void;
  onIncludeConfigSnapshot: (snapshotId: string) => void;
  onExcludeConfigSnapshot: (snapshotId: string) => void;
  onCreatePresetSnapshot: (preset: string) => void;
  onEditConfigSnapshot: (snapshotId: string) => void;
  onDuplicateConfigSnapshot: (snapshotId: string) => void;
  onTrainingSearchChange: Dispatch<SetStateAction<TrainingSearchState>>;
  activeTrainingJob: TrainingJob | undefined;
  progressError: string;
  onActiveJobIdChange: (jobId: string | null) => void;
  onJobChange: (job: TrainingJob | undefined) => void;
};

export function useTrainingPanelViewModel({
  models,
  presets,
  datasetOptions,
  experimentTaskOptions = [],
  configSections,
  selectedModelType,
  selectedModel,
  selectedPreset,
  selectedTrainingPresets,
  selectedExperimentTask = "",
  selectedDatasets,
  overrides,
  allConfigSnapshots,
  configSnapshotCount,
  selectedTrainingSnapshotIds,
  monitorOptions,
  snapshotOverrideWarning,
  selectedMonitors,
  monitorsLoading,
  searchAxes,
  searchLoading,
  trainingSearch,
  trainingEnabled,
  canOpenFullConfig,
  onOpenFullConfig,
  onSetMonitors,
  onSelectAllMonitors,
  onClearMonitors,
  onSelectModelType,
  onSelectModel,
  onSelectPreset,
  onSetTrainingPresets,
  onSetTrainingSnapshotSelection,
  onToggleTrainingPreset,
  onToggleDraftTrainingPreset,
  onExcludeDraftTrainingPreset,
  onMakeTrainingPresetPrimary,
  onSelectAllTrainingPresets,
  onSelectPrimaryTrainingPreset,
  onSelectExperimentTask,
  onSetDatasets,
  onToggleDataset,
  onSelectAllDatasets,
  onSelectFirstDataset,
  onRemoveConfigSnapshot,
  onIncludeConfigSnapshot,
  onExcludeConfigSnapshot,
  onCreatePresetSnapshot,
  onEditConfigSnapshot,
  onDuplicateConfigSnapshot,
  onTrainingSearchChange,
  activeTrainingJob,
  progressError,
  onActiveJobIdChange,
  onJobChange,
}: TrainingPanelViewModelInput) {
  const logFolder = useTrainingLogFolderState({ enabled: true });
  const options = useTrainingPanelOptions({
    models,
    selectedModelType,
    presets,
    monitorOptions,
  });
  const selectedTrainingSnapshotIdSet = useMemo(
    () => new Set(selectedTrainingSnapshotIds),
    [selectedTrainingSnapshotIds],
  );
  const selectedTrainingSnapshots = useMemo(
    () =>
      allConfigSnapshots.filter((snapshot) =>
        selectedTrainingSnapshotIdSet.has(snapshot.id),
      ),
    [allConfigSnapshots, selectedTrainingSnapshotIdSet],
  );
  const requestState = useTrainingRequestState({
    configSections,
    overrides,
    configSnapshotCount: allConfigSnapshots.length || configSnapshotCount,
    selectedTrainingSnapshots,
    selectedModelType,
    selectedModel,
    selectedPreset,
    selectedTrainingPresets,
    selectedExperimentTask,
    selectedDatasets,
    trainingSearch,
    searchAxes,
    searchLoading,
    trainingEnabled,
    logFolder: logFolder.value,
  });
  const training = useTrainingJobController({
    selectedModelType,
    selectedModel,
    selectedPreset,
    selectedTrainingPresets,
    selectedExperimentTask,
    selectedDatasets,
    effectiveOverrides: requestState.effectiveOverrides,
    logFolder: logFolder.value,
    selectedMonitors,
    trainingSearch: requestState.effectiveTrainingSearch,
    searchPayload: requestState.searchPayload,
    submittedRunPlan: requestState.snapshotRunPlan,
    canPlan: requestState.canPlan,
    hasValidLogFolder: logFolder.isValid,
    plannedRunCount: requestState.plannedRunCount,
    activeTrainingJob,
    progressError,
    onActiveJobIdChange,
    onJobChange,
    onJobStarted: () => {},
  });
  const clusterGrowth = useMemo(
    () => buildClusterGrowth(training.job),
    [training.job],
  );
  const jobStatus = training.job?.status ?? "idle";
  const currentPreset =
    training.job?.currentPreset ??
    training.job?.preset ??
    selectedTrainingPresets[0] ??
    "";
  const currentDataset =
    training.job?.currentDataset ?? selectedDatasets[0] ?? "No dataset";
  const epochStep =
    training.job?.epoch !== null && training.job?.epoch !== undefined
      ? `epoch ${training.job.epoch}${
          training.job.step !== null && training.job.step !== undefined
            ? ` / step ${training.job.step}`
            : ""
        }`
      : "waiting";
  const searchModeLabel = requestState.searchModeLabel;
  const activeSearchLabel =
    requestState.effectiveTrainingSearch.mode === "off" ? "" : `${searchModeLabel} search`;
  const presetCountLabel = `${requestState.selectedTrainingPresetCount} preset${
    requestState.selectedTrainingPresetCount === 1 ? "" : "s"
  }`;
  const monitorCount = `${selectedMonitors.length} / ${monitorOptions.length}`;
  const datasetCountLabel = `${selectedDatasets.length} dataset${
    selectedDatasets.length === 1 ? "" : "s"
  }`;
  const plannedRunLabel = `${training.displayedRunCount} planned run${
    training.displayedRunCount === 1 ? "" : "s"
  }`;

  return {
    input: {
      datasetOptions,
      experimentTaskOptions,
      selectedModelType,
      selectedModel,
      selectedPreset,
      selectedTrainingPresets,
      selectedExperimentTask,
      selectedTrainingSnapshotIds,
      selectedDatasets,
      overrides,
      allConfigSnapshots,
      configSnapshotCount,
      monitorOptions,
      snapshotOverrideWarning,
      selectedMonitors,
      monitorsLoading,
      searchAxes,
      searchLoading,
      trainingEnabled,
      canOpenFullConfig,
      onOpenFullConfig,
      onSelectModelType,
      onSelectModel,
      onSelectPreset,
      onSetTrainingPresets,
      onSetTrainingSnapshotSelection,
      onToggleTrainingPreset,
      onToggleDraftTrainingPreset,
      onExcludeDraftTrainingPreset,
      onMakeTrainingPresetPrimary,
      onSelectAllTrainingPresets,
      onSelectPrimaryTrainingPreset,
      onSelectExperimentTask,
      onSetDatasets,
      onToggleDataset,
      onSelectAllDatasets,
      onSelectFirstDataset,
      onSetMonitors,
      onSelectAllMonitors,
      onClearMonitors,
      onRemoveConfigSnapshot,
      onIncludeConfigSnapshot,
      onExcludeConfigSnapshot,
      onCreatePresetSnapshot,
      onEditConfigSnapshot,
      onDuplicateConfigSnapshot,
      onTrainingSearchChange,
    },
    logFolder: {
      mode: logFolder.mode,
      setMode: logFolder.setMode,
      existingValue: logFolder.existingValue,
      setExistingValue: logFolder.setExistingValue,
      newValue: logFolder.newValue,
      setNewValue: logFolder.setNewValue,
      options: logFolder.options,
      isLoading: logFolder.isLoading,
      existingHelp: logFolder.existingHelp,
      newValid: logFolder.newValid,
      newError: logFolder.newError,
    },
    options,
    request: {
      activeConfigSnapshotCount: requestState.activeConfigSnapshotCount,
      effectiveTrainingSearch: requestState.effectiveTrainingSearch,
      searchConflictKeys: requestState.searchConflictKeys,
      trainingSearchValidation: requestState.trainingSearchValidation,
      searchLockSummary: requestState.searchLockSummary,
      selectedTrainingPresetCount: requestState.selectedTrainingPresetCount,
      activeSearchAxisCount: requestState.activeSearchAxisCount,
      canRequestTraining: requestState.canRequestTraining,
      searchModeLabel,
    },
    training,
    status: {
      jobStatus,
      currentPreset,
      currentDataset,
      epochStep,
      metricLabel: metricLabel(training.job),
      clusterGrowth,
      activeSearchLabel,
      logFolderLabel: logFolder.label,
      presetCountLabel,
      monitorCount,
      datasetCountLabel,
      plannedRunLabel,
    },
  };
}

export type TrainingPanelViewModel = ReturnType<typeof useTrainingPanelViewModel>;
