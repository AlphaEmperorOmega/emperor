import {
  type Dispatch,
  type SetStateAction,
  useMemo,
  useState,
} from "react";
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
import {
  type FullConfigDialogMode,
} from "@/features/viewer/state/use-viewer-workspace-shell";
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
  configSections: ConfigSection[];
  selectedModelType: string;
  selectedModel: string;
  selectedPreset: string;
  selectedTrainingPresets: string[];
  selectedDatasets: string[];
  overrides: OverrideValues;
  allConfigSnapshots: ConfigSnapshot[];
  configSnapshotCount: number;
  selectedTrainingSnapshotIds: string[];
  monitorOptions: MonitorOption[];
  selectedMonitors: string[];
  monitorsLoading: boolean;
  searchAxes: SearchAxis[];
  searchLoading: boolean;
  trainingSearch: TrainingSearchState;
  trainingEnabled: boolean;
  trainingLockedByHistoricalSelection: boolean;
  historicalTrainingLockExperiment: string;
  canOpenFullConfig: boolean;
  onToggleMonitor: (monitor: string) => void;
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
  onSetDatasets: (datasets: string[]) => void;
  onToggleDataset: (dataset: string) => void;
  onSelectAllDatasets: () => void;
  onSelectFirstDataset: () => void;
  onResetOverrides: () => void;
  onOpenFullConfig: (mode?: FullConfigDialogMode) => void;
  onRemoveConfigSnapshot: (snapshotId: string) => void;
  onIncludeConfigSnapshot: (snapshotId: string) => void;
  onExcludeConfigSnapshot: (snapshotId: string) => void;
  onEditPresetAsSnapshot: (preset: string) => void;
  onEditConfigSnapshotCopy: (snapshotId: string) => void;
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
  configSections,
  selectedModelType,
  selectedModel,
  selectedPreset,
  selectedTrainingPresets,
  selectedDatasets,
  overrides,
  allConfigSnapshots,
  configSnapshotCount,
  selectedTrainingSnapshotIds,
  monitorOptions,
  selectedMonitors,
  monitorsLoading,
  searchAxes,
  searchLoading,
  trainingSearch,
  trainingEnabled,
  trainingLockedByHistoricalSelection,
  historicalTrainingLockExperiment,
  canOpenFullConfig,
  onToggleMonitor,
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
  onSetDatasets,
  onToggleDataset,
  onSelectAllDatasets,
  onSelectFirstDataset,
  onResetOverrides,
  onOpenFullConfig,
  onRemoveConfigSnapshot,
  onIncludeConfigSnapshot,
  onExcludeConfigSnapshot,
  onEditPresetAsSnapshot,
  onEditConfigSnapshotCopy,
  onTrainingSearchChange,
  activeTrainingJob,
  progressError,
  onActiveJobIdChange,
  onJobChange,
}: TrainingPanelViewModelInput) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isProgressOpen, setIsProgressOpen] = useState(false);
  const logFolder = useTrainingLogFolderState({ enabled: isExpanded });
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
    selectedDatasets,
    trainingSearch,
    searchAxes,
    searchLoading,
    trainingEnabled,
    trainingLockedByHistoricalSelection,
    historicalTrainingLockExperiment,
    logFolder: logFolder.value,
  });
  const training = useTrainingJobController({
    selectedModelType,
    selectedModel,
    selectedPreset,
    selectedTrainingPresets,
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
    onJobStarted: () => setIsExpanded(true),
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
  const progressButtonLabel = training.isProgressPlanning
    ? "Planning..."
    : training.progressPlanError
      ? "Plan error"
      : training.progressRunPlanSummary
        ? `${training.progressRunPlanSummary.completedRuns} / ${training.progressRunPlanSummary.totalRuns} runs · ${training.progressRunPlanSummary.remainingEpochs} epochs left`
        : "Progress";

  function changeMonitors(nextMonitors: string[]) {
    const changedMonitor = monitorOptions.find(
      (monitor) =>
        selectedMonitors.includes(monitor.name) !==
        nextMonitors.includes(monitor.name),
    );
    if (changedMonitor) {
      onToggleMonitor(changedMonitor.name);
    }
  }

  return {
    input: {
      datasetOptions,
      configSections,
      selectedModelType,
      selectedModel,
      selectedPreset,
      selectedTrainingPresets,
      selectedTrainingSnapshotIds,
      selectedDatasets,
      overrides,
      allConfigSnapshots,
      configSnapshotCount,
      monitorOptions,
      selectedMonitors,
      monitorsLoading,
      searchAxes,
      searchLoading,
      trainingEnabled,
      canOpenFullConfig,
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
      onSetDatasets,
      onToggleDataset,
      onSelectAllDatasets,
      onSelectFirstDataset,
      onResetOverrides,
      onOpenFullConfig,
      onRemoveConfigSnapshot,
      onIncludeConfigSnapshot,
      onExcludeConfigSnapshot,
      onEditPresetAsSnapshot,
      onEditConfigSnapshotCopy,
      onTrainingSearchChange,
    },
    ui: {
      isExpanded,
      toggleExpanded: () => setIsExpanded((current) => !current),
      isProgressOpen,
      openProgress: () => setIsProgressOpen(true),
      closeProgress: () => setIsProgressOpen(false),
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
      fieldCount: requestState.fieldCount,
      overrideCount: requestState.overrideCount,
      hasConfigSnapshots: requestState.hasConfigSnapshots,
      activeConfigSnapshotCount: requestState.activeConfigSnapshotCount,
      effectiveTrainingSearch: requestState.effectiveTrainingSearch,
      selectedFieldSummary: requestState.selectedFieldSummary,
      searchConflictKeys: requestState.searchConflictKeys,
      trainingSearchValidation: requestState.trainingSearchValidation,
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
      historicalTrainingLockMessage: requestState.historicalTrainingLockMessage,
      activeSearchLabel,
      logFolderLabel: logFolder.label,
      presetCountLabel,
      monitorCount,
      datasetCountLabel,
      plannedRunLabel,
      progressButtonLabel,
    },
    actions: {
      changeMonitors,
    },
  };
}

export type TrainingPanelViewModel = ReturnType<typeof useTrainingPanelViewModel>;
