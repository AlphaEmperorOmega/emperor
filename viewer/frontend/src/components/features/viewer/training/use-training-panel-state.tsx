import { useMemo, useState } from "react";
import {
  type MonitorOption,
  type Preset,
  type SearchAxis,
  type TrainingJob,
} from "@/lib/api";
import { useLogExperimentsQuery } from "@/hooks/use-log-queries";
import { type ConfigSection, type OverrideValues } from "@/lib/config";
import {
  buildConfigSnapshotRunPlan,
  type ConfigSnapshot,
} from "@/lib/config-snapshots";
import { buildClusterGrowth } from "@/lib/cluster-growth";
import {
  DEFAULT_TRAINING_SEARCH_STATE,
  buildEffectiveOverrides,
  buildTrainingSearchPayload,
  estimatePlannedRuns,
  searchOverrideConflictKeys,
  selectedSearchAxisCount,
  trainingSearchModeLabel,
  validateTrainingSearch,
  type TrainingSearchState,
} from "@/lib/training-search";
import { metricLabel, overrideSummary } from "@/lib/training/summary";
import { type MultiSelectDropdownOption } from "@/components/features/viewer/screen/multi-select-dropdown";
import { useTrainingJobController } from "@/components/features/viewer/training/use-training-job-controller";

const LOG_FOLDER_RE = /^[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*$/;

export type LogFolderMode = "existing" | "new";

type UseTrainingPanelStateInput = {
  models: string[];
  presets: Preset[];
  configSections: ConfigSection[];
  selectedModel: string;
  selectedPreset: string;
  selectedTrainingPresets: string[];
  selectedDatasets: string[];
  overrides: OverrideValues;
  configSnapshots: ConfigSnapshot[];
  configSnapshotCount: number;
  deselectedSnapshotIds: string[];
  monitorOptions: MonitorOption[];
  selectedMonitors: string[];
  searchAxes: SearchAxis[];
  searchLoading: boolean;
  trainingSearch: TrainingSearchState;
  trainingEnabled: boolean;
  trainingLockedByHistoricalSelection: boolean;
  historicalTrainingLockExperiment: string;
  onToggleMonitor: (monitor: string) => void;
  activeJobId: string | null;
  onActiveJobIdChange: (jobId: string | null) => void;
  onJobChange: (job: TrainingJob | undefined) => void;
};

export function useTrainingPanelState({
  models,
  presets,
  configSections,
  selectedModel,
  selectedPreset,
  selectedTrainingPresets,
  selectedDatasets,
  overrides,
  configSnapshots,
  configSnapshotCount,
  deselectedSnapshotIds,
  monitorOptions,
  selectedMonitors,
  searchAxes,
  searchLoading,
  trainingSearch,
  trainingEnabled,
  trainingLockedByHistoricalSelection,
  historicalTrainingLockExperiment,
  onToggleMonitor,
  activeJobId,
  onActiveJobIdChange,
  onJobChange,
}: UseTrainingPanelStateInput) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [logFolderMode, setLogFolderMode] =
    useState<LogFolderMode>("existing");
  const [selectedExistingLogFolder, setSelectedExistingLogFolder] =
    useState("");
  const [newLogFolder, setNewLogFolder] = useState("");
  const [isProgressOpen, setIsProgressOpen] = useState(false);
  const logExperimentsQuery = useLogExperimentsQuery();

  const configFields = useMemo(
    () => configSections.flatMap((section) => section.fields),
    [configSections],
  );
  const modelOptions = useMemo(
    () => models.map((model) => ({ value: model, label: model })),
    [models],
  );
  const presetOptions = useMemo(
    () => presets.map((preset) => ({ value: preset.name, label: preset.name })),
    [presets],
  );
  const trainingMonitorOptions = useMemo<MultiSelectDropdownOption[]>(
    () =>
      monitorOptions.map((monitor) => ({
        value: monitor.name,
        label: monitor.label,
        description: monitor.description,
        meta:
          monitor.kinds.length > 0 ? (
            <span>{monitor.kinds.join(" / ")}</span>
          ) : undefined,
      })),
    [monitorOptions],
  );
  const fieldCount = configFields.length;
  const overrideCount = Object.keys(overrides).length;
  // Only snapshots the user kept checked feed the training run plan.
  const runnableConfigSnapshots = useMemo(
    () =>
      configSnapshots.filter(
        (snapshot) => !deselectedSnapshotIds.includes(snapshot.id),
      ),
    [configSnapshots, deselectedSnapshotIds],
  );
  const hasConfigSnapshots = configSnapshotCount > 0;
  const effectiveTrainingSearch = hasConfigSnapshots
    ? DEFAULT_TRAINING_SEARCH_STATE
    : trainingSearch;
  const effectiveOverrides = useMemo<OverrideValues>(
    () =>
      hasConfigSnapshots
        ? {}
        : buildEffectiveOverrides(overrides, effectiveTrainingSearch),
    [effectiveTrainingSearch, hasConfigSnapshots, overrides],
  );
  const selectedFieldSummary = useMemo(
    () => overrideSummary(configFields, effectiveOverrides),
    [configFields, effectiveOverrides],
  );
  const searchConflictKeys = useMemo(
    () => searchOverrideConflictKeys(overrides, effectiveTrainingSearch),
    [effectiveTrainingSearch, overrides],
  );
  const trainingSearchValidation = useMemo(
    () => validateTrainingSearch(effectiveTrainingSearch, searchAxes),
    [effectiveTrainingSearch, searchAxes],
  );
  const selectedTrainingPresetCount = selectedTrainingPresets.length;
  const activeSearchAxisCount = selectedSearchAxisCount(effectiveTrainingSearch);
  const logFolderOptions = logExperimentsQuery.data?.experiments ?? [];
  const existingLogFolderValid = Boolean(
    selectedExistingLogFolder &&
      logFolderOptions.some(
        (option) => option.experiment === selectedExistingLogFolder,
      ),
  );
  const newLogFolderValid = LOG_FOLDER_RE.test(newLogFolder);
  const newLogFolderError =
    newLogFolder.length === 0
      ? "Enter a folder name."
      : newLogFolderValid
        ? ""
        : "Use letters and numbers separated by single underscores.";
  const logFolder =
    logFolderMode === "existing" ? selectedExistingLogFolder : newLogFolder;
  const hasValidLogFolder =
    logFolderMode === "existing" ? existingLogFolderValid : newLogFolderValid;
  const logFolderLabel = hasValidLogFolder
    ? `logs/${logFolder}`
    : "Choose log folder";
  const existingLogFolderHelp = logExperimentsQuery.isLoading
    ? "Loading folders"
    : logFolderOptions.length === 0
      ? "No safe experiment folders found"
      : "Select a top-level logs folder";
  const canRequestTraining =
    trainingEnabled && !trainingLockedByHistoricalSelection;
  const historicalTrainingLockMessage = trainingLockedByHistoricalSelection
    ? historicalTrainingLockExperiment
      ? `Cannot perform training while experiment ${historicalTrainingLockExperiment} is selected.`
      : "Cannot perform training while a historical experiment is selected."
    : "";
  const searchPayload = useMemo(
    () =>
      hasConfigSnapshots
        ? undefined
        : buildTrainingSearchPayload(effectiveTrainingSearch),
    [effectiveTrainingSearch, hasConfigSnapshots],
  );
  const snapshotRunPlan = useMemo(
    () =>
      hasConfigSnapshots
        ? buildConfigSnapshotRunPlan({
            model: selectedModel,
            selectedPreset,
            selectedTrainingPresets,
            selectedDatasets,
            snapshots: runnableConfigSnapshots,
            fields: configFields,
            logFolder,
          })
        : undefined,
    [
      configFields,
      runnableConfigSnapshots,
      hasConfigSnapshots,
      logFolder,
      selectedDatasets,
      selectedModel,
      selectedPreset,
      selectedTrainingPresets,
    ],
  );
  const plannedRunCount = useMemo(
    () =>
      snapshotRunPlan?.summary.totalRuns ??
      estimatePlannedRuns(
        effectiveTrainingSearch,
        selectedDatasets.length,
        selectedTrainingPresetCount,
      ),
    [
      effectiveTrainingSearch,
      selectedDatasets.length,
      selectedTrainingPresetCount,
      snapshotRunPlan,
    ],
  );
  const canPlan = Boolean(
    canRequestTraining &&
      (hasConfigSnapshots
        ? selectedModel && selectedPreset && snapshotRunPlan
        : selectedModel &&
            selectedPreset &&
            selectedTrainingPresetCount > 0 &&
            selectedDatasets.length > 0 &&
            trainingSearchValidation.ready &&
            (effectiveTrainingSearch.mode === "off" || !searchLoading)),
  );
  const training = useTrainingJobController({
    selectedModel,
    selectedPreset,
    selectedTrainingPresets,
    selectedDatasets,
    effectiveOverrides,
    logFolder,
    selectedMonitors,
    trainingSearch: effectiveTrainingSearch,
    searchPayload,
    submittedRunPlan: snapshotRunPlan,
    canPlan,
    hasValidLogFolder,
    plannedRunCount,
    activeJobId,
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
  const searchModeLabel = trainingSearchModeLabel(effectiveTrainingSearch.mode);
  const activeSearchLabel =
    effectiveTrainingSearch.mode === "off" ? "" : `${searchModeLabel} search`;
  const presetCountLabel = `${selectedTrainingPresetCount} preset${
    selectedTrainingPresetCount === 1 ? "" : "s"
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
    ui: {
      isExpanded,
      toggleExpanded: () => setIsExpanded((current) => !current),
      isProgressOpen,
      openProgress: () => setIsProgressOpen(true),
      closeProgress: () => setIsProgressOpen(false),
    },
    logFolder: {
      mode: logFolderMode,
      setMode: setLogFolderMode,
      existingValue: selectedExistingLogFolder,
      setExistingValue: setSelectedExistingLogFolder,
      newValue: newLogFolder,
      setNewValue: setNewLogFolder,
      options: logFolderOptions,
      isLoading: logExperimentsQuery.isLoading,
      existingHelp: existingLogFolderHelp,
      newValid: newLogFolderValid,
      newError: newLogFolderError,
    },
    options: {
      modelOptions,
      presetOptions,
      trainingMonitorOptions,
    },
    request: {
      fieldCount,
      overrideCount,
      hasConfigSnapshots,
      effectiveTrainingSearch,
      selectedFieldSummary,
      searchConflictKeys,
      trainingSearchValidation,
      selectedTrainingPresetCount,
      activeSearchAxisCount,
      canRequestTraining,
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
      historicalTrainingLockMessage,
      activeSearchLabel,
      logFolderLabel,
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
