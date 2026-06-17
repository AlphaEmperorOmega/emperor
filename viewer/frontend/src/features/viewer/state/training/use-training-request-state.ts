import { useMemo } from "react";
import { type SearchAxis } from "@/lib/api";
import { type ConfigSection, type OverrideValues } from "@/lib/config";
import {
  buildConfigSnapshotRunPlan,
  type ConfigSnapshot,
} from "@/lib/config-snapshots";
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
import { overrideSummary } from "@/lib/training/summary";

type TrainingRequestStateInput = {
  configSections: ConfigSection[];
  overrides: OverrideValues;
  configSnapshotCount: number;
  selectedTrainingSnapshots: ConfigSnapshot[];
  selectedModelType: string;
  selectedModel: string;
  selectedPreset: string;
  selectedTrainingPresets: string[];
  selectedDatasets: string[];
  trainingSearch: TrainingSearchState;
  searchAxes: SearchAxis[];
  searchLoading: boolean;
  trainingEnabled: boolean;
  trainingLockedByHistoricalSelection: boolean;
  historicalTrainingLockExperiment: string;
  logFolder: string;
};

export function useTrainingRequestState({
  configSections,
  overrides,
  configSnapshotCount,
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
  logFolder,
}: TrainingRequestStateInput) {
  const configFields = useMemo(
    () => configSections.flatMap((section) => section.fields),
    [configSections],
  );
  const fieldCount = configFields.length;
  const overrideCount = Object.keys(overrides).length;
  const hasConfigSnapshots = configSnapshotCount > 0;
  const activeConfigSnapshotCount = selectedTrainingSnapshots.length;
  const hasActiveConfigSnapshots = activeConfigSnapshotCount > 0;
  const effectiveTrainingSearch = hasActiveConfigSnapshots
    ? DEFAULT_TRAINING_SEARCH_STATE
    : trainingSearch;
  const effectiveOverrides = useMemo<OverrideValues>(
    () =>
      hasActiveConfigSnapshots
        ? {}
        : buildEffectiveOverrides(overrides, effectiveTrainingSearch),
    [effectiveTrainingSearch, hasActiveConfigSnapshots, overrides],
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
  const canRequestTraining =
    trainingEnabled && !trainingLockedByHistoricalSelection;
  const historicalTrainingLockMessage = trainingLockedByHistoricalSelection
    ? historicalTrainingLockExperiment
      ? `Cannot perform training while experiment ${historicalTrainingLockExperiment} is selected.`
      : "Cannot perform training while a historical experiment is selected."
    : "";
  const searchPayload = useMemo(
    () =>
      hasActiveConfigSnapshots
        ? undefined
        : buildTrainingSearchPayload(effectiveTrainingSearch),
    [effectiveTrainingSearch, hasActiveConfigSnapshots],
  );
  const snapshotRunPlan = useMemo(
    () =>
      hasActiveConfigSnapshots
        ? buildConfigSnapshotRunPlan({
            modelType: selectedModelType,
            model: selectedModel,
            selectedPreset,
            selectedTrainingPresets,
            selectedDatasets,
            snapshots: selectedTrainingSnapshots,
            fields: configFields,
            logFolder,
          })
        : undefined,
    [
      configFields,
      hasActiveConfigSnapshots,
      logFolder,
      selectedDatasets,
      selectedTrainingSnapshots,
      selectedModelType,
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
      (hasActiveConfigSnapshots
        ? selectedModel &&
          (selectedPreset || selectedTrainingSnapshots.length > 0) &&
          selectedDatasets.length > 0 &&
          snapshotRunPlan
        : selectedModel &&
            selectedPreset &&
            selectedTrainingPresetCount > 0 &&
            selectedDatasets.length > 0 &&
            trainingSearchValidation.ready &&
            (effectiveTrainingSearch.mode === "off" || !searchLoading)),
  );
  const searchModeLabel = trainingSearchModeLabel(effectiveTrainingSearch.mode);

  return {
    fieldCount,
    overrideCount,
    hasConfigSnapshots,
    activeConfigSnapshotCount,
    effectiveTrainingSearch,
    effectiveOverrides,
    selectedFieldSummary,
    searchConflictKeys,
    trainingSearchValidation,
    selectedTrainingPresetCount,
    activeSearchAxisCount,
    canRequestTraining,
    historicalTrainingLockMessage,
    searchPayload,
    snapshotRunPlan,
    plannedRunCount,
    canPlan,
    searchModeLabel,
  };
}
