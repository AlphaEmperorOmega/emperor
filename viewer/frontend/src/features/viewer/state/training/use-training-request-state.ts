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
  configSnapshots: ConfigSnapshot[];
  configSnapshotCount: number;
  deselectedSnapshotIds: string[];
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
  configSnapshots,
  configSnapshotCount,
  deselectedSnapshotIds,
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
  const runnableConfigSnapshots = useMemo(
    () =>
      configSnapshots.filter(
        (snapshot) => !deselectedSnapshotIds.includes(snapshot.id),
      ),
    [configSnapshots, deselectedSnapshotIds],
  );
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
      hasConfigSnapshots,
      logFolder,
      runnableConfigSnapshots,
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
  const searchModeLabel = trainingSearchModeLabel(effectiveTrainingSearch.mode);

  return {
    fieldCount,
    overrideCount,
    hasConfigSnapshots,
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
