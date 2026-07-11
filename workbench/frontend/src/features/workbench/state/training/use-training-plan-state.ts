import { useMemo } from "react";
import { type SearchAxis } from "@/lib/api";
import {
  configSectionsFields,
  effectivePresetOverrides,
  type ConfigSection,
  type OverrideValues,
} from "@/lib/config";
import {
  buildConfigSnapshotRunPlan,
  type ConfigSnapshot,
} from "@/lib/config-snapshots";
import {
  DEFAULT_TRAINING_SEARCH_STATE,
  buildEffectiveOverrides,
  buildTrainingSearchPayload,
  deriveTrainingSearchLockSummary,
  effectiveUnlockedTrainingSearch,
  estimatePlannedRuns,
  searchOverrideConflictKeys,
  selectedSearchAxisCount,
  trainingSearchModeLabel,
  validateTrainingSearch,
  type TrainingSearchState,
} from "@/lib/training-search";

type TrainingPlanStateInput = {
  configSections: ConfigSection[];
  overrides: OverrideValues;
  selectedTrainingSnapshots: ConfigSnapshot[];
  selectedModelType: string;
  selectedModel: string;
  selectedPreset: string;
  selectedTrainingPresets: string[];
  selectedExperimentTask?: string;
  selectedDatasets: string[];
  trainingSearch: TrainingSearchState;
  searchAxes: SearchAxis[];
  searchLoading: boolean;
  trainingEnabled: boolean;
  logFolder: string;
};

export function useTrainingPlanState({
  configSections,
  overrides,
  selectedTrainingSnapshots,
  selectedModelType,
  selectedModel,
  selectedPreset,
  selectedTrainingPresets,
  selectedExperimentTask = "",
  selectedDatasets,
  trainingSearch,
  searchAxes,
  searchLoading,
  trainingEnabled,
  logFolder,
}: TrainingPlanStateInput) {
  const configFields = useMemo(
    () => configSectionsFields(configSections),
    [configSections],
  );
  const editablePresetOverrides = useMemo(
    () => effectivePresetOverrides(configFields, overrides),
    [configFields, overrides],
  );
  const activeConfigSnapshotCount = selectedTrainingSnapshots.length;
  const hasActiveConfigSnapshots = activeConfigSnapshotCount > 0;
  const baseTrainingSearch = hasActiveConfigSnapshots
    ? DEFAULT_TRAINING_SEARCH_STATE
    : trainingSearch;
  const searchLockSummary = useMemo(
    () => deriveTrainingSearchLockSummary(baseTrainingSearch, searchAxes),
    [baseTrainingSearch, searchAxes],
  );
  const effectiveTrainingSearch = useMemo(
    () => effectiveUnlockedTrainingSearch(baseTrainingSearch, searchAxes),
    [baseTrainingSearch, searchAxes],
  );
  const effectiveOverrides = useMemo<OverrideValues>(
    () =>
      hasActiveConfigSnapshots
        ? {}
        : buildEffectiveOverrides(editablePresetOverrides, effectiveTrainingSearch),
    [
      editablePresetOverrides,
      effectiveTrainingSearch,
      hasActiveConfigSnapshots,
    ],
  );
  const searchConflictKeys = useMemo(
    () => searchOverrideConflictKeys(editablePresetOverrides, effectiveTrainingSearch),
    [editablePresetOverrides, effectiveTrainingSearch],
  );
  const trainingSearchValidation = useMemo(
    () =>
      validateTrainingSearch(effectiveTrainingSearch, searchAxes, {
        allowEmptySelected: searchLockSummary.skippedSelectedAxisCount > 0,
      }),
    [
      effectiveTrainingSearch,
      searchAxes,
      searchLockSummary.skippedSelectedAxisCount,
    ],
  );
  const selectedTrainingPresetCount = selectedTrainingPresets.length;
  const activeSearchAxisCount = selectedSearchAxisCount(effectiveTrainingSearch);
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
            selectedExperimentTask,
            selectedDatasets,
            snapshots: selectedTrainingSnapshots,
            fields: configFields,
            bulkOverrides: editablePresetOverrides,
            logFolder,
          })
        : undefined,
    [
      configFields,
      hasActiveConfigSnapshots,
      logFolder,
      editablePresetOverrides,
      selectedDatasets,
      selectedExperimentTask,
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
        {
          emptySearchRunsAsBase:
            searchLockSummary.skippedSelectedAxisCount > 0,
        },
      ),
    [
      effectiveTrainingSearch,
      searchLockSummary.skippedSelectedAxisCount,
      selectedDatasets.length,
      selectedTrainingPresetCount,
      snapshotRunPlan,
    ],
  );
  const canPlan = Boolean(
    trainingEnabled &&
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
    activeConfigSnapshotCount,
    effectiveTrainingSearch,
    effectiveOverrides,
    searchConflictKeys,
    trainingSearchValidation,
    searchLockSummary,
    selectedTrainingPresetCount,
    activeSearchAxisCount,
    searchPayload,
    snapshotRunPlan,
    plannedRunCount,
    canPlan,
    searchModeLabel,
  };
}
