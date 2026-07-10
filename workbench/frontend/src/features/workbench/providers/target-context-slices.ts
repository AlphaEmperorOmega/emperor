import { useRef } from "react";
import { type TargetConfigContextValue } from "@/features/workbench/state/use-workbench-state";

const TARGET_CATALOG_KEYS = [
  "capabilities",
  "apiOnline",
  "models",
  "modelsLoading",
  "isModelsError",
  "modelsError",
] as const satisfies readonly (keyof TargetConfigContextValue)[];

const MODEL_TARGET_KEYS = [
  "selectedModelType",
  "selectModelType",
  "selectedModel",
  "selectModel",
  "selectedTargetMode",
  "activateTargetPresetMode",
  "activateTargetSnapshotMode",
  "activateTargetExperimentMode",
  "selectedPreset",
  "selectPreset",
  "selectTargetPreset",
  "selectedExperimentRunId",
  "selectedExperimentTarget",
  "selectedExperimentName",
  "selectedExperimentPreset",
  "selectedExperimentDataset",
  "selectedPresetMeta",
  "selectedDatasets",
  "setDatasetSelection",
  "toggleDataset",
  "selectAllDatasets",
  "selectFirstDataset",
  "selectExperimentTask",
  "presetOverrides",
  "effectivePresetOverrides",
  "activeOverrides",
  "activeOverrideScope",
  "activeOverrideScopeLabel",
  "inactiveLockedOverrides",
  "inactiveLockedOverrideCount",
  "overrides",
  "configSections",
  "overrideCount",
  "presetOwnedFieldCount",
  "fieldCount",
  "updateOverride",
  "clearOverride",
  "updatePreview",
  "resetOverrides",
  "resetOverridesPreservingTargetSelection",
  "presets",
  "presetsReady",
  "isPresetsError",
  "presetsError",
  "selectedExperimentTask",
  "experimentTaskOptions",
  "datasets",
  "isDatasetsError",
  "datasetsError",
  "targetMonitors",
  "targetMonitorsLoading",
  "isSchemaReady",
  "schemaLoading",
  "isSchemaError",
  "schemaError",
] as const satisfies readonly (keyof TargetConfigContextValue)[];

const TRAINING_TARGET_KEYS = [
  "selectedTrainingModelType",
  "selectTrainingModelType",
  "selectedTrainingModel",
  "selectTrainingModel",
  "selectedTrainingPrimaryPreset",
  "selectTrainingPrimaryPreset",
  "selectedTrainingPresets",
  "setTrainingPresetSelection",
  "selectedTrainingSnapshotIds",
  "setTrainingSnapshotSelection",
  "toggleTrainingPreset",
  "toggleDraftTrainingPreset",
  "excludeDraftTrainingPreset",
  "makeTrainingPresetPrimary",
  "selectAllTrainingPresets",
  "selectPrimaryTrainingPreset",
  "selectedTrainingDatasets",
  "setTrainingDatasetSelection",
  "toggleTrainingDataset",
  "selectAllTrainingDatasets",
  "selectFirstTrainingDataset",
  "selectTrainingExperimentTask",
  "selectedMonitors",
  "selectedTrainingMonitors",
  "toggleMonitor",
  "setMonitorSelection",
  "selectAllMonitors",
  "clearMonitors",
  "trainingOverrides",
  "trainingBulkOverrides",
  "trainingConfigSections",
  "trainingOverrideCount",
  "trainingPresetOwnedFieldCount",
  "trainingFieldCount",
  "trainingInactiveLockedOverrideCount",
  "trainingSearch",
  "setTrainingSearch",
  "snapshotOverrideWarning",
  "allTrainingConfigSnapshots",
  "allTrainingConfigSnapshotCount",
  "removeConfigSnapshot",
  "includeConfigSnapshot",
  "excludeConfigSnapshot",
  "prepareTrainingPresetSnapshotDraft",
  "prepareTrainingSelectedSnapshotEdit",
  "updateTrainingOverride",
  "clearTrainingOverride",
  "resetTrainingOverrides",
  "trainingPresets",
  "selectedTrainingExperimentTask",
  "trainingExperimentTaskOptions",
  "trainingDatasets",
  "monitors",
  "trainingMonitors",
  "monitorsLoading",
  "trainingMonitorsLoading",
  "isTrainingSchemaReady",
  "trainingSchemaLoading",
  "searchAxes",
  "trainingSearchAxes",
  "searchAxesLoading",
  "trainingSearchAxesLoading",
] as const satisfies readonly (keyof TargetConfigContextValue)[];

const TARGET_SNAPSHOT_KEYS = [
  "selectedSnapshotId",
  "selectedConfigSnapshot",
  "selectTargetSnapshot",
  "prepareSelectedSnapshotEdit",
  "selectedTrainingSnapshotIds",
  "selectedTrainingSnapshots",
  "setTrainingSnapshotSelection",
  "snapshotEditorDraft",
  "snapshotOverrideWarning",
  "configSnapshots",
  "allConfigSnapshots",
  "configSnapshotLibrary",
  "configSnapshotGroups",
  "allConfigSnapshotGroups",
  "configSnapshotCount",
  "allConfigSnapshotCount",
  "trainingConfigSnapshots",
  "allTrainingConfigSnapshots",
  "trainingConfigSnapshotGroups",
  "allTrainingConfigSnapshotGroups",
  "trainingConfigSnapshotCount",
  "allTrainingConfigSnapshotCount",
  "configSnapshotLibraryCount",
  "addConfigSnapshot",
  "removeConfigSnapshot",
  "renameConfigSnapshot",
  "updateSelectedConfigSnapshot",
  "loadConfigSnapshot",
  "includeConfigSnapshot",
  "excludeConfigSnapshot",
  "toggleConfigSnapshotRunSelection",
  "preparePresetSnapshotDraft",
  "prepareTrainingPresetSnapshotDraft",
  "prepareTrainingSelectedSnapshotEdit",
  "updateSnapshotEditorDraftOverride",
  "clearSnapshotEditorDraftOverride",
  "resetSnapshotEditorDraft",
  "libraryLoading",
  "isLibraryError",
  "libraryError",
] as const satisfies readonly (keyof TargetConfigContextValue)[];

type CatalogKey = (typeof TARGET_CATALOG_KEYS)[number];
type ModelTargetKey = (typeof MODEL_TARGET_KEYS)[number];
type TrainingTargetKey = (typeof TRAINING_TARGET_KEYS)[number];
type TargetSnapshotKey = (typeof TARGET_SNAPSHOT_KEYS)[number];

export type TargetCatalogContextValue = Pick<
  TargetConfigContextValue,
  CatalogKey
>;
export type ModelTargetContextValue = Pick<
  TargetConfigContextValue,
  ModelTargetKey
>;
export type TrainingTargetContextValue = Pick<
  TargetConfigContextValue,
  TrainingTargetKey
>;
export type TargetSnapshotsContextValue = Pick<
  TargetConfigContextValue,
  TargetSnapshotKey
>;

type CoveredTargetKey =
  | CatalogKey
  | ModelTargetKey
  | TrainingTargetKey
  | TargetSnapshotKey;
type MissingTargetKey = Exclude<keyof TargetConfigContextValue, CoveredTargetKey>;

// A compile-time architecture budget: adding a field to the old aggregate
// Interface requires assigning it to one of the focused domain Interfaces.
export const TARGET_CONTEXT_KEYS_COVER_ALL_FIELDS: MissingTargetKey extends never
  ? true
  : never = true;

function pickTargetFields<
  Keys extends readonly (keyof TargetConfigContextValue)[],
>(target: TargetConfigContextValue, keys: Keys) {
  return Object.fromEntries(keys.map((key) => [key, target[key]])) as Pick<
    TargetConfigContextValue,
    Keys[number]
  >;
}

function shallowValuesEqual<Target extends Record<PropertyKey, unknown>>(
  left: Target,
  right: Target,
) {
  const keys = Reflect.ownKeys(left);
  return (
    keys.length === Reflect.ownKeys(right).length &&
    keys.every((key) => Object.is(left[key], right[key]))
  );
}

/**
 * Preserves a slice reference while all of its public values are unchanged.
 * This lets React Context isolate consumers even though the orchestration hook
 * still produces a compatibility aggregate during the migration.
 */
function useShallowStableSlice<Target extends Record<PropertyKey, unknown>>(
  value: Target,
) {
  const stableValue = useRef(value);
  if (!shallowValuesEqual(stableValue.current, value)) {
    stableValue.current = value;
  }
  return stableValue.current;
}

export function useTargetContextSlices(target: TargetConfigContextValue) {
  const catalog = useShallowStableSlice(
    pickTargetFields(target, TARGET_CATALOG_KEYS),
  );
  const model = useShallowStableSlice(
    pickTargetFields(target, MODEL_TARGET_KEYS),
  );
  const training = useShallowStableSlice(
    pickTargetFields(target, TRAINING_TARGET_KEYS),
  );
  const snapshots = useShallowStableSlice(
    pickTargetFields(target, TARGET_SNAPSHOT_KEYS),
  );

  return { catalog, model, training, snapshots };
}
