import type { ConfigField, Dataset, DatasetGroup, Preset } from "@/lib/api/models";
import {
  configSectionsFields,
  groupConfigFieldsBySectionPath,
  type ConfigSection,
} from "@/lib/config";
import { presetOwnedRuntimeDefaultCount } from "@/features/workbench/state/runtime-defaults/runtime-defaults";
import {
  groupConfigSnapshotsByPreset,
  type ConfigSnapshot,
  type ConfigSnapshotGroup,
} from "@/lib/config-snapshots";

const EMPTY_DATASETS: Dataset[] = [];

export type ModelPackageSelectionInput = {
  datasets?: Dataset[];
  presets?: Preset[];
  schemaFields?: ConfigField[];
  configSnapshots: ConfigSnapshot[];
  selectedModelType: string;
  selectedModel: string;
  selectedPreset: string;
};

export type ModelPackageSelection = {
  datasetNames: string[];
  presetNames: string[];
  selectedPresetMeta: Preset | undefined;
  configSections: ConfigSection[];
  configFields: ConfigField[];
  modelConfigSnapshots: ConfigSnapshot[];
  modelConfigSnapshotGroups: ConfigSnapshotGroup[];
  presetOwnedFieldCount: number;
  fieldCount: number;
};

export function deriveModelPackageSelection(
  input: ModelPackageSelectionInput,
): ModelPackageSelection {
  const datasetNames = (input.datasets ?? []).map((dataset) => dataset.name);
  const presetNames = (input.presets ?? []).map((preset) => preset.name);
  const selectedPresetMeta = (input.presets ?? []).find(
    (preset) => preset.name === input.selectedPreset,
  );
  const configSections = groupConfigFieldsBySectionPath(input.schemaFields ?? []);
  const configFields = configSectionsFields(configSections);
  const modelConfigSnapshots = input.configSnapshots.filter(
    (snapshot) =>
      snapshot.modelType === input.selectedModelType &&
      snapshot.model === input.selectedModel,
  );
  const modelConfigSnapshotGroups = groupConfigSnapshotsByPreset(
    modelConfigSnapshots,
    presetNames,
  );
  return {
    datasetNames,
    presetNames,
    selectedPresetMeta,
    configSections,
    configFields,
    modelConfigSnapshots,
    modelConfigSnapshotGroups,
    presetOwnedFieldCount: configSections.reduce(
      (total, section) =>
        total +
        presetOwnedRuntimeDefaultCount(configSectionsFields([section])),
      0,
    ),
    fieldCount: configFields.length,
  };
}

export function experimentTaskOptions(groups: DatasetGroup[]) {
  return groups.map((group) => ({
    value: group.experimentTask,
    label: group.label || group.experimentTask,
  }));
}

export function normalizeExperimentTask(
  current: string,
  defaultExperimentTask: string,
  groups: DatasetGroup[],
) {
  const taskNames = groups.map((group) => group.experimentTask);
  if (current && taskNames.includes(current)) {
    return current;
  }
  if (defaultExperimentTask && taskNames.includes(defaultExperimentTask)) {
    return defaultExperimentTask;
  }
  return taskNames[0] ?? "";
}

export function datasetsForExperimentTask(
  groups: DatasetGroup[],
  experimentTask: string,
) {
  return (
    groups.find((group) => group.experimentTask === experimentTask)?.datasets ??
    EMPTY_DATASETS
  );
}
