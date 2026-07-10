import { type ConfigField, type Dataset, type Preset } from "@/lib/api";
import {
  configSectionsFields,
  groupConfigFieldsBySectionPath,
  presetOwnedCount,
  type ConfigSection,
} from "@/lib/config";
import {
  groupConfigSnapshotsByPreset,
  selectedConfigSnapshots,
  type ConfigSnapshot,
  type ConfigSnapshotGroup,
} from "@/lib/config-snapshots";

export type TargetSelectionInput = {
  datasets?: Dataset[];
  presets?: Preset[];
  schemaFields?: ConfigField[];
  configSnapshots: ConfigSnapshot[];
  selectedModelType: string;
  selectedModel: string;
  selectedPreset: string;
  selectedTrainingPresets: string[];
};

export type TargetSelectionState = {
  datasetNames: string[];
  presetNames: string[];
  selectedPresetMeta: Preset | undefined;
  configSections: ConfigSection[];
  configFields: ConfigField[];
  modelConfigSnapshots: ConfigSnapshot[];
  modelConfigSnapshotGroups: ConfigSnapshotGroup[];
  visibleConfigSnapshots: ConfigSnapshot[];
  configSnapshotGroups: ConfigSnapshotGroup[];
  presetOwnedFieldCount: number;
  fieldCount: number;
};

export function deriveTargetSelectionState(
  input: TargetSelectionInput,
): TargetSelectionState {
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
  const visibleConfigSnapshots = selectedConfigSnapshots(
    modelConfigSnapshots,
    input.selectedModelType,
    input.selectedModel,
    input.selectedTrainingPresets,
  );
  const configSnapshotGroups = groupConfigSnapshotsByPreset(
    visibleConfigSnapshots,
    input.selectedTrainingPresets,
  );
  const presetOwnedFieldCount = configSections.reduce(
    (total, section) => total + presetOwnedCount(configSectionsFields([section])),
    0,
  );
  const fieldCount = configFields.length;

  return {
    datasetNames,
    presetNames,
    selectedPresetMeta,
    configSections,
    configFields,
    modelConfigSnapshots,
    modelConfigSnapshotGroups,
    visibleConfigSnapshots,
    configSnapshotGroups,
    presetOwnedFieldCount,
    fieldCount,
  };
}
