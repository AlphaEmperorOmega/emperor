import { type ConfigField, type Dataset, type Preset } from "@/lib/api";
import {
  normalizeConfigFieldForDisplay,
  presetOwnedCount,
  type ConfigSection,
  type OverrideValues,
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
  overrides: OverrideValues;
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
  overrideCount: number;
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

  const groups = new Map<string, ConfigField[]>();
  for (const field of input.schemaFields ?? []) {
    const displayField = normalizeConfigFieldForDisplay(field);
    groups.set(displayField.section, [
      ...(groups.get(displayField.section) ?? []),
      displayField,
    ]);
  }
  const configSections = Array.from(groups, ([title, fields]) => ({ title, fields }));
  const configFields = configSections.flatMap((section) => section.fields);
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
    (total, section) => total + presetOwnedCount(section.fields),
    0,
  );
  const fieldCount = configSections.reduce(
    (total, section) => total + section.fields.length,
    0,
  );

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
    overrideCount: Object.keys(input.overrides).length,
    presetOwnedFieldCount,
    fieldCount,
  };
}
