import { type ConfigField, type Dataset, type Preset } from "@/lib/api";
import { presetOwnedCount, type ConfigSection, type OverrideValues } from "@/lib/config";
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
    const section = field.section || "General";
    groups.set(section, [...(groups.get(section) ?? []), field]);
  }
  const configSections = Array.from(groups, ([title, fields]) => ({ title, fields }));
  const configFields = configSections.flatMap((section) => section.fields);
  const visibleConfigSnapshots = selectedConfigSnapshots(
    input.configSnapshots,
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
    visibleConfigSnapshots,
    configSnapshotGroups,
    overrideCount: Object.keys(input.overrides).length,
    presetOwnedFieldCount,
    fieldCount,
  };
}
