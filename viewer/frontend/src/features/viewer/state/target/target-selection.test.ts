import { describe, expect, it } from "vitest";
import {
  deriveTargetSelectionState,
} from "@/features/viewer/state/target/target-selection";
import {
  type ConfigField,
  type Dataset,
  type Preset,
} from "@/lib/api";
import { type ConfigSnapshot } from "@/lib/config-snapshots";

function field(overrides: Partial<ConfigField> & Pick<ConfigField, "key">): ConfigField {
  return {
    key: overrides.key,
    configKey: overrides.configKey ?? overrides.key.toUpperCase(),
    flag: overrides.flag ?? `--${overrides.key.replace(/_/g, "-")}`,
    label: overrides.label ?? overrides.key,
    section: overrides.section ?? "General",
    type: overrides.type ?? "int",
    default: overrides.default ?? 0,
    nullable: overrides.nullable ?? false,
    choices: overrides.choices ?? [],
    locked: overrides.locked ?? false,
    lockedValue: overrides.lockedValue,
    lockedReason: overrides.lockedReason,
  };
}

function dataset(name: string): Dataset {
  return {
    name,
    label: name,
    inputDim: 8,
    outputDim: 2,
  };
}

function preset(name: string, description = ""): Preset {
  return {
    name,
    label: name,
    description,
  };
}

function snapshot(overrides: Partial<ConfigSnapshot> & Pick<ConfigSnapshot, "id">): ConfigSnapshot {
  return {
    id: overrides.id,
    name: overrides.name ?? overrides.id,
    model: overrides.model ?? "linear",
    preset: overrides.preset ?? "baseline",
    overrides: overrides.overrides ?? {},
    createdAt: overrides.createdAt ?? "2026-06-01T00:00:00.000Z",
  };
}

describe("target selection", () => {
  it("derives target, preset, dataset, and config counts", () => {
    const state = deriveTargetSelectionState({
      datasets: [dataset("Mnist"), dataset("Cifar10")],
      presets: [preset("baseline"), preset("fast", "Fast preset")],
      schemaFields: [
        field({ key: "hidden_dim", section: "Model" }),
        field({
          key: "layer_norm",
          label: "Layer Norm",
          section: "Preset",
          locked: true,
          lockedValue: true,
        }),
        field({ key: "dropout", section: "" }),
      ],
      configSnapshots: [
        snapshot({ id: "fast", preset: "fast" }),
        snapshot({ id: "baseline", preset: "baseline" }),
        snapshot({ id: "other-preset", preset: "rare" }),
        snapshot({ id: "other-model", model: "bert", preset: "fast" }),
      ],
      selectedModel: "linear",
      selectedPreset: "fast",
      selectedTrainingPresets: ["fast", "baseline"],
      overrides: { hidden_dim: "128", dropout: "0.1" },
    });

    expect(state.datasetNames).toEqual(["Mnist", "Cifar10"]);
    expect(state.presetNames).toEqual(["baseline", "fast"]);
    expect(state.selectedPresetMeta?.description).toBe("Fast preset");
    expect(state.configSections.map((section) => section.title)).toEqual([
      "Model",
      "Preset",
      "General",
    ]);
    expect(state.configFields.map((configField) => configField.key)).toEqual([
      "hidden_dim",
      "layer_norm",
      "dropout",
    ]);
    expect(state.overrideCount).toBe(2);
    expect(state.presetOwnedFieldCount).toBe(1);
    expect(state.fieldCount).toBe(3);
    expect(state.modelConfigSnapshots.map((item) => item.id)).toEqual([
      "fast",
      "baseline",
      "other-preset",
    ]);
    expect(state.visibleConfigSnapshots.map((item) => item.id)).toEqual([
      "fast",
      "baseline",
    ]);
    expect(state.configSnapshotGroups).toEqual([
      { preset: "fast", snapshots: [snapshot({ id: "fast", preset: "fast" })] },
      {
        preset: "baseline",
        snapshots: [snapshot({ id: "baseline", preset: "baseline" })],
      },
    ]);
    expect(state.modelConfigSnapshotGroups).toEqual([
      { preset: "baseline", snapshots: [snapshot({ id: "baseline", preset: "baseline" })] },
      { preset: "fast", snapshots: [snapshot({ id: "fast", preset: "fast" })] },
      { preset: "rare", snapshots: [snapshot({ id: "other-preset", preset: "rare" })] },
    ]);
  });

  it("normalizes recurrent controller schema sections for viewer display", () => {
    const state = deriveTargetSelectionState({
      datasets: [],
      presets: [],
      schemaFields: [
        field({
          key: "recurrent_flag",
          section: "Recurrent Layer Options",
          type: "bool",
        }),
        field({
          key: "recurrent_layer_norm_position",
          section: "Recurrent Layer Options",
          type: "enum",
          default: "DISABLED",
          choices: ["DISABLED", "BEFORE", "DEFAULT", "AFTER"],
        }),
        field({
          key: "recurrent_gate_hidden_dim",
          section: "Recurrent Gate Stack Options",
        }),
        field({
          key: "recurrent_halting_threshold",
          section: "Recurrent Halting Options",
          type: "float",
        }),
      ],
      configSnapshots: [],
      selectedModel: "linear",
      selectedPreset: "baseline",
      selectedTrainingPresets: [],
      overrides: {},
    });

    expect(state.configSections.map((section) => section.title)).toEqual([
      "Recurrent Layer Options",
    ]);
    expect(state.configSections[0].fields.map((configField) => configField.key))
      .toEqual([
        "recurrent_flag",
        "recurrent_layer_norm_position",
        "recurrent_gate_hidden_dim",
        "recurrent_halting_threshold",
      ]);
    expect(
      state.configSections[0].fields.map((configField) => configField.section),
    ).toEqual([
      "Recurrent Layer Options",
      "Recurrent Layer Options",
      "Recurrent Layer Options",
      "Recurrent Layer Options",
    ]);
  });
});
