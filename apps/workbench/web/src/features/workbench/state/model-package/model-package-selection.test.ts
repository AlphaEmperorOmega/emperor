import { describe, expect, it } from "vitest";
import {
  deriveModelPackageSelection,
} from "@/features/workbench/state/model-package/model-package-selection";
import type { ConfigField, Dataset, Preset } from "@/lib/api/models";
import { configSectionsFields } from "@/lib/config";
import { type ConfigSnapshot } from "@/lib/config-snapshots";

function field(overrides: Partial<ConfigField> & Pick<ConfigField, "key">): ConfigField {
  const section = overrides.section ?? "General";
  return {
    key: overrides.key,
    configKey: overrides.configKey ?? overrides.key.toUpperCase(),
    flag: overrides.flag ?? `--${overrides.key.replace(/_/g, "-")}`,
    label: overrides.label ?? overrides.key,
    section,
    sectionPath: overrides.sectionPath ?? [section || "General"],
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
    modelType: overrides.modelType ?? "linears",
    model: overrides.model ?? "linear",
    preset: overrides.preset ?? "baseline",
    overrides: overrides.overrides ?? {},
    createdAt: overrides.createdAt ?? "2026-06-01T00:00:00.000Z",
  };
}

describe("target selection", () => {
  it("derives target, preset, dataset, and config counts", () => {
    const state = deriveModelPackageSelection({
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
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedPreset: "fast",
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
    expect(state.presetOwnedFieldCount).toBe(1);
    expect(state.fieldCount).toBe(3);
    expect(state.modelConfigSnapshots.map((item) => item.id)).toEqual([
      "fast",
      "baseline",
      "other-preset",
    ]);
    expect(state.modelConfigSnapshotGroups).toEqual([
      { preset: "baseline", snapshots: [snapshot({ id: "baseline", preset: "baseline" })] },
      { preset: "fast", snapshots: [snapshot({ id: "fast", preset: "fast" })] },
      { preset: "rare", snapshots: [snapshot({ id: "other-preset", preset: "rare" })] },
    ]);
  });

  it("preserves recurrent controller schema sections from config comments", () => {
    const state = deriveModelPackageSelection({
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
          key: "recurrent_gate_stack_hidden_dim",
          section: "Recurrent Gate Stack Options",
        }),
        field({
          key: "recurrent_halting_threshold",
          section: "Recurrent Halting Options",
          type: "float",
        }),
      ],
      configSnapshots: [],
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedPreset: "baseline",
    });

    expect(state.configSections.map((section) => section.title)).toEqual([
      "Recurrent Layer Options",
      "Recurrent Gate Stack Options",
      "Recurrent Halting Options",
    ]);
    expect(configSectionsFields(state.configSections).map((field) => field.key))
      .toEqual([
        "recurrent_flag",
        "recurrent_layer_norm_position",
        "recurrent_gate_stack_hidden_dim",
        "recurrent_halting_threshold",
      ]);
    expect(
      configSectionsFields(state.configSections).map(
        (configField) => configField.section,
      ),
    ).toEqual([
      "Recurrent Layer Options",
      "Recurrent Layer Options",
      "Recurrent Gate Stack Options",
      "Recurrent Halting Options",
    ]);
  });

  it("builds nested config sections from backend section paths", () => {
    const state = deriveModelPackageSelection({
      datasets: [],
      presets: [],
      schemaFields: [
        field({
          key: "recurrent_flag",
          section: "Recurrent Layer Options",
          sectionPath: ["Recurrent Layer Options"],
        }),
        field({
          key: "recurrent_stack_gate_flag",
          section: "Recurrent Gate Options",
          sectionPath: ["Recurrent Layer Options", "Recurrent Gate Options"],
        }),
        field({
          key: "recurrent_gate_stack_hidden_dim",
          section: "Recurrent Gate Stack Options",
          sectionPath: [
            "Recurrent Layer Options",
            "Recurrent Gate Options",
            "Recurrent Gate Stack Options",
          ],
        }),
      ],
      configSnapshots: [],
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedPreset: "baseline",
    });

    expect(state.configSections).toHaveLength(1);
    expect(state.configSections[0].title).toBe("Recurrent Layer Options");
    expect(state.configSections[0].children?.[0].title).toBe(
      "Recurrent Gate Options",
    );
    expect(state.configSections[0].children?.[0].children?.[0].title).toBe(
      "Recurrent Gate Stack Options",
    );
    expect(configSectionsFields(state.configSections).map((item) => item.key))
      .toEqual([
        "recurrent_flag",
        "recurrent_stack_gate_flag",
        "recurrent_gate_stack_hidden_dim",
      ]);
  });
});
