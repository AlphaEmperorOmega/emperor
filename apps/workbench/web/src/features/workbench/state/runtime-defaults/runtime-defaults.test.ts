import { describe, expect, it } from "vitest";
import { type ConfigField } from "@/lib/api/models";
import {
  effectivePresetOverrides,
  inactivePresetOwnedOverrideKeys,
  isEnabledRuntimeDefaultValue,
  runtimeDefaultsEditor,
  runtimeDefaultsMetrics,
} from "@/features/workbench/state/runtime-defaults/runtime-defaults";

function field(
  overrides: Partial<ConfigField> & Pick<ConfigField, "key">,
): ConfigField {
  const section = overrides.section ?? "General";
  return {
    key: overrides.key,
    configKey: overrides.configKey ?? overrides.key.toUpperCase(),
    flag: overrides.flag ?? `--${overrides.key.replace(/_/g, "-")}`,
    label: overrides.label ?? overrides.key,
    section,
    sectionPath: overrides.sectionPath ?? [section],
    type: overrides.type ?? "int",
    default: "default" in overrides ? overrides.default ?? null : 0,
    nullable: overrides.nullable ?? false,
    choices: overrides.choices ?? [],
    locked: overrides.locked ?? false,
    lockedValue: overrides.lockedValue,
    lockedReason: overrides.lockedReason,
  };
}

const hiddenDim = field({
  key: "hidden_dim",
  configKey: "HIDDEN_DIM",
  type: "int",
  default: 256,
});

const adaptiveOptionFields = [
  field({
    key: "weight_option_flag",
    type: "bool",
    default: false,
    choices: [true, false],
  }),
  field({
    key: "weight_option",
    type: "class",
    default: null,
    nullable: true,
    choices: [
      "SingleModelDynamicWeightConfig",
      "DualModelDynamicWeightConfig",
    ],
  }),
];

describe("Runtime Defaults", () => {
  it("canonicalizes replacement keys while retaining inactive preset-owned values", () => {
    const presetOwnedLayerWidth = field({
      key: "layer_width",
      configKey: "LAYER_WIDTH",
      type: "int",
      default: 64,
      locked: true,
      lockedValue: 96,
    });

    expect(
      runtimeDefaultsEditor.replace([hiddenDim, presetOwnedLayerWidth], {
        HIDDEN_DIM: "128",
        LAYER_WIDTH: "192",
      }),
    ).toEqual({
      hidden_dim: "128",
      layer_width: "192",
    });
  });

  it("suppresses default-equivalent edits with token-aware removal", () => {
    expect(
      runtimeDefaultsEditor.edit(
        [hiddenDim],
        { HIDDEN_DIM: "128", unknown: "kept" },
        "hidden-dim",
        "256",
      ),
    ).toEqual({ unknown: "kept" });
  });

  it("repairs paired adaptive options after semantic edits and clears", () => {
    const enabled = runtimeDefaultsEditor.edit(
      adaptiveOptionFields,
      {},
      "WEIGHT_OPTION_FLAG",
      "true",
    );

    expect(enabled).toEqual({
      weight_option_flag: "true",
      weight_option: "SingleModelDynamicWeightConfig",
    });
    expect(
      runtimeDefaultsEditor.clear(
        adaptiveOptionFields,
        enabled,
        "weight-option-flag",
      ),
    ).toEqual({});
  });

  it("preserves object identity when normalization or an edit is a no-op", () => {
    const current = { hidden_dim: "128" };

    expect(runtimeDefaultsEditor.normalize([hiddenDim], current)).toBe(current);
    expect(
      runtimeDefaultsEditor.edit([hiddenDim], current, "hidden_dim", "128"),
    ).toBe(current);
    expect(
      runtimeDefaultsEditor.edit([], current, "HIDDEN-DIM", "128"),
    ).toBe(current);
    expect(
      runtimeDefaultsEditor.clear([hiddenDim], current, "missing"),
    ).toBe(current);
  });

  it("projects preset-owned overrides without mutating the canonical draft", () => {
    const presetOwnedField = field({
      key: "layer_norm",
      locked: true,
      lockedValue: true,
    });
    const overrides = {
      hidden_dim: "128",
      layer_norm: "false",
    };

    expect(
      inactivePresetOwnedOverrideKeys(
        [hiddenDim, presetOwnedField],
        overrides,
      ),
    ).toEqual(["layer_norm"]);
    expect(
      effectivePresetOverrides([hiddenDim, presetOwnedField], overrides),
    ).toEqual({ hidden_dim: "128" });
    expect(overrides).toEqual({
      hidden_dim: "128",
      layer_norm: "false",
    });
  });

  it("reports shared field metrics and enabled-value semantics", () => {
    const presetOwnedField = field({
      key: "layer_norm",
      locked: true,
      lockedValue: true,
    });

    expect(
      runtimeDefaultsMetrics(
        [hiddenDim, presetOwnedField],
        { hidden_dim: "128" },
      ),
    ).toEqual({
      fieldCount: 2,
      overrideCount: 1,
      presetCount: 1,
      state: "override-and-preset",
    });
    expect(["true", "1", "yes", "on"].every(isEnabledRuntimeDefaultValue)).toBe(
      true,
    );
    expect(isEnabledRuntimeDefaultValue("false")).toBe(false);
  });
});
