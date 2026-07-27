import { describe, expect, it } from "vitest";
import { type ConfigField } from "@/lib/api/models";
import {
  applicableConfigFields,
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
    applicableWhen: overrides.applicableWhen ?? [],
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

const recurrentVariantFields = [
  field({
    key: "recurrent_composition_option",
    configKey: "RECURRENT_COMPOSITION_OPTION",
    type: "class",
    default: "RecurrentLayerConfig",
    choices: [
      "HierarchicalReasoningModelRecurrentConfig",
      "RecurrentLayerConfig",
      "TinyRecursiveModelRecurrentConfig",
    ],
  }),
  field({
    key: "recurrent_no_gradient_transition_count",
    default: null,
    nullable: true,
  }),
  field({
    key: "recurrent_max_steps",
    default: 2,
    applicableWhen: [
      {
        key: "RECURRENT_COMPOSITION_OPTION",
        values: ["RecurrentLayerConfig"],
      },
    ],
  }),
  field({
    key: "recurrent_reinject_original_hidden_flag",
    type: "bool",
    default: false,
    choices: [true, false],
    applicableWhen: [
      {
        key: "RECURRENT_COMPOSITION_OPTION",
        values: ["RecurrentLayerConfig"],
      },
    ],
  }),
  field({
    key: "recurrent_latent_updates_per_answer_update",
    default: 2,
    applicableWhen: [
      {
        key: "RECURRENT_COMPOSITION_OPTION",
        values: ["TinyRecursiveModelRecurrentConfig"],
      },
    ],
  }),
  field({
    key: "recurrent_answer_update_count",
    default: 2,
    applicableWhen: [
      {
        key: "RECURRENT_COMPOSITION_OPTION",
        values: ["TinyRecursiveModelRecurrentConfig"],
      },
    ],
  }),
  field({
    key: "recurrent_high_cycles",
    default: 2,
    applicableWhen: [
      {
        key: "RECURRENT_COMPOSITION_OPTION",
        values: ["HierarchicalReasoningModelRecurrentConfig"],
      },
    ],
  }),
  field({
    key: "recurrent_low_cycles",
    default: 2,
    applicableWhen: [
      {
        key: "RECURRENT_COMPOSITION_OPTION",
        values: ["HierarchicalReasoningModelRecurrentConfig"],
      },
    ],
  }),
  field({
    key: "recurrent_initialization_standard_deviation",
    type: "float",
    default: 1,
    applicableWhen: [
      {
        key: "RECURRENT_COMPOSITION_OPTION",
        values: [
          "TinyRecursiveModelRecurrentConfig",
          "HierarchicalReasoningModelRecurrentConfig",
        ],
      },
    ],
  }),
];

describe("Runtime Defaults", () => {
  it("selects recurrent fields generically for Standard, TRM, and HRM", () => {
    const keysFor = (overrides: Record<string, string>) =>
      applicableConfigFields(recurrentVariantFields, overrides).map(
        (item) => item.key,
      );

    expect(keysFor({})).toEqual([
      "recurrent_composition_option",
      "recurrent_no_gradient_transition_count",
      "recurrent_max_steps",
      "recurrent_reinject_original_hidden_flag",
    ]);
    expect(
      keysFor({
        recurrent_composition_option: "TinyRecursiveModelRecurrentConfig",
      }),
    ).toEqual([
      "recurrent_composition_option",
      "recurrent_no_gradient_transition_count",
      "recurrent_latent_updates_per_answer_update",
      "recurrent_answer_update_count",
      "recurrent_initialization_standard_deviation",
    ]);
    expect(
      keysFor({
        recurrent_composition_option:
          "HierarchicalReasoningModelRecurrentConfig",
      }),
    ).toEqual([
      "recurrent_composition_option",
      "recurrent_no_gradient_transition_count",
      "recurrent_high_cycles",
      "recurrent_low_cycles",
      "recurrent_initialization_standard_deviation",
    ]);
  });

  it("discards incompatible recurrent overrides as variants change", () => {
    const standard = runtimeDefaultsEditor.replace(recurrentVariantFields, {
      recurrent_max_steps: "5",
      recurrent_latent_updates_per_answer_update: "4",
    });
    expect(standard).toEqual({ recurrent_max_steps: "5" });

    const trm = runtimeDefaultsEditor.edit(
      recurrentVariantFields,
      standard,
      "RECURRENT_COMPOSITION_OPTION",
      "TinyRecursiveModelRecurrentConfig",
    );
    expect(trm).toEqual({
      recurrent_composition_option: "TinyRecursiveModelRecurrentConfig",
    });
    const configuredTrm = runtimeDefaultsEditor.edit(
      recurrentVariantFields,
      trm,
      "recurrent_latent_updates_per_answer_update",
      "4",
    );

    const hrm = runtimeDefaultsEditor.edit(
      recurrentVariantFields,
      configuredTrm,
      "recurrent_composition_option",
      "HierarchicalReasoningModelRecurrentConfig",
    );
    expect(hrm).toEqual({
      recurrent_composition_option:
        "HierarchicalReasoningModelRecurrentConfig",
    });
    expect(
      runtimeDefaultsEditor.edit(
        recurrentVariantFields,
        hrm,
        "recurrent_composition_option",
        "RecurrentLayerConfig",
      ),
    ).toEqual({});
  });

  it("uses a locked controller value before an override or default", () => {
    const lockedFields = recurrentVariantFields.map((item) =>
      item.key === "recurrent_composition_option"
        ? {
            ...item,
            locked: true,
            lockedValue: "HierarchicalReasoningModelRecurrentConfig",
          }
        : item,
    );

    const applicableKeys = applicableConfigFields(lockedFields, {
      recurrent_composition_option: "TinyRecursiveModelRecurrentConfig",
    }).map((item) => item.key);

    expect(applicableKeys).toContain("recurrent_high_cycles");
    expect(applicableKeys).not.toContain(
      "recurrent_latent_updates_per_answer_update",
    );
  });

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
