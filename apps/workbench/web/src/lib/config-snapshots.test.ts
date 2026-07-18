import { describe, expect, it } from "vitest";
import {
  configSnapshotOverrideCount,
  configSnapshotOverrideCountLabel,
  configSnapshotOverrideEntries,
  createConfigSnapshot,
  groupConfigSnapshotsByPreset,
  validateConfigSnapshotCandidate,
  validateConfigSnapshotName,
} from "@/lib/config-snapshots";
import type { ConfigField } from "@/lib/api/models";

type ConfigFieldFixture = Omit<ConfigField, "sectionPath"> &
  Partial<Pick<ConfigField, "sectionPath">>;

function withSectionPaths(fields: ConfigFieldFixture[]): ConfigField[] {
  return fields.map((field) => ({
    ...field,
    sectionPath: field.sectionPath ?? [field.section || "General"],
  }));
}

const fields: ConfigField[] = withSectionPaths([
  {
    key: "hidden_dim",
    configKey: "HIDDEN_DIM",
    flag: "--hidden-dim",
    label: "Hidden Dim",
    section: "Model",
    type: "int",
    default: 64,
    nullable: false,
    choices: [],
    locked: false,
  },
  {
    key: "num_layers",
    configKey: "NUM_LAYERS",
    flag: "--num-layers",
    label: "Num Layers",
    section: "Model",
    type: "int",
    default: 2,
    nullable: false,
    choices: [],
    locked: false,
  },
  {
    key: "activation",
    configKey: "ACTIVATION",
    flag: "--activation",
    label: "Activation",
    section: "Model",
    type: "str",
    default: "RELU",
    nullable: false,
    choices: ["RELU", "GELU"],
    locked: false,
  },
  {
    key: "num_epochs",
    configKey: "NUM_EPOCHS",
    flag: "--num-epochs",
    label: "Epochs",
    section: "Training",
    type: "int",
    default: 10,
    nullable: false,
    choices: [],
    locked: false,
  },
  {
    key: "layer_norm",
    configKey: "LAYER_NORM",
    flag: "--layer-norm",
    label: "Layer Norm",
    section: "Preset",
    type: "bool",
    default: false,
    nullable: false,
    choices: [],
    locked: true,
    lockedValue: true,
    lockedReason: "Preset controlled",
  },
]);

const adaptiveFields: ConfigField[] = withSectionPaths([
  {
    key: "weight_option_flag",
    configKey: "WEIGHT_OPTION_FLAG",
    flag: "--weight-option-flag",
    label: "Weight Option Flag",
    section: "Weight Options",
    type: "bool",
    default: false,
    nullable: false,
    choices: [true, false],
    locked: false,
  },
  {
    key: "weight_option",
    configKey: "WEIGHT_OPTION",
    flag: "--weight-option",
    label: "Weight Option",
    section: "Weight Options",
    type: "class",
    default: null,
    nullable: true,
    choices: ["SingleModelDynamicWeightConfig"],
    locked: false,
  },
  {
    key: "bias_option_flag",
    configKey: "BIAS_OPTION_FLAG",
    flag: "--bias-option-flag",
    label: "Bias Option Flag",
    section: "Bias Options",
    type: "bool",
    default: false,
    nullable: false,
    choices: [true, false],
    locked: false,
  },
  {
    key: "bias_option",
    configKey: "BIAS_OPTION",
    flag: "--bias-option",
    label: "Bias Option",
    section: "Bias Options",
    type: "class",
    default: null,
    nullable: true,
    choices: ["AdditiveDynamicBiasConfig"],
    locked: false,
  },
  {
    key: "diagonal_option_flag",
    configKey: "DIAGONAL_OPTION_FLAG",
    flag: "--diagonal-option-flag",
    label: "Diagonal Option Flag",
    section: "Diagonal Options",
    type: "bool",
    default: false,
    nullable: false,
    choices: [true, false],
    locked: false,
  },
  {
    key: "diagonal_option",
    configKey: "DIAGONAL_OPTION",
    flag: "--diagonal-option",
    label: "Diagonal Option",
    section: "Diagonal Options",
    type: "class",
    default: null,
    nullable: true,
    choices: ["StandardDynamicDiagonalConfig"],
    locked: false,
  },
  {
    key: "mask_option_flag",
    configKey: "MASK_OPTION_FLAG",
    flag: "--mask-option-flag",
    label: "Mask Option Flag",
    section: "Mask Options",
    type: "bool",
    default: false,
    nullable: false,
    choices: [true, false],
    locked: false,
  },
  {
    key: "row_mask_option",
    configKey: "ROW_MASK_OPTION",
    flag: "--row-mask-option",
    label: "Row Mask Option",
    section: "Mask Options",
    type: "class",
    default: null,
    nullable: true,
    choices: ["DiagonalAxisMaskConfig"],
    locked: false,
  },
]);

function makeSnapshot(overrides: Record<string, string>, name = "snapshot") {
  const result = createConfigSnapshot({
    id: `snap-${name}`,
    name,
    modelType: "linears",
    model: "linear",
    preset: "baseline",
    fields,
    overrides,
    snapshots: [],
    createdAt: "2026-06-04T00:00:00.000Z",
  });
  if (!result.ok) {
    throw new Error(result.error);
  }
  return result.snapshot;
}

describe("config snapshots", () => {
  it("counts and labels snapshot overrides", () => {
    const snapshot = makeSnapshot(
      { hidden_dim: "128", num_layers: "4" },
      "counted",
    );

    expect(configSnapshotOverrideCount(snapshot)).toBe(2);
    expect(configSnapshotOverrideCountLabel(0)).toBe("0 overrides");
    expect(configSnapshotOverrideCountLabel(1)).toBe("1 override");
    expect(configSnapshotOverrideCountLabel(2)).toBe("2 overrides");
  });

  it("groups snapshots by preset without changing snapshot order", () => {
    const firstBaselineSnapshot = makeSnapshot(
      { hidden_dim: "128" },
      "first-baseline",
    );
    const wideSnapshot = {
      ...makeSnapshot({ hidden_dim: "256" }, "wide"),
      preset: "wide",
    };
    const secondBaselineSnapshot = makeSnapshot(
      { hidden_dim: "512" },
      "second-baseline",
    );

    expect(
      groupConfigSnapshotsByPreset(
        [firstBaselineSnapshot, wideSnapshot, secondBaselineSnapshot],
        ["baseline", "wide"],
      ),
    ).toEqual([
      {
        preset: "baseline",
        snapshots: [firstBaselineSnapshot, secondBaselineSnapshot],
      },
      { preset: "wide", snapshots: [wideSnapshot] },
    ]);
  });

  it("rejects default-equivalent snapshots", () => {
    const result = createConfigSnapshot({
      id: "snap-default",
      name: "default",
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      fields,
      overrides: { hidden_dim: "64" },
      snapshots: [],
      createdAt: "2026-06-04T00:00:00.000Z",
    });

    expect(result).toMatchObject({
      ok: false,
      error: "Change at least one non-default field before adding a snapshot.",
    });
  });

  it("excludes normalized default-equivalent snapshot entries", () => {
    const normalizationFields: ConfigField[] = [
      {
        ...fields[0],
        default: 256,
      },
      {
        ...fields[0],
        key: "dropout",
        configKey: "DROPOUT",
        flag: "--dropout",
        label: "Dropout",
        type: "float",
        default: 0.2,
      },
      {
        ...fields[0],
        key: "use_bias",
        configKey: "USE_BIAS",
        flag: "--use-bias",
        label: "Use Bias",
        type: "bool",
        default: false,
        choices: [true, false],
      },
      {
        ...fields[0],
        key: "optional_hidden_dim",
        configKey: "OPTIONAL_HIDDEN_DIM",
        flag: "--optional-hidden-dim",
        label: "Optional Hidden Dim",
        default: null,
        nullable: true,
      },
    ];

    expect(
      configSnapshotOverrideEntries(normalizationFields, {
        hidden_dim: "256",
        dropout: "0.2",
        use_bias: "false",
        optional_hidden_dim: "",
      }),
    ).toEqual({ entries: [], lockedFields: [] });
  });

  it("rejects empty snapshot names", () => {
    const result = createConfigSnapshot({
      id: "snap-empty-name",
      name: "   ",
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      fields,
      overrides: { hidden_dim: "128" },
      snapshots: [],
      createdAt: "2026-06-04T00:00:00.000Z",
    });

    expect(result).toMatchObject({
      ok: false,
      error: "Config snapshot name cannot be empty.",
    });
  });

  it("rejects duplicate config identity and duplicate names", () => {
    const existing = makeSnapshot({ hidden_dim: "128" }, "same name");
    const duplicate = createConfigSnapshot({
      id: "snap-dup",
      name: "different name",
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      fields,
      overrides: { hidden_dim: "128" },
      snapshots: [existing],
      createdAt: "2026-06-04T00:00:00.000Z",
    });
    const duplicateNameDifferentConfig = createConfigSnapshot({
      id: "snap-name-dup",
      name: " Same Name ",
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      fields,
      overrides: { hidden_dim: "256" },
      snapshots: [existing],
      createdAt: "2026-06-04T00:00:00.000Z",
    });

    expect(duplicate).toMatchObject({
      ok: false,
      error: "A snapshot with these config values already exists.",
    });
    expect(duplicateNameDifferentConfig).toMatchObject({
      ok: false,
      error: "A snapshot with this name already exists.",
    });
  });

  it("allows edit validation to exclude the snapshot being updated", () => {
    const existing = makeSnapshot({ hidden_dim: "128" }, "existing");

    const nameResult = validateConfigSnapshotName({
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      name: " existing ",
      snapshots: [existing],
      excludeSnapshotId: existing.id,
    });
    const configResult = validateConfigSnapshotCandidate({
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      fields,
      overrides: { hidden_dim: "128" },
      snapshots: [existing],
      excludeSnapshotId: existing.id,
    });

    expect(nameResult).toMatchObject({ ok: true, name: "existing" });
    expect(configResult.ok).toBe(true);
  });

  it("rejects locked-field overrides defensively", () => {
    const result = createConfigSnapshot({
      id: "snap-locked",
      name: "locked",
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      fields,
      overrides: { layer_norm: "true" },
      snapshots: [],
      createdAt: "2026-06-04T00:00:00.000Z",
    });

    expect(result).toMatchObject({
      ok: false,
      error: "Snapshots cannot include preset-locked fields: Layer Norm.",
    });
  });

  it("rejects adaptive option flags without matching options", () => {
    const cases = [
      ["weight_option_flag", "weight_option"],
      ["bias_option_flag", "bias_option"],
      ["diagonal_option_flag", "diagonal_option"],
      ["mask_option_flag", "row_mask_option"],
    ];

    for (const [flagKey, optionKey] of cases) {
      const result = validateConfigSnapshotCandidate({
        modelType: "linears",
        model: "linear_adaptive",
        preset: "baseline",
        fields: adaptiveFields,
        overrides: { [flagKey]: "true" },
        snapshots: [],
      });

      expect(result).toMatchObject({
        ok: false,
        error: `Invalid config snapshot overrides: ${optionKey} must be set when ${flagKey} is True.`,
      });
    }
  });

  it("accepts adaptive option flags with matching options", () => {
    const cases = [
      {
        flagKey: "weight_option_flag",
        optionKey: "weight_option",
        optionValue: "SingleModelDynamicWeightConfig",
      },
      {
        flagKey: "bias_option_flag",
        optionKey: "bias_option",
        optionValue: "AdditiveDynamicBiasConfig",
      },
      {
        flagKey: "diagonal_option_flag",
        optionKey: "diagonal_option",
        optionValue: "StandardDynamicDiagonalConfig",
      },
      {
        flagKey: "mask_option_flag",
        optionKey: "row_mask_option",
        optionValue: "DiagonalAxisMaskConfig",
      },
    ];

    for (const { flagKey, optionKey, optionValue } of cases) {
      const result = validateConfigSnapshotCandidate({
        modelType: "linears",
        model: "linear_adaptive",
        preset: "baseline",
        fields: adaptiveFields,
        overrides: { [flagKey]: "true", [optionKey]: optionValue },
        snapshots: [],
      });

      expect(result).toMatchObject({
        ok: true,
        overrides: { [flagKey]: "true", [optionKey]: optionValue },
      });
    }
  });
});
