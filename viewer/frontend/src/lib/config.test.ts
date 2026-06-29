import { describe, expect, it } from "vitest";

import {
  boundaryProjectorFieldGroups,
  configFieldSelectOptions,
  defaultConfigFieldValue,
  deriveNestedConfigSections,
  disabledConfigFieldReasons,
  effectivePresetOverrides,
  filterConfigSectionsForSearch,
  isDefaultConfigFieldValue,
  lockedOverrideKeys,
  normalizeAdaptiveOptionOverrides,
  normalizeConfigFieldValue,
  overrideDigest,
  overrideValueForConfigField,
  type ConfigSection,
} from "@/lib/config";
import { type ConfigField } from "@/lib/api";

function field(overrides: Partial<ConfigField> & Pick<ConfigField, "key">): ConfigField {
  return {
    key: overrides.key,
    configKey: overrides.configKey ?? overrides.key.toUpperCase(),
    flag: overrides.flag ?? `--${overrides.key.replace(/_/g, "-")}`,
    label: overrides.label ?? overrides.key,
    section: overrides.section ?? "Gate Stack Options",
    type: overrides.type ?? "int",
    default: "default" in overrides ? overrides.default ?? null : 0,
    nullable: overrides.nullable ?? false,
    choices: overrides.choices ?? [],
    locked: overrides.locked ?? false,
    lockedValue: overrides.lockedValue,
    lockedReason: overrides.lockedReason,
  };
}

function stackFieldFromCanonical(
  canonicalFields: ConfigField[],
  overrides: Partial<ConfigField> & Pick<ConfigField, "key">,
): ConfigField {
  const suffix = overrides.key.split("_stack_", 2)[1];
  const canonical = canonicalFields.find((item) => item.key === `stack_${suffix}`);
  if (!canonical) {
    throw new Error(`Missing canonical stack fixture for ${overrides.key}`);
  }

  return field({
    ...overrides,
    type: canonical.type,
    choices: canonical.choices,
  });
}

const adaptiveOptionFields = [
  field({
    key: "weight_option_flag",
    type: "bool",
    default: false,
    choices: [true, false],
    section: "Weight Generator Options",
  }),
  field({
    key: "weight_option",
    type: "class",
    default: null,
    nullable: true,
    choices: ["SingleModelDynamicWeightConfig", "DualModelDynamicWeightConfig"],
    section: "Weight Generator Options",
  }),
  field({
    key: "bias_option_flag",
    type: "bool",
    default: false,
    choices: [true, false],
    section: "Bias Generator Options",
  }),
  field({
    key: "bias_option",
    type: "class",
    default: null,
    nullable: true,
    choices: ["AffineTransformDynamicBiasConfig", "AdditiveDynamicBiasConfig"],
    section: "Bias Generator Options",
  }),
  field({
    key: "diagonal_option_flag",
    type: "bool",
    default: false,
    choices: [true, false],
    section: "Diagonal Generator Options",
  }),
  field({
    key: "diagonal_option",
    type: "class",
    default: null,
    nullable: true,
    choices: ["StandardDynamicDiagonalConfig", "AntiDynamicDiagonalConfig"],
    section: "Diagonal Generator Options",
  }),
  field({
    key: "mask_option_flag",
    type: "bool",
    default: false,
    choices: [true, false],
    section: "Mask Options",
  }),
  field({
    key: "row_mask_option",
    type: "class",
    default: null,
    nullable: true,
    choices: ["DiagonalAxisMaskConfig", "OuterProductMaskConfig"],
    section: "Mask Options",
  }),
];

describe("config field value normalization", () => {
  it("matches int defaults against equivalent string input", () => {
    const intField = field({
      key: "stack_hidden_dim",
      type: "int",
      default: 256,
    });

    expect(normalizeConfigFieldValue(intField, "256")).toBe("256");
    expect(isDefaultConfigFieldValue(intField, "256")).toBe(true);
  });

  it("matches float defaults against equivalent string input", () => {
    const floatField = field({
      key: "dropout",
      type: "float",
      default: 0.2,
    });

    expect(normalizeConfigFieldValue(floatField, "0.2")).toBe("0.2");
    expect(isDefaultConfigFieldValue(floatField, "0.2")).toBe(true);
  });

  it("preserves invalid numeric text as a raw override value", () => {
    const intField = field({
      key: "stack_hidden_dim",
      type: "int",
      default: 256,
    });
    const floatField = field({
      key: "dropout",
      type: "float",
      default: 0.2,
    });

    expect(normalizeConfigFieldValue(intField, "abc")).toBe("abc");
    expect(isDefaultConfigFieldValue(intField, "abc")).toBe(false);
    expect(overrideValueForConfigField(floatField, "abc")).toBe("abc");
  });

  it("matches bool defaults against equivalent string input", () => {
    const boolField = field({
      key: "gate_flag",
      type: "bool",
      default: false,
    });

    expect(normalizeConfigFieldValue(boolField, "false")).toBe("false");
    expect(isDefaultConfigFieldValue(boolField, "false")).toBe(true);
  });

  it("normalizes nullable null and empty UI values consistently", () => {
    const nullableField = field({
      key: "optional_hidden_dim",
      type: "int",
      default: null,
      nullable: true,
    });

    expect(defaultConfigFieldValue(nullableField)).toBe("null");
    expect(normalizeConfigFieldValue(nullableField, null)).toBe("null");
    expect(normalizeConfigFieldValue(nullableField, "")).toBe("null");
    expect(isDefaultConfigFieldValue(nullableField, "")).toBe(true);
    expect(overrideValueForConfigField(nullableField, "")).toBe("");
  });
});

describe("adaptive option override helpers", () => {
  it("selects the first concrete option when an adaptive flag is enabled", () => {
    const cases = [
      {
        flagKey: "weight_option_flag",
        optionKey: "weight_option",
        optionValue: "SingleModelDynamicWeightConfig",
      },
      {
        flagKey: "bias_option_flag",
        optionKey: "bias_option",
        optionValue: "AffineTransformDynamicBiasConfig",
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
      const normalized = normalizeAdaptiveOptionOverrides(adaptiveOptionFields, {
        [flagKey]: "true",
      });

      expect(normalized).toMatchObject({
        [flagKey]: "true",
        [optionKey]: optionValue,
      });
    }
  });

  it("clears only the paired adaptive option when a flag is disabled", () => {
    const normalized = normalizeAdaptiveOptionOverrides(adaptiveOptionFields, {
      weight_option_flag: "false",
      weight_option: "DualModelDynamicWeightConfig",
      stack_hidden_dim: "128",
    });

    expect(normalized).toEqual({
      weight_option_flag: "false",
      stack_hidden_dim: "128",
    });
  });

  it("clears paired adaptive options when the flag is at its inactive default", () => {
    const normalized = normalizeAdaptiveOptionOverrides(adaptiveOptionFields, {
      row_mask_option: "OuterProductMaskConfig",
      stack_hidden_dim: "128",
    });

    expect(normalized).toEqual({ stack_hidden_dim: "128" });
  });

  it("omits the nullable None option only while the adaptive flag is enabled", () => {
    const optionField = adaptiveOptionFields.find(
      (item) => item.key === "weight_option",
    );
    if (!optionField) {
      throw new Error("Missing weight option field fixture");
    }

    expect(configFieldSelectOptions(optionField, {}).at(0)).toEqual({
      value: "",
      label: "None",
    });
    expect(
      configFieldSelectOptions(optionField, { weight_option_flag: "true" }).map(
        (option) => option.label,
      ),
    ).toEqual([
      "SingleModelDynamicWeightConfig",
      "DualModelDynamicWeightConfig",
    ]);
  });
});

describe("preset override helpers", () => {
  it("filters locked fields from effective preset overrides without mutating the draft", () => {
    const unlockedField = field({ key: "stack_hidden_dim" });
    const lockedField = field({
      key: "layer_norm",
      locked: true,
      lockedValue: true,
    });
    const overrides = {
      stack_hidden_dim: "128",
      layer_norm: "false",
    };

    expect(lockedOverrideKeys([unlockedField, lockedField], overrides)).toEqual([
      "layer_norm",
    ]);
    expect(effectivePresetOverrides([unlockedField, lockedField], overrides)).toEqual({
      stack_hidden_dim: "128",
    });
    expect(overrides).toEqual({
      stack_hidden_dim: "128",
      layer_norm: "false",
    });
  });

  it("builds a stable digest independent of object insertion order", () => {
    expect(overrideDigest({ b: "2", a: "1" })).toBe(
      overrideDigest({ a: "1", b: "2" }),
    );
  });
});

describe("config section controls", () => {
  it("keeps exact comment-section ancestors visible for nested search matches", () => {
    const sections: ConfigSection[] = [
      {
        title: "Recurrent Layer Options",
        fields: [
          field({
            key: "recurrent_flag",
            type: "bool",
            default: false,
            section: "Recurrent Layer Options",
          }),
        ],
      },
      {
        title: "Recurrent Gate Options",
        fields: [
          field({
            key: "recurrent_gate_flag",
            type: "bool",
            default: false,
            section: "Recurrent Gate Options",
          }),
        ],
      },
      {
        title: "Recurrent Gate Stack Options",
        fields: [
          field({
            key: "recurrent_gate_stack_hidden_dim",
            label: "recurrent gate stack hidden dim",
            section: "Recurrent Gate Stack Options",
          }),
        ],
      },
    ];

    const filtered = filterConfigSectionsForSearch(sections, {
      query: "recurrent gate stack hidden",
    });
    const [recurrentSection] = deriveNestedConfigSections(filtered, sections);

    expect(filtered.map((section) => section.title)).toEqual([
      "Recurrent Layer Options",
      "Recurrent Gate Options",
      "Recurrent Gate Stack Options",
    ]);
    expect(recurrentSection.title).toBe("Recurrent Layer Options");
    expect(recurrentSection.fields.map((item) => item.key)).toEqual([
      "recurrent_flag",
      "recurrent_gate_flag",
      "recurrent_gate_stack_hidden_dim",
    ]);
    expect(recurrentSection.children?.[0]?.title).toBe("Recurrent Gate Options");
    expect(recurrentSection.children?.[0]?.children?.[0]?.title).toBe(
      "Recurrent Gate Stack Options",
    );
  });

  it("uses independent stack flags to enable controller stack fields", () => {
    const sections: ConfigSection[] = [
      {
        title: "Gate Options",
        fields: [
          field({
            key: "gate_flag",
            type: "bool",
            default: true,
            section: "Gate Options",
          }),
        ],
      },
      {
        title: "Gate Stack Options",
        fields: [
          field({
            key: "gate_stack_independent_flag",
            type: "bool",
            default: false,
            section: "Gate Stack Options",
          }),
          field({
            key: "gate_stack_hidden_dim",
            default: null,
            nullable: true,
            section: "Gate Stack Options",
          }),
        ],
      },
    ];

    const [gateSection] = deriveNestedConfigSections(sections);
    const gateStackSection = gateSection.children?.find(
      (section) => section.title === "Gate Stack Options",
    );

    expect(gateSection.children?.map((section) => section.title)).toEqual([
      "Gate Stack Options",
    ]);
    expect(gateStackSection?.controlFieldKey).toBe("gate_stack_independent_flag");

    const disabledByDefault = disabledConfigFieldReasons(sections, {});
    expect(disabledByDefault.has("gate_stack_independent_flag")).toBe(false);
    expect(disabledByDefault.get("gate_stack_hidden_dim")).toContain(
      "gate_stack_independent_flag",
    );

    const enabled = disabledConfigFieldReasons(sections, {
      gate_stack_independent_flag: "true",
    });
    expect(enabled.has("gate_stack_hidden_dim")).toBe(false);
  });

  it("preserves canonical boolean metadata on derived stack child fields", () => {
    const canonicalStackFields = [
      field({
        key: "stack_bias_flag",
        section: "Layer Stack Options",
        type: "bool",
        default: true,
        choices: [true, false],
      }),
      field({
        key: "stack_apply_output_pipeline_flag",
        section: "Layer Stack Options",
        type: "bool",
        default: false,
        choices: [true, false],
      }),
    ];
    const sections: ConfigSection[] = [
      {
        title: "Gate Options",
        fields: [
          field({
            key: "gate_flag",
            type: "bool",
            default: true,
            section: "Gate Options",
          }),
        ],
      },
      {
        title: "Gate Stack Options",
        fields: [
          field({
            key: "gate_stack_independent_flag",
            type: "bool",
            default: false,
            section: "Gate Stack Options",
          }),
          stackFieldFromCanonical(canonicalStackFields, {
            key: "gate_stack_bias_flag",
            default: true,
            nullable: true,
            section: "Gate Stack Options",
          }),
          stackFieldFromCanonical(canonicalStackFields, {
            key: "gate_stack_apply_output_pipeline_flag",
            default: null,
            nullable: true,
            section: "Gate Stack Options",
          }),
        ],
      },
    ];

    const [gateSection] = deriveNestedConfigSections(sections);
    const gateStackSection = gateSection.children?.find(
      (section) => section.title === "Gate Stack Options",
    );
    const gateBias = gateStackSection?.fields.find(
      (item) => item.key === "gate_stack_bias_flag",
    );
    const gatePipeline = gateStackSection?.fields.find(
      (item) => item.key === "gate_stack_apply_output_pipeline_flag",
    );

    expect(gateBias?.type).toBe("bool");
    expect(gateBias?.choices).toEqual([true, false]);
    expect(gateBias?.default).toBe(true);
    expect(gateBias?.nullable).toBe(true);
    expect(gatePipeline?.type).toBe("bool");
    expect(gatePipeline?.choices).toEqual([true, false]);
    expect(gatePipeline?.default).toBeNull();
    expect(gatePipeline?.nullable).toBe(true);
  });

  it("derives stack child sections from config field prefixes", () => {
    const sections: ConfigSection[] = [
      {
        title: "Experimental Options",
        fields: [
          field({
            key: "custom_rate",
            type: "float",
            default: 0.1,
            section: "Experimental Options",
          }),
          field({
            key: "custom_stack_independent_flag",
            type: "bool",
            default: false,
            section: "Experimental Options",
          }),
          field({
            key: "custom_stack_hidden_dim",
            default: null,
            nullable: true,
            section: "Experimental Options",
          }),
          field({
            key: "custom_stack_num_layers",
            default: null,
            nullable: true,
            section: "Experimental Options",
          }),
        ],
      },
    ];

    const [section] = deriveNestedConfigSections(sections);
    const stackSection = section.children?.find(
      (child) => child.title === "Custom Stack Options",
    );

    expect(stackSection?.controlFieldKey).toBe("custom_stack_independent_flag");
    expect(stackSection?.fields.map((item) => item.key)).toEqual([
      "custom_stack_independent_flag",
      "custom_stack_hidden_dim",
      "custom_stack_num_layers",
    ]);

    const disabledByDefault = disabledConfigFieldReasons(sections, {});
    expect(disabledByDefault.has("custom_stack_independent_flag")).toBe(false);
    expect(disabledByDefault.get("custom_stack_hidden_dim")).toContain(
      "custom_stack_independent_flag",
    );
    expect(disabledByDefault.has("custom_rate")).toBe(false);
  });

  it("keeps layer stack fields on their owning section", () => {
    const sections: ConfigSection[] = [
      {
        title: "Layer Stack Options",
        fields: [
          field({
            key: "stack_hidden_dim",
            section: "Layer Stack Options",
          }),
          field({
            key: "stack_num_layers",
            section: "Layer Stack Options",
          }),
          field({
            key: "stack_activation",
            type: "enum",
            default: "GELU",
            choices: ["GELU", "MISH"],
            section: "Layer Stack Options",
          }),
        ],
      },
    ];

    const [section] = deriveNestedConfigSections(sections);

    expect(section.title).toBe("Layer Stack Options");
    expect(section.fields.map((item) => item.key)).toEqual([
      "stack_hidden_dim",
      "stack_num_layers",
      "stack_activation",
    ]);
    expect(section.children?.some((child) => child.title === "Layer Stack"))
      .not.toBe(true);
  });

  it("keeps layer stack submodule fields on their owning section", () => {
    const sections: ConfigSection[] = [
      {
        title: "Layer Stack Submodule Options",
        fields: [
          field({
            key: "submodule_stack_hidden_dim",
            section: "Layer Stack Submodule Options",
          }),
          field({
            key: "submodule_stack_activation",
            type: "enum",
            default: "GELU",
            choices: ["GELU", "MISH"],
            section: "Layer Stack Submodule Options",
          }),
        ],
      },
    ];

    const [section] = deriveNestedConfigSections(sections);

    expect(section.title).toBe("Layer Stack Submodule Options");
    expect(section.fields.map((item) => item.key)).toEqual([
      "submodule_stack_hidden_dim",
      "submodule_stack_activation",
    ]);
    expect(
      section.children?.some((child) => child.title === "Submodule Stack Options"),
    ).not.toBe(true);
  });

  it("keeps adaptive submodule stack fields on their comment-titled section", () => {
    const sections: ConfigSection[] = [
      {
        title: "Adaptive Submodule Stack Options",
        fields: [
          field({
            key: "adaptive_submodule_stack_hidden_dim",
            section: "Adaptive Submodule Stack Options",
          }),
          field({
            key: "adaptive_submodule_stack_num_layers",
            section: "Adaptive Submodule Stack Options",
          }),
        ],
      },
    ];

    const [section] = deriveNestedConfigSections(sections);

    expect(section.title).toBe("Adaptive Submodule Stack Options");
    expect(section.children).toBeUndefined();
    expect(section.fields.map((item) => item.key)).toEqual([
      "adaptive_submodule_stack_hidden_dim",
      "adaptive_submodule_stack_num_layers",
    ]);
    expect(disabledConfigFieldReasons(sections, {}).size).toBe(0);
  });

  it("uses adaptive option flags and generator stack flags for component sections", () => {
    const sections: ConfigSection[] = [
      {
        title: "Weight Generator Options",
        fields: [
          field({
            key: "weight_option_flag",
            type: "bool",
            default: false,
            section: "Weight Generator Options",
          }),
          field({
            key: "weight_option",
            type: "class",
            default: null,
            nullable: true,
            section: "Weight Generator Options",
          }),
        ],
      },
      {
        title: "Weight Generator Stack Options",
        fields: [
          field({
            key: "weight_generator_stack_independent_flag",
            type: "bool",
            default: false,
            section: "Weight Generator Stack Options",
          }),
          field({
            key: "weight_generator_stack_hidden_dim",
            default: null,
            nullable: true,
            section: "Weight Generator Stack Options",
          }),
        ],
      },
      {
        title: "Mask Options",
        fields: [
          field({
            key: "mask_option_flag",
            type: "bool",
            default: false,
            section: "Mask Options",
          }),
          field({
            key: "row_mask_option",
            type: "class",
            default: null,
            nullable: true,
            section: "Mask Options",
          }),
        ],
      },
      {
        title: "Mask Stack Options",
        fields: [
          field({
            key: "mask_generator_stack_independent_flag",
            type: "bool",
            default: false,
            section: "Mask Stack Options",
          }),
          field({
            key: "mask_generator_stack_hidden_dim",
            default: null,
            nullable: true,
            section: "Mask Stack Options",
          }),
        ],
      },
    ];

    const [weightSection, maskSection] = deriveNestedConfigSections(sections);
    const weightGeneratorSection = weightSection.children?.find(
      (section) => section.title === "Weight Generator Stack Options",
    );
    const maskGeneratorSection = maskSection.children?.find(
      (section) => section.title === "Mask Stack Options",
    );

    expect(weightSection.controlFieldKey).toBe("weight_option_flag");
    expect(weightGeneratorSection?.controlFieldKey).toBe(
      "weight_generator_stack_independent_flag",
    );
    expect(maskSection.controlFieldKey).toBe("mask_option_flag");
    expect(maskGeneratorSection?.controlFieldKey).toBe(
      "mask_generator_stack_independent_flag",
    );

    const disabledByDefault = disabledConfigFieldReasons(sections, {});
    expect(disabledByDefault.has("weight_option_flag")).toBe(false);
    expect(disabledByDefault.get("weight_option")).toContain("weight_option_flag");
    expect(
      disabledByDefault.get("weight_generator_stack_independent_flag"),
    ).toContain("weight_option_flag");
    expect(disabledByDefault.get("row_mask_option")).toContain(
      "mask_option_flag",
    );

    const componentEnabled = disabledConfigFieldReasons(sections, {
      weight_option_flag: "true",
      mask_option_flag: "true",
    });
    expect(componentEnabled.has("weight_option")).toBe(false);
    expect(componentEnabled.has("weight_generator_stack_independent_flag")).toBe(
      false,
    );
    expect(componentEnabled.get("weight_generator_stack_hidden_dim")).toContain(
      "weight_generator_stack_independent_flag",
    );
    expect(componentEnabled.get("mask_generator_stack_hidden_dim")).toContain(
      "mask_generator_stack_independent_flag",
    );

    const stackEnabled = disabledConfigFieldReasons(sections, {
      weight_option_flag: "true",
      weight_generator_stack_independent_flag: "true",
      mask_option_flag: "true",
      mask_generator_stack_independent_flag: "true",
    });
    expect(stackEnabled.has("weight_generator_stack_hidden_dim")).toBe(false);
    expect(stackEnabled.has("mask_generator_stack_hidden_dim")).toBe(false);
  });

  it("uses divider groups for boundary projector sections", () => {
    const sections: ConfigSection[] = [
      {
        title: "Input Boundary Projector Options",
        fields: [
          field({
            key: "input_layer_weight_option",
            type: "class",
            default: null,
            nullable: true,
            section: "Input Boundary Projector Options",
          }),
          field({
            key: "input_layer_bias_option",
            type: "class",
            default: null,
            nullable: true,
            section: "Input Boundary Projector Options",
          }),
          field({
            key: "input_layer_diagonal_option",
            type: "class",
            default: null,
            nullable: true,
            section: "Input Boundary Projector Options",
          }),
          field({
            key: "input_layer_row_mask_option",
            type: "class",
            default: null,
            nullable: true,
            section: "Input Boundary Projector Options",
          }),
        ],
      },
      {
        title: "Output Boundary Projector Options",
        fields: [
          field({
            key: "output_layer_weight_option",
            type: "class",
            default: null,
            nullable: true,
            section: "Output Boundary Projector Options",
          }),
        ],
      },
    ];

    const [inputSection, outputSection] = deriveNestedConfigSections(sections);
    const groups = boundaryProjectorFieldGroups(
      inputSection.title,
      inputSection.fields,
    );

    expect(inputSection.controlFieldKey).toBeUndefined();
    expect(outputSection.controlFieldKey).toBeUndefined();
    expect(inputSection.children ?? []).toEqual([]);
    expect(outputSection.children ?? []).toEqual([]);
    expect(groups?.map((group) => group.title)).toEqual([
      "Weight",
      "Bias",
      "Diagonal",
      "Mask",
    ]);
    expect(groups?.map((group) => group.fields.map((item) => item.key))).toEqual([
      ["input_layer_weight_option"],
      ["input_layer_bias_option"],
      ["input_layer_diagonal_option"],
      ["input_layer_row_mask_option"],
    ]);

    const disabledByDefault = disabledConfigFieldReasons(sections, {});
    expect(disabledByDefault.has("input_layer_weight_option")).toBe(false);
    expect(disabledByDefault.has("output_layer_weight_option")).toBe(false);
  });
});
