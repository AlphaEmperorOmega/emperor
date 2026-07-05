import { describe, expect, it } from "vitest";

import {
  boundaryProjectorFieldGroups,
  configFieldSelectOptions,
  defaultConfigFieldValue,
  deriveNestedConfigSections,
  disabledConfigFieldReasons,
  effectivePresetOverrides,
  filterConfigSectionsForSearch,
  inheritedStackSectionHint,
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

function inheritedStackHintSection(
  title: string,
  controlFieldKey: string,
  controlFieldLabel: string,
): ConfigSection {
  return {
    title,
    controlFieldKey,
    fields: [
      field({
        key: controlFieldKey,
        label: controlFieldLabel,
        type: "bool",
        default: false,
        choices: [true, false],
        section: title,
      }),
    ],
  };
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

  it("selects and clears router adaptive options with their own flags", () => {
    const routerFields = [
      field({
        key: "router_weight_option_flag",
        type: "bool",
        default: false,
        choices: [true, false],
        section: "Router Weight Generator Options",
      }),
      field({
        key: "router_weight_option",
        type: "class",
        default: null,
        nullable: true,
        choices: [
          "DualModelDynamicWeightConfig",
          "LayeredWeightedBankDynamicWeightConfig",
        ],
        section: "Router Weight Generator Options",
      }),
    ];

    expect(
      normalizeAdaptiveOptionOverrides(routerFields, {
        router_weight_option_flag: "true",
      }),
    ).toMatchObject({
      router_weight_option_flag: "true",
      router_weight_option: "DualModelDynamicWeightConfig",
    });

    expect(
      normalizeAdaptiveOptionOverrides(routerFields, {
        router_weight_option_flag: "false",
        router_weight_option: "LayeredWeightedBankDynamicWeightConfig",
      }),
    ).toEqual({ router_weight_option_flag: "false" });
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
  it("reports router stack inheritance while its independent flag is off", () => {
    const section = inheritedStackHintSection(
      "Router Stack Options",
      "sampler_stack_independent_flag",
      "sampler stack independent flag",
    );

    expect(inheritedStackSectionHint(section, {})).toEqual({
      label: "Inherits Submodule Stack",
      title:
        "Uses Layer Stack Submodule Options while sampler stack independent flag is off. Enable it to use Router Stack Options values.",
      sourceTitle: "Layer Stack Submodule Options",
      isCustom: false,
    });
  });

  it("reports router stack customization while its independent flag is on", () => {
    const section = inheritedStackHintSection(
      "Router Stack Options",
      "sampler_stack_independent_flag",
      "sampler stack independent flag",
    );

    expect(
      inheritedStackSectionHint(section, {
        sampler_stack_independent_flag: "true",
      }),
    ).toEqual({
      label: "Custom Stack",
      title:
        "Uses Router Stack Options values while sampler stack independent flag is on. Disable it to inherit Layer Stack Submodule Options.",
      sourceTitle: "Layer Stack Submodule Options",
      isCustom: true,
    });
  });

  it("reports immediate inherited sources for recurrent stack sections", () => {
    const recurrentGateSection = inheritedStackHintSection(
      "Recurrent Gate Stack Options",
      "recurrent_gate_stack_independent_flag",
      "recurrent gate stack independent flag",
    );
    const recurrentHaltingSection = inheritedStackHintSection(
      "Recurrent Halting Stack Options",
      "recurrent_halting_stack_independent_flag",
      "recurrent halting stack independent flag",
    );

    expect(inheritedStackSectionHint(recurrentGateSection, {})).toMatchObject({
      label: "Inherits Gate Stack",
      sourceTitle: "Gate Stack Options",
      isCustom: false,
    });
    expect(inheritedStackSectionHint(recurrentHaltingSection, {})).toMatchObject({
      label: "Inherits Halting Stack",
      sourceTitle: "Halting Stack Options",
      isCustom: false,
    });
  });

  it("reports expert controller stack inheritance from expert stack options", () => {
    const expertGateSection = inheritedStackHintSection(
      "Expert Gate Stack Options",
      "expert_gate_stack_independent_flag",
      "expert gate stack independent flag",
    );
    const expertRecurrentGateSection = inheritedStackHintSection(
      "Expert Recurrent Gate Stack Options",
      "expert_recurrent_gate_stack_independent_flag",
      "expert recurrent gate stack independent flag",
    );

    expect(inheritedStackSectionHint(expertGateSection, {})).toMatchObject({
      label: "Inherits Expert Stack",
      sourceTitle: "Expert Stack Options",
      isCustom: false,
    });
    expect(
      inheritedStackSectionHint(expertRecurrentGateSection, {
        expert_recurrent_gate_stack_independent_flag: "true",
      }),
    ).toMatchObject({
      label: "Custom Stack",
      sourceTitle: "Expert Stack Options",
      isCustom: true,
    });
  });

  it("does not report inheritance hints for non-inherited sections", () => {
    const section: ConfigSection = {
      title: "Layer Stack Options",
      fields: [
        field({
          key: "stack_hidden_dim",
          section: "Layer Stack Options",
        }),
      ],
    };

    expect(inheritedStackSectionHint(section, {})).toBeUndefined();
  });

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

  it("nests experts sampler router sections under the sampler model section", () => {
    const sections: ConfigSection[] = [
      {
        title: "Sampler Model Options",
        fields: [
          field({
            key: "sampler_threshold",
            type: "float",
            default: 0,
            section: "Sampler Model Options",
          }),
          field({
            key: "sampler_switch_loss_weight",
            type: "float",
            default: 0,
            section: "Sampler Model Options",
          }),
        ],
      },
      {
        title: "Router Options",
        fields: [
          field({
            key: "router_noisy_topk_flag",
            type: "bool",
            default: false,
            choices: [true, false],
            section: "Router Options",
          }),
        ],
      },
      {
        title: "Router Stack Options",
        fields: [
          field({
            key: "sampler_stack_independent_flag",
            type: "bool",
            default: false,
            choices: [true, false],
            section: "Router Stack Options",
          }),
          field({
            key: "sampler_stack_hidden_dim",
            default: null,
            nullable: true,
            section: "Router Stack Options",
          }),
          field({
            key: "sampler_stack_num_layers",
            default: null,
            nullable: true,
            section: "Router Stack Options",
          }),
          field({
            key: "sampler_bias_flag",
            type: "bool",
            default: true,
            choices: [true, false],
            section: "Router Stack Options",
          }),
        ],
      },
    ];

    const [samplerSection] = deriveNestedConfigSections(sections);
    const routerSection = samplerSection.children?.[0];
    const routerStackSection = routerSection?.children?.[0];

    expect(samplerSection.title).toBe("Sampler Model Options");
    expect(samplerSection.fields.map((item) => item.key)).toEqual([
      "sampler_threshold",
      "sampler_switch_loss_weight",
    ]);
    expect(samplerSection.children?.map((section) => section.title)).toEqual([
      "Router Options",
    ]);
    expect(routerSection?.fields.map((item) => item.key)).toEqual([
      "router_noisy_topk_flag",
    ]);
    expect(routerSection?.children?.map((section) => section.title)).toEqual([
      "Router Stack Options",
    ]);
    expect(routerStackSection?.controlFieldKey).toBe(
      "sampler_stack_independent_flag",
    );
    expect(routerStackSection?.fields.map((item) => item.key)).toEqual([
      "sampler_stack_independent_flag",
      "sampler_stack_hidden_dim",
      "sampler_stack_num_layers",
      "sampler_bias_flag",
    ]);
    expect(routerStackSection?.children).toBeUndefined();

    const disabledByDefault = disabledConfigFieldReasons(sections, {});
    expect(disabledByDefault.has("sampler_stack_independent_flag")).toBe(false);
    expect(disabledByDefault.get("sampler_stack_hidden_dim")).toContain(
      "sampler_stack_independent_flag",
    );

    const enabled = disabledConfigFieldReasons(sections, {
      sampler_stack_independent_flag: "true",
    });
    expect(enabled.has("sampler_stack_hidden_dim")).toBe(false);
  });

  it("keeps experts sampler and router ancestors visible for router stack search matches", () => {
    const sections: ConfigSection[] = [
      {
        title: "Sampler Model Options",
        fields: [
          field({
            key: "sampler_threshold",
            type: "float",
            default: 0,
            section: "Sampler Model Options",
          }),
        ],
      },
      {
        title: "Router Options",
        fields: [
          field({
            key: "router_noisy_topk_flag",
            type: "bool",
            default: false,
            choices: [true, false],
            section: "Router Options",
          }),
        ],
      },
      {
        title: "Router Stack Options",
        fields: [
          field({
            key: "sampler_stack_independent_flag",
            type: "bool",
            default: false,
            choices: [true, false],
            section: "Router Stack Options",
          }),
          field({
            key: "sampler_stack_hidden_dim",
            label: "sampler stack hidden dim",
            default: null,
            nullable: true,
            section: "Router Stack Options",
          }),
        ],
      },
    ];

    const filtered = filterConfigSectionsForSearch(sections, {
      query: "sampler stack hidden dim",
    });
    const [samplerSection] = deriveNestedConfigSections(filtered, sections);

    expect(filtered.map((section) => section.title)).toEqual([
      "Sampler Model Options",
      "Router Options",
      "Router Stack Options",
    ]);
    expect(samplerSection.title).toBe("Sampler Model Options");
    expect(samplerSection.children?.[0]?.title).toBe("Router Options");
    expect(samplerSection.children?.[0]?.children?.[0]?.title).toBe(
      "Router Stack Options",
    );
  });

  it("nests expert stack options under mixture of experts model options", () => {
    const mixtureFields = [
      field({
        key: "expert_top_k",
        type: "int",
        default: 2,
        section: "Mixture Of Experts Model Options",
      }),
      field({
        key: "expert_num_experts",
        type: "int",
        default: 4,
        section: "Mixture Of Experts Model Options",
      }),
      field({
        key: "expert_compute_expert_mixture_flag",
        type: "bool",
        default: true,
        choices: [true, false],
        section: "Mixture Of Experts Model Options",
      }),
      field({
        key: "expert_weighted_parameters_flag",
        type: "bool",
        default: true,
        choices: [true, false],
        section: "Mixture Of Experts Model Options",
      }),
    ];
    const expertStackFields = [
      field({
        key: "expert_stack_hidden_dim",
        type: "int",
        default: 256,
        section: "Expert Stack Options",
      }),
      field({
        key: "expert_stack_num_layers",
        type: "int",
        default: 2,
        section: "Expert Stack Options",
      }),
      field({
        key: "expert_stack_activation",
        type: "enum",
        default: "GELU",
        choices: ["GELU", "TANH"],
        section: "Expert Stack Options",
      }),
      field({
        key: "expert_stack_residual_connection_option",
        type: "enum",
        default: "DISABLED",
        choices: ["DISABLED", "RESIDUAL"],
        section: "Expert Stack Options",
      }),
      field({
        key: "expert_stack_dropout_probability",
        type: "float",
        default: 0,
        section: "Expert Stack Options",
      }),
      field({
        key: "expert_stack_layer_norm_position",
        type: "enum",
        default: "DISABLED",
        choices: ["DISABLED", "BEFORE", "AFTER"],
        section: "Expert Stack Options",
      }),
      field({
        key: "expert_stack_last_layer_bias_option",
        type: "enum",
        default: "DEFAULT",
        choices: ["DEFAULT", "DISABLED"],
        section: "Expert Stack Options",
      }),
      field({
        key: "expert_stack_apply_output_pipeline_flag",
        type: "bool",
        default: true,
        choices: [true, false],
        section: "Expert Stack Options",
      }),
      field({
        key: "expert_bias_flag",
        type: "bool",
        default: true,
        choices: [true, false],
        section: "Expert Stack Options",
      }),
    ];
    const sections: ConfigSection[] = [
      {
        title: "Mixture Of Experts Model Options",
        fields: mixtureFields,
      },
      {
        title: "Expert Stack Options",
        fields: expertStackFields,
      },
    ];

    const [mixtureSection] = deriveNestedConfigSections(sections);
    const expertStackSection = mixtureSection.children?.[0];

    expect(mixtureSection.title).toBe("Mixture Of Experts Model Options");
    expect(mixtureSection.fields.map((item) => item.key)).toEqual(
      mixtureFields.map((item) => item.key),
    );
    expect(mixtureSection.children?.map((section) => section.title)).toEqual([
      "Expert Stack Options",
    ]);
    expect(expertStackSection?.controlFieldKey).toBeUndefined();
    expect(expertStackSection?.fields.map((item) => item.key)).toEqual(
      expertStackFields.map((item) => item.key),
    );
    expect(expertStackSection?.children).toBeUndefined();
    expect(disabledConfigFieldReasons(sections, {}).size).toBe(0);
  });

  it("nests expert-internal controller sections under expert stack options", () => {
    const sections: ConfigSection[] = [
      {
        title: "Mixture Of Experts Model Options",
        fields: [
          field({
            key: "expert_top_k",
            type: "int",
            default: 2,
            section: "Mixture Of Experts Model Options",
          }),
        ],
      },
      {
        title: "Expert Stack Options",
        fields: [
          field({
            key: "expert_stack_hidden_dim",
            type: "int",
            default: 32,
            section: "Expert Stack Options",
          }),
        ],
      },
      {
        title: "Expert Gate Options",
        fields: [
          field({
            key: "expert_gate_flag",
            label: "expert gate flag",
            type: "bool",
            default: false,
            choices: [true, false],
            section: "Expert Gate Options",
          }),
        ],
      },
      {
        title: "Expert Gate Stack Options",
        fields: [
          field({
            key: "expert_gate_stack_independent_flag",
            label: "expert gate stack independent flag",
            type: "bool",
            default: false,
            choices: [true, false],
            section: "Expert Gate Stack Options",
          }),
          field({
            key: "expert_gate_stack_hidden_dim",
            default: null,
            nullable: true,
            section: "Expert Gate Stack Options",
          }),
        ],
      },
      {
        title: "Expert Memory Options",
        fields: [
          field({
            key: "expert_memory_flag",
            type: "bool",
            default: false,
            choices: [true, false],
            section: "Expert Memory Options",
          }),
        ],
      },
      {
        title: "Expert Memory Stack Options",
        fields: [
          field({
            key: "expert_memory_stack_independent_flag",
            type: "bool",
            default: false,
            choices: [true, false],
            section: "Expert Memory Stack Options",
          }),
          field({
            key: "expert_memory_stack_hidden_dim",
            default: null,
            nullable: true,
            section: "Expert Memory Stack Options",
          }),
        ],
      },
      {
        title: "Expert Recurrent Layer Options",
        fields: [
          field({
            key: "expert_recurrent_flag",
            type: "bool",
            default: false,
            choices: [true, false],
            section: "Expert Recurrent Layer Options",
          }),
        ],
      },
      {
        title: "Expert Recurrent Gate Options",
        fields: [
          field({
            key: "expert_recurrent_gate_flag",
            type: "bool",
            default: false,
            choices: [true, false],
            section: "Expert Recurrent Gate Options",
          }),
        ],
      },
      {
        title: "Expert Recurrent Gate Stack Options",
        fields: [
          field({
            key: "expert_recurrent_gate_stack_independent_flag",
            type: "bool",
            default: false,
            choices: [true, false],
            section: "Expert Recurrent Gate Stack Options",
          }),
          field({
            key: "expert_recurrent_gate_stack_hidden_dim",
            default: null,
            nullable: true,
            section: "Expert Recurrent Gate Stack Options",
          }),
        ],
      },
    ];

    const [mixtureSection] = deriveNestedConfigSections(sections);
    const expertStackSection = mixtureSection.children?.[0];
    const expertGateSection = expertStackSection?.children?.find(
      (section) => section.title === "Expert Gate Options",
    );
    const expertMemorySection = expertStackSection?.children?.find(
      (section) => section.title === "Expert Memory Options",
    );
    const expertRecurrentSection = expertStackSection?.children?.find(
      (section) => section.title === "Expert Recurrent Layer Options",
    );

    expect(mixtureSection.title).toBe("Mixture Of Experts Model Options");
    expect(expertStackSection?.children?.map((section) => section.title)).toEqual([
      "Expert Gate Options",
      "Expert Memory Options",
      "Expert Recurrent Layer Options",
    ]);
    expect(expertGateSection?.children?.[0]?.title).toBe(
      "Expert Gate Stack Options",
    );
    expect(expertMemorySection?.children?.[0]?.title).toBe(
      "Expert Memory Stack Options",
    );
    expect(expertRecurrentSection?.children?.[0]?.title).toBe(
      "Expert Recurrent Gate Options",
    );
    expect(
      expertRecurrentSection?.children?.[0]?.children?.[0]?.title,
    ).toBe("Expert Recurrent Gate Stack Options");

    const disabledByDefault = disabledConfigFieldReasons(sections, {});
    expect(disabledByDefault.get("expert_gate_stack_independent_flag")).toContain(
      "expert gate flag",
    );

    const gateEnabled = disabledConfigFieldReasons(sections, {
      expert_gate_flag: "true",
    });
    expect(gateEnabled.has("expert_gate_stack_independent_flag")).toBe(false);
    expect(gateEnabled.get("expert_gate_stack_hidden_dim")).toContain(
      "expert gate stack independent flag",
    );

    const customGateStack = disabledConfigFieldReasons(sections, {
      expert_gate_flag: "true",
      expert_gate_stack_independent_flag: "true",
    });
    expect(customGateStack.has("expert_gate_stack_hidden_dim")).toBe(false);
  });

  it("keeps mixture and expert stack ancestors visible for expert stack search matches", () => {
    const sections: ConfigSection[] = [
      {
        title: "Mixture Of Experts Model Options",
        fields: [
          field({
            key: "expert_top_k",
            type: "int",
            default: 2,
            section: "Mixture Of Experts Model Options",
          }),
        ],
      },
      {
        title: "Expert Stack Options",
        fields: [
          field({
            key: "expert_stack_hidden_dim",
            label: "expert stack hidden dim",
            type: "int",
            default: 256,
            section: "Expert Stack Options",
          }),
          field({
            key: "expert_bias_flag",
            label: "expert bias flag",
            type: "bool",
            default: true,
            choices: [true, false],
            section: "Expert Stack Options",
          }),
        ],
      },
    ];

    const filtered = filterConfigSectionsForSearch(sections, {
      query: "expert bias flag",
    });
    const [mixtureSection] = deriveNestedConfigSections(filtered, sections);

    expect(filtered.map((section) => section.title)).toEqual([
      "Mixture Of Experts Model Options",
      "Expert Stack Options",
    ]);
    expect(mixtureSection.title).toBe("Mixture Of Experts Model Options");
    expect(mixtureSection.fields.map((item) => item.key)).toEqual([
      "expert_bias_flag",
    ]);
    expect(mixtureSection.children?.[0]?.title).toBe("Expert Stack Options");
    expect(mixtureSection.children?.[0]?.fields.map((item) => item.key)).toEqual([
      "expert_bias_flag",
    ]);
  });

  it("keeps the expert stack path visible for expert controller search matches", () => {
    const sections: ConfigSection[] = [
      {
        title: "Mixture Of Experts Model Options",
        fields: [
          field({
            key: "expert_top_k",
            type: "int",
            default: 2,
            section: "Mixture Of Experts Model Options",
          }),
        ],
      },
      {
        title: "Expert Stack Options",
        fields: [
          field({
            key: "expert_stack_hidden_dim",
            section: "Expert Stack Options",
          }),
        ],
      },
      {
        title: "Expert Recurrent Layer Options",
        fields: [
          field({
            key: "expert_recurrent_flag",
            type: "bool",
            default: false,
            choices: [true, false],
            section: "Expert Recurrent Layer Options",
          }),
        ],
      },
      {
        title: "Expert Recurrent Gate Options",
        fields: [
          field({
            key: "expert_recurrent_gate_flag",
            type: "bool",
            default: false,
            choices: [true, false],
            section: "Expert Recurrent Gate Options",
          }),
        ],
      },
      {
        title: "Expert Recurrent Gate Stack Options",
        fields: [
          field({
            key: "expert_recurrent_gate_stack_hidden_dim",
            label: "expert recurrent gate stack hidden dim",
            default: null,
            nullable: true,
            section: "Expert Recurrent Gate Stack Options",
          }),
        ],
      },
    ];

    const filtered = filterConfigSectionsForSearch(sections, {
      query: "expert recurrent gate stack hidden dim",
    });
    const [mixtureSection] = deriveNestedConfigSections(filtered, sections);

    expect(filtered.map((section) => section.title)).toEqual([
      "Mixture Of Experts Model Options",
      "Expert Stack Options",
      "Expert Recurrent Layer Options",
      "Expert Recurrent Gate Options",
      "Expert Recurrent Gate Stack Options",
    ]);
    expect(mixtureSection.children?.[0]?.title).toBe("Expert Stack Options");
    expect(mixtureSection.children?.[0]?.children?.[0]?.title).toBe(
      "Expert Recurrent Layer Options",
    );
    expect(
      mixtureSection.children?.[0]?.children?.[0]?.children?.[0]?.title,
    ).toBe("Expert Recurrent Gate Options");
    expect(
      mixtureSection.children?.[0]?.children?.[0]?.children?.[0]?.children?.[0]
        ?.title,
    ).toBe("Expert Recurrent Gate Stack Options");
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

  it("keeps adaptive generator stack fields on their comment-titled section", () => {
    const sections: ConfigSection[] = [
      {
        title: "Adaptive Generator Stack Options",
        fields: [
          field({
            key: "adaptive_generator_stack_hidden_dim",
            section: "Adaptive Generator Stack Options",
          }),
          field({
            key: "adaptive_generator_stack_num_layers",
            section: "Adaptive Generator Stack Options",
          }),
        ],
      },
    ];

    const [section] = deriveNestedConfigSections(sections);

    expect(section.title).toBe("Adaptive Generator Stack Options");
    expect(section.children).toBeUndefined();
    expect(section.fields.map((item) => item.key)).toEqual([
      "adaptive_generator_stack_hidden_dim",
      "adaptive_generator_stack_num_layers",
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

  it("nests expert and router adaptive sections under their stack owners", () => {
    const sections: ConfigSection[] = [
      {
        title: "Mixture Of Experts Model Options",
        fields: [
          field({
            key: "expert_top_k",
            type: "int",
            default: 2,
            section: "Mixture Of Experts Model Options",
          }),
        ],
      },
      {
        title: "Expert Stack Options",
        fields: [
          field({
            key: "expert_stack_num_layers",
            type: "int",
            default: 2,
            section: "Expert Stack Options",
          }),
        ],
      },
      {
        title: "Weight Generator Options",
        fields: [
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
            choices: [true, false],
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
        title: "Sampler Model Options",
        fields: [
          field({
            key: "sampler_threshold",
            type: "float",
            default: 0,
            section: "Sampler Model Options",
          }),
        ],
      },
      {
        title: "Router Options",
        fields: [
          field({
            key: "router_noisy_topk_flag",
            type: "bool",
            default: false,
            choices: [true, false],
            section: "Router Options",
          }),
        ],
      },
      {
        title: "Router Stack Options",
        fields: [
          field({
            key: "sampler_stack_independent_flag",
            type: "bool",
            default: false,
            choices: [true, false],
            section: "Router Stack Options",
          }),
        ],
      },
      {
        title: "Router Weight Generator Options",
        fields: [
          field({
            key: "router_weight_option_flag",
            type: "bool",
            default: false,
            choices: [true, false],
            section: "Router Weight Generator Options",
          }),
          field({
            key: "router_weight_option",
            type: "class",
            default: null,
            nullable: true,
            section: "Router Weight Generator Options",
          }),
        ],
      },
      {
        title: "Router Weight Generator Stack Options",
        fields: [
          field({
            key: "router_weight_generator_stack_independent_flag",
            type: "bool",
            default: false,
            choices: [true, false],
            section: "Router Weight Generator Stack Options",
          }),
          field({
            key: "router_weight_generator_stack_hidden_dim",
            label: "router weight generator stack hidden dim",
            default: null,
            nullable: true,
            section: "Router Weight Generator Stack Options",
          }),
        ],
      },
    ];

    const [mixtureSection, samplerSection] = deriveNestedConfigSections(sections);
    const expertSection = mixtureSection.children?.find(
      (section) => section.title === "Expert Stack Options",
    );
    const expertWeightSection = expertSection?.children?.find(
      (section) => section.title === "Weight Generator Options",
    );
    const expertWeightStackSection = expertWeightSection?.children?.[0];
    const routerStackSection =
      samplerSection.children?.[0]?.children?.find(
        (section) => section.title === "Router Stack Options",
      );
    const routerWeightSection = routerStackSection?.children?.find(
      (section) => section.title === "Router Weight Generator Options",
    );
    const routerWeightStackSection = routerWeightSection?.children?.[0];

    expect(expertWeightSection?.controlFieldKey).toBe("weight_option_flag");
    expect(expertWeightStackSection?.controlFieldKey).toBe(
      "weight_generator_stack_independent_flag",
    );
    expect(routerWeightSection?.controlFieldKey).toBe(
      "router_weight_option_flag",
    );
    expect(routerWeightStackSection?.controlFieldKey).toBe(
      "router_weight_generator_stack_independent_flag",
    );
    expect(inheritedStackSectionHint(routerWeightStackSection!, {})).toMatchObject({
      sourceTitle: "Adaptive Generator Stack Options",
      isCustom: false,
    });

    const filtered = filterConfigSectionsForSearch(sections, {
      query: "router weight generator stack hidden dim",
    });
    expect(filtered.map((section) => section.title)).toEqual([
      "Sampler Model Options",
      "Router Options",
      "Router Stack Options",
      "Router Weight Generator Options",
      "Router Weight Generator Stack Options",
    ]);
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
            choices: ["DualModelDynamicWeightConfig"],
            section: "Input Boundary Projector Options",
          }),
          field({
            key: "input_layer_weight_decay_warmup_batches",
            type: "int",
            default: 0,
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
      ["input_layer_weight_option", "input_layer_weight_decay_warmup_batches"],
      ["input_layer_bias_option"],
      ["input_layer_diagonal_option"],
      ["input_layer_row_mask_option"],
    ]);
    expect(groups?.[0]?.controlField?.key).toBe("input_layer_weight_option");
    expect(groups?.[0]?.isEnabled).toBe(false);

    const disabledByDefault = disabledConfigFieldReasons(sections, {});
    expect(disabledByDefault.has("input_layer_weight_option")).toBe(false);
    expect(
      disabledByDefault.get("input_layer_weight_decay_warmup_batches"),
    ).toContain("input_layer_weight_option");
    expect(disabledByDefault.has("output_layer_weight_option")).toBe(false);

    const enabledGroups = boundaryProjectorFieldGroups(
      inputSection.title,
      inputSection.fields,
      { input_layer_weight_option: "DualModelDynamicWeightConfig" },
    );
    expect(enabledGroups?.[0]?.isEnabled).toBe(true);
    expect(
      disabledConfigFieldReasons(sections, {
        input_layer_weight_option: "DualModelDynamicWeightConfig",
      }).has("input_layer_weight_decay_warmup_batches"),
    ).toBe(false);

    const inputWeightOption = inputSection.fields.find(
      (item) => item.key === "input_layer_weight_option",
    );
    if (!inputWeightOption) {
      throw new Error("Missing input boundary weight option fixture");
    }
    expect(configFieldSelectOptions(inputWeightOption, {}).at(0)).toEqual({
      value: "",
      label: "None",
    });
    expect(
      configFieldSelectOptions(inputWeightOption, {
        input_layer_weight_option: "DualModelDynamicWeightConfig",
      }).map((option) => option.label),
    ).toEqual(["DualModelDynamicWeightConfig"]);
  });
});
