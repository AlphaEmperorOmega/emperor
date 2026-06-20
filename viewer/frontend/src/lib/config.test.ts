import { describe, expect, it } from "vitest";

import {
  boundaryProjectorFieldGroups,
  defaultConfigFieldValue,
  deriveNestedConfigSections,
  disabledConfigFieldReasons,
  effectivePresetOverrides,
  isDefaultConfigFieldValue,
  lockedOverrideKeys,
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

describe("config field value normalization", () => {
  it("matches int defaults against equivalent string input", () => {
    const intField = field({
      key: "hidden_dim",
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

describe("preset override helpers", () => {
  it("filters locked fields from effective preset overrides without mutating the draft", () => {
    const unlockedField = field({ key: "hidden_dim" });
    const lockedField = field({
      key: "layer_norm",
      locked: true,
      lockedValue: true,
    });
    const overrides = {
      hidden_dim: "128",
      layer_norm: "false",
    };

    expect(lockedOverrideKeys([unlockedField, lockedField], overrides)).toEqual([
      "layer_norm",
    ]);
    expect(effectivePresetOverrides([unlockedField, lockedField], overrides)).toEqual({
      hidden_dim: "128",
    });
    expect(overrides).toEqual({
      hidden_dim: "128",
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
  it("uses independent stack flags to enable controller stack fields", () => {
    const sections: ConfigSection[] = [
      {
        title: "Gate Stack Options",
        fields: [
          field({
            key: "gate_flag",
            type: "bool",
            default: true,
          }),
          field({
            key: "gate_stack_independent_flag",
            type: "bool",
            default: false,
          }),
          field({
            key: "gate_stack_hidden_dim",
            default: null,
            nullable: true,
          }),
        ],
      },
    ];

    const [gateSection] = deriveNestedConfigSections(sections);
    const gateStackSection = gateSection.children?.find(
      (section) => section.title === "Gate Model Stack",
    );

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

  it("derives an inner stack section from a stack configuration section", () => {
    const sections: ConfigSection[] = [
      {
        title: "Adaptive Generator Stack Options",
        fields: [
          field({
            key: "adaptive_stack_hidden_dim",
            section: "Adaptive Generator Stack Options",
          }),
          field({
            key: "adaptive_stack_num_layers",
            section: "Adaptive Generator Stack Options",
          }),
        ],
      },
    ];

    const [section] = deriveNestedConfigSections(sections);
    const stackSection = section.children?.find(
      (child) => child.title === "Adaptive Generator Stack",
    );

    expect(stackSection?.controlFieldKey).toBeUndefined();
    expect(stackSection?.fields.map((item) => item.key)).toEqual([
      "adaptive_stack_hidden_dim",
      "adaptive_stack_num_layers",
    ]);
    expect(disabledConfigFieldReasons(sections, {}).size).toBe(0);
  });

  it("uses adaptive option flags and generator stack flags for component sections", () => {
    const sections: ConfigSection[] = [
      {
        title: "Weight Options",
        fields: [
          field({
            key: "weight_option_flag",
            type: "bool",
            default: false,
            section: "Weight Options",
          }),
          field({
            key: "weight_option",
            type: "class",
            default: null,
            nullable: true,
            section: "Weight Options",
          }),
          field({
            key: "weight_generator_stack_independent_flag",
            type: "bool",
            default: false,
            section: "Weight Options",
          }),
          field({
            key: "weight_generator_stack_hidden_dim",
            default: null,
            nullable: true,
            section: "Weight Options",
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
          field({
            key: "mask_generator_stack_independent_flag",
            type: "bool",
            default: false,
            section: "Mask Options",
          }),
          field({
            key: "mask_generator_stack_hidden_dim",
            default: null,
            nullable: true,
            section: "Mask Options",
          }),
        ],
      },
    ];

    const [weightSection, maskSection] = deriveNestedConfigSections(sections);
    const weightGeneratorSection = weightSection.children?.find(
      (section) => section.title === "Weight Generator Stack Options",
    );
    const maskGeneratorSection = maskSection.children?.find(
      (section) => section.title === "Mask Generator Stack Options",
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

  it("uses adaptive flags and divider groups for boundary projector sections", () => {
    const sections: ConfigSection[] = [
      {
        title: "Input Boundary Projector Options",
        fields: [
          field({
            key: "input_layer_adaptive_flag",
            type: "bool",
            default: false,
            section: "Input Boundary Projector Options",
          }),
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
          field({
            key: "input_layer_adaptive_generator_stack_hidden_dim",
            section: "Input Boundary Projector Options",
          }),
        ],
      },
      {
        title: "Output Boundary Projector Options",
        fields: [
          field({
            key: "output_layer_adaptive_flag",
            type: "bool",
            default: false,
            section: "Output Boundary Projector Options",
          }),
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
    const inputBodyFields = inputSection.fields.filter(
      (item) => item.key !== inputSection.controlFieldKey,
    );
    const groups = boundaryProjectorFieldGroups(
      inputSection.title,
      inputBodyFields,
    );

    expect(inputSection.controlFieldKey).toBe("input_layer_adaptive_flag");
    expect(outputSection.controlFieldKey).toBe("output_layer_adaptive_flag");
    expect(inputSection.children ?? []).toEqual([]);
    expect(outputSection.children ?? []).toEqual([]);
    expect(groups?.map((group) => group.title)).toEqual([
      "Weight",
      "Bias",
      "Diagonal",
      "Mask",
      "Adaptive Generator Stack",
    ]);
    expect(groups?.map((group) => group.fields.map((item) => item.key))).toEqual([
      ["input_layer_weight_option"],
      ["input_layer_bias_option"],
      ["input_layer_diagonal_option"],
      ["input_layer_row_mask_option"],
      ["input_layer_adaptive_generator_stack_hidden_dim"],
    ]);

    const disabledByDefault = disabledConfigFieldReasons(sections, {});
    expect(disabledByDefault.has("input_layer_adaptive_flag")).toBe(false);
    expect(disabledByDefault.get("input_layer_weight_option")).toContain(
      "input_layer_adaptive_flag",
    );
    expect(disabledByDefault.get("output_layer_weight_option")).toContain(
      "output_layer_adaptive_flag",
    );

    const enabled = disabledConfigFieldReasons(sections, {
      input_layer_adaptive_flag: "true",
      output_layer_adaptive_flag: "true",
    });
    expect(enabled.has("input_layer_weight_option")).toBe(false);
    expect(enabled.has("output_layer_weight_option")).toBe(false);
  });
});
