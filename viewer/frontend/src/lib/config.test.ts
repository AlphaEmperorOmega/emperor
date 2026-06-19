import { describe, expect, it } from "vitest";

import {
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
});
