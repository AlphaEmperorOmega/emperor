import { describe, expect, it } from "vitest";

import {
  deriveNestedConfigSections,
  disabledConfigFieldReasons,
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
    default: overrides.default ?? 0,
    nullable: overrides.nullable ?? false,
    choices: overrides.choices ?? [],
    locked: overrides.locked ?? false,
    lockedValue: overrides.lockedValue,
    lockedReason: overrides.lockedReason,
  };
}

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
