import { describe, expect, it } from "vitest";
import type { ConfigField } from "@/lib/api/models";
import { type ConfigSection } from "@/lib/config";
import { presentRuntimeDefaultsSchema } from "@/features/workbench/state/runtime-defaults/runtime-defaults-presentation";

function field(
  overrides: Partial<ConfigField> &
    Pick<ConfigField, "key" | "section" | "sectionPath">,
): ConfigField {
  return {
    key: overrides.key,
    configKey: overrides.configKey ?? overrides.key.toUpperCase(),
    flag: overrides.flag ?? `--${overrides.key.replace(/_/g, "-")}`,
    label: overrides.label ?? overrides.key,
    section: overrides.section,
    sectionPath: overrides.sectionPath,
    description: overrides.description,
    type: overrides.type ?? "int",
    default: Object.prototype.hasOwnProperty.call(overrides, "default")
      ? (overrides.default ?? null)
      : 0,
    nullable: overrides.nullable ?? false,
    choices: overrides.choices ?? [],
    locked: overrides.locked ?? false,
    lockedValue: overrides.lockedValue,
    lockedReason: overrides.lockedReason,
  };
}

function schemaSections(): ConfigSection[] {
  const globalHidden = field({
    key: "hidden_dim",
    label: "Hidden Dim",
    section: "Global",
    sectionPath: ["Global"],
    default: 64,
    locked: true,
    lockedValue: 96,
    lockedReason: "Selected preset owns this value.",
  });
  const layerStackDepth = field({
    key: "stack_depth",
    label: "Stack Depth",
    section: "Layer Stack Options",
    sectionPath: ["Layer Stack Options"],
    default: 2,
  });
  const gateFlag = field({
    key: "stack_gate_flag",
    label: "Gate Flag",
    section: "Gate Options",
    sectionPath: ["Gate Options"],
    type: "bool",
    default: false,
    choices: [true, false],
  });
  const gateScale = field({
    key: "gate_scale",
    label: "Gate Scale",
    section: "Gate Options",
    sectionPath: ["Gate Options"],
    type: "float",
    default: 1,
  });
  const gateStackFlag = field({
    key: "gate_stack_independent_flag",
    label: "Gate Stack Independent Flag",
    section: "Gate Stack Options",
    sectionPath: ["Gate Options", "Gate Stack Options"],
    type: "bool",
    default: false,
    choices: [true, false],
  });
  const gateStackDepth = field({
    key: "gate_stack_depth",
    label: "Gate Stack Depth",
    section: "Gate Stack Options",
    sectionPath: ["Gate Options", "Gate Stack Options"],
    default: 2,
  });
  const inputWeightOption = field({
    key: "input_layer_weight_option",
    label: "Input Layer Weight Option",
    section: "Input Boundary Model Options",
    sectionPath: ["Input Boundary Model Options"],
    type: "class",
    default: null,
    nullable: true,
    choices: ["DynamicWeightConfig"],
  });
  const inputWeightRank = field({
    key: "input_layer_weight_rank",
    label: "Input Layer Weight Rank",
    section: "Input Boundary Model Options",
    sectionPath: ["Input Boundary Model Options"],
    default: 4,
  });

  return [
    { title: "Global", fields: [globalHidden] },
    { title: "Layer Stack Options", fields: [layerStackDepth] },
    { title: "Gate Options", fields: [gateFlag, gateScale] },
    {
      title: "Gate Stack Options",
      fields: [gateStackFlag, gateStackDepth],
    },
    {
      title: "Input Boundary Model Options",
      fields: [inputWeightOption, inputWeightRank],
    },
  ];
}

describe("Runtime Defaults schema presentation", () => {
  it("assembles navigation counts, nested disablement, inheritance, and field groups", () => {
    const presentation = presentRuntimeDefaultsSchema({
      sections: schemaSections(),
      overrides: { stack_depth: "4" },
      search: { query: "" },
    });

    expect(presentation.fieldCount).toBe(8);
    expect(presentation.presetOwnedFieldCount).toBe(1);
    expect(presentation.sections.map((section) => section.title)).toEqual([
      "Global",
      "Layer Stack Options",
      "Gate Options",
      "Input Boundary Model Options",
    ]);
    expect(presentation.defaultOpenSectionTitles).toEqual([
      "Global",
      "Layer Stack Options",
    ]);

    const global = presentation.sections[0]!;
    expect(global.treeMetrics).toEqual({
      fieldCount: 1,
      overrideCount: 0,
      presetCount: 1,
      state: "preset",
    });

    const layerStack = presentation.sections[1]!;
    expect(layerStack.displayTitle).toBe("Layer Hidden Stack Options");
    expect(layerStack.directMetrics.state).toBe("override");
    expect(layerStack.inheritedField).toMatchObject({
      label: "Hidden Dim",
      sourceTitle: "Global",
      field: { value: "96", isPresetOwned: true },
    });

    const gate = presentation.sections[2]!;
    expect(gate.isDisabled).toBe(true);
    expect(gate.controlField).toMatchObject({
      key: "stack_gate_flag",
      label: "Enabled",
      isEnabledValue: false,
      disabledReason: undefined,
    });
    expect(gate.treeMetrics.fieldCount).toBe(4);
    expect(gate.bodyFields.find((item) => item.key === "gate_scale")?.disabledReason)
      .toBe("Enable Gate Flag before editing Gate Options.");
    expect(gate.children).toHaveLength(1);
    expect(gate.children[0]).toMatchObject({
      title: "Gate Stack Options",
      isDisabled: true,
      disabledReason: "Enable Gate Flag before editing Gate Options.",
    });

    const inputBoundary = presentation.sections[3]!;
    const weightGroup = inputBoundary.fieldGroups?.[0];
    expect(weightGroup).toMatchObject({
      title: "Weight",
      isEnabled: false,
      isSwitchDisabled: false,
      firstConcreteOption: "DynamicWeightConfig",
      metrics: {
        fieldCount: 2,
        overrideCount: 0,
        presetCount: 0,
        state: "default",
      },
    });
    expect(
      weightGroup?.fields.find((item) => item.key === "input_layer_weight_rank")
        ?.disabledReason,
    ).toBe("Select Input Layer Weight Option before editing weight boundary settings.");
  });

  it("keeps the matching leaf and required controls in one searchable tree", () => {
    const presentation = presentRuntimeDefaultsSchema({
      sections: schemaSections(),
      overrides: {},
      search: { query: "", selectedFieldKey: "gate_stack_depth" },
    });

    expect(presentation.isSearchActive).toBe(true);
    expect(presentation.searchOpenKey).toBe("gate_stack_depth\u0000");
    expect(presentation.defaultOpenSectionTitles).toEqual(["Gate Options"]);
    expect(presentation.sections).toHaveLength(1);

    const gate = presentation.sections[0]!;
    expect(gate.fields.map((item) => item.key)).toEqual(["stack_gate_flag"]);
    expect(gate.children[0]?.fields.map((item) => item.key)).toEqual([
      "gate_stack_independent_flag",
      "gate_stack_depth",
    ]);
    expect(gate.children[0]?.inheritedField).toBeUndefined();

    const searchOption = presentation.search.options.find(
      (option) => option.key === "gate_stack_depth",
    );
    expect(searchOption).toMatchObject({
      sectionTitle: "Gate Stack Options",
      rootSectionTitle: "Gate Options",
      field: {
        key: "gate_stack_depth",
        value: "2",
        disabledReason: "Enable Gate Flag before editing Gate Options.",
      },
    });
    expect(
      searchOption && presentation.search.matchesQuery(searchOption, "STACK DEPTH"),
    ).toBe(true);
    expect(
      searchOption && presentation.search.matchesQuery(searchOption, "not present"),
    ).toBe(false);
  });

  it("re-derives controlled and inherited states from current overrides", () => {
    const presentation = presentRuntimeDefaultsSchema({
      sections: schemaSections(),
      overrides: {
        stack_gate_flag: "true",
        gate_stack_independent_flag: "false",
      },
      search: { query: "" },
    });

    const gate = presentation.sections.find(
      (section) => section.title === "Gate Options",
    )!;
    const gateStack = gate.children[0]!;
    expect(gate.isDisabled).toBe(false);
    expect(gate.treeMetrics).toMatchObject({
      overrideCount: 2,
      state: "override",
    });
    expect(gateStack).toMatchObject({
      isDisabled: true,
      disabledReason:
        "Enable Gate Stack Independent Flag before editing Gate Stack Options.",
      stackInheritanceHint: {
        isCustom: false,
        sourceTitle: "Layer Stack Submodule Options",
      },
    });
    expect(presentation.defaultOpenSectionTitles).toEqual(["Global", "Gate Options"]);
  });
});
