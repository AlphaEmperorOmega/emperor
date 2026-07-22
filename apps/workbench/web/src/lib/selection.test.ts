import { describe, expect, it } from "vitest";
import {
  configValueEquals,
  modelNameForId,
  modelsForType,
  modelTypeForId,
  modelTypeOptions,
  normalizePrimarySelection,
  normalizeSelection,
  selectionValuesEqual,
  uniqueValidValues,
  valueIsSelected,
} from "@/lib/selection";

describe("modelTypeForId", () => {
  it("returns the explicit model type", () => {
    expect(modelTypeForId({ modelType: "linears", model: "linear" })).toBe(
      "linears",
    );
    expect(modelTypeForId({ modelType: "bert", model: "linear" })).toBe(
      "bert",
    );
  });
});

describe("modelNameForId", () => {
  it("returns the explicit model name", () => {
    expect(modelNameForId({ modelType: "linears", model: "linear" })).toBe(
      "linear",
    );
    expect(
      modelNameForId({ modelType: "vit", model: "expert_linear" }),
    ).toBe("expert_linear");
  });
});

describe("modelTypeOptions", () => {
  it("deduplicates types in catalog order and formats labels", () => {
    expect(
      modelTypeOptions([
        { modelType: "linears", model: "linear" },
        { modelType: "linears", model: "linear_adaptive" },
        { modelType: "experts", model: "linear" },
        { modelType: "bert", model: "linear" },
        { modelType: "vit", model: "linear" },
        { modelType: "transformer", model: "linear" },
      ]),
    ).toEqual([
      { value: "linears", label: "Linears" },
      { value: "experts", label: "Experts" },
      { value: "bert", label: "Bert" },
      { value: "vit", label: "Vit" },
      { value: "transformer", label: "Transformer" },
    ]);
  });
});

describe("modelsForType", () => {
  it("filters explicit model identities by type", () => {
    const catalog = [
      { modelType: "linears", model: "linear" },
      { modelType: "linears", model: "linear_adaptive" },
      { modelType: "experts", model: "linear" },
      { modelType: "bert", model: "linear" },
      { modelType: "bert", model: "linear_adaptive" },
      { modelType: "bert", model: "expert_linear" },
      { modelType: "bert", model: "expert_linear_adaptive" },
      { modelType: "vit", model: "linear" },
      { modelType: "vit", model: "linear_adaptive" },
      { modelType: "vit", model: "expert_linear" },
      { modelType: "vit", model: "expert_linear_adaptive" },
      { modelType: "transformer", model: "linear" },
      { modelType: "transformer", model: "linear_adaptive" },
      { modelType: "transformer", model: "expert_linear" },
      { modelType: "transformer", model: "expert_linear_adaptive" },
    ];

    expect(modelsForType(catalog, "linears")).toEqual([
      { modelType: "linears", model: "linear" },
      { modelType: "linears", model: "linear_adaptive" },
    ]);

    expect(modelsForType(catalog, "bert")).toEqual([
      { modelType: "bert", model: "linear" },
      { modelType: "bert", model: "linear_adaptive" },
      { modelType: "bert", model: "expert_linear" },
      { modelType: "bert", model: "expert_linear_adaptive" },
    ]);

    expect(modelsForType(catalog, "vit")).toEqual([
      { modelType: "vit", model: "linear" },
      { modelType: "vit", model: "linear_adaptive" },
      { modelType: "vit", model: "expert_linear" },
      { modelType: "vit", model: "expert_linear_adaptive" },
    ]);

    expect(modelsForType(catalog, "transformer")).toEqual([
      { modelType: "transformer", model: "linear" },
      { modelType: "transformer", model: "linear_adaptive" },
      { modelType: "transformer", model: "expert_linear" },
      { modelType: "transformer", model: "expert_linear_adaptive" },
    ]);
  });

  it("returns the full catalog when no type is selected", () => {
    const catalog = [
      { modelType: "linears", model: "linear" },
      { modelType: "experts", model: "linear" },
    ];
    expect(modelsForType(catalog, "")).toEqual(catalog);
  });
});

describe("uniqueValidValues", () => {
  it("filters invalid values, removes duplicates, and preserves input order", () => {
    expect(
      uniqueValidValues(["b", "x", "a", "b", "c", "a"], ["a", "b", "c"]),
    ).toEqual(["b", "a", "c"]);
  });
});

describe("normalizeSelection", () => {
  it("returns unique valid selected values when present", () => {
    expect(normalizeSelection(["b", "x", "a", "b"], ["a", "b", "c"])).toEqual([
      "b",
      "a",
    ]);
  });

  it("uses the first valid fallback when selections are invalid", () => {
    expect(
      normalizeSelection(["x"], ["a", "b", "c"], ["z", "b", "a"]),
    ).toEqual(["b"]);
  });

  it("uses the first valid value when no fallback matches", () => {
    expect(normalizeSelection(["x"], ["a", "b"], ["z"])).toEqual(["a"]);
  });

  it("uses the first valid value when fallback is empty", () => {
    expect(normalizeSelection([], ["a", "b"])).toEqual(["a"]);
  });

  it("returns an empty selection when there are no valid values", () => {
    expect(normalizeSelection(["a"], [])).toEqual([]);
  });
});

describe("normalizePrimarySelection", () => {
  it("filters invalid values, removes duplicates, and preserves input order", () => {
    expect(
      normalizePrimarySelection(["b", "x", "a", "b"], ["a", "b", "c"]),
    ).toEqual(["b", "a"]);
  });

  it("moves an already-selected primary value to the front", () => {
    expect(
      normalizePrimarySelection(["a", "c", "b"], ["a", "b", "c"], "b"),
    ).toEqual(["b", "a", "c"]);
  });

  it("does not auto-add the primary value when other valid selections exist", () => {
    expect(normalizePrimarySelection(["a"], ["a", "b"], "b")).toEqual(["a"]);
  });

  it("uses a valid primary value as fallback when selections are empty or invalid", () => {
    expect(normalizePrimarySelection([], ["a", "b"], "b")).toEqual(["b"]);
    expect(normalizePrimarySelection(["x"], ["a", "b"], "b")).toEqual(["b"]);
  });

  it("returns an empty selection without valid values or a valid primary fallback", () => {
    expect(normalizePrimarySelection(["a"], [], "a")).toEqual([]);
    expect(normalizePrimarySelection(["x"], ["a", "b"])).toEqual([]);
    expect(normalizePrimarySelection(["x"], ["a", "b"], "z")).toEqual([]);
  });
});

describe("selectionValuesEqual", () => {
  it("compares ordered selection values", () => {
    expect(selectionValuesEqual(["a", "b"], ["a", "b"])).toBe(true);
    expect(selectionValuesEqual(["a", "b"], ["b", "a"])).toBe(false);
    expect(selectionValuesEqual(["a"], ["a", "b"])).toBe(false);
  });
});

describe("configValueEquals", () => {
  it("uses type-sensitive string comparison", () => {
    expect(configValueEquals(1, 1)).toBe(true);
    expect(configValueEquals(1, "1")).toBe(false);
    expect(configValueEquals(true, true)).toBe(true);
    expect(configValueEquals(true, "true")).toBe(false);
    expect(configValueEquals("value", "value")).toBe(true);
    expect(configValueEquals("value", "other")).toBe(false);
  });

  it("handles null equality separately from the string null", () => {
    expect(configValueEquals(null, null)).toBe(true);
    expect(configValueEquals(null, "null")).toBe(false);
  });
});

describe("valueIsSelected", () => {
  it("matches selected string, number, boolean, and null values", () => {
    expect(valueIsSelected(["fast", 3, false, null], "fast")).toBe(true);
    expect(valueIsSelected(["fast", 3, false, null], 3)).toBe(true);
    expect(valueIsSelected(["fast", 3, false, null], false)).toBe(true);
    expect(valueIsSelected(["fast", 3, false, null], null)).toBe(true);
  });

  it("keeps comparisons type-sensitive", () => {
    expect(valueIsSelected([1], "1")).toBe(false);
    expect(valueIsSelected(["1"], 1)).toBe(false);
    expect(valueIsSelected([null], "null")).toBe(false);
    expect(valueIsSelected(["null"], null)).toBe(false);
  });

  it("returns false for an empty list", () => {
    expect(valueIsSelected([], "fast")).toBe(false);
  });
});
