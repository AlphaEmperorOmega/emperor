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
  it("uses the first public ID path segment as the model type", () => {
    expect(modelTypeForId("linears/linear")).toBe("linears");
    expect(modelTypeForId("bert/linear")).toBe("bert");
    expect(modelTypeForId("vit/linear")).toBe("vit");
  });

  it("falls back to a shared legacy type for flat or malformed IDs", () => {
    expect(modelTypeForId("linear")).toBe("models");
    expect(modelTypeForId("/linear")).toBe("models");
    expect(modelTypeForId("")).toBe("models");
  });
});

describe("modelNameForId", () => {
  it("returns the public ID suffix when a type prefix is present", () => {
    expect(modelNameForId("linears/linear")).toBe("linear");
    expect(modelNameForId("bert/linear")).toBe("linear");
    expect(modelNameForId("vit/expert_linear")).toBe("expert_linear");
  });

  it("preserves flat IDs", () => {
    expect(modelNameForId("linear")).toBe("linear");
  });
});

describe("modelTypeOptions", () => {
  it("deduplicates types in catalog order and formats labels", () => {
    expect(
      modelTypeOptions([
        "linears/linear",
        "linears/linear_adaptive",
        "experts/linear",
        "bert/linear",
        "vit/linear",
      ]),
    ).toEqual([
      { value: "linears", label: "Linears" },
      { value: "experts", label: "Experts" },
      { value: "bert", label: "Bert" },
      { value: "vit", label: "Vit" },
    ]);
  });
});

describe("modelsForType", () => {
  it("filters public model IDs by type", () => {
    const catalog = [
      "linears/linear",
      "linears/linear_adaptive",
      "experts/linear",
      "bert/linear",
      "bert/linear_adaptive",
      "bert/expert_linear",
      "bert/expert_linear_adaptive",
      "vit/linear",
      "vit/linear_adaptive",
      "vit/expert_linear",
      "vit/expert_linear_adaptive",
    ];

    expect(
      modelsForType(catalog, "linears"),
    ).toEqual(["linears/linear", "linears/linear_adaptive"]);

    expect(modelsForType(catalog, "bert")).toEqual([
      "bert/linear",
      "bert/linear_adaptive",
      "bert/expert_linear",
      "bert/expert_linear_adaptive",
    ]);

    expect(modelsForType(catalog, "vit")).toEqual([
      "vit/linear",
      "vit/linear_adaptive",
      "vit/expert_linear",
      "vit/expert_linear_adaptive",
    ]);
  });

  it("returns the full catalog when no type is selected", () => {
    expect(modelsForType(["linears/linear", "experts/linear"], ""))
      .toEqual(["linears/linear", "experts/linear"]);
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
