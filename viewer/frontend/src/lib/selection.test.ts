import { describe, expect, it } from "vitest";
import {
  configValueEquals,
  normalizePrimarySelection,
  normalizeSelection,
  selectionValuesEqual,
  toggleSetValue,
  uniqueValidValues,
  valueIsSelected,
} from "@/lib/selection";

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

describe("toggleSetValue", () => {
  it("adds a missing value without mutating the original set", () => {
    const original = new Set(["a"]);

    const next = toggleSetValue(original, "b");

    expect(next).toEqual(new Set(["a", "b"]));
    expect(original).toEqual(new Set(["a"]));
  });

  it("removes an existing value without mutating the original set", () => {
    const original = new Set(["a", "b"]);

    const next = toggleSetValue(original, "a");

    expect(next).toEqual(new Set(["b"]));
    expect(original).toEqual(new Set(["a", "b"]));
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
