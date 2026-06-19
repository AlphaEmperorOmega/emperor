import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useTargetOverridesState } from "@/features/viewer/state/target/use-target-overrides";

// Characterization tests for model/preset selection + override mutation semantics.

describe("useTargetOverridesState", () => {
  it("starts empty", () => {
    const { result } = renderHook(() => useTargetOverridesState());
    expect(result.current.selectedModel).toBe("");
    expect(result.current.selectedPreset).toBe("");
    expect(result.current.overrides).toEqual({});
  });

  it("selectModel resets preset and overrides", () => {
    const { result } = renderHook(() => useTargetOverridesState());

    act(() => result.current.selectPreset("p1"));
    act(() => result.current.updateOverride("lr", "0.1"));
    act(() => result.current.selectModel("linear"));

    expect(result.current.selectedModel).toBe("linear");
    expect(result.current.selectedPreset).toBe("");
    expect(result.current.overrides).toEqual({});
  });

  it("selectPreset preserves overrides and keeps the model", () => {
    const { result } = renderHook(() => useTargetOverridesState());

    act(() => result.current.selectModel("linear"));
    act(() => result.current.updateOverride("lr", "0.1"));
    act(() => result.current.selectPreset("p2"));

    expect(result.current.selectedModel).toBe("linear");
    expect(result.current.selectedPreset).toBe("p2");
    expect(result.current.overrides).toEqual({ lr: "0.1" });
  });

  it("updateOverride merges keys, clearOverride removes one, clearOverrides empties", () => {
    const { result } = renderHook(() => useTargetOverridesState());

    act(() => result.current.updateOverride("a", "1"));
    act(() => result.current.updateOverride("b", "2"));
    expect(result.current.overrides).toEqual({ a: "1", b: "2" });

    act(() => result.current.updateOverride("a", "3"));
    expect(result.current.overrides).toEqual({ a: "3", b: "2" });

    act(() => result.current.clearOverride("a"));
    expect(result.current.overrides).toEqual({ b: "2" });

    act(() => result.current.clearOverrides());
    expect(result.current.overrides).toEqual({});
  });
});
