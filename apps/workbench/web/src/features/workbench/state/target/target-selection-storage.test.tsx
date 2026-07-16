import { act } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  getPersistedTargetSelectionServerSnapshot,
  getPersistedTargetSelectionSnapshot,
  setPersistedTargetSelection,
  subscribePersistedTargetSelection,
  TARGET_SELECTION_STORAGE_KEY,
} from "@/features/workbench/state/target/target-selection-storage";

const selection = {
  selectedModelType: "linears",
  selectedModel: "linear",
  selectedPreset: "fast",
  selectedTargetMode: "preset",
  selectedSnapshotId: "",
} as const;

afterEach(() => {
  vi.restoreAllMocks();
  setPersistedTargetSelection(selection);
  window.localStorage.clear();
  getPersistedTargetSelectionSnapshot();
});

describe("persisted target selection store", () => {
  it("uses a deterministic empty server snapshot", () => {
    expect(getPersistedTargetSelectionServerSnapshot()).toBeNull();
  });

  it("publishes stable immutable snapshots through its explicit setter", () => {
    const listener = vi.fn();
    const unsubscribe = subscribePersistedTargetSelection(listener);

    act(() => setPersistedTargetSelection(selection));

    const first = getPersistedTargetSelectionSnapshot();
    const second = getPersistedTargetSelectionSnapshot();
    expect(first).toBe(second);
    expect(first).toEqual(selection);
    expect(Object.isFrozen(first)).toBe(true);
    expect(listener).toHaveBeenCalledTimes(1);

    act(() => setPersistedTargetSelection(selection));
    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
  });

  it("observes cross-document storage changes", () => {
    const listener = vi.fn();
    const unsubscribe = subscribePersistedTargetSelection(listener);
    const next = { ...selection, selectedPreset: "accurate" };
    window.localStorage.setItem(
      TARGET_SELECTION_STORAGE_KEY,
      JSON.stringify(next),
    );

    act(() => {
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: TARGET_SELECTION_STORAGE_KEY,
          newValue: JSON.stringify(next),
          storageArea: null,
        }),
      );
    });

    expect(getPersistedTargetSelectionSnapshot()).toEqual(next);
    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
  });

  it("keeps the in-memory snapshot when storage is unavailable", () => {
    vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
      throw new Error("write unavailable");
    });

    act(() => setPersistedTargetSelection(selection));

    expect(getPersistedTargetSelectionSnapshot()).toEqual(selection);
  });
});
