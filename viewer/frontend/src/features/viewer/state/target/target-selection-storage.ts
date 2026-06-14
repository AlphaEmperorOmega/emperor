type PersistedTargetMode = "preset" | "snapshot";

export type PersistedTargetSelection = {
  selectedModel: string;
  selectedPreset: string;
  selectedTargetMode: PersistedTargetMode;
  selectedSnapshotId: string;
};

const TARGET_SELECTION_STORAGE_KEY = "emperor.viewer.targetSelection.v1";
let fallbackTargetSelectionValue: string | null = null;

function getLocalStorage() {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function stringValue(value: unknown) {
  return typeof value === "string" ? value : "";
}

function parseTargetSelection(value: string | null): PersistedTargetSelection | null {
  if (!value) {
    return null;
  }
  try {
    const parsed = JSON.parse(value) as Record<string, unknown>;
    const selectedModel = stringValue(parsed.selectedModel).trim();
    const selectedPreset = stringValue(parsed.selectedPreset).trim();
    const selectedTargetMode =
      parsed.selectedTargetMode === "snapshot" ? "snapshot" : "preset";
    const selectedSnapshotId =
      selectedTargetMode === "snapshot"
        ? stringValue(parsed.selectedSnapshotId).trim()
        : "";
    if (!selectedModel || !selectedPreset) {
      return null;
    }
    return {
      selectedModel,
      selectedPreset,
      selectedTargetMode,
      selectedSnapshotId,
    };
  } catch {
    return null;
  }
}

export function readPersistedTargetSelection() {
  const storage = getLocalStorage();
  try {
    const storedValue = storage?.getItem?.(TARGET_SELECTION_STORAGE_KEY);
    if (storedValue) {
      return parseTargetSelection(storedValue);
    }
  } catch {
    // Fall through to the in-memory fallback below.
  }
  return parseTargetSelection(fallbackTargetSelectionValue);
}

export function writePersistedTargetSelection(
  selection: PersistedTargetSelection,
) {
  const value = JSON.stringify(selection);
  fallbackTargetSelectionValue = value;
  const storage = getLocalStorage();
  if (!storage?.setItem) {
    return;
  }
  try {
    storage.setItem(TARGET_SELECTION_STORAGE_KEY, value);
  } catch {
    // Persistence is best-effort; storage failures should not break the viewer.
  }
}

export function clearPersistedTargetSelection() {
  fallbackTargetSelectionValue = null;
  const storage = getLocalStorage();
  try {
    storage?.removeItem?.(TARGET_SELECTION_STORAGE_KEY);
  } catch {
    // Clearing is best-effort for the same reason writes are.
  }
}
