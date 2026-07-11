type PersistedTargetMode = "preset" | "snapshot";

export type PersistedTargetSelection = {
  selectedModelType: string;
  selectedModel: string;
  selectedPreset: string;
  selectedTargetMode: PersistedTargetMode;
  selectedSnapshotId: string;
};

const TARGET_SELECTION_STORAGE_KEY = "emperor.workbench.targetSelection.v1";

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

function splitLegacyModelId(modelId: string) {
  const separatorIndex = modelId.indexOf("/");
  if (separatorIndex <= 0 || separatorIndex === modelId.length - 1) {
    return { modelType: "models", model: modelId };
  }
  return {
    modelType: modelId.slice(0, separatorIndex).trim() || "models",
    model: modelId.slice(separatorIndex + 1).trim() || modelId,
  };
}

function parseTargetSelection(value: string | null): PersistedTargetSelection | null {
  if (!value) {
    return null;
  }
  try {
    const parsed = JSON.parse(value) as Record<string, unknown>;
    const rawModel = stringValue(parsed.selectedModel).trim();
    const explicitModelType = stringValue(parsed.selectedModelType).trim();
    const legacyIdentity = splitLegacyModelId(rawModel);
    const selectedModelType = explicitModelType || legacyIdentity.modelType;
    const selectedModel = explicitModelType ? rawModel : legacyIdentity.model;
    const selectedPreset = stringValue(parsed.selectedPreset).trim();
    const selectedTargetMode =
      parsed.selectedTargetMode === "snapshot" ? "snapshot" : "preset";
    const selectedSnapshotId =
      selectedTargetMode === "snapshot"
        ? stringValue(parsed.selectedSnapshotId).trim()
        : "";
    if (!selectedModelType || !selectedModel || !selectedPreset) {
      return null;
    }
    return {
      selectedModelType,
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
    return parseTargetSelection(
      storage?.getItem?.(TARGET_SELECTION_STORAGE_KEY) ?? null,
    );
  } catch {
    return null;
  }
}

export function writePersistedTargetSelection(
  selection: PersistedTargetSelection,
) {
  const value = JSON.stringify(selection);
  const storage = getLocalStorage();
  if (!storage?.setItem) {
    return;
  }
  try {
    storage.setItem(TARGET_SELECTION_STORAGE_KEY, value);
  } catch {
    // Persistence is best-effort; storage failures should not break the workbench.
  }
}
