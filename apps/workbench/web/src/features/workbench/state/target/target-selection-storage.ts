type PersistedTargetMode = "preset" | "snapshot";

export type PersistedTargetSelection = Readonly<{
  selectedModelType: string;
  selectedModel: string;
  selectedPreset: string;
  selectedTargetMode: PersistedTargetMode;
  selectedSnapshotId: string;
}>;

export const TARGET_SELECTION_STORAGE_KEY =
  "emperor.workbench.targetSelection.v1";

const listeners = new Set<() => void>();
let snapshot: PersistedTargetSelection | null = null;
let snapshotRawValue: string | null | undefined;
let storageFallbackActive = false;

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

function isModelIdentitySegment(value: string) {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(value);
}

function parseTargetSelection(
  value: string | null,
): PersistedTargetSelection | null {
  if (!value) {
    return null;
  }
  try {
    const parsed = JSON.parse(value) as Record<string, unknown>;
    const selectedModelType = stringValue(parsed.selectedModelType).trim();
    const selectedModel = stringValue(parsed.selectedModel).trim();
    const selectedPreset = stringValue(parsed.selectedPreset).trim();
    const selectedTargetMode =
      parsed.selectedTargetMode === "snapshot" ? "snapshot" : "preset";
    const selectedSnapshotId =
      selectedTargetMode === "snapshot"
        ? stringValue(parsed.selectedSnapshotId).trim()
        : "";
    if (
      !isModelIdentitySegment(selectedModelType) ||
      !isModelIdentitySegment(selectedModel) ||
      !selectedPreset
    ) {
      return null;
    }
    return Object.freeze({
      selectedModelType,
      selectedModel,
      selectedPreset,
      selectedTargetMode,
      selectedSnapshotId,
    });
  } catch {
    return null;
  }
}

function snapshotFromRawValue(value: string | null) {
  if (snapshotRawValue === value) {
    return snapshot;
  }
  snapshotRawValue = value;
  snapshot = parseTargetSelection(value);
  return snapshot;
}

function storedSnapshot() {
  if (storageFallbackActive) {
    return snapshotRawValue === undefined ? null : snapshot;
  }
  const storage = getLocalStorage();
  if (!storage) {
    return snapshotRawValue === undefined ? null : snapshot;
  }
  try {
    return snapshotFromRawValue(
      storage.getItem(TARGET_SELECTION_STORAGE_KEY),
    );
  } catch {
    return snapshotRawValue === undefined ? null : snapshot;
  }
}

function notifyListeners() {
  for (const listener of listeners) {
    listener();
  }
}

function publishRawValue(value: string | null) {
  const previous = snapshot;
  const previousRawValue = snapshotRawValue;
  const next = snapshotFromRawValue(value);
  if (previousRawValue !== snapshotRawValue || previous !== next) {
    notifyListeners();
  }
}

function handleStorage(event: StorageEvent) {
  if (
    event.key !== TARGET_SELECTION_STORAGE_KEY &&
    event.key !== null
  ) {
    return;
  }
  try {
    if (event.storageArea !== null && event.storageArea !== window.localStorage) {
      return;
    }
  } catch {
    return;
  }
  storageFallbackActive = false;
  if (event.key === TARGET_SELECTION_STORAGE_KEY) {
    publishRawValue(event.newValue);
  } else {
    try {
      publishRawValue(
        window.localStorage.getItem(TARGET_SELECTION_STORAGE_KEY),
      );
    } catch {
      // Keep the last readable snapshot when storage is unavailable.
    }
  }
}

export function subscribePersistedTargetSelection(listener: () => void) {
  listeners.add(listener);
  if (typeof window !== "undefined" && listeners.size === 1) {
    window.addEventListener("storage", handleStorage);
  }
  return () => {
    listeners.delete(listener);
    if (typeof window !== "undefined" && listeners.size === 0) {
      window.removeEventListener("storage", handleStorage);
    }
  };
}

export function getPersistedTargetSelectionSnapshot() {
  return storedSnapshot();
}

export function getPersistedTargetSelectionServerSnapshot() {
  return null;
}

export function setPersistedTargetSelection(
  selection: PersistedTargetSelection,
) {
  const value = JSON.stringify(selection);
  const previousRawValue = snapshotRawValue;
  snapshotFromRawValue(value);
  const storage = getLocalStorage();
  if (storage?.setItem) {
    try {
      storage.setItem(TARGET_SELECTION_STORAGE_KEY, value);
      storageFallbackActive = false;
    } catch {
      storageFallbackActive = true;
    }
  } else {
    storageFallbackActive = true;
  }
  if (previousRawValue !== value) {
    notifyListeners();
  }
}
