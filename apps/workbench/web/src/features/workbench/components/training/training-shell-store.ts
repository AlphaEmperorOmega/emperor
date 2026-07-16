export type TrainingShell = "posix" | "powershell";

export const TRAINING_SHELL_STORAGE_KEY =
  "emperor.workbench.trainingShell";
export const TRAINING_SHELL_CHANGE_EVENT =
  "emperor:training-shell-change";

const listeners = new Set<() => void>();
let snapshot: TrainingShell | null = null;
let storageFallbackActive = false;

function isTrainingShell(value: unknown): value is TrainingShell {
  return value === "posix" || value === "powershell";
}

function notifyListeners() {
  for (const listener of listeners) {
    listener();
  }
}

function suggestedShell(): TrainingShell {
  if (typeof navigator === "undefined") {
    return "posix";
  }
  const navigatorWithUserAgentData = navigator as Navigator & {
    userAgentData?: { platform?: string };
  };
  const platform = [
    navigatorWithUserAgentData.userAgentData?.platform,
    navigator.platform,
    navigator.userAgent,
  ]
    .filter(Boolean)
    .join(" ");
  return /Windows|Win32|Win64/i.test(platform) ? "powershell" : "posix";
}

function storedShell(): {
  available: boolean;
  shell: TrainingShell | null;
} {
  if (typeof window === "undefined") {
    return { available: false, shell: null };
  }
  try {
    const value = window.localStorage.getItem(TRAINING_SHELL_STORAGE_KEY);
    return {
      available: true,
      shell: isTrainingShell(value) ? value : null,
    };
  } catch {
    return { available: false, shell: null };
  }
}

function readSnapshot() {
  if (storageFallbackActive) {
    return snapshot ?? suggestedShell();
  }
  const stored = storedShell();
  if (!stored.available) {
    return snapshot ?? suggestedShell();
  }
  return stored.shell ?? suggestedShell();
}

function publishStoredShell() {
  const next = readSnapshot();
  if (snapshot === next) {
    return;
  }
  snapshot = next;
  notifyListeners();
}

function handleStorage(event: StorageEvent) {
  if (event.key !== TRAINING_SHELL_STORAGE_KEY && event.key !== null) {
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
  publishStoredShell();
}

function handleTrainingShellChange(event: Event) {
  if (!(event instanceof CustomEvent) || !isTrainingShell(event.detail)) {
    storageFallbackActive = false;
  }
  const next =
    event instanceof CustomEvent && isTrainingShell(event.detail)
      ? event.detail
      : readSnapshot();
  if (snapshot === next) {
    return;
  }
  snapshot = next;
  notifyListeners();
}

export function subscribeTrainingShell(listener: () => void) {
  listeners.add(listener);
  if (typeof window !== "undefined" && listeners.size === 1) {
    window.addEventListener(
      TRAINING_SHELL_CHANGE_EVENT,
      handleTrainingShellChange,
    );
    window.addEventListener("storage", handleStorage);
  }
  return () => {
    listeners.delete(listener);
    if (typeof window !== "undefined" && listeners.size === 0) {
      window.removeEventListener(
        TRAINING_SHELL_CHANGE_EVENT,
        handleTrainingShellChange,
      );
      window.removeEventListener("storage", handleStorage);
    }
  };
}

export function getTrainingShellSnapshot() {
  snapshot = readSnapshot();
  return snapshot;
}

export function getTrainingShellServerSnapshot(): TrainingShell {
  return "posix";
}

export function setTrainingShell(nextShell: TrainingShell) {
  const changed = snapshot !== nextShell;
  snapshot = nextShell;
  if (typeof window !== "undefined") {
    try {
      window.localStorage.setItem(TRAINING_SHELL_STORAGE_KEY, nextShell);
      storageFallbackActive = false;
    } catch {
      storageFallbackActive = true;
    }
    window.dispatchEvent(
      new CustomEvent<TrainingShell>(TRAINING_SHELL_CHANGE_EVENT, {
        detail: nextShell,
      }),
    );
  } else {
    storageFallbackActive = true;
  }
  if (changed) {
    notifyListeners();
  }
}
