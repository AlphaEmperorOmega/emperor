export const WORKBENCH_API_BASE_URL_STORAGE_KEY =
  "emperor.workbench.apiBaseUrl";
export const WORKBENCH_AUTH_TOKEN_STORAGE_KEY =
  "emperor.workbench.authToken";

export type WorkbenchStorageAvailability = "available" | "unavailable";
export type WorkbenchStorageKind = "localStorage" | "sessionStorage";

type StorageReadResult = Readonly<{
  availability: WorkbenchStorageAvailability;
  value: string | null;
  message: string | null;
}>;

type StoragePersistenceResult =
  | Readonly<{ ok: true; rollback: () => string | null }>
  | Readonly<{ ok: false; message: string }>;

function storageFor(kind: WorkbenchStorageKind) {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const storage = window[kind];
    return typeof storage?.getItem === "function" ? storage : null;
  } catch {
    return null;
  }
}

function writeStorageValue(
  storage: Storage,
  key: string,
  value: string | null,
) {
  if (value === null) {
    storage.removeItem(key);
  } else {
    storage.setItem(key, value);
  }
  if (storage.getItem(key) !== value) {
    throw new Error("Browser storage did not retain the requested value.");
  }
}

export function readConnectionStorageValue(
  kind: WorkbenchStorageKind,
  key: string,
  unavailableMessage: string,
): StorageReadResult {
  const storage = storageFor(kind);
  if (!storage) {
    return {
      availability: "unavailable",
      value: null,
      message: unavailableMessage,
    };
  }
  try {
    return {
      availability: "available",
      value: storage.getItem(key),
      message: null,
    };
  } catch {
    return {
      availability: "unavailable",
      value: null,
      message: unavailableMessage,
    };
  }
}

export function writeConnectionStorageValue(
  kind: WorkbenchStorageKind,
  key: string,
  value: string | null,
) {
  const storage = storageFor(kind);
  if (!storage) {
    return false;
  }
  try {
    writeStorageValue(storage, key, value);
    return true;
  } catch {
    return false;
  }
}

export function persistConnectionStorageValue({
  kind,
  key,
  value,
  failureMessage,
  rollbackFailureMessage,
}: {
  kind: WorkbenchStorageKind;
  key: string;
  value: string | null;
  failureMessage: string;
  rollbackFailureMessage: string;
}): StoragePersistenceResult {
  const storage = storageFor(kind);
  if (!storage) {
    return { ok: false, message: failureMessage };
  }

  let previousValue: string | null = null;
  let previousValueRead = false;
  try {
    previousValue = storage.getItem(key);
    previousValueRead = true;
    writeStorageValue(storage, key, value);
  } catch {
    if (previousValueRead) {
      try {
        writeStorageValue(storage, key, previousValue);
      } catch {
        // The caller records that persistence is unavailable.
      }
    }
    return { ok: false, message: failureMessage };
  }

  return {
    ok: true,
    rollback: () => {
      try {
        writeStorageValue(storage, key, previousValue);
        return null;
      } catch {
        return rollbackFailureMessage;
      }
    },
  };
}
