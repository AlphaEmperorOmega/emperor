import {
  workbenchConnectionRuntimeStateForActions,
  WORKBENCH_API_BASE_URL_STORAGE_KEY,
} from "@/lib/api/_connection-runtime";

const AUTH_TOKEN_STORAGE_KEY = "emperor.workbench.authToken";

type RuntimeWriteResult =
  | { ok: true }
  | { ok: false; message: string };

function storageAdapter(kind: "localStorage" | "sessionStorage") {
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

function persistStorageValue(
  storage: Storage | null,
  area: "apiBaseUrl" | "sessionToken",
  key: string,
  value: string | null,
  failureMessage: string,
): RuntimeWriteResult {
  const current = workbenchConnectionRuntimeStateForActions();
  if (!storage) {
    current.storage[area] = "unavailable";
    current.storageMessages[area] = failureMessage;
    return { ok: false, message: failureMessage };
  }
  try {
    if (value === null) {
      storage.removeItem(key);
    } else {
      storage.setItem(key, value);
    }
    if (storage.getItem(key) !== value) {
      throw new Error("Browser storage did not retain the requested value.");
    }
    current.storage[area] = "available";
    current.storageMessages[area] = null;
    return { ok: true };
  } catch {
    current.storage[area] = "unavailable";
    current.storageMessages[area] = failureMessage;
    return { ok: false, message: failureMessage };
  }
}

export function persistWorkbenchApiBaseUrl(url: string): RuntimeWriteResult {
  return persistStorageValue(
    storageAdapter("localStorage"),
    "apiBaseUrl",
    WORKBENCH_API_BASE_URL_STORAGE_KEY,
    url,
    "This browser could not persist the API base URL. Check browser storage permissions.",
  );
}

export function persistDefaultWorkbenchApiBaseUrl(): RuntimeWriteResult {
  return persistStorageValue(
    storageAdapter("localStorage"),
    "apiBaseUrl",
    WORKBENCH_API_BASE_URL_STORAGE_KEY,
    null,
    "This browser could not clear the saved API base URL. Check browser storage permissions.",
  );
}

export function persistWorkbenchAuthToken(token: string): RuntimeWriteResult {
  return persistStorageValue(
    storageAdapter("sessionStorage"),
    "sessionToken",
    AUTH_TOKEN_STORAGE_KEY,
    token,
    "This browser could not store a session token. Check browser storage permissions.",
  );
}

export function persistClearedWorkbenchAuthToken(): RuntimeWriteResult {
  return persistStorageValue(
    storageAdapter("sessionStorage"),
    "sessionToken",
    AUTH_TOKEN_STORAGE_KEY,
    null,
    "This browser could not clear the session token. Check browser storage permissions.",
  );
}

export function beginWorkbenchConnectionTransition() {
  const current = workbenchConnectionRuntimeStateForActions();
  current.revision += 1;
  current.transitioning = true;
  current.verifiedRevision = null;
  current.authenticationProbeGeneration += 1;
  return current.revision;
}

export function commitWorkbenchApiBaseUrl(url: string) {
  const current = workbenchConnectionRuntimeStateForActions();
  current.apiBaseUrl = url;
  current.authMode = "unknown";
  current.verifiedRevision = null;
}

export function commitWorkbenchAuthToken(token: string | null) {
  const current = workbenchConnectionRuntimeStateForActions();
  current.authToken = token;
  current.authMode = "unknown";
  current.verifiedRevision = null;
}

export function finishWorkbenchConnectionTransition() {
  workbenchConnectionRuntimeStateForActions().transitioning = false;
}
