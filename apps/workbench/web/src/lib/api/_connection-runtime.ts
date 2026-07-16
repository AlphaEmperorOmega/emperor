import {
  readConnectionStorageValue,
  writeConnectionStorageValue,
  WORKBENCH_API_BASE_URL_STORAGE_KEY,
  WORKBENCH_AUTH_TOKEN_STORAGE_KEY,
  type WorkbenchStorageAvailability,
} from "@/lib/api/_connection-storage";

export {
  WORKBENCH_API_BASE_URL_STORAGE_KEY,
  WORKBENCH_AUTH_TOKEN_STORAGE_KEY,
  type WorkbenchStorageAvailability,
} from "@/lib/api/_connection-storage";

export const WORKBENCH_API_URL_ENV_NAME = "NEXT_PUBLIC_WORKBENCH_API_URL";
const WORKBENCH_API_ALLOWED_ORIGINS_ENV_NAME =
  "NEXT_PUBLIC_WORKBENCH_API_ALLOWED_ORIGINS";
const DEFAULT_WORKBENCH_API_BASE_URL = "http://127.0.0.1:9999";

type WorkbenchApiOriginLock = {
  locked: boolean;
  allowedOrigins: Set<string>;
};

type RuntimeStorageStatus = Readonly<{
  apiBaseUrl: WorkbenchStorageAvailability;
  sessionToken: WorkbenchStorageAvailability;
}>;

type RuntimeStorageMessages = Readonly<{
  apiBaseUrl: string | null;
  sessionToken: string | null;
}>;

type WorkbenchConnectionRuntimeState = Readonly<{
  apiBaseUrl: string;
  authToken: string | null;
  revision: number;
  lifecycleActive: boolean;
  transitioning: boolean;
  transitionArea: WorkbenchConnectionStorageArea | null;
  storage: RuntimeStorageStatus;
  storageMessages: RuntimeStorageMessages;
  authMode: "unknown" | "none" | "bearer";
  verifiedRevision: number | null;
  authenticationProbeGeneration: number;
}>;

export type WorkbenchConnectionStorageArea =
  | "apiBaseUrl"
  | "sessionToken";

export type WorkbenchConnectionRuntimeSnapshot = Readonly<{
  apiBaseUrl: string;
  configurationError: string | null;
  hasAuthToken: boolean;
  lifecycleActive: boolean;
  isChanging: boolean;
  transitionArea: WorkbenchConnectionStorageArea | null;
  revision: number;
  authenticationVerified: boolean;
  storage: RuntimeStorageStatus;
  storageMessages: RuntimeStorageMessages;
}>;

type BaseUrlValidationResult =
  | { ok: true; value: string }
  | { ok: false; message: string };

const listeners = new Set<() => void>();
let runtimeState: WorkbenchConnectionRuntimeState | null = null;
let runtimeSnapshot: WorkbenchConnectionRuntimeSnapshot | null = null;

function workbenchConnectionChangedError() {
  const error = new Error(
    "Workbench connection changed while the request was in flight.",
  );
  error.name = "WorkbenchConnectionChangedError";
  return error;
}

function normalizeWorkbenchApiBaseUrl(url: string) {
  const trimmedUrl = url.trim();
  if (!trimmedUrl) {
    return null;
  }
  try {
    const parsedUrl = new URL(trimmedUrl);
    if (parsedUrl.protocol !== "http:" && parsedUrl.protocol !== "https:") {
      return null;
    }
    if (
      parsedUrl.username ||
      parsedUrl.password ||
      parsedUrl.search ||
      parsedUrl.hash
    ) {
      return null;
    }
  } catch {
    return null;
  }
  return trimmedUrl.replace(/\/+$/, "");
}

const configuredApiUrlValue = process.env.NEXT_PUBLIC_WORKBENCH_API_URL ?? "";
const configuredApiBaseUrl = normalizeWorkbenchApiBaseUrl(configuredApiUrlValue);
export const WORKBENCH_API_BASE_URL =
  configuredApiBaseUrl ?? DEFAULT_WORKBENCH_API_BASE_URL;
const configurationError =
  configuredApiUrlValue.trim() && !configuredApiBaseUrl
    ? `${WORKBENCH_API_URL_ENV_NAME} must be an absolute HTTP(S) URL without credentials, query, or fragment.`
    : null;

function originFromWorkbenchApiBaseUrl(url: string) {
  const normalizedUrl = normalizeWorkbenchApiBaseUrl(url);
  return normalizedUrl ? new URL(normalizedUrl).origin : null;
}

function parseAllowedOriginValues(rawValue: string) {
  const trimmedValue = rawValue.trim();
  if (!trimmedValue) {
    return [];
  }
  if (trimmedValue.startsWith("[")) {
    try {
      const parsed = JSON.parse(trimmedValue);
      if (Array.isArray(parsed)) {
        return parsed.map(String);
      }
    } catch {
      return [];
    }
  }
  return trimmedValue.split(",");
}

function parseWorkbenchApiAllowedOrigins(rawValue: string) {
  const origins = parseAllowedOriginValues(rawValue)
    .map((value) => originFromWorkbenchApiBaseUrl(value))
    .filter((origin): origin is string => Boolean(origin));
  return Array.from(new Set(origins));
}

function isLocalWorkbenchApiOrigin(origin: string) {
  try {
    const hostname = new URL(origin).hostname;
    return ["localhost", "127.0.0.1", "0.0.0.0", "::1", "[::1]"].includes(
      hostname,
    );
  } catch {
    return false;
  }
}

function createWorkbenchApiOriginLock(): WorkbenchApiOriginLock {
  if (configurationError) {
    return { locked: true, allowedOrigins: new Set() };
  }
  const explicitAllowedOrigins =
    process.env.NEXT_PUBLIC_WORKBENCH_API_ALLOWED_ORIGINS ?? "";
  if (explicitAllowedOrigins.trim()) {
    return {
      locked: true,
      allowedOrigins: new Set(
        parseWorkbenchApiAllowedOrigins(explicitAllowedOrigins),
      ),
    };
  }
  const configuredOrigin = originFromWorkbenchApiBaseUrl(WORKBENCH_API_BASE_URL);
  if (configuredOrigin && !isLocalWorkbenchApiOrigin(configuredOrigin)) {
    return { locked: true, allowedOrigins: new Set([configuredOrigin]) };
  }
  return { locked: false, allowedOrigins: new Set() };
}

const workbenchApiOriginLock = createWorkbenchApiOriginLock();

function getWorkbenchApiAllowedOrigins() {
  return Array.from(workbenchApiOriginLock.allowedOrigins);
}

function isWorkbenchApiBaseUrlAllowed(url: string) {
  if (!workbenchApiOriginLock.locked) {
    return true;
  }
  const origin = originFromWorkbenchApiBaseUrl(url);
  return origin !== null && workbenchApiOriginLock.allowedOrigins.has(origin);
}

function disallowedBaseUrlMessage(url: string) {
  const origin = originFromWorkbenchApiBaseUrl(url) ?? "the requested origin";
  const allowedOrigins = getWorkbenchApiAllowedOrigins();
  const allowedText =
    allowedOrigins.length > 0 ? allowedOrigins.join(", ") : "no allowed origins";
  return (
    `Workbench API origin ${origin} is not allowed by this build. ` +
    `Configure ${WORKBENCH_API_ALLOWED_ORIGINS_ENV_NAME} or rebuild with ` +
    `${WORKBENCH_API_URL_ENV_NAME}. Allowed origins: ${allowedText}.`
  );
}

export function validateWorkbenchApiBaseUrl(
  url: string,
): BaseUrlValidationResult {
  const normalizedUrl = normalizeWorkbenchApiBaseUrl(url);
  if (!normalizedUrl) {
    return {
      ok: false,
      message:
        "Workbench API base URL must be absolute HTTP(S) without credentials, query, or fragment.",
    };
  }
  if (!isWorkbenchApiBaseUrlAllowed(normalizedUrl)) {
    return { ok: false, message: disallowedBaseUrlMessage(normalizedUrl) };
  }
  return { ok: true, value: normalizedUrl };
}

function assertWorkbenchApiBaseUrlAllowed(url: string) {
  if (configurationError) {
    throw new Error(configurationError);
  }
  const result = validateWorkbenchApiBaseUrl(url);
  if (!result.ok) {
    throw new Error(result.message);
  }
}

function readStoredApiBaseUrl() {
  const read = readConnectionStorageValue(
    "localStorage",
    WORKBENCH_API_BASE_URL_STORAGE_KEY,
    "Local storage is unavailable. The configured API base URL is in use.",
  );
  if (read.availability === "unavailable" || !read.value) {
    return {
      value: WORKBENCH_API_BASE_URL,
      availability: read.availability,
      message: read.message,
    };
  }

  const validation = validateWorkbenchApiBaseUrl(read.value);
  if (!validation.ok) {
    const removed = writeConnectionStorageValue(
      "localStorage",
      WORKBENCH_API_BASE_URL_STORAGE_KEY,
      null,
    );
    return {
      value: WORKBENCH_API_BASE_URL,
      availability: removed ? ("available" as const) : ("unavailable" as const),
      message: removed
        ? "The saved API base URL was invalid or disallowed and was removed. The configured default is in use."
        : "The saved API base URL was invalid or disallowed, but this browser could not remove it. The configured default is in use.",
    };
  }

  if (validation.value !== read.value) {
    const persisted = writeConnectionStorageValue(
      "localStorage",
      WORKBENCH_API_BASE_URL_STORAGE_KEY,
      validation.value,
    );
    return {
      value: validation.value,
      availability: persisted
        ? ("available" as const)
        : ("unavailable" as const),
      message: persisted
        ? null
        : "This browser could not persist the normalized API base URL. The normalized value is in use for this session.",
    };
  }

  return {
    value: validation.value,
    availability: "available" as const,
    message: null,
  };
}

function readStoredAuthToken() {
  return readConnectionStorageValue(
    "sessionStorage",
    WORKBENCH_AUTH_TOKEN_STORAGE_KEY,
    "Session storage is unavailable. Bearer sign-in and logout cannot be completed in this browser context.",
  );
}

function authenticationVerified(current: WorkbenchConnectionRuntimeState) {
  return (
    current.verifiedRevision === current.revision &&
    (current.authMode === "none" ||
      (current.authMode === "bearer" && current.authToken !== null))
  );
}

function snapshotFromState(
  current: WorkbenchConnectionRuntimeState,
): WorkbenchConnectionRuntimeSnapshot {
  return Object.freeze({
    apiBaseUrl: current.apiBaseUrl,
    configurationError,
    hasAuthToken: current.authToken !== null,
    lifecycleActive: current.lifecycleActive,
    isChanging: current.transitioning,
    transitionArea: current.transitionArea,
    revision: current.revision,
    authenticationVerified: authenticationVerified(current),
    storage: current.storage,
    storageMessages: current.storageMessages,
  });
}

function freezeState(
  current: Omit<
    WorkbenchConnectionRuntimeState,
    "storage" | "storageMessages"
  > & {
    storage: RuntimeStorageStatus;
    storageMessages: RuntimeStorageMessages;
  },
): WorkbenchConnectionRuntimeState {
  return Object.freeze({
    ...current,
    storage: Object.freeze({ ...current.storage }),
    storageMessages: Object.freeze({ ...current.storageMessages }),
  });
}

function createRuntimeState(
  revision: number,
  lifecycleActive = false,
): WorkbenchConnectionRuntimeState {
  const api = readStoredApiBaseUrl();
  const auth = readStoredAuthToken();
  return freezeState({
    apiBaseUrl: api.value,
    authToken: auth.value,
    revision,
    lifecycleActive,
    transitioning: false,
    transitionArea: null,
    storage: {
      apiBaseUrl: api.availability,
      sessionToken: auth.availability,
    },
    storageMessages: {
      apiBaseUrl: api.message,
      sessionToken: auth.message,
    },
    authMode: "unknown",
    verifiedRevision: null,
    authenticationProbeGeneration: 0,
  });
}

function state() {
  if (!runtimeState) {
    runtimeState = createRuntimeState(0);
    runtimeSnapshot = snapshotFromState(runtimeState);
  }
  return runtimeState;
}

function publish(next: WorkbenchConnectionRuntimeState) {
  runtimeState = next;
  runtimeSnapshot = snapshotFromState(next);
  for (const listener of Array.from(listeners)) {
    listener();
  }
}

function replaceState(
  update: (current: WorkbenchConnectionRuntimeState) => WorkbenchConnectionRuntimeState,
) {
  const current = state();
  const next = update(current);
  if (next !== current) {
    publish(next);
  }
  return next;
}

/** Reload browser-backed identity at a provider lifecycle boundary. */
export function loadWorkbenchConnectionRuntime() {
  const nextRevision = runtimeState ? runtimeState.revision + 1 : 0;
  publish(createRuntimeState(nextRevision, true));
  return getWorkbenchConnectionRuntimeSnapshot();
}

export function deactivateWorkbenchConnectionRuntime() {
  replaceState((current) =>
    freezeState({
      ...current,
      lifecycleActive: false,
      authMode: "unknown",
      verifiedRevision: null,
      authenticationProbeGeneration:
        current.authenticationProbeGeneration + 1,
      storage: current.storage,
      storageMessages: current.storageMessages,
    }),
  );
}

export function subscribeWorkbenchConnectionRuntime(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getWorkbenchConnectionRuntimeSnapshot() {
  state();
  return runtimeSnapshot!;
}

const serverRuntimeSnapshot = Object.freeze({
  apiBaseUrl: WORKBENCH_API_BASE_URL,
  configurationError,
  hasAuthToken: false,
  lifecycleActive: false,
  isChanging: false,
  transitionArea: null,
  revision: 0,
  authenticationVerified: false,
  storage: Object.freeze({
    apiBaseUrl: "unavailable" as const,
    sessionToken: "unavailable" as const,
  }),
  storageMessages: Object.freeze({
    apiBaseUrl: null,
    sessionToken: null,
  }),
});

export function getWorkbenchConnectionRuntimeServerSnapshot() {
  return serverRuntimeSnapshot;
}

export function reportWorkbenchConnectionStorage(
  area: WorkbenchConnectionStorageArea,
  availability: WorkbenchStorageAvailability,
  message: string | null,
) {
  replaceState((current) =>
    freezeState({
      ...current,
      storage: { ...current.storage, [area]: availability },
      storageMessages: { ...current.storageMessages, [area]: message },
    }),
  );
}

export function beginWorkbenchConnectionRuntimeTransition(
  area: WorkbenchConnectionStorageArea,
) {
  replaceState((current) =>
    freezeState({
      ...current,
      revision: current.revision + 1,
      transitioning: true,
      transitionArea: area,
      storage: { ...current.storage, [area]: "available" },
      storageMessages: { ...current.storageMessages, [area]: null },
      authMode: "unknown",
      verifiedRevision: null,
      authenticationProbeGeneration:
        current.authenticationProbeGeneration + 1,
    }),
  );
}

export function completeWorkbenchConnectionRuntimeTransition(
  area: WorkbenchConnectionStorageArea,
  identity: string | null,
) {
  replaceState((current) =>
    freezeState({
      ...current,
      apiBaseUrl:
        area === "apiBaseUrl" ? (identity ?? WORKBENCH_API_BASE_URL) : current.apiBaseUrl,
      authToken: area === "sessionToken" ? identity : current.authToken,
      transitioning: false,
      transitionArea: null,
      storage: current.storage,
      storageMessages: current.storageMessages,
    }),
  );
}

export function abortWorkbenchConnectionRuntimeTransition(
  area: WorkbenchConnectionStorageArea,
  rollbackMessage: string | null,
) {
  replaceState((current) =>
    freezeState({
      ...current,
      transitioning: false,
      transitionArea: null,
      storage: {
        ...current.storage,
        [area]: rollbackMessage ? "unavailable" : "available",
      },
      storageMessages: {
        ...current.storageMessages,
        [area]: rollbackMessage,
      },
    }),
  );
}

export function observeWorkbenchAuthMode(mode: "none" | "bearer") {
  replaceState((current) => {
    const verifiedRevision = mode === "none" ? current.revision : null;
    if (
      current.authMode === mode &&
      current.verifiedRevision === verifiedRevision
    ) {
      return current;
    }
    return freezeState({
      ...current,
      authMode: mode,
      verifiedRevision,
      storage: current.storage,
      storageMessages: current.storageMessages,
    });
  });
}

export function clearWorkbenchAuthenticationObservation() {
  replaceState((current) => {
    if (current.authMode === "unknown" && current.verifiedRevision === null) {
      return current;
    }
    return freezeState({
      ...current,
      authMode: "unknown",
      verifiedRevision: null,
      authenticationProbeGeneration:
        current.authenticationProbeGeneration + 1,
      storage: current.storage,
      storageMessages: current.storageMessages,
    });
  });
}

export function confirmWorkbenchAuthentication(
  revision: number,
  probeGeneration: number | null,
) {
  replaceState((current) => {
    if (
      probeGeneration === null ||
      current.transitioning ||
      current.revision !== revision ||
      current.authenticationProbeGeneration !== probeGeneration ||
      current.authMode !== "bearer"
    ) {
      return current;
    }
    return freezeState({
      ...current,
      verifiedRevision: revision,
      storage: current.storage,
      storageMessages: current.storageMessages,
    });
  });
}

export function captureWorkbenchConnectionRequest(
  path: string,
  authenticationProbe = false,
) {
  let current = state();
  const isPublicRead = path === "/health" || path === "/capabilities";
  if (
    current.transitioning &&
    !(current.transitionArea === "sessionToken" && isPublicRead)
  ) {
    throw workbenchConnectionChangedError();
  }
  assertWorkbenchApiBaseUrlAllowed(current.apiBaseUrl);
  let probeGeneration: number | null = null;

  if (authenticationProbe && current.authMode !== "none") {
    current = replaceState((snapshot) =>
      freezeState({
        ...snapshot,
        authenticationProbeGeneration:
          snapshot.authenticationProbeGeneration + 1,
        verifiedRevision: null,
        storage: snapshot.storage,
        storageMessages: snapshot.storageMessages,
      }),
    );
    probeGeneration = current.authenticationProbeGeneration;
  }

  if (
    !isPublicRead &&
    !authenticationProbe &&
    !authenticationVerified(current)
  ) {
    throw new Error(
      "Workbench authentication must be verified before protected data can be read.",
    );
  }

  return {
    apiBaseUrl: current.apiBaseUrl,
    authToken: current.authToken,
    revision: current.revision,
    authenticationProbeGeneration: probeGeneration,
  } as const;
}

export function assertWorkbenchConnectionRequestCurrent(revision: number) {
  const current = state();
  if (current.transitioning || current.revision !== revision) {
    throw workbenchConnectionChangedError();
  }
}
