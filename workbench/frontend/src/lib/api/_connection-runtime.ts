export const WORKBENCH_API_URL_ENV_NAME = "NEXT_PUBLIC_WORKBENCH_API_URL";
const WORKBENCH_API_ALLOWED_ORIGINS_ENV_NAME =
  "NEXT_PUBLIC_WORKBENCH_API_ALLOWED_ORIGINS";
const DEFAULT_WORKBENCH_API_BASE_URL = "http://127.0.0.1:9999";
export const WORKBENCH_API_BASE_URL_STORAGE_KEY =
  "emperor.workbench.apiBaseUrl";
export const WORKBENCH_AUTH_TOKEN_STORAGE_KEY =
  "emperor.workbench.authToken";

export type WorkbenchStorageAvailability = "available" | "unavailable";

type WorkbenchApiOriginLock = {
  locked: boolean;
  allowedOrigins: Set<string>;
};

type RuntimeStorageStatus = {
  apiBaseUrl: WorkbenchStorageAvailability;
  sessionToken: WorkbenchStorageAvailability;
};

type RuntimeStorageMessages = {
  apiBaseUrl: string | null;
  sessionToken: string | null;
};

type WorkbenchConnectionRuntimeState = {
  apiBaseUrl: string;
  authToken: string | null;
  revision: number;
  transitioning: boolean;
  storage: RuntimeStorageStatus;
  storageMessages: RuntimeStorageMessages;
  authMode: "unknown" | "none" | "bearer";
  verifiedRevision: number | null;
  authenticationProbeGeneration: number;
};

export type WorkbenchConnectionRuntimeSnapshot = Readonly<{
  apiBaseUrl: string;
  configurationError: string | null;
  hasAuthToken: boolean;
  storage: Readonly<RuntimeStorageStatus>;
  storageMessages: Readonly<RuntimeStorageMessages>;
}>;

type BaseUrlValidationResult =
  | { ok: true; value: string }
  | { ok: false; message: string };

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
    if (parsedUrl.username || parsedUrl.password || parsedUrl.search || parsedUrl.hash) {
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
let runtimeState: WorkbenchConnectionRuntimeState | null = null;

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

const localStorageAdapter = () => storageAdapter("localStorage");
const sessionStorageAdapter = () => storageAdapter("sessionStorage");

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

function readStoredApiBaseUrl(storage: Storage | null) {
  if (!storage) {
    return {
      value: WORKBENCH_API_BASE_URL,
      availability: "unavailable" as const,
      message:
        "Local storage is unavailable. API base URL changes cannot be persisted in this browser context.",
    };
  }
  try {
    const stored = storage.getItem(WORKBENCH_API_BASE_URL_STORAGE_KEY);
    if (!stored) {
      return {
        value: WORKBENCH_API_BASE_URL,
        availability: "available" as const,
        message: null,
      };
    }
    const validation = validateWorkbenchApiBaseUrl(stored);
    if (!validation.ok) {
      try {
        storage.removeItem(WORKBENCH_API_BASE_URL_STORAGE_KEY);
        return {
          value: WORKBENCH_API_BASE_URL,
          availability: "available" as const,
          message:
            "The saved API base URL was invalid or disallowed and was removed. The configured default is in use.",
        };
      } catch {
        return {
          value: WORKBENCH_API_BASE_URL,
          availability: "unavailable" as const,
          message:
            "The saved API base URL was invalid or disallowed, but this browser could not remove it. The configured default is in use.",
        };
      }
    }
    if (validation.value !== stored) {
      try {
        storage.setItem(WORKBENCH_API_BASE_URL_STORAGE_KEY, validation.value);
      } catch {
        return {
          value: validation.value,
          availability: "unavailable" as const,
          message:
            "This browser could not persist the normalized API base URL. The normalized value is in use for this session.",
        };
      }
    }
    return {
      value: validation.value,
      availability: "available" as const,
      message: null,
    };
  } catch {
    return {
      value: WORKBENCH_API_BASE_URL,
      availability: "unavailable" as const,
      message:
        "Local storage is unavailable. The configured API base URL is in use.",
    };
  }
}

function readStoredAuthToken(storage: Storage | null) {
  if (!storage) {
    return {
      value: null,
      availability: "unavailable" as const,
      message:
        "Session storage is unavailable. Bearer sign-in and logout cannot be completed in this browser context.",
    };
  }
  try {
    return {
      value: storage.getItem(WORKBENCH_AUTH_TOKEN_STORAGE_KEY),
      availability: "available" as const,
      message: null,
    };
  } catch {
    return {
      value: null,
      availability: "unavailable" as const,
      message:
        "Session storage is unavailable. Bearer sign-in and logout cannot be completed in this browser context.",
    };
  }
}

/** Reloads browser-backed identity once at the Workbench provider boundary. */
export function loadWorkbenchConnectionRuntime() {
  runtimeState = createRuntimeState(runtimeState?.revision ?? 0);
  return workbenchConnectionRuntimeSnapshot();
}

function createRuntimeState(revision: number): WorkbenchConnectionRuntimeState {
  const api = readStoredApiBaseUrl(localStorageAdapter());
  const auth = readStoredAuthToken(sessionStorageAdapter());
  return {
    apiBaseUrl: api.value,
    authToken: auth.value,
    revision,
    transitioning: false,
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
  };
}

function state() {
  runtimeState ??= createRuntimeState(0);
  return runtimeState;
}

export function workbenchConnectionRuntimeSnapshot(): WorkbenchConnectionRuntimeSnapshot {
  const current = state();
  return {
    apiBaseUrl: current.apiBaseUrl,
    configurationError,
    hasAuthToken: current.authToken !== null,
    storage: { ...current.storage },
    storageMessages: { ...current.storageMessages },
  };
}

/** Scoped mutable capability for the one lazy Workbench Connection Implementation. */
export function withWorkbenchConnectionRuntimeTransition<Result>(
  implementation: (current: WorkbenchConnectionRuntimeState) => Result,
) {
  return implementation(state());
}

export function isWorkbenchAuthenticationVerified() {
  const current = state();
  return (
    current.verifiedRevision === current.revision &&
    (current.authMode === "none" ||
      (current.authMode === "bearer" && current.authToken !== null))
  );
}

export function observeWorkbenchAuthMode(mode: "none" | "bearer") {
  const current = state();
  if (current.authMode !== mode) {
    current.authMode = mode;
    current.verifiedRevision = null;
  }
  if (mode === "none") {
    current.verifiedRevision = current.revision;
  }
}

export function clearWorkbenchAuthenticationObservation() {
  const current = state();
  if (current.authMode !== "unknown" || current.verifiedRevision !== null) {
    current.authenticationProbeGeneration += 1;
  }
  current.authMode = "unknown";
  current.verifiedRevision = null;
}

export function confirmWorkbenchAuthentication(
  revision: number,
  probeGeneration: number | null,
) {
  const current = state();
  if (
    probeGeneration !== null &&
    !current.transitioning &&
    current.revision === revision &&
    current.authenticationProbeGeneration === probeGeneration &&
    current.authMode === "bearer"
  ) {
    current.verifiedRevision = revision;
  }
}

export function captureWorkbenchConnectionRequest(
  path: string,
  authenticationProbe = false,
) {
  const current = state();
  if (current.transitioning) {
    throw workbenchConnectionChangedError();
  }
  assertWorkbenchApiBaseUrlAllowed(current.apiBaseUrl);
  const isPublicRead = path === "/health" || path === "/capabilities";
  let probeGeneration: number | null = null;
  if (authenticationProbe && current.authMode !== "none") {
    current.authenticationProbeGeneration += 1;
    current.verifiedRevision = null;
    probeGeneration = current.authenticationProbeGeneration;
  }
  const protectedAccessReady = isWorkbenchAuthenticationVerified();
  if (!isPublicRead && !authenticationProbe && !protectedAccessReady) {
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
