import { z } from "zod";

import { getSessionAuthToken } from "@/lib/auth-token";

export const VIEWER_API_URL_ENV_NAME = "NEXT_PUBLIC_VIEWER_API_URL";
export const DEFAULT_VIEWER_API_BASE_URL = "http://127.0.0.1:9999";
export const VIEWER_API_BASE_URL_STORAGE_KEY = "emperor.viewer.apiBaseUrl";
let runtimeViewerApiBaseUrlOverride: string | null | undefined;

export function normalizeViewerApiBaseUrl(url: string) {
  const trimmedUrl = url.trim();
  if (!trimmedUrl) {
    return null;
  }
  try {
    const parsedUrl = new URL(trimmedUrl);
    if (parsedUrl.protocol !== "http:" && parsedUrl.protocol !== "https:") {
      return null;
    }
    if (parsedUrl.search || parsedUrl.hash) {
      return null;
    }
  } catch {
    return null;
  }
  return trimmedUrl.replace(/\/+$/, "");
}

function defaultViewerApiBaseUrl() {
  return (
    normalizeViewerApiBaseUrl(process.env.NEXT_PUBLIC_VIEWER_API_URL ?? "") ??
    DEFAULT_VIEWER_API_BASE_URL
  );
}

export const VIEWER_API_BASE_URL = defaultViewerApiBaseUrl();

function getLocalStorage() {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const storage = window.localStorage;
    return typeof storage?.getItem === "function" ? storage : null;
  } catch {
    return null;
  }
}

function readStoredViewerApiBaseUrl() {
  const storage = getLocalStorage();
  try {
    const storedUrl = storage?.getItem(VIEWER_API_BASE_URL_STORAGE_KEY);
    if (!storedUrl) {
      return null;
    }
    const normalizedUrl = normalizeViewerApiBaseUrl(storedUrl);
    if (!normalizedUrl) {
      try {
        storage?.removeItem?.(VIEWER_API_BASE_URL_STORAGE_KEY);
      } catch {
        // Invalid persisted values should not break API setup.
      }
      return null;
    }
    if (storedUrl !== normalizedUrl) {
      try {
        storage?.setItem?.(VIEWER_API_BASE_URL_STORAGE_KEY, normalizedUrl);
      } catch {
        // Normalization persistence is best-effort.
      }
    }
    return normalizedUrl;
  } catch {
    return null;
  }
}

export function getViewerApiBaseUrl() {
  if (runtimeViewerApiBaseUrlOverride !== undefined) {
    return runtimeViewerApiBaseUrlOverride ?? VIEWER_API_BASE_URL;
  }
  return (
    readStoredViewerApiBaseUrl() ??
    VIEWER_API_BASE_URL
  );
}

export function setViewerApiBaseUrl(url: string) {
  const normalizedUrl = normalizeViewerApiBaseUrl(url);
  if (!normalizedUrl) {
    throw new Error(
      "Viewer API base URL must be an absolute http:// or https:// URL without a query string or fragment.",
    );
  }
  runtimeViewerApiBaseUrlOverride = normalizedUrl;
  const storage = getLocalStorage();
  try {
    storage?.setItem?.(VIEWER_API_BASE_URL_STORAGE_KEY, normalizedUrl);
  } catch {
    // Runtime switching should continue even if persistence is unavailable.
  }
  return normalizedUrl;
}

export function resetViewerApiBaseUrl() {
  const storage = getLocalStorage();
  let clearedStoredUrl = !storage;
  try {
    storage?.removeItem?.(VIEWER_API_BASE_URL_STORAGE_KEY);
    clearedStoredUrl = true;
  } catch {
    // Clearing persistence is best-effort for locked-down browser contexts.
  }
  runtimeViewerApiBaseUrlOverride = clearedStoredUrl ? undefined : null;
  return VIEWER_API_BASE_URL;
}

const errorBodySchema = z.object({ detail: z.unknown() }).partial();

type ApiErrorInit = {
  status: number;
  method: string;
  path: string;
  detail: string;
  baseUrl: string;
};

export type UnauthorizedApiError = Error & {
  status: 401;
  method: string;
  path: string;
  detail: string;
};

class ApiError extends Error {
  readonly status: number;
  readonly method: string;
  readonly path: string;
  readonly detail: string;

  constructor({ status, method, path, detail, baseUrl }: ApiErrorInit) {
    const messageDetail = detail || "Request failed";
    super(
      `${method} ${path} from ${baseUrl} failed with ${status}: ${messageDetail}`,
    );
    this.name = "ApiError";
    this.status = status;
    this.method = method;
    this.path = path;
    this.detail = messageDetail;
  }
}

export function isUnauthorizedApiError(error: unknown): error is UnauthorizedApiError {
  return error instanceof ApiError && error.status === 401;
}

function requestMethod(init?: RequestInit) {
  return String(init?.method ?? "GET").toUpperCase();
}

function formatIssuePath(path: Array<string | number>) {
  return path.length > 0 ? path.map(String).join(".") : "<root>";
}

function formatZodIssues(issues: z.ZodIssue[]) {
  const visibleIssues = issues
    .slice(0, 5)
    .map((issue) => `${formatIssuePath(issue.path)}: ${issue.message}`);
  if (issues.length > visibleIssues.length) {
    visibleIssues.push(`${issues.length - visibleIssues.length} more issue(s)`);
  }
  return visibleIssues.join("; ");
}

function detailText(detail: unknown) {
  if (typeof detail === "string") {
    return detail;
  }
  if (detail === null || detail === undefined) {
    return "";
  }
  try {
    return JSON.stringify(detail);
  } catch {
    return String(detail);
  }
}

function requestHeaders(
  initHeaders?: HeadersInit,
  contentType: string | null = "application/json",
) {
  const headers = new Headers(
    contentType ? { "content-type": contentType } : undefined,
  );
  if (initHeaders) {
    new Headers(initHeaders).forEach((value, key) => {
      headers.set(key, value);
    });
  }
  const token = getSessionAuthToken();
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  return headers;
}

async function parseJsonResponse<TSchema extends z.ZodTypeAny>(
  {
    path,
    method,
    apiBaseUrl,
    response,
    schema,
  }: {
    path: string;
    method: string;
    apiBaseUrl: string;
    response: Response;
    schema: TSchema;
  },
): Promise<z.output<TSchema>> {
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = errorBodySchema.safeParse(await response.json());
      if (payload.success && payload.data.detail) {
        detail = detailText(payload.data.detail);
      }
    } catch {
      // Response was not JSON; keep status text.
    }
    throw new ApiError({
      status: response.status,
      method,
      path,
      detail,
      baseUrl: apiBaseUrl,
    });
  }
  const payload = await response.json();
  const parsed = schema.safeParse(payload);
  if (!parsed.success) {
    throw new Error(
      `Invalid API response for ${method} ${path} from ${apiBaseUrl}: ${formatZodIssues(
        parsed.error.issues,
      )}`,
    );
  }
  return parsed.data;
}

export async function requestJson<TSchema extends z.ZodTypeAny>(
  path: string,
  schema: TSchema,
  init?: RequestInit,
): Promise<z.output<TSchema>> {
  const method = requestMethod(init);
  const request = {
    ...init,
    headers: requestHeaders(init?.headers),
  };
  const apiBaseUrl = getViewerApiBaseUrl();
  const response = await fetch(`${apiBaseUrl}${path}`, request);
  return parseJsonResponse({ path, method, apiBaseUrl, response, schema });
}

export async function requestMultipartJson<TSchema extends z.ZodTypeAny>(
  path: string,
  schema: TSchema,
  formData: FormData,
  init?: Omit<RequestInit, "body">,
): Promise<z.output<TSchema>> {
  const method = requestMethod({ method: init?.method ?? "POST" });
  const request = {
    ...init,
    method,
    body: formData,
    headers: requestHeaders(init?.headers, null),
  };
  const apiBaseUrl = getViewerApiBaseUrl();
  const response = await fetch(`${apiBaseUrl}${path}`, request);
  return parseJsonResponse({ path, method, apiBaseUrl, response, schema });
}
