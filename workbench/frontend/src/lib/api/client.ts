import { z } from "zod";
import {
  assertWorkbenchConnectionRequestCurrent,
  captureWorkbenchConnectionRequest,
  confirmWorkbenchAuthentication,
} from "@/lib/api/_connection-runtime";

export const WORKBENCH_MUTATION_HEADER_NAME = "X-Workbench-Mutation";
export const WORKBENCH_MUTATION_HEADER_VALUE = "true";

const errorBodySchema = z.object({ detail: z.unknown() }).partial();

type ApiErrorInit = {
  status: number;
  method: string;
  path: string;
  detail: string;
  baseUrl: string;
  authToken: string | null;
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

  constructor({ status, method, path, detail, baseUrl, authToken }: ApiErrorInit) {
    const messageDetail = redactBearerToken(detail || "Request failed", authToken);
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
  method: string,
  authToken: string | null,
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
  headers.delete("authorization");
  if (authToken) {
    headers.set("Authorization", `Bearer ${authToken}`);
  }
  if (["GET", "HEAD", "OPTIONS"].includes(method)) {
    headers.delete(WORKBENCH_MUTATION_HEADER_NAME);
  } else {
    headers.set(WORKBENCH_MUTATION_HEADER_NAME, WORKBENCH_MUTATION_HEADER_VALUE);
  }
  return headers;
}

function redactBearerToken(value: string, authToken: string | null) {
  if (!authToken) {
    return value;
  }
  return value.split(authToken).join("[REDACTED]");
}

function redactedError(error: unknown, authToken: string | null) {
  const safeError = new Error(
    redactBearerToken(
      error instanceof Error ? error.message : String(error),
      authToken,
    ),
  );
  safeError.name = error instanceof Error ? error.name : "Error";
  return safeError;
}

function throwIfAborted(signal?: AbortSignal | null) {
  if (!signal?.aborted) {
    return;
  }
  throw new DOMException("The operation was aborted.", "AbortError");
}

async function parseJsonResponse<TSchema extends z.ZodTypeAny>(
  {
    path,
    method,
    apiBaseUrl,
    response,
    schema,
    requestRevision,
    authToken,
  }: {
    path: string;
    method: string;
    apiBaseUrl: string;
    response: Response;
    schema: TSchema;
    requestRevision: number;
    authToken: string | null;
  },
): Promise<z.output<TSchema>> {
  assertWorkbenchConnectionRequestCurrent(requestRevision);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const errorBody = await response.json();
      const payload = errorBodySchema.safeParse(errorBody);
      if (payload.success && payload.data.detail) {
        detail = detailText(payload.data.detail);
      }
    } catch {
      // Response was not JSON; keep status text.
    }
    assertWorkbenchConnectionRequestCurrent(requestRevision);
    throw new ApiError({
      status: response.status,
      method,
      path,
      detail,
      baseUrl: apiBaseUrl,
      authToken,
    });
  }
  let payload: unknown;
  try {
    payload = await response.json();
  } catch (error) {
    assertWorkbenchConnectionRequestCurrent(requestRevision);
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw redactedError(error, authToken);
  }
  assertWorkbenchConnectionRequestCurrent(requestRevision);
  let parsed: z.SafeParseReturnType<unknown, z.output<TSchema>>;
  try {
    parsed = schema.safeParse(payload);
  } catch (error) {
    throw redactedError(error, authToken);
  }
  if (!parsed.success) {
    throw new Error(
      redactBearerToken(
        `Invalid API response for ${method} ${path} from ${apiBaseUrl}: ${formatZodIssues(
          parsed.error.issues,
        )}`,
        authToken,
      ),
    );
  }
  return parsed.data;
}

export async function requestJson<TSchema extends z.ZodTypeAny>(
  path: string,
  schema: TSchema,
  init?: RequestInit,
  policy: { authenticationProbe?: boolean } = {},
): Promise<z.output<TSchema>> {
  throwIfAborted(init?.signal);
  const method = requestMethod(init);
  const connection = captureWorkbenchConnectionRequest(
    path,
    policy.authenticationProbe,
  );
  const apiBaseUrl = connection.apiBaseUrl;
  let request: RequestInit;
  try {
    request = {
      ...init,
      headers: requestHeaders(method, connection.authToken, init?.headers),
    };
  } catch (error) {
    throw redactedError(error, connection.authToken);
  }
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}${path}`, request);
  } catch (error) {
    assertWorkbenchConnectionRequestCurrent(connection.revision);
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw redactedError(error, connection.authToken);
  }
  const parsed = await parseJsonResponse({
    path,
    method,
    apiBaseUrl,
    response,
    schema,
    requestRevision: connection.revision,
    authToken: connection.authToken,
  });
  if (policy.authenticationProbe) {
    confirmWorkbenchAuthentication(
      connection.revision,
      connection.authenticationProbeGeneration,
    );
  }
  return parsed;
}

export async function requestMultipartJson<TSchema extends z.ZodTypeAny>(
  path: string,
  schema: TSchema,
  formData: FormData,
  init?: Omit<RequestInit, "body">,
): Promise<z.output<TSchema>> {
  throwIfAborted(init?.signal);
  const method = requestMethod({ method: init?.method ?? "POST" });
  const connection = captureWorkbenchConnectionRequest(path);
  const apiBaseUrl = connection.apiBaseUrl;
  let request: RequestInit;
  try {
    request = {
      ...init,
      method,
      body: formData,
      headers: requestHeaders(method, connection.authToken, init?.headers, null),
    };
  } catch (error) {
    throw redactedError(error, connection.authToken);
  }
  let response: Response;
  try {
    response = await fetch(`${apiBaseUrl}${path}`, request);
  } catch (error) {
    assertWorkbenchConnectionRequestCurrent(connection.revision);
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw redactedError(error, connection.authToken);
  }
  return parseJsonResponse({
    path,
    method,
    apiBaseUrl,
    response,
    schema,
    requestRevision: connection.revision,
    authToken: connection.authToken,
  });
}
