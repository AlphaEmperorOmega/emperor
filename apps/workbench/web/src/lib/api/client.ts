import {
  assertWorkbenchConnectionRequestCurrent,
  captureWorkbenchConnectionRequest,
  confirmWorkbenchAuthentication,
} from "@/lib/api/_connection-runtime";

export const WORKBENCH_MUTATION_HEADER_NAME = "X-Workbench-Mutation";
export const WORKBENCH_MUTATION_HEADER_VALUE = "true";
export const IDEMPOTENCY_HEADER_NAME = "Idempotency-Key";

export type MutationRequestOptions = Readonly<{
  idempotencyKey: string;
}>;

type ResponseValidationIssue = Readonly<{
  path: readonly PropertyKey[];
  message: string;
}>;

type ResponseValidationResult<T> =
  | Readonly<{ success: true; data: T }>
  | Readonly<{
      success: false;
      error: Readonly<{ issues: readonly ResponseValidationIssue[] }>;
    }>;

export type ResponseSchema<T> = Readonly<{
  safeParse: (value: unknown) => ResponseValidationResult<T>;
}>;

export function createMutationRequestOptions(): MutationRequestOptions {
  return { idempotencyKey: mutationRequestId() };
}

function mutationRequestId() {
  if (typeof globalThis.crypto?.randomUUID === "function") {
    return globalThis.crypto.randomUUID();
  }
  const bytes = new Uint8Array(16);
  globalThis.crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0"));
  return [
    hex.slice(0, 4).join(""),
    hex.slice(4, 6).join(""),
    hex.slice(6, 8).join(""),
    hex.slice(8, 10).join(""),
    hex.slice(10).join(""),
  ].join("-");
}

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

function formatIssuePath(path: readonly PropertyKey[]) {
  return path.length > 0 ? path.map(String).join(".") : "<root>";
}

function formatValidationIssues(issues: readonly ResponseValidationIssue[]) {
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
  mutation?: MutationRequestOptions,
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
  headers.delete(WORKBENCH_MUTATION_HEADER_NAME);
  headers.delete(IDEMPOTENCY_HEADER_NAME);
  if (mutation) {
    if (["GET", "HEAD", "OPTIONS"].includes(method)) {
      throw new Error(`${method} requests cannot be declared as mutations.`);
    }
    if (!mutation.idempotencyKey) {
      throw new Error("Mutation requests require an idempotency key.");
    }
    headers.set(WORKBENCH_MUTATION_HEADER_NAME, WORKBENCH_MUTATION_HEADER_VALUE);
    headers.set(IDEMPOTENCY_HEADER_NAME, mutation.idempotencyKey);
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

async function parseJsonResponse<T>(
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
    schema: ResponseSchema<T>;
    requestRevision: number;
    authToken: string | null;
  },
): Promise<T> {
  assertWorkbenchConnectionRequestCurrent(requestRevision);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const errorBody = await response.json();
      if (
        typeof errorBody === "object" &&
        errorBody !== null &&
        !Array.isArray(errorBody) &&
        "detail" in errorBody &&
        errorBody.detail
      ) {
        detail = detailText(errorBody.detail);
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
  let parsed: ResponseValidationResult<T>;
  try {
    parsed = schema.safeParse(payload);
  } catch (error) {
    throw redactedError(error, authToken);
  }
  if (!parsed.success) {
    throw new Error(
      redactBearerToken(
        `Invalid API response for ${method} ${path} from ${apiBaseUrl}: ${formatValidationIssues(
          parsed.error.issues,
        )}`,
        authToken,
      ),
    );
  }
  return parsed.data;
}

type RequestExecution =
  | Readonly<{
      bodyKind: "json";
      init?: RequestInit;
      authenticationProbe?: boolean;
      mutation?: MutationRequestOptions;
    }>
  | Readonly<{
      bodyKind: "multipart";
      formData: FormData;
      init?: Omit<RequestInit, "body">;
      mutation?: MutationRequestOptions;
    }>;

async function executeRequest<T>(
  path: string,
  schema: ResponseSchema<T>,
  execution: RequestExecution,
): Promise<T> {
  throwIfAborted(execution.init?.signal);
  const method =
    execution.bodyKind === "multipart"
      ? requestMethod({ method: execution.init?.method ?? "POST" })
      : requestMethod(execution.init);
  const authenticationProbe =
    execution.bodyKind === "json" && execution.authenticationProbe;
  const connection = captureWorkbenchConnectionRequest(
    path,
    authenticationProbe,
  );
  const apiBaseUrl = connection.apiBaseUrl;
  let request: RequestInit;
  try {
    request =
      execution.bodyKind === "multipart"
        ? {
            ...execution.init,
            method,
            body: execution.formData,
            headers: requestHeaders(
              method,
              connection.authToken,
              execution.init?.headers,
              null,
              execution.mutation,
            ),
          }
        : {
            ...execution.init,
            headers: requestHeaders(
              method,
              connection.authToken,
              execution.init?.headers,
              "application/json",
              execution.mutation,
            ),
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
  if (authenticationProbe) {
    confirmWorkbenchAuthentication(
      connection.revision,
      connection.authenticationProbeGeneration,
    );
  }
  return parsed;
}

export function requestJson<T>(
  path: string,
  schema: ResponseSchema<T>,
  init?: RequestInit,
  policy: {
    authenticationProbe?: boolean;
    mutation?: MutationRequestOptions;
  } = {},
): Promise<T> {
  return executeRequest(path, schema, {
    bodyKind: "json",
    init,
    ...policy,
  });
}

export function requestMultipartJson<T>(
  path: string,
  schema: ResponseSchema<T>,
  formData: FormData,
  init?: Omit<RequestInit, "body">,
  policy: { mutation?: MutationRequestOptions } = {},
): Promise<T> {
  return executeRequest(path, schema, {
    bodyKind: "multipart",
    formData,
    init,
    ...policy,
  });
}
