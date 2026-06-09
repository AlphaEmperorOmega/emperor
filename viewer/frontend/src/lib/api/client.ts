import { z } from "zod";

import { getSessionAuthToken } from "@/lib/auth-token";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_VIEWER_API_URL ?? "http://127.0.0.1:9999";

const errorBodySchema = z.object({ detail: z.unknown() }).partial();

type ApiErrorInit = {
  status: number;
  method: string;
  path: string;
  detail: string;
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

  constructor({ status, method, path, detail }: ApiErrorInit) {
    const messageDetail = detail || "Request failed";
    super(`${method} ${path} from ${API_BASE_URL} failed with ${status}: ${messageDetail}`);
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

function requestHeaders(initHeaders?: HeadersInit) {
  const headers = new Headers({ "content-type": "application/json" });
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
  const response = await fetch(`${API_BASE_URL}${path}`, request);
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
    });
  }
  const payload = await response.json();
  const parsed = schema.safeParse(payload);
  if (!parsed.success) {
    throw new Error(
      `Invalid API response for ${method} ${path} from ${API_BASE_URL}: ${formatZodIssues(
        parsed.error.issues,
      )}`,
    );
  }
  return parsed.data;
}
