import { afterEach, describe, expect, it, vi } from "vitest";

type FetchFn = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

type ClientEnv = {
  apiUrl?: string;
  allowedOrigins?: string;
};

function fakeResponse() {
  return {
    ok: true,
    status: 200,
    statusText: "OK",
    json: () => Promise.resolve({ status: "ok" }),
  } as unknown as Response;
}

async function importApiClient({
  apiUrl = "",
  allowedOrigins = "",
}: ClientEnv = {}) {
  vi.resetModules();
  vi.stubEnv("NEXT_PUBLIC_WORKBENCH_API_URL", apiUrl);
  vi.stubEnv("NEXT_PUBLIC_WORKBENCH_API_ALLOWED_ORIGINS", allowedOrigins);
  return import("@/lib/api/client");
}

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
  vi.resetModules();
  window.localStorage.clear();
  window.sessionStorage.clear();
});

describe("Workbench API origin lock", () => {
  it("keeps local development runtime API switching unlocked by default", async () => {
    const client = await importApiClient();

    expect(client.getWorkbenchApiAllowedOrigins()).toEqual([]);

    const apiBaseUrl = client.setWorkbenchApiBaseUrl(
      " https://api.example.test/workbench/// ",
    );

    expect(apiBaseUrl).toBe("https://api.example.test/workbench");
    expect(client.getWorkbenchApiBaseUrl()).toBe("https://api.example.test/workbench");
    expect(
      window.localStorage.getItem(client.WORKBENCH_API_BASE_URL_STORAGE_KEY),
    ).toBe("https://api.example.test/workbench");
  });

  it("locks hosted builds to the configured non-local API origin by default", async () => {
    const client = await importApiClient({
      apiUrl: "https://api.example.test/workbench",
    });

    expect(client.WORKBENCH_API_BASE_URL).toBe("https://api.example.test/workbench");
    expect(client.getWorkbenchApiAllowedOrigins()).toEqual([
      "https://api.example.test",
    ]);

    window.localStorage.setItem(
      client.WORKBENCH_API_BASE_URL_STORAGE_KEY,
      "https://other-api.example.test/workbench",
    );

    expect(client.getWorkbenchApiBaseUrl()).toBe("https://api.example.test/workbench");
    expect(
      window.localStorage.getItem(client.WORKBENCH_API_BASE_URL_STORAGE_KEY),
    ).toBeNull();
    expect(() =>
      client.setWorkbenchApiBaseUrl("https://other-api.example.test/workbench"),
    ).toThrow(/not allowed by this hosted build/i);
  });

  it("honors an explicit hosted API origin allowlist", async () => {
    const client = await importApiClient({
      apiUrl: "https://api.example.test/workbench",
      allowedOrigins: JSON.stringify([
        "https://api.example.test",
        "https://backup-api.example.test",
      ]),
    });

    expect(client.getWorkbenchApiAllowedOrigins()).toEqual([
      "https://api.example.test",
      "https://backup-api.example.test",
    ]);

    expect(client.setWorkbenchApiBaseUrl("https://backup-api.example.test/v2")).toBe(
      "https://backup-api.example.test/v2",
    );
    expect(() =>
      client.setWorkbenchApiBaseUrl("https://other-api.example.test"),
    ).toThrow(/allowed origins: https:\/\/api\.example\.test/i);
  });

  it("does not call fetch when the configured API origin is outside the explicit allowlist", async () => {
    const fetchMock = vi.fn<FetchFn>(() => Promise.resolve(fakeResponse()));
    vi.stubGlobal("fetch", fetchMock);

    await importApiClient({
      apiUrl: "https://api.example.test/workbench",
      allowedOrigins: "https://allowed-api.example.test",
    });
    const auth = await import("@/lib/auth-token");
    const health = await import("@/lib/api/health");
    auth.setSessionAuthToken("hosted-secret");

    await expect(health.fetchHealth()).rejects.toThrow(
      /origin https:\/\/api\.example\.test is not allowed/i,
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
