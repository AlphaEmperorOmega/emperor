import { QueryClient } from "@tanstack/react-query";
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
  const [runtime, actions] = await Promise.all([
    import("@/lib/api/_connection-runtime"),
    import("@/features/workbench/providers/_workbench-connection-actions"),
  ]);
  const environment = {
    publishRuntime: () => undefined,
    queryClient: new QueryClient(),
    resetProtectedState: () => undefined,
    setIsChanging: () => undefined,
  };
  return {
    ...runtime,
    signIn: (token: string) => actions.signIn(environment, token),
    useApiBaseUrl: (url: string) => actions.useApiBaseUrl(environment, url),
  };
}

type ConnectionRuntime = Awaited<ReturnType<typeof importApiClient>>;

async function useRuntimeApiBaseUrl(runtime: ConnectionRuntime, url: string) {
  const outcome = await runtime.useApiBaseUrl(url);
  if (!outcome.ok) {
    throw new Error(outcome.message);
  }
  return runtime.workbenchConnectionRuntimeSnapshot().apiBaseUrl;
}

async function useRuntimeAuthToken(runtime: ConnectionRuntime, token: string) {
  const outcome = await runtime.signIn(token);
  if (!outcome.ok) {
    throw new Error(outcome.message);
  }
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

    const apiBaseUrl = await useRuntimeApiBaseUrl(
      client,
      " https://api.example.test/workbench/// ",
    );

    expect(apiBaseUrl).toBe("https://api.example.test/workbench");
    expect(client.workbenchConnectionRuntimeSnapshot().apiBaseUrl).toBe(
      "https://api.example.test/workbench",
    );
    expect(
      window.localStorage.getItem(client.WORKBENCH_API_BASE_URL_STORAGE_KEY),
    ).toBe("https://api.example.test/workbench");
  });

  it("locks hosted builds to the configured non-local API origin by default", async () => {
    const client = await importApiClient({
      apiUrl: "https://api.example.test/workbench",
    });

    expect(client.WORKBENCH_API_BASE_URL).toBe("https://api.example.test/workbench");
    window.localStorage.setItem(
      client.WORKBENCH_API_BASE_URL_STORAGE_KEY,
      "https://other-api.example.test/workbench",
    );

    expect(client.loadWorkbenchConnectionRuntime().apiBaseUrl).toBe(
      "https://api.example.test/workbench",
    );
    expect(
      window.localStorage.getItem(client.WORKBENCH_API_BASE_URL_STORAGE_KEY),
    ).toBeNull();
    await expect(
      useRuntimeApiBaseUrl(
        client,
        "https://other-api.example.test/workbench",
      ),
    ).rejects.toThrow(/not allowed by this build/i);
  });

  it("honors an explicit hosted API origin allowlist", async () => {
    const client = await importApiClient({
      apiUrl: "https://api.example.test/workbench",
      allowedOrigins: JSON.stringify([
        "https://api.example.test",
        "https://backup-api.example.test",
      ]),
    });

    await expect(
      useRuntimeApiBaseUrl(client, "https://backup-api.example.test/v2"),
    ).resolves.toBe("https://backup-api.example.test/v2");
    await expect(
      useRuntimeApiBaseUrl(client, "https://other-api.example.test"),
    ).rejects.toThrow(/allowed origins: https:\/\/api\.example\.test/i);
  });

  it("honors one explicitly allowed hosted origin", async () => {
    const client = await importApiClient({
      apiUrl: "https://api.example.test/workbench",
      allowedOrigins: "https://backup-api.example.test",
    });

    await expect(
      useRuntimeApiBaseUrl(client, "https://backup-api.example.test/v2"),
    ).resolves.toBe("https://backup-api.example.test/v2");
    await expect(
      useRuntimeApiBaseUrl(client, "https://api.example.test/workbench"),
    ).rejects.toThrow(/not allowed/i);
  });

  it("rejects URL user information before persistence or network activity", async () => {
    const fetchMock = vi.fn<FetchFn>(() => Promise.resolve(fakeResponse()));
    vi.stubGlobal("fetch", fetchMock);
    const client = await importApiClient();

    expect(
      client.validateWorkbenchApiBaseUrl(
        "https://operator:secret@api.example.test/workbench",
      ),
    ).toMatchObject({ ok: false });
    await expect(
      useRuntimeApiBaseUrl(
        client,
        "https://operator:secret@api.example.test/workbench",
      ),
    ).rejects.toThrow(/without credentials/i);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(window.localStorage.length).toBe(0);
  });

  it("fails closed when the configured hosted API URL is invalid", async () => {
    const fetchMock = vi.fn<FetchFn>(() => Promise.resolve(fakeResponse()));
    vi.stubGlobal("fetch", fetchMock);
    await importApiClient({ apiUrl: "not-an-absolute-url" });
    const health = await import("@/lib/api/health");

    await expect(health.fetchHealth()).rejects.toThrow(
      /NEXT_PUBLIC_WORKBENCH_API_URL must be an absolute/i,
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("does not call fetch when the configured API origin is outside the explicit allowlist", async () => {
    const fetchMock = vi.fn<FetchFn>(() => Promise.resolve(fakeResponse()));
    vi.stubGlobal("fetch", fetchMock);

    const runtime = await importApiClient({
      apiUrl: "https://api.example.test/workbench",
      allowedOrigins: "https://allowed-api.example.test",
    });
    const health = await import("@/lib/api/health");
    await useRuntimeAuthToken(runtime, "hosted-secret");

    await expect(health.fetchHealth()).rejects.toThrow(
      /origin https:\/\/api\.example\.test is not allowed/i,
    );
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
