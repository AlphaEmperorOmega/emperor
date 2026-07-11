import { QueryClient, QueryClientProvider, useQuery } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { type ReactNode } from "react";
import { z } from "zod";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  WorkbenchConnectionProvider,
  isWorkbenchProtectedAccessReady,
  useRegisterWorkbenchConnectionReset,
  useWorkbenchCapabilities,
  useWorkbenchConnection,
} from "@/features/workbench/providers/workbench-connection-provider";
import { requestJson } from "@/lib/api/client";
import { workbenchQueryKeys } from "@/lib/query-keys";

type FetchFn = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

const capabilities = {
  authMode: "bearer",
  trainingEnabled: true,
  trainingCancellationCapability: "unsupported",
  logDeletionEnabled: true,
  configSnapshotsEnabled: true,
  historicalLogsEnabled: true,
  liveMonitorDataEnabled: true,
  historicalMonitorDataEnabled: true,
  uploadsEnabled: true,
  maxUploadSize: null,
  dataSourcesEnabled: false,
  dataSources: [],
};

function jsonResponse(body: unknown, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 401 ? "Unauthorized" : "OK",
    json: () => Promise.resolve(body),
  } as Response;
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void;
  const promise = new Promise<T>((promiseResolve) => {
    resolve = promiseResolve;
  });
  return { promise, resolve };
}

function connectionFetch({
  onModels,
  capabilityResponse = capabilities,
}: {
  onModels?: (
    authorization: string | null,
    input: RequestInfo | URL,
    init?: RequestInit,
  ) => Response | Promise<Response>;
  capabilityResponse?: typeof capabilities;
} = {}) {
  return vi.fn<FetchFn>(async (input, init) => {
    const url = String(input);
    if (url.endsWith("/health")) {
      return jsonResponse({ status: "ok" });
    }
    if (url.endsWith("/capabilities")) {
      return jsonResponse(capabilityResponse);
    }
    if (url.endsWith("/models")) {
      const authorization = new Headers(init?.headers).get("authorization");
      if (onModels) {
        return onModels(authorization, input, init);
      }
      if (authorization === "Bearer accepted" || authorization === "Bearer replacement") {
        return jsonResponse({ models: [] });
      }
      return jsonResponse({ detail: "Rejected credential" }, 401);
    }
    throw new Error(`Unexpected request: ${url}`);
  });
}

function wrapper(queryClient: QueryClient) {
  return function ConnectionWrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <WorkbenchConnectionProvider>{children}</WorkbenchConnectionProvider>
      </QueryClientProvider>
    );
  };
}

function createQueryClient() {
  return new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: Infinity } },
  });
}

beforeEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  window.localStorage.clear();
  window.sessionStorage.clear();
});

describe("Workbench Connection caller Interface", () => {
  it("reports backend-auth-disabled without requiring a bearer token", async () => {
    vi.stubGlobal(
      "fetch",
      connectionFetch({
        capabilityResponse: { ...capabilities, authMode: "none" },
      }),
    );
    const queryClient = createQueryClient();
    const { result } = renderHook(
      () => ({
        connection: useWorkbenchConnection(),
        capabilities: useWorkbenchCapabilities(),
      }),
      { wrapper: wrapper(queryClient) },
    );

    await waitFor(() => {
      expect(result.current.connection.authentication).toMatchObject({
        mode: "none",
        state: "disabled",
        hasToken: false,
      });
    });
    expect(result.current.capabilities.capabilities).not.toHaveProperty(
      "authMode",
    );
  });

  it("holds protected callers until the bearer probe verifies the current identity", async () => {
    const modelProbe = deferred<Response>();
    const fetchMock = connectionFetch({
      onModels: () => modelProbe.promise,
    });
    const defaultFetch = fetchMock.getMockImplementation()!;
    let protectedRequestCount = 0;
    fetchMock.mockImplementation((input, init) => {
      if (String(input).endsWith("/protected-after-auth")) {
        protectedRequestCount += 1;
        return Promise.resolve(jsonResponse({ ready: true }));
      }
      return defaultFetch(input, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const queryClient = createQueryClient();
    const payloadSchema = z.object({ ready: z.boolean() });
    const { result } = renderHook(
      () => {
        const connection = useWorkbenchConnection();
        const protectedQuery = useQuery({
          queryKey: ["protected-after-auth"],
          queryFn: ({ signal }) =>
            requestJson("/protected-after-auth", payloadSchema, { signal }),
          enabled: isWorkbenchProtectedAccessReady(connection),
          retry: false,
        });
        return { connection, protectedQuery };
      },
      { wrapper: wrapper(queryClient) },
    );

    await waitFor(() => {
      expect(result.current.connection.authentication.state).toBe(
        "unauthenticated",
      );
    });
    await act(async () => {
      await result.current.connection.actions.signIn("accepted");
    });
    await waitFor(() => {
      expect(result.current.connection.authentication.state).toBe("checking");
    });
    expect(protectedRequestCount).toBe(0);

    await act(async () => {
      modelProbe.resolve(jsonResponse({ models: [] }));
      await modelProbe.promise;
    });
    await waitFor(() => {
      expect(result.current.protectedQuery.data).toEqual({ ready: true });
    });
    expect(protectedRequestCount).toBe(1);
  });

  it("re-verifies a cached catalog when the provider remounts", async () => {
    const remountProbe = deferred<Response>();
    let modelRequestCount = 0;
    const fetchMock = connectionFetch({
      onModels: () => {
        modelRequestCount += 1;
        return modelRequestCount === 1
          ? jsonResponse({ models: [] })
          : remountProbe.promise;
      },
    });
    const defaultFetch = fetchMock.getMockImplementation()!;
    let protectedRequestCount = 0;
    fetchMock.mockImplementation((input, init) => {
      if (String(input).endsWith("/protected-after-remount")) {
        protectedRequestCount += 1;
        return Promise.resolve(jsonResponse({ ready: true }));
      }
      return defaultFetch(input, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const queryClient = createQueryClient();
    const first = renderHook(() => useWorkbenchConnection(), {
      wrapper: wrapper(queryClient),
    });

    await waitFor(() => {
      expect(first.result.current.authentication.state).toBe("unauthenticated");
    });
    await act(async () => {
      await first.result.current.actions.signIn("accepted");
    });
    await waitFor(() => {
      expect(first.result.current.authentication.state).toBe("authenticated");
    });
    first.unmount();

    const payloadSchema = z.object({ ready: z.boolean() });
    const second = renderHook(
      () => {
        const connection = useWorkbenchConnection();
        const protectedQuery = useQuery({
          queryKey: ["protected-after-remount"],
          queryFn: ({ signal }) =>
            requestJson("/protected-after-remount", payloadSchema, { signal }),
          enabled: isWorkbenchProtectedAccessReady(connection),
          retry: false,
        });
        return { connection, protectedQuery };
      },
      { wrapper: wrapper(queryClient) },
    );

    await waitFor(() => {
      expect(second.result.current.connection.authentication.state).toBe(
        "checking",
      );
      expect(modelRequestCount).toBe(2);
    });
    expect(protectedRequestCount).toBe(0);

    await act(async () => {
      remountProbe.resolve(jsonResponse({ models: [] }));
      await remountProbe.promise;
    });
    await waitFor(() => {
      expect(second.result.current.protectedQuery.data).toEqual({ ready: true });
    });
    expect(protectedRequestCount).toBe(1);
  });

  it("closes private protected access while a bearer re-probe is pending", async () => {
    const recheck = deferred<Response>();
    let modelRequestCount = 0;
    const fetchMock = connectionFetch({
      onModels: () => {
        modelRequestCount += 1;
        return modelRequestCount === 1
          ? jsonResponse({ models: [] })
          : recheck.promise;
      },
    });
    const defaultFetch = fetchMock.getMockImplementation()!;
    let protectedRequestCount = 0;
    fetchMock.mockImplementation((input, init) => {
      if (String(input).endsWith("/protected-during-recheck")) {
        protectedRequestCount += 1;
        return Promise.resolve(jsonResponse({ ready: true }));
      }
      return defaultFetch(input, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const queryClient = createQueryClient();
    const { result } = renderHook(() => useWorkbenchConnection(), {
      wrapper: wrapper(queryClient),
    });
    const payloadSchema = z.object({ ready: z.boolean() });

    await waitFor(() => {
      expect(result.current.authentication.state).toBe("unauthenticated");
    });
    await act(async () => {
      await result.current.actions.signIn("accepted");
    });
    await waitFor(() => {
      expect(result.current.authentication.state).toBe("authenticated");
    });

    let retryPromise: Promise<void> | undefined;
    act(() => {
      retryPromise = result.current.actions.retry();
    });
    await waitFor(() => {
      expect(modelRequestCount).toBe(2);
      expect(result.current.authentication.state).toBe("checking");
    });
    await expect(
      requestJson("/protected-during-recheck", payloadSchema),
    ).rejects.toThrow(/must be verified/i);
    expect(protectedRequestCount).toBe(0);

    await act(async () => {
      recheck.resolve(jsonResponse({ models: [] }));
      await retryPromise;
    });
    await waitFor(() => {
      expect(result.current.authentication.state).toBe("authenticated");
    });
    await expect(
      requestJson("/protected-during-recheck", payloadSchema),
    ).resolves.toEqual({ ready: true });
    expect(protectedRequestCount).toBe(1);
  });

  it("fails closed after a background capability failure and recovers on retry", async () => {
    const fetchMock = connectionFetch();
    const defaultFetch = fetchMock.getMockImplementation()!;
    let capabilityRequestCount = 0;
    let protectedRequestCount = 0;
    fetchMock.mockImplementation((input, init) => {
      const url = String(input);
      if (url.endsWith("/capabilities")) {
        capabilityRequestCount += 1;
        if (capabilityRequestCount === 2) {
          return Promise.resolve(jsonResponse({ detail: "temporary" }, 503));
        }
      }
      if (url.endsWith("/protected-after-capability-failure")) {
        protectedRequestCount += 1;
        return Promise.resolve(jsonResponse({ ready: true }));
      }
      return defaultFetch(input, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const queryClient = createQueryClient();
    const { result } = renderHook(() => useWorkbenchConnection(), {
      wrapper: wrapper(queryClient),
    });
    const payloadSchema = z.object({ ready: z.boolean() });

    await waitFor(() => {
      expect(result.current.authentication.state).toBe("unauthenticated");
    });
    await act(async () => {
      await result.current.actions.signIn("accepted");
    });
    await waitFor(() => {
      expect(result.current.authentication.state).toBe("authenticated");
    });

    await act(async () => {
      await result.current.actions.retry();
    });
    await waitFor(() => {
      expect(result.current.authentication.state).toBe("capability-failed");
    });
    await expect(
      requestJson("/protected-after-capability-failure", payloadSchema),
    ).rejects.toThrow(/must be verified/i);
    expect(protectedRequestCount).toBe(0);

    await act(async () => {
      await result.current.actions.retry();
    });
    await waitFor(() => {
      expect(result.current.authentication.state).toBe("authenticated");
    });
    await expect(
      requestJson("/protected-after-capability-failure", payloadSchema),
    ).resolves.toEqual({ ready: true });
    expect(protectedRequestCount).toBe(1);
  });

  it("re-probes when a live backend changes from disabled auth to bearer", async () => {
    let authMode: "none" | "bearer" = "none";
    let modelRequestCount = 0;
    const fetchMock = connectionFetch({
      onModels: () => {
        modelRequestCount += 1;
        return jsonResponse({ models: [] });
      },
    });
    const defaultFetch = fetchMock.getMockImplementation()!;
    fetchMock.mockImplementation((input, init) => {
      if (String(input).endsWith("/capabilities")) {
        return Promise.resolve(jsonResponse({ ...capabilities, authMode }));
      }
      return defaultFetch(input, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const queryClient = createQueryClient();
    const { result } = renderHook(() => useWorkbenchConnection(), {
      wrapper: wrapper(queryClient),
    });

    await waitFor(() => {
      expect(result.current.authentication.state).toBe("disabled");
      expect(modelRequestCount).toBeGreaterThan(0);
    });
    await act(async () => {
      await result.current.actions.signIn("accepted");
    });
    await waitFor(() => {
      expect(result.current.authentication.state).toBe("disabled");
    });
    const requestCountBeforeModeChange = modelRequestCount;

    authMode = "bearer";
    await act(async () => {
      await queryClient.invalidateQueries({
        queryKey: workbenchQueryKeys.capabilities(),
      });
    });

    await waitFor(() => {
      expect(modelRequestCount).toBeGreaterThan(requestCountBeforeModeChange);
      expect(result.current.authentication.state).toBe("authenticated");
    });
  });

  it("falls back from an invalid saved URL with an actionable caller status", async () => {
    window.localStorage.setItem(
      "emperor.workbench.apiBaseUrl",
      "ftp://invalid.example.test",
    );
    vi.stubGlobal("fetch", connectionFetch());
    const queryClient = createQueryClient();
    const { result } = renderHook(() => useWorkbenchConnection(), {
      wrapper: wrapper(queryClient),
    });

    expect(result.current.connection.apiBaseUrl).toBe("http://127.0.0.1:9999");
    expect(result.current.storage.message).toMatch(/invalid or disallowed/i);
    expect(
      window.localStorage.getItem("emperor.workbench.apiBaseUrl"),
    ).toBeNull();
  });

  it("renders an actionable state when browser storage getters are unavailable", () => {
    const localDescriptor = Object.getOwnPropertyDescriptor(window, "localStorage");
    const sessionDescriptor = Object.getOwnPropertyDescriptor(
      window,
      "sessionStorage",
    );
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      get: () => {
        throw new DOMException("blocked", "SecurityError");
      },
    });
    Object.defineProperty(window, "sessionStorage", {
      configurable: true,
      get: () => {
        throw new DOMException("blocked", "SecurityError");
      },
    });
    vi.stubGlobal("fetch", connectionFetch());

    try {
      const queryClient = createQueryClient();
      const { result } = renderHook(() => useWorkbenchConnection(), {
        wrapper: wrapper(queryClient),
      });
      expect(result.current.storage).toMatchObject({
        apiBaseUrl: "unavailable",
        sessionToken: "unavailable",
      });
      expect(result.current.storage.message).toMatch(/session storage is unavailable/i);
      expect(result.current.connection.apiBaseUrl).toBe(
        "http://127.0.0.1:9999",
      );
    } finally {
      Object.defineProperty(window, "localStorage", localDescriptor!);
      Object.defineProperty(window, "sessionStorage", sessionDescriptor!);
    }
  });

  it("owns sign-in, rejection, replacement, and logout as one lifecycle", async () => {
    const queryClient = createQueryClient();
    queryClient.setQueryData(["protected", "old"], { secret: "old" });
    queryClient.getMutationCache().build(queryClient, {
      mutationKey: ["protected-mutation", "old"],
      mutationFn: async () => ({ secret: "old" }),
    });
    let resolveAccepted!: (response: Response) => void;
    const acceptedProbe = new Promise<Response>((resolve) => {
      resolveAccepted = resolve;
    });
    const cacheAtProbe: Array<{ query: unknown; mutations: number }> = [];
    const fetchMock = connectionFetch({
      onModels: (authorization) => {
        cacheAtProbe.push({
          query: queryClient.getQueryData(["protected", "old"]),
          mutations: queryClient.getMutationCache().getAll().length,
        });
        if (authorization === "Bearer wrong") {
          return jsonResponse({ detail: "Rejected credential" }, 401);
        }
        if (authorization === "Bearer accepted") {
          return acceptedProbe;
        }
        if (authorization === "Bearer replacement") {
          return jsonResponse({ models: [] });
        }
        return jsonResponse({ detail: "Rejected credential" }, 401);
      },
    });
    vi.stubGlobal("fetch", fetchMock);
    const reset = vi.fn();
    const { result } = renderHook(
      () => {
        const connection = useWorkbenchConnection();
        const capabilityProjection = useWorkbenchCapabilities();
        useRegisterWorkbenchConnectionReset(reset);
        return { connection, capabilityProjection };
      },
      { wrapper: wrapper(queryClient) },
    );

    await waitFor(() => {
      expect(result.current.connection.authentication.state).toBe(
        "unauthenticated",
      );
    });
    expect(Object.keys(result.current.connection).sort()).toEqual([
      "actions",
      "authentication",
      "connection",
      "storage",
    ]);
    expect(Object.keys(result.current.connection.connection).sort()).toEqual([
      "apiBaseUrl",
      "configurationError",
      "isChanging",
      "isOnline",
    ]);
    expect(Object.keys(result.current.connection.authentication).sort()).toEqual([
      "hasToken",
      "mode",
      "state",
    ]);
    expect(result.current.connection).not.toHaveProperty("token");
    expect(result.current.connection).not.toHaveProperty("revision");
    expect(Object.keys(result.current.capabilityProjection)).toEqual([
      "capabilities",
    ]);
    expect(result.current.capabilityProjection.capabilities).not.toHaveProperty(
      "authMode",
    );

    await act(async () => {
      expect(await result.current.connection.actions.signIn("wrong")).toEqual({
        ok: true,
      });
    });
    await waitFor(() => {
      expect(result.current.connection.authentication.state).toBe("rejected");
    });
    expect(result.current.connection.authentication.hasToken).toBe(true);

    expect(cacheAtProbe[0]).toEqual({ query: undefined, mutations: 0 });

    await act(async () => {
      expect(await result.current.connection.actions.signIn("accepted")).toEqual({
        ok: true,
      });
    });
    expect(result.current.connection.authentication.state).toBe("checking");
    expect(JSON.stringify(result.current)).not.toContain("wrong");
    expect(JSON.stringify(result.current)).not.toContain("accepted");
    await act(async () => {
      resolveAccepted(jsonResponse({ models: [] }));
      await acceptedProbe;
    });
    await waitFor(() => {
      expect(result.current.connection.authentication.state).toBe("authenticated");
    });

    await act(async () => {
      expect(
        await result.current.connection.actions.signIn("replacement"),
      ).toEqual({ ok: true });
    });
    await waitFor(() => {
      expect(result.current.connection.authentication.state).toBe("authenticated");
    });

    await act(async () => {
      expect(await result.current.connection.actions.logout()).toEqual({ ok: true });
    });
    await waitFor(() => {
      expect(result.current.connection.authentication.state).toBe(
        "unauthenticated",
      );
    });
    expect(result.current.connection.authentication.hasToken).toBe(false);
    expect(reset).toHaveBeenCalledTimes(4);
    expect(
      fetchMock.mock.calls.some(([, init]) =>
        new Headers(init?.headers).get("authorization")?.includes("replacement"),
      ),
    ).toBe(true);
    expect(JSON.stringify(result.current)).not.toContain("replacement");
  });

  it("validates a base URL before reset and replaces the active preview through a semantic reset", async () => {
    vi.stubGlobal("fetch", connectionFetch());
    const queryClient = createQueryClient();
    const replaceConnectionPreview = vi.fn();
    const { result } = renderHook(
      () => {
        const connection = useWorkbenchConnection();
        useRegisterWorkbenchConnectionReset(replaceConnectionPreview);
        return connection;
      },
      { wrapper: wrapper(queryClient) },
    );

    await waitFor(() => {
      expect(result.current.authentication.state).toBe("unauthenticated");
    });

    await act(async () => {
      const invalid = await result.current.actions.useApiBaseUrl("ftp://invalid");
      expect(invalid.ok).toBe(false);
    });
    expect(replaceConnectionPreview).not.toHaveBeenCalled();

    await act(async () => {
      expect(
        await result.current.actions.useApiBaseUrl(
          " https://backup.example.test/workbench/// ",
        ),
      ).toEqual({ ok: true });
    });
    expect(result.current.connection.apiBaseUrl).toBe(
      "https://backup.example.test/workbench",
    );
    expect(replaceConnectionPreview).toHaveBeenCalledTimes(1);
  });

  it("quarantines an obsolete protected read even when fetch ignores cancellation", async () => {
    let resolveOldResponse: ((response: Response) => void) | undefined;
    let oldSignal: AbortSignal | null | undefined;
    let protectedRequestCount = 0;
    const fetchMock = connectionFetch();
    const normalFetch = fetchMock.getMockImplementation();
    fetchMock.mockImplementation((input, init) => {
      if (String(input).endsWith("/protected-race")) {
        protectedRequestCount += 1;
        if (protectedRequestCount > 1) {
          return Promise.resolve(jsonResponse({ source: "new-backend" }));
        }
        oldSignal = init?.signal;
        return new Promise<Response>((resolve) => {
          resolveOldResponse = resolve;
        });
      }
      return normalFetch!(input, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const queryClient = createQueryClient();
    const payloadSchema = z.object({ source: z.string() });
    const { result } = renderHook(
      () => {
        const connection = useWorkbenchConnection();
        const protectedQuery = useQuery({
          queryKey: ["protected-race"],
          queryFn: ({ signal }) =>
            requestJson("/protected-race", payloadSchema, { signal }),
          enabled: connection.authentication.state === "authenticated",
          retry: false,
        });
        return { connection, protectedQuery };
      },
      { wrapper: wrapper(queryClient) },
    );

    await waitFor(() => {
      expect(result.current.connection.authentication.state).toBe(
        "unauthenticated",
      );
    });
    await act(async () => {
      await result.current.connection.actions.signIn("accepted");
    });
    await waitFor(() => expect(resolveOldResponse).toBeTypeOf("function"));
    await act(async () => {
      await result.current.connection.actions.useApiBaseUrl(
        "https://next.example.test",
      );
    });
    expect(oldSignal?.aborted).toBe(true);
    await act(async () => {
      resolveOldResponse?.(jsonResponse({ source: "old-backend" }));
      await Promise.resolve();
    });

    await waitFor(() => {
      expect(queryClient.getQueryData(["protected-race"])).toEqual({
        source: "new-backend",
      });
    });
    expect(result.current.protectedQuery.data).not.toEqual({ source: "old-backend" });
  });

  it.each(["replacement", "logout"] as const)(
    "quarantines an active protected read during token %s",
    async (transition) => {
      type DeferredRequest = {
        signal: AbortSignal | null | undefined;
        resolve: (response: Response) => void;
      };
      const protectedRequests: DeferredRequest[] = [];
      let protectedRequestCount = 0;
      const baseFetch = connectionFetch();
      const baseImplementation = baseFetch.getMockImplementation()!;
      const fetchMock = vi.fn<FetchFn>((input, init) => {
        if (!String(input).endsWith("/protected-race")) {
          return baseImplementation(input, init);
        }
        protectedRequestCount += 1;
        if (protectedRequestCount === 1) {
          return Promise.resolve(jsonResponse({ source: "current" }));
        }
        if (protectedRequestCount > 2) {
          return Promise.resolve(jsonResponse({ source: "replacement" }));
        }
        return new Promise<Response>((resolve) => {
          protectedRequests.push({ signal: init?.signal, resolve });
        });
      });
      vi.stubGlobal("fetch", fetchMock);
      const queryClient = createQueryClient();
      const payloadSchema = z.object({ source: z.string() });
      const { result } = renderHook(
        () => {
          const connection = useWorkbenchConnection();
          const protectedQuery = useQuery({
            queryKey: ["protected-token-race"],
            queryFn: ({ signal }) =>
              requestJson("/protected-race", payloadSchema, { signal }),
            enabled: connection.authentication.state === "authenticated",
            retry: false,
          });
          return { connection, protectedQuery };
        },
        { wrapper: wrapper(queryClient) },
      );

      await waitFor(() => {
        expect(result.current.connection.authentication.state).toBe(
          "unauthenticated",
        );
      });
      await act(async () => {
        await result.current.connection.actions.signIn("accepted");
      });
      await waitFor(() => {
        expect(result.current.protectedQuery.data).toEqual({ source: "current" });
      });

      act(() => {
        void result.current.protectedQuery.refetch();
      });
      await waitFor(() => expect(protectedRequests).toHaveLength(1));

      await act(async () => {
        if (transition === "replacement") {
          await result.current.connection.actions.signIn("replacement");
        } else {
          await result.current.connection.actions.logout();
        }
      });
      expect(protectedRequests[0]?.signal?.aborted).toBe(true);

      if (transition === "replacement") {
        await waitFor(() => {
          expect(result.current.protectedQuery.data).toEqual({
            source: "replacement",
          });
        });
      } else {
        await waitFor(() => {
          expect(result.current.connection.authentication.state).toBe(
            "unauthenticated",
          );
          expect(result.current.protectedQuery.data).toBeUndefined();
        });
      }

      await act(async () => {
        protectedRequests[0]?.resolve(jsonResponse({ source: "obsolete" }));
        await Promise.resolve();
      });
      expect(JSON.stringify(queryClient.getQueryCache().getAll().map((query) => query.state.data)))
        .not.toContain("obsolete");
      expect(result.current.protectedQuery.data).not.toEqual({ source: "obsolete" });
    },
  );

  it("keeps the previous identity and caches when browser storage rejects a transition", async () => {
    vi.stubGlobal("fetch", connectionFetch());
    const queryClient = createQueryClient();
    queryClient.setQueryData(["protected", "current"], { retained: true });
    const reset = vi.fn();
    const { result } = renderHook(
      () => {
        const connection = useWorkbenchConnection();
        useRegisterWorkbenchConnectionReset(reset);
        return connection;
      },
      { wrapper: wrapper(queryClient) },
    );

    await waitFor(() => {
      expect(result.current.authentication.state).toBe("unauthenticated");
    });
    const originalUrl = result.current.connection.apiBaseUrl;
    const localStorageDescriptor = Object.getOwnPropertyDescriptor(
      window,
      "localStorage",
    );
    const blockedLocalStorage = {
      ...window.localStorage,
      getItem: window.localStorage.getItem.bind(window.localStorage),
      removeItem: window.localStorage.removeItem.bind(window.localStorage),
      setItem: () => undefined,
    } as Storage;
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: blockedLocalStorage,
    });

    await act(async () => {
      const outcome = await result.current.actions.useApiBaseUrl(
        "https://blocked.example.test",
      );
      expect(outcome.ok).toBe(false);
    });
    expect(result.current.connection.apiBaseUrl).toBe(originalUrl);
    expect(result.current.storage.apiBaseUrl).toBe("unavailable");
    expect(queryClient.getQueryData(["protected", "current"])).toEqual({
      retained: true,
    });
    expect(reset).not.toHaveBeenCalled();
    Object.defineProperty(window, "localStorage", localStorageDescriptor!);

    const sessionStorageDescriptor = Object.getOwnPropertyDescriptor(
      window,
      "sessionStorage",
    );
    const blockedSessionStorage = {
      ...window.sessionStorage,
      getItem: window.sessionStorage.getItem.bind(window.sessionStorage),
      removeItem: window.sessionStorage.removeItem.bind(window.sessionStorage),
      setItem: () => {
        throw new DOMException("blocked", "SecurityError");
      },
    } as Storage;
    Object.defineProperty(window, "sessionStorage", {
      configurable: true,
      value: blockedSessionStorage,
    });
    await act(async () => {
      const outcome = await result.current.actions.signIn("never-published");
      expect(outcome.ok).toBe(false);
    });
    expect(result.current.authentication.hasToken).toBe(false);
    expect(result.current.storage.sessionToken).toBe("unavailable");
    expect(reset).not.toHaveBeenCalled();
    Object.defineProperty(window, "sessionStorage", sessionStorageDescriptor!);
  });

  it("keeps an authenticated identity intact when replacement or logout storage fails", async () => {
    const fetchMock = connectionFetch();
    vi.stubGlobal("fetch", fetchMock);
    const queryClient = createQueryClient();
    const reset = vi.fn();
    const { result } = renderHook(
      () => {
        const connection = useWorkbenchConnection();
        useRegisterWorkbenchConnectionReset(reset);
        return connection;
      },
      { wrapper: wrapper(queryClient) },
    );

    await waitFor(() => {
      expect(result.current.authentication.state).toBe("unauthenticated");
    });
    await act(async () => {
      await result.current.actions.signIn("accepted");
    });
    await waitFor(() => {
      expect(result.current.authentication.state).toBe("authenticated");
    });
    reset.mockClear();
    queryClient.setQueryData(["protected", "retained"], { retained: true });
    queryClient.getMutationCache().build(queryClient, {
      mutationKey: ["protected-mutation", "retained"],
      mutationFn: async () => ({ retained: true }),
    });
    const requestCount = fetchMock.mock.calls.length;
    const originalDescriptor = Object.getOwnPropertyDescriptor(
      window,
      "sessionStorage",
    );
    const blockedStorage = {
      length: 1,
      clear: vi.fn(),
      getItem: vi.fn(() => "accepted"),
      key: vi.fn(() => "emperor.workbench.authToken"),
      removeItem: vi.fn(() => {
        throw new DOMException("blocked", "SecurityError");
      }),
      setItem: vi.fn(() => {
        throw new DOMException("blocked", "SecurityError");
      }),
    } as Storage;
    Object.defineProperty(window, "sessionStorage", {
      configurable: true,
      value: blockedStorage,
    });

    try {
      await act(async () => {
        expect(await result.current.actions.signIn("replacement")).toMatchObject({
          ok: false,
        });
      });
      expect(result.current.authentication).toMatchObject({
        state: "authenticated",
        hasToken: true,
      });
      expect(result.current.storage.sessionToken).toBe("unavailable");
      expect(queryClient.getQueryData(["protected", "retained"])).toEqual({
        retained: true,
      });
      expect(queryClient.getMutationCache().getAll()).toHaveLength(1);
      expect(fetchMock.mock.calls).toHaveLength(requestCount);
      expect(reset).not.toHaveBeenCalled();

      await act(async () => {
        expect(await result.current.actions.logout()).toMatchObject({ ok: false });
      });
      expect(result.current.authentication).toMatchObject({
        state: "authenticated",
        hasToken: true,
      });
      expect(queryClient.getQueryData(["protected", "retained"])).toEqual({
        retained: true,
      });
      expect(fetchMock.mock.calls).toHaveLength(requestCount);
      expect(reset).not.toHaveBeenCalled();
    } finally {
      Object.defineProperty(window, "sessionStorage", originalDescriptor!);
    }
  });

  it("does not reset identity when local storage silently retains the saved URL", async () => {
    vi.stubGlobal("fetch", connectionFetch());
    const queryClient = createQueryClient();
    const reset = vi.fn();
    const { result } = renderHook(
      () => {
        const connection = useWorkbenchConnection();
        useRegisterWorkbenchConnectionReset(reset);
        return connection;
      },
      { wrapper: wrapper(queryClient) },
    );
    await waitFor(() => {
      expect(result.current.authentication.state).toBe("unauthenticated");
    });
    await act(async () => {
      await result.current.actions.useApiBaseUrl("https://saved.example.test");
    });
    reset.mockClear();
    const originalDescriptor = Object.getOwnPropertyDescriptor(
      window,
      "localStorage",
    );
    const retainedStorage = {
      length: 1,
      clear: vi.fn(),
      getItem: vi.fn(() => "https://saved.example.test"),
      key: vi.fn(() => "emperor.workbench.apiBaseUrl"),
      removeItem: vi.fn(),
      setItem: vi.fn(),
    } as Storage;
    Object.defineProperty(window, "localStorage", {
      configurable: true,
      value: retainedStorage,
    });

    try {
      await act(async () => {
        expect(await result.current.actions.resetApiBaseUrl()).toMatchObject({
          ok: false,
        });
      });
      expect(result.current.connection.apiBaseUrl).toBe(
        "https://saved.example.test",
      );
      expect(result.current.storage.apiBaseUrl).toBe("unavailable");
      expect(reset).not.toHaveBeenCalled();
    } finally {
      Object.defineProperty(window, "localStorage", originalDescriptor!);
    }
  });
});
