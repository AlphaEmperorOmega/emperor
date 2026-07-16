// @vitest-environment jsdom

import { QueryClient } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  logout,
  signIn,
  useApiBaseUrl as applyApiBaseUrl,
  type WorkbenchConnectionActionEnvironment,
} from "@/features/workbench/providers/_workbench-connection-actions";
import {
  assertWorkbenchConnectionRequestCurrent,
  captureWorkbenchConnectionRequest,
  confirmWorkbenchAuthentication,
  loadWorkbenchConnectionRuntime,
  observeWorkbenchAuthMode,
  getWorkbenchConnectionRuntimeSnapshot,
  subscribeWorkbenchConnectionRuntime,
  WORKBENCH_API_BASE_URL_STORAGE_KEY,
} from "@/lib/api/_connection-runtime";

function connectionEnvironment(): WorkbenchConnectionActionEnvironment {
  return {
    queryClient: new QueryClient(),
    resetProtectedState: vi.fn(),
  };
}

beforeEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  loadWorkbenchConnectionRuntime();
  observeWorkbenchAuthMode("none");
});

afterEach(() => {
  vi.unstubAllGlobals();
  window.localStorage.clear();
  window.sessionStorage.clear();
  loadWorkbenchConnectionRuntime();
});

describe("private Workbench connection storage", () => {
  it("publishes immutable snapshots for every runtime transition", async () => {
    const environment = connectionEnvironment();
    const initial = getWorkbenchConnectionRuntimeSnapshot();
    const observed: ReturnType<
      typeof getWorkbenchConnectionRuntimeSnapshot
    >[] = [];
    const unsubscribe = subscribeWorkbenchConnectionRuntime(() => {
      observed.push(getWorkbenchConnectionRuntimeSnapshot());
    });

    await signIn(environment, "session-credential");
    unsubscribe();

    expect(observed.length).toBeGreaterThanOrEqual(2);
    expect(observed.at(-1)).not.toBe(initial);
    expect(observed.at(-1)).toMatchObject({
      hasAuthToken: true,
      isChanging: false,
    });
    expect(Object.isFrozen(observed.at(-1))).toBe(true);
    expect(Object.isFrozen(observed.at(-1)?.storage)).toBe(true);
    expect(JSON.stringify(observed)).not.toContain("session-credential");
  });

  it("stores, reads at request time, and clears the browser-session token", async () => {
    const environment = connectionEnvironment();
    expect(getWorkbenchConnectionRuntimeSnapshot().hasAuthToken).toBe(false);

    await expect(signIn(environment, "session-credential")).resolves.toEqual({
      ok: true,
    });
    observeWorkbenchAuthMode("none");

    expect(getWorkbenchConnectionRuntimeSnapshot().hasAuthToken).toBe(true);
    expect(captureWorkbenchConnectionRequest("/models").authToken).toBe(
      "session-credential",
    );
    expect(JSON.stringify(getWorkbenchConnectionRuntimeSnapshot())).not.toContain(
      "session-credential",
    );

    await expect(logout(environment)).resolves.toEqual({ ok: true });
    observeWorkbenchAuthMode("none");
    expect(getWorkbenchConnectionRuntimeSnapshot().hasAuthToken).toBe(false);
    expect(captureWorkbenchConnectionRequest("/models").authToken).toBeNull();
  });

  it("fails closed until a bearer probe verifies the current revision", async () => {
    const environment = connectionEnvironment();
    observeWorkbenchAuthMode("bearer");
    await signIn(environment, "session-credential");
    observeWorkbenchAuthMode("bearer");

    expect(() => captureWorkbenchConnectionRequest("/logs/runs"))
      .toThrow(/must be verified/i);
    const firstProbe = captureWorkbenchConnectionRequest("/models", true);

    confirmWorkbenchAuthentication(
      firstProbe.revision,
      firstProbe.authenticationProbeGeneration,
    );
    expect(captureWorkbenchConnectionRequest("/logs/runs").authToken).toBe(
      "session-credential",
    );

    const recheck = captureWorkbenchConnectionRequest("/models", true);
    expect(() => captureWorkbenchConnectionRequest("/logs/runs"))
      .toThrow(/must be verified/i);
    confirmWorkbenchAuthentication(
      recheck.revision,
      recheck.authenticationProbeGeneration,
    );
    expect(captureWorkbenchConnectionRequest("/logs/runs").authToken).toBe(
      "session-credential",
    );
  });

  it("restores identity but keeps the failed transition revision quarantined", async () => {
    const environment = connectionEnvironment();
    await applyApiBaseUrl(environment, "https://current.example.test");
    observeWorkbenchAuthMode("none");
    const previousRequest = captureWorkbenchConnectionRequest("/models");
    vi.spyOn(environment.queryClient, "cancelQueries").mockRejectedValueOnce(
      new Error("consumer quarantine failed"),
    );

    await expect(
      applyApiBaseUrl(environment, "https://next.example.test"),
    ).rejects.toThrow("consumer quarantine failed");

    expect(getWorkbenchConnectionRuntimeSnapshot().apiBaseUrl).toBe(
      "https://current.example.test",
    );
    expect(
      window.localStorage.getItem(WORKBENCH_API_BASE_URL_STORAGE_KEY),
    ).toBe("https://current.example.test");
    expect(() =>
      assertWorkbenchConnectionRequestCurrent(previousRequest.revision),
    ).toThrow(/connection changed/i);
    expect(() => captureWorkbenchConnectionRequest("/models")).toThrow(
      /must be verified/i,
    );
    observeWorkbenchAuthMode("none");
    expect(captureWorkbenchConnectionRequest("/models").apiBaseUrl).toBe(
      "https://current.example.test",
    );
  });

  it("reports unavailable storage without throwing when window is missing", async () => {
    const environment = connectionEnvironment();
    vi.stubGlobal("window", undefined);

    expect(() => loadWorkbenchConnectionRuntime()).not.toThrow();
    expect(getWorkbenchConnectionRuntimeSnapshot().storage).toEqual({
      apiBaseUrl: "unavailable",
      sessionToken: "unavailable",
    });
    await expect(signIn(environment, "unpublished")).resolves.toMatchObject({
      ok: false,
    });
    expect(getWorkbenchConnectionRuntimeSnapshot().hasAuthToken).toBe(false);
  });
});
