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
  workbenchConnectionRuntimeSnapshot,
  WORKBENCH_API_BASE_URL_STORAGE_KEY,
} from "@/lib/api/_connection-runtime";

function connectionEnvironment(): WorkbenchConnectionActionEnvironment {
  return {
    publishRuntime: vi.fn(),
    queryClient: new QueryClient(),
    resetProtectedState: vi.fn(),
    setIsChanging: vi.fn(),
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
  it("stores, reads at request time, and clears the browser-session token", async () => {
    const environment = connectionEnvironment();
    expect(workbenchConnectionRuntimeSnapshot().hasAuthToken).toBe(false);

    await expect(signIn(environment, "session-credential")).resolves.toEqual({
      ok: true,
    });
    observeWorkbenchAuthMode("none");

    expect(workbenchConnectionRuntimeSnapshot().hasAuthToken).toBe(true);
    expect(captureWorkbenchConnectionRequest("/models").authToken).toBe(
      "session-credential",
    );
    expect(JSON.stringify(workbenchConnectionRuntimeSnapshot())).not.toContain(
      "session-credential",
    );

    await expect(logout(environment)).resolves.toEqual({ ok: true });
    observeWorkbenchAuthMode("none");
    expect(workbenchConnectionRuntimeSnapshot().hasAuthToken).toBe(false);
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

    expect(workbenchConnectionRuntimeSnapshot().apiBaseUrl).toBe(
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
    expect(workbenchConnectionRuntimeSnapshot().storage).toEqual({
      apiBaseUrl: "unavailable",
      sessionToken: "unavailable",
    });
    await expect(signIn(environment, "unpublished")).resolves.toMatchObject({
      ok: false,
    });
    expect(workbenchConnectionRuntimeSnapshot().hasAuthToken).toBe(false);
  });
});
