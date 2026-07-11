// @vitest-environment jsdom

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  captureWorkbenchConnectionRequest,
  confirmWorkbenchAuthentication,
  loadWorkbenchConnectionRuntime,
  observeWorkbenchAuthMode,
  workbenchConnectionRuntimeSnapshot,
} from "@/lib/api/_connection-runtime";
import {
  beginWorkbenchConnectionTransition,
  commitWorkbenchAuthToken,
  finishWorkbenchConnectionTransition,
  persistClearedWorkbenchAuthToken,
  persistWorkbenchAuthToken,
} from "@/lib/api/_connection-runtime-actions";

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
  it("stores, reads at request time, and clears the browser-session token", () => {
    expect(workbenchConnectionRuntimeSnapshot().hasAuthToken).toBe(false);

    expect(persistWorkbenchAuthToken("session-credential")).toEqual({ ok: true });
    beginWorkbenchConnectionTransition();
    commitWorkbenchAuthToken("session-credential");
    finishWorkbenchConnectionTransition();
    observeWorkbenchAuthMode("none");

    expect(workbenchConnectionRuntimeSnapshot().hasAuthToken).toBe(true);
    expect(captureWorkbenchConnectionRequest("/models").authToken).toBe(
      "session-credential",
    );
    expect(JSON.stringify(workbenchConnectionRuntimeSnapshot())).not.toContain(
      "session-credential",
    );

    expect(persistClearedWorkbenchAuthToken()).toEqual({ ok: true });
    beginWorkbenchConnectionTransition();
    commitWorkbenchAuthToken(null);
    finishWorkbenchConnectionTransition();
    observeWorkbenchAuthMode("none");
    expect(workbenchConnectionRuntimeSnapshot().hasAuthToken).toBe(false);
    expect(captureWorkbenchConnectionRequest("/models").authToken).toBeNull();
  });

  it("fails closed until a bearer probe verifies the current revision", () => {
    observeWorkbenchAuthMode("bearer");
    expect(persistWorkbenchAuthToken("session-credential")).toEqual({ ok: true });
    const revision = beginWorkbenchConnectionTransition();
    commitWorkbenchAuthToken("session-credential");
    finishWorkbenchConnectionTransition();
    observeWorkbenchAuthMode("bearer");

    expect(() => captureWorkbenchConnectionRequest("/logs/runs"))
      .toThrow(/must be verified/i);
    const firstProbe = captureWorkbenchConnectionRequest("/models", true);
    expect(firstProbe.revision).toBe(revision);

    confirmWorkbenchAuthentication(
      revision,
      firstProbe.authenticationProbeGeneration,
    );
    expect(captureWorkbenchConnectionRequest("/logs/runs").authToken).toBe(
      "session-credential",
    );

    const recheck = captureWorkbenchConnectionRequest("/models", true);
    expect(() => captureWorkbenchConnectionRequest("/logs/runs"))
      .toThrow(/must be verified/i);
    confirmWorkbenchAuthentication(
      revision,
      recheck.authenticationProbeGeneration,
    );
    expect(captureWorkbenchConnectionRequest("/logs/runs").authToken).toBe(
      "session-credential",
    );
  });

  it("reports unavailable storage without throwing when window is missing", () => {
    vi.stubGlobal("window", undefined);

    expect(() => loadWorkbenchConnectionRuntime()).not.toThrow();
    expect(workbenchConnectionRuntimeSnapshot().storage).toEqual({
      apiBaseUrl: "unavailable",
      sessionToken: "unavailable",
    });
    expect(persistWorkbenchAuthToken("unpublished")).toMatchObject({ ok: false });
    expect(workbenchConnectionRuntimeSnapshot().hasAuthToken).toBe(false);
  });
});
