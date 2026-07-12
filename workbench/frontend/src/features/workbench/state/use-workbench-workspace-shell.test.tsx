import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { useWorkbenchWorkspaceShell } from "@/features/workbench/state/use-workbench-workspace-shell";
import { parseWorkbenchWorkspace } from "@/types/workbench";

describe("useWorkbenchWorkspaceShell", () => {
  beforeEach(() => {
    window.history.replaceState(null, "", "/?theme=night#workspace-content");
  });

  it.each([
    ["model", "model"],
    ["training", "training"],
    ["logs", "logs"],
    ["invalid", "model"],
    [undefined, "model"],
  ] as const)("parses %s as the %s workspace", (value, expected) => {
    expect(parseWorkbenchWorkspace(value)).toBe(expected);
  });

  it.each(["training", "logs"] as const)(
    "activates the deferred provider for a direct %s link",
    (workspace) => {
      const { result } = renderHook(() =>
        useWorkbenchWorkspaceShell(workspace),
      );

      expect(result.current.screen.activeWorkspace).toBe(workspace);
      expect(result.current.deferredWorkspaceOrder).toEqual([workspace]);
    },
  );

  it("pushes workspace navigation while preserving unrelated URL state", () => {
    const pushState = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useWorkbenchWorkspaceShell());

    act(() => result.current.screen.onChangeWorkspace("logs"));

    const currentUrl = new URL(window.location.href);
    expect(currentUrl.searchParams.get("workspace")).toBe("logs");
    expect(currentUrl.searchParams.get("theme")).toBe("night");
    expect(currentUrl.hash).toBe("#workspace-content");
    expect(pushState).toHaveBeenCalledTimes(1);

    pushState.mockRestore();
  });

  it("synchronizes Back and Forward state without pushing another entry", () => {
    const pushState = vi.spyOn(window.history, "pushState");
    const { result } = renderHook(() => useWorkbenchWorkspaceShell("logs"));

    act(() => {
      result.current.screen.importLogsDialog.open();
      result.current.screen.apiConnectionDialog.open();
    });
    expect(result.current.screen.importLogsDialog.isOpen).toBe(true);

    window.history.replaceState(
      null,
      "",
      "/?theme=night&workspace=training#workspace-content",
    );
    act(() => window.dispatchEvent(new PopStateEvent("popstate")));

    expect(result.current.screen.activeWorkspace).toBe("training");
    expect(result.current.deferredWorkspaceOrder).toEqual(["logs", "training"]);
    expect(result.current.screen.importLogsDialog.isOpen).toBe(false);
    expect(result.current.screen.apiConnectionDialog.isOpen).toBe(false);
    expect(pushState).not.toHaveBeenCalled();

    window.history.replaceState(
      null,
      "",
      "/?theme=night&workspace=not-a-workspace#workspace-content",
    );
    act(() => window.dispatchEvent(new PopStateEvent("popstate")));
    expect(result.current.screen.activeWorkspace).toBe("model");
    expect(pushState).not.toHaveBeenCalled();

    pushState.mockRestore();
  });

  it("remembers the first Logs activation after switching away", () => {
    const { result } = renderHook(() => useWorkbenchWorkspaceShell());

    expect(result.current.logsWorkspaceActivated).toBe(false);
    expect(result.current.deferredWorkspaceOrder).toEqual([]);

    act(() => result.current.screen.onChangeWorkspace("logs"));
    expect(result.current.screen.activeWorkspace).toBe("logs");
    expect(result.current.logsWorkspaceActivated).toBe(true);
    expect(result.current.deferredWorkspaceOrder).toEqual(["logs"]);

    act(() => result.current.screen.onChangeWorkspace("model"));
    expect(result.current.screen.activeWorkspace).toBe("model");
    expect(result.current.logsWorkspaceActivated).toBe(true);
    expect(result.current.deferredWorkspaceOrder).toEqual(["logs"]);
  });

  it("keeps training activated after switching away", () => {
    const { result } = renderHook(() => useWorkbenchWorkspaceShell());

    expect(result.current.trainingWorkspaceActivated).toBe(false);
    expect(result.current.deferredWorkspaceOrder).toEqual([]);

    act(() => result.current.screen.onChangeWorkspace("training"));
    expect(result.current.screen.activeWorkspace).toBe("training");
    expect(result.current.trainingWorkspaceActivated).toBe(true);
    expect(result.current.deferredWorkspaceOrder).toEqual(["training"]);

    act(() => result.current.screen.onChangeWorkspace("model"));
    expect(result.current.screen.activeWorkspace).toBe("model");
    expect(result.current.trainingWorkspaceActivated).toBe(true);
    expect(result.current.deferredWorkspaceOrder).toEqual(["training"]);
  });

  it("keeps deferred workspaces in first-activation order", () => {
    const { result } = renderHook(() => useWorkbenchWorkspaceShell());

    act(() => result.current.screen.onChangeWorkspace("logs"));
    act(() => result.current.screen.onChangeWorkspace("training"));
    act(() => result.current.screen.onChangeWorkspace("logs"));

    expect(result.current.deferredWorkspaceOrder).toEqual(["logs", "training"]);
  });
});
