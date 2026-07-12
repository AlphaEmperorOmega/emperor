import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { useWorkbenchWorkspaceShell } from "@/features/workbench/state/use-workbench-workspace-shell";

describe("useWorkbenchWorkspaceShell", () => {
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
