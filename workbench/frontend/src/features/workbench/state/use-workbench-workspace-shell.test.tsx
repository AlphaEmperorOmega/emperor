import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { useWorkbenchWorkspaceShell } from "@/features/workbench/state/use-workbench-workspace-shell";

describe("useWorkbenchWorkspaceShell", () => {
  it("remembers the first Logs activation after switching away", () => {
    const { result } = renderHook(() => useWorkbenchWorkspaceShell());

    expect(result.current.logsWorkspaceActivated).toBe(false);

    act(() => result.current.screen.onChangeWorkspace("logs"));
    expect(result.current.screen.activeWorkspace).toBe("logs");
    expect(result.current.logsWorkspaceActivated).toBe(true);

    act(() => result.current.screen.onChangeWorkspace("model"));
    expect(result.current.screen.activeWorkspace).toBe("model");
    expect(result.current.logsWorkspaceActivated).toBe(true);
  });
});
