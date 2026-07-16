import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  startTransition: vi.fn((update: () => void) => update()),
}));

vi.mock("react", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react")>();
  return { ...actual, startTransition: mocks.startTransition };
});

import { useWorkbenchWorkspaceShell } from "@/features/workbench/state/use-workbench-workspace-shell";

describe("Workbench workspace transitions", () => {
  beforeEach(() => {
    mocks.startTransition.mockClear();
    window.history.replaceState(null, "", "/?theme=night#workspace-content");
  });

  it("schedules user navigation and Back/Forward synchronization as transitions", () => {
    const { result } = renderHook(() => useWorkbenchWorkspaceShell());

    act(() => result.current.screen.onChangeWorkspace("logs"));
    expect(mocks.startTransition).toHaveBeenCalledTimes(1);
    expect(result.current.screen.activeWorkspace).toBe("logs");

    act(() => {
      window.history.replaceState(
        null,
        "",
        "/?theme=night&workspace=training#workspace-content",
      );
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    expect(mocks.startTransition).toHaveBeenCalledTimes(2);
    expect(result.current.screen.activeWorkspace).toBe("training");
  });
});
