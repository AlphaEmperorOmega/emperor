import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  useLogExperimentsQuery: vi.fn(),
}));

vi.mock("@/features/workbench/state/logs/use-log-queries", () => ({
  useLogExperimentsQuery: mocks.useLogExperimentsQuery,
}));

import { useTrainingLogFolderState } from "@/features/workbench/state/training/use-training-log-folder-state";

beforeEach(() => {
  mocks.useLogExperimentsQuery.mockReset().mockReturnValue({
    data: {
      experiments: [
        { experiment: "scratch", runCount: 2, relativePath: "scratch" },
      ],
    },
    isLoading: false,
  });
});

describe("Training Log folder ownership", () => {
  it("owns existing/new selection, validation, and connection reset commands", () => {
    const { result } = renderHook(() => useTrainingLogFolderState());

    expect(result.current.state).toMatchObject({
      mode: "existing",
      value: "",
      isValid: false,
    });
    act(() => result.current.actions.selectExisting("scratch"));
    expect(result.current.state).toMatchObject({
      value: "scratch",
      isValid: true,
    });

    act(() => {
      result.current.actions.selectMode("new");
      result.current.actions.nameNew("bad folder");
    });
    expect(result.current.state).toMatchObject({
      mode: "new",
      value: "bad folder",
      isValid: false,
      newError: "Use letters and numbers separated by single underscores.",
    });
    act(() => result.current.actions.nameNew("new_folder"));
    expect(result.current.state).toMatchObject({
      value: "new_folder",
      isValid: true,
      newError: "",
    });

    act(() => result.current.clearForConnectionChange());
    expect(result.current.state).toMatchObject({
      mode: "existing",
      existingValue: "",
      newValue: "",
      value: "",
      isValid: false,
    });
  });
});
