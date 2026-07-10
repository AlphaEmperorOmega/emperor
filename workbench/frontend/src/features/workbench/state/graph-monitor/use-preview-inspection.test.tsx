import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { type ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  inspectModel: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  inspectModel: mocks.inspectModel,
}));

import {
  type PreviewInspectionRequest,
  usePreviewInspectionState,
} from "@/features/workbench/state/graph-monitor/use-preview-inspection";
import { type InspectResponse } from "@/lib/api";

type Deferred<T> = {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason: unknown) => void;
};

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });
  return { promise, resolve, reject };
}

function previewRequest(preset: string): PreviewInspectionRequest {
  return {
    modelType: "linears",
    model: "linear",
    preset,
    experimentTask: "image-classification",
    dataset: "Mnist",
    overrides: {},
  };
}

function previewResponse(preset: string): InspectResponse {
  return {
    modelType: "linears",
    model: "linear",
    preset,
    parameterCount: 0,
    parameterSizeBytes: 0,
    nodes: [],
    edges: [],
  };
}

describe("usePreviewInspectionState", () => {
  beforeEach(() => {
    mocks.inspectModel.mockReset();
  });

  it("aborts a superseded inspection without exposing cancellation as an error", async () => {
    const pendingByPreset = new Map<string, Deferred<InspectResponse>>();
    const signalsByPreset = new Map<string, AbortSignal | undefined>();
    mocks.inspectModel.mockImplementation(
      (
        input: { preset: string },
        options: { signal?: AbortSignal } = {},
      ) => {
        const pending = deferred<InspectResponse>();
        pendingByPreset.set(input.preset, pending);
        signalsByPreset.set(input.preset, options.signal);
        options.signal?.addEventListener(
          "abort",
          () => pending.reject(new DOMException("Aborted", "AbortError")),
          { once: true },
        );
        return pending.promise;
      },
    );
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    const { result } = renderHook(() => usePreviewInspectionState(), {
      wrapper: ({ children }: { children: ReactNode }) => (
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
      ),
    });

    act(() => result.current.requestPreview(previewRequest("baseline")));
    await waitFor(() => expect(pendingByPreset.has("baseline")).toBe(true));

    act(() => result.current.requestPreview(previewRequest("fast")));

    await waitFor(() => {
      expect(signalsByPreset.get("baseline")?.aborted).toBe(true);
      expect(pendingByPreset.has("fast")).toBe(true);
    });
    await act(async () => {
      pendingByPreset.get("fast")?.resolve(previewResponse("fast"));
      await pendingByPreset.get("fast")?.promise;
    });

    await waitFor(() => expect(result.current.graph?.preset).toBe("fast"));
    expect(result.current.previewInspection).toMatchObject({
      isBuilding: false,
      isError: false,
      error: null,
    });
  });
});
