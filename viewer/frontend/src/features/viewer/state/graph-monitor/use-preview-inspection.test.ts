import { createElement, type ReactNode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";

// inspectModel is mocked so we control resolution order and exercise the
// stale-response race guard (requestIdRef) in usePreviewInspectionState.
const { inspectModelMock } = vi.hoisted(() => ({ inspectModelMock: vi.fn() }));
vi.mock("@/lib/api", () => ({ inspectModel: inspectModelMock }));

import { usePreviewInspectionState } from "@/features/viewer/state/graph-monitor/use-preview-inspection";

type Deferred<T> = {
  promise: Promise<T>;
  resolve: (value: T) => void;
};

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res) => {
    resolve = res;
  });
  return { promise, resolve };
}

function renderPreview() {
  const client = new QueryClient({
    defaultOptions: { mutations: { retry: false } },
  });
  return renderHook(() => usePreviewInspectionState(), {
    wrapper: ({ children }: { children: ReactNode }) =>
      createElement(QueryClientProvider, { client }, children),
  });
}

const request = (model: string) => ({ model, preset: "p", overrides: {} });

afterEach(() => {
  inspectModelMock.mockReset();
});

describe("usePreviewInspectionState", () => {
  it("clears the graph immediately and stores a successful response", async () => {
    const d = deferred<unknown>();
    inspectModelMock.mockReturnValueOnce(d.promise);
    const graph = {
      model: "a",
      preset: "p",
      parameterCount: 0,
      parameterSizeBytes: 0,
      nodes: [],
      edges: [],
    };

    const { result } = renderPreview();
    act(() => result.current.requestPreview(request("a")));

    expect(result.current.graph).toBeUndefined();
    await waitFor(() => expect(result.current.previewInspection.isBuilding).toBe(true));

    await act(async () => {
      d.resolve(graph);
      await d.promise;
    });

    await waitFor(() => expect(result.current.graph).toEqual(graph));
  });

  it("discards a stale response so the latest request wins", async () => {
    const first = deferred<unknown>();
    const second = deferred<unknown>();
    inspectModelMock.mockReturnValueOnce(first.promise).mockReturnValueOnce(second.promise);
    const graphA = {
      model: "A",
      preset: "p",
      parameterCount: 1,
      parameterSizeBytes: 4,
      nodes: [],
      edges: [],
    };
    const graphB = {
      model: "B",
      preset: "p",
      parameterCount: 2,
      parameterSizeBytes: 8,
      nodes: [],
      edges: [],
    };

    const { result } = renderPreview();
    act(() => result.current.requestPreview(request("A")));
    act(() => result.current.requestPreview(request("B")));

    // Newer request resolves first.
    await act(async () => {
      second.resolve(graphB);
      await second.promise;
    });
    await waitFor(() => expect(result.current.graph).toEqual(graphB));

    // Older request resolves later and must NOT clobber the newer graph.
    await act(async () => {
      first.resolve(graphA);
      await first.promise;
    });
    await waitFor(() => expect(result.current.graph).toEqual(graphB));
    expect(result.current.graph).not.toEqual(graphA);
  });
});
