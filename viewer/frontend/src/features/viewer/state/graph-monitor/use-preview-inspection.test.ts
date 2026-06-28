import { createElement, type ReactNode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";

// inspectModel is mocked so we control resolution order and exercise the
// stale-response race guard (requestIdRef) in usePreviewInspectionState.
const { inspectModelMock } = vi.hoisted(() => ({
  inspectModelMock: vi.fn(),
}));
vi.mock("@/lib/api", () => ({
  inspectModel: inspectModelMock,
}));

import { usePreviewInspectionState } from "@/features/viewer/state/graph-monitor/use-preview-inspection";

type Deferred<T> = {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason: unknown) => void;
};

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function renderPreview() {
  const client = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return renderHook(() => usePreviewInspectionState(), {
    wrapper: ({ children }: { children: ReactNode }) =>
      createElement(QueryClientProvider, { client }, children),
  });
}

const request = (model: string) => ({
  modelType: "linears",
  model,
  preset: "p",
  overrides: {},
});

afterEach(() => {
  inspectModelMock.mockReset();
});

describe("usePreviewInspectionState", () => {
  it("clears the graph immediately and stores a successful response", async () => {
    const d = deferred<unknown>();
    inspectModelMock.mockReturnValueOnce(d.promise);
    const graph = {
      modelType: "linears",
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
      modelType: "linears",
      model: "A",
      preset: "p",
      parameterCount: 1,
      parameterSizeBytes: 4,
      nodes: [],
      edges: [],
    };
    const graphB = {
      modelType: "linears",
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

  it("clears the current graph while a different preview builds", async () => {
    const next = deferred<unknown>();
    const graphA = {
      modelType: "linears",
      model: "A",
      preset: "p",
      parameterCount: 1,
      parameterSizeBytes: 4,
      nodes: [],
      edges: [],
    };
    const graphB = {
      modelType: "linears",
      model: "B",
      preset: "p",
      parameterCount: 2,
      parameterSizeBytes: 8,
      nodes: [],
      edges: [],
    };
    inspectModelMock.mockResolvedValueOnce(graphA).mockReturnValueOnce(next.promise);

    const { result } = renderPreview();
    act(() => result.current.requestPreview(request("A")));
    await waitFor(() => expect(result.current.graph).toEqual(graphA));

    act(() => result.current.requestPreview(request("B")));

    expect(result.current.graph).toBeUndefined();
    await waitFor(() => expect(result.current.previewInspection.isBuilding).toBe(true));

    await act(async () => {
      next.resolve(graphB);
      await next.promise;
    });
    await waitFor(() => expect(result.current.graph).toEqual(graphB));
  });

  it("reuses a cached preview response for repeated identical requests", async () => {
    const graph = {
      modelType: "linears",
      model: "A",
      preset: "p",
      parameterCount: 1,
      parameterSizeBytes: 4,
      nodes: [],
      edges: [],
    };
    inspectModelMock.mockResolvedValueOnce(graph);

    const { result } = renderPreview();
    act(() => result.current.requestPreview(request("A")));
    await waitFor(() => expect(result.current.graph).toEqual(graph));

    inspectModelMock.mockClear();
    act(() => result.current.requestPreview(request("A")));

    await waitFor(() =>
      expect(result.current.previewInspection.isBuilding).toBe(false),
    );
    expect(result.current.graph).toEqual(graph);
    expect(inspectModelMock).not.toHaveBeenCalled();
  });

  it("deduplicates identical in-flight preview requests", async () => {
    const pending = deferred<unknown>();
    const graph = {
      modelType: "linears",
      model: "A",
      preset: "p",
      parameterCount: 1,
      parameterSizeBytes: 4,
      nodes: [],
      edges: [],
    };
    inspectModelMock.mockReturnValueOnce(pending.promise);

    const { result } = renderPreview();
    act(() => result.current.requestPreview(request("A")));
    act(() => result.current.requestPreview(request("A")));

    expect(inspectModelMock).toHaveBeenCalledTimes(1);

    await act(async () => {
      pending.resolve(graph);
      await pending.promise;
    });
    await waitFor(() => expect(result.current.graph).toEqual(graph));
  });

  it("rejects inspect responses whose identity does not match the request", async () => {
    inspectModelMock.mockResolvedValueOnce({
      modelType: "linears",
      model: "B",
      preset: "p",
      parameterCount: 1,
      parameterSizeBytes: 4,
      nodes: [],
      edges: [],
    });

    const { result } = renderPreview();
    act(() => result.current.requestPreview(request("A")));

    await waitFor(() => expect(result.current.previewInspection.isError).toBe(true));
    expect(result.current.graph).toBeUndefined();
    expect(result.current.previewInspection.error).toBeInstanceOf(Error);
    expect(String(result.current.previewInspection.error)).toContain(
      "requested linears/A/p, received linears/B/p",
    );
  });

  it("accepts canonical preset names returned for historical run preset labels", async () => {
    inspectModelMock.mockResolvedValueOnce({
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      parameterCount: 1,
      parameterSizeBytes: 4,
      nodes: [],
      edges: [],
    });

    const { result } = renderPreview();
    act(() =>
      result.current.requestPreview({
        modelType: "linears",
        model: "linear",
        preset: "BASELINE",
        dataset: "Mnist",
        overrides: {},
        targetMode: "experiment",
        targetId: "run-1",
        logRunId: "run-1",
      }),
    );

    await waitFor(() => expect(result.current.graph?.preset).toBe("baseline"));
    expect(result.current.previewInspection.isError).toBe(false);
  });

  it("passes logRunId through for experiment preview requests", async () => {
    inspectModelMock.mockResolvedValueOnce({
      modelType: "linears",
      model: "A",
      preset: "p",
      parameterCount: 1,
      parameterSizeBytes: 4,
      nodes: [],
      edges: [],
    });

    const { result } = renderPreview();
    act(() =>
      result.current.requestPreview({
        ...request("A"),
        targetMode: "experiment",
        targetId: "run-1",
        logRunId: "run-1",
      }),
    );

    await waitFor(() => expect(result.current.graph?.model).toBe("A"));
    expect(inspectModelMock).toHaveBeenCalledWith({
      modelType: "linears",
      model: "A",
      preset: "p",
      overrides: {},
      logRunId: "run-1",
    });
  });

  it("clears preview state and ignores in-flight responses", async () => {
    const pending = deferred<unknown>();
    inspectModelMock.mockReturnValueOnce(pending.promise);
    const graphA = {
      modelType: "linears",
      model: "A",
      preset: "p",
      parameterCount: 0,
      parameterSizeBytes: 0,
      nodes: [],
      edges: [],
    };

    const { result } = renderPreview();
    act(() => result.current.requestPreview(request("A")));
    await waitFor(() => expect(result.current.previewInspection.isBuilding).toBe(true));

    act(() => result.current.clearPreview());

    expect(result.current.graph).toBeUndefined();
    expect(result.current.previewRequest).toBeNull();
    expect(result.current.previewRequestKey).toBeNull();

    await act(async () => {
      pending.resolve(graphA);
      await pending.promise;
    });
    await waitFor(() => expect(result.current.graph).toBeUndefined());
  });
});
