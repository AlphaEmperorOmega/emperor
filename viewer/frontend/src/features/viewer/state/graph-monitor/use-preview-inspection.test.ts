import { createElement, type ReactNode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, describe, expect, it, vi } from "vitest";

// inspectModel is mocked so we control resolution order and exercise the
// stale-response race guard (requestIdRef) in usePreviewInspectionState.
const { inspectModelMock, inspectOperationGraphMock } = vi.hoisted(() => ({
  inspectModelMock: vi.fn(),
  inspectOperationGraphMock: vi.fn(),
}));
vi.mock("@/lib/api", () => ({
  inspectModel: inspectModelMock,
  inspectOperationGraph: inspectOperationGraphMock,
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
  inspectOperationGraphMock.mockReset();
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

  it("rejects inspect responses whose identity does not match the request", async () => {
    inspectModelMock.mockResolvedValueOnce({
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
      "requested A/p, received B/p",
    );
  });

  it("rejects operation graph responses whose identity does not match the request", async () => {
    inspectModelMock.mockImplementation((input: { model: string; preset: string }) => {
      return Promise.resolve({
        model: input.model,
        preset: input.preset,
        parameterCount: 0,
        parameterSizeBytes: 0,
        nodes: [],
        edges: [],
      });
    });
    inspectOperationGraphMock.mockResolvedValueOnce({
      model: "B",
      preset: "p",
      source: "torch-export",
      status: "ok",
      nodes: [],
      edges: [],
      warnings: [],
    });

    const { result } = renderPreview();
    act(() => result.current.requestPreview(request("A")));
    await waitFor(() => expect(result.current.graph?.model).toBe("A"));
    act(() => result.current.requestOperationGraph(request("A")));

    await waitFor(() => expect(result.current.operationInspection.isError).toBe(true));
    expect(result.current.operationGraph).toBeUndefined();
    expect(result.current.operationGraphRequestKey).toBeNull();
    expect(result.current.operationGraphFailedRequestKey).not.toBeNull();
    expect(result.current.operationInspection.error).toBeInstanceOf(Error);
    expect(String(result.current.operationInspection.error)).toContain(
      "requested A/p, received B/p",
    );
  });

  it("clears preview state and ignores in-flight responses", async () => {
    const operation = deferred<unknown>();
    inspectModelMock.mockImplementation((input: { model: string; preset: string }) => {
      return Promise.resolve({
        model: input.model,
        preset: input.preset,
        parameterCount: 0,
        parameterSizeBytes: 0,
        nodes: [],
        edges: [],
      });
    });
    inspectOperationGraphMock.mockReturnValueOnce(operation.promise);
    const operationA = {
      model: "A",
      preset: "p",
      source: "torch-export",
      status: "ok",
      nodes: [],
      edges: [],
      warnings: [],
    };

    const { result } = renderPreview();
    act(() => result.current.requestPreview(request("A")));
    await waitFor(() => expect(result.current.graph?.model).toBe("A"));
    act(() => result.current.requestOperationGraph(request("A")));
    await waitFor(() =>
      expect(result.current.operationInspection.isBuilding).toBe(true),
    );

    act(() => result.current.clearPreview());

    expect(result.current.graph).toBeUndefined();
    expect(result.current.operationGraph).toBeUndefined();
    expect(result.current.previewRequest).toBeNull();
    expect(result.current.previewRequestKey).toBeNull();
    expect(result.current.operationGraphRequestKey).toBeNull();
    expect(result.current.operationGraphInFlightRequestKey).toBeNull();
    expect(result.current.operationGraphFailedRequestKey).toBeNull();
    expect(result.current.operationInspection.isError).toBe(false);

    await act(async () => {
      operation.resolve(operationA);
      await operation.promise;
    });
    await waitFor(() => expect(result.current.operationGraph).toBeUndefined());
  });

  it("clears and guards operation graph responses independently", async () => {
    const first = deferred<unknown>();
    const second = deferred<unknown>();
    inspectModelMock.mockImplementation((input: { model: string; preset: string }) => {
      return Promise.resolve({
        model: input.model,
        preset: input.preset,
        parameterCount: 0,
        parameterSizeBytes: 0,
        nodes: [],
        edges: [],
      });
    });
    inspectOperationGraphMock
      .mockReturnValueOnce(first.promise)
      .mockReturnValueOnce(second.promise);
    const operationA = {
      model: "A",
      preset: "p",
      source: "torch-export",
      status: "ok",
      nodes: [],
      edges: [],
      warnings: [],
    };
    const operationA2 = {
      ...operationA,
      warnings: ["latest"],
    };
    const { result } = renderPreview();
    act(() => result.current.requestPreview(request("A")));
    await waitFor(() => expect(result.current.graph?.model).toBe("A"));
    act(() => result.current.requestOperationGraph(request("A")));
    act(() => result.current.requestOperationGraph(request("A")));

    await act(async () => {
      second.resolve(operationA2);
      await second.promise;
    });
    await waitFor(() => expect(result.current.operationGraph).toEqual(operationA2));

    await act(async () => {
      first.resolve(operationA);
      await first.promise;
    });
    await waitFor(() => expect(result.current.operationGraph).toEqual(operationA2));

    act(() => result.current.requestOperationGraph(request("B")));
    expect(inspectOperationGraphMock).toHaveBeenCalledTimes(2);
    expect(result.current.operationGraph).toEqual(operationA2);

    act(() => result.current.requestPreview(request("C")));
    expect(result.current.operationGraph).toBeUndefined();
    expect(result.current.operationGraphRequestKey).toBeNull();
    await waitFor(() => expect(result.current.graph?.model).toBe("C"));
  });

  it("marks failed operation graph requests retryable for the current preview", async () => {
    const failed = deferred<unknown>();
    const retry = deferred<unknown>();
    inspectModelMock.mockImplementation((input: { model: string; preset: string }) => {
      return Promise.resolve({
        model: input.model,
        preset: input.preset,
        parameterCount: 0,
        parameterSizeBytes: 0,
        nodes: [],
        edges: [],
      });
    });
    inspectOperationGraphMock
      .mockReturnValueOnce(failed.promise)
      .mockReturnValueOnce(retry.promise);
    const operationA = {
      model: "A",
      preset: "p",
      source: "torch-export",
      status: "ok",
      nodes: [],
      edges: [],
      warnings: [],
    };

    const { result } = renderPreview();
    act(() => result.current.requestPreview(request("A")));
    await waitFor(() => expect(result.current.graph?.model).toBe("A"));
    act(() => result.current.requestOperationGraph(request("A")));

    await act(async () => {
      failed.reject(new Error("trace failed"));
      await failed.promise.catch(() => undefined);
    });
    await waitFor(() =>
      expect(result.current.operationInspection.isError).toBe(true),
    );
    expect(result.current.operationGraphFailedRequestKey).not.toBeNull();
    expect(result.current.operationGraphRequestKey).toBeNull();

    act(() => result.current.requestOperationGraph(request("A")));
    await waitFor(() =>
      expect(result.current.operationInspection.isError).toBe(false),
    );
    await waitFor(() => expect(inspectOperationGraphMock).toHaveBeenCalledTimes(2));

    await act(async () => {
      retry.resolve(operationA);
      await retry.promise;
    });
    await waitFor(() => expect(result.current.operationGraph).toEqual(operationA));
    expect(result.current.operationGraphFailedRequestKey).toBeNull();
  });
});
