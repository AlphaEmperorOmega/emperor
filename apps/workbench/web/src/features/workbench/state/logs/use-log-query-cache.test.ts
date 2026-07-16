import { createElement, type ReactNode } from "react";
import { act, renderHook } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";
import { useLogQueryCache } from "@/features/workbench/state/logs/use-log-query-cache";
import {
  LOG_ARTIFACTS_QUERY_KEY,
  LOG_CHECKPOINTS_QUERY_KEY,
  LOG_EXPERIMENTS_QUERY_KEY,
  LOG_MEDIA_QUERY_KEY,
  LOG_RUNS_QUERY_KEY,
  LOG_SCALARS_QUERY_KEY,
  LOG_TAGS_QUERY_KEY,
} from "@/lib/query-keys";

function renderLogQueryCache() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  const invalidateSpy = vi.spyOn(client, "invalidateQueries");
  const removeSpy = vi.spyOn(client, "removeQueries");
  const hook = renderHook(() => useLogQueryCache(), {
    wrapper: ({ children }: { children: ReactNode }) =>
      createElement(QueryClientProvider, { client }, children),
  });
  return { client, invalidateSpy, removeSpy, ...hook };
}

describe("useLogQueryCache", () => {
  it("refreshes log lists and removes affected run detail caches after mutations", async () => {
    const { client, invalidateSpy, removeSpy, result } = renderLogQueryCache();
    client.setQueryData([...LOG_TAGS_QUERY_KEY, ["run-1"]], "run-1-tags");
    client.setQueryData([...LOG_TAGS_QUERY_KEY, ["run-2"]], "run-2-tags");
    client.setQueryData([...LOG_CHECKPOINTS_QUERY_KEY, ["run-1"]], "run-1-checkpoints");
    client.setQueryData([...LOG_ARTIFACTS_QUERY_KEY, "run-1"], "run-1-artifacts");
    client.setQueryData([...LOG_MEDIA_QUERY_KEY, ["run-1"], ["image"], []], "run-1-media");
    client.setQueryData([...LOG_SCALARS_QUERY_KEY, ["run-1"], ["loss"]], "run-1-loss");
    client.setQueryData([...LOG_SCALARS_QUERY_KEY, ["run-2"], ["loss"]], "run-2-loss");

    await act(async () => {
      await result.current.refreshAfterMutation({ runId: "run-1" });
    });

    expect(invalidateSpy).toHaveBeenCalledTimes(2);
    expect(invalidateSpy).toHaveBeenNthCalledWith(1, {
      queryKey: LOG_EXPERIMENTS_QUERY_KEY,
    });
    expect(invalidateSpy).toHaveBeenNthCalledWith(2, {
      queryKey: LOG_RUNS_QUERY_KEY,
    });
    expect(removeSpy).toHaveBeenCalledTimes(5);
    expect(client.getQueryData([...LOG_TAGS_QUERY_KEY, ["run-1"]])).toBeUndefined();
    expect(client.getQueryData([...LOG_CHECKPOINTS_QUERY_KEY, ["run-1"]])).toBeUndefined();
    expect(client.getQueryData([...LOG_ARTIFACTS_QUERY_KEY, "run-1"])).toBeUndefined();
    expect(
      client.getQueryData([...LOG_MEDIA_QUERY_KEY, ["run-1"], ["image"], []]),
    ).toBeUndefined();
    expect(
      client.getQueryData([...LOG_SCALARS_QUERY_KEY, ["run-1"], ["loss"]]),
    ).toBeUndefined();
    expect(client.getQueryData([...LOG_TAGS_QUERY_KEY, ["run-2"]])).toBe("run-2-tags");
    expect(client.getQueryData([...LOG_SCALARS_QUERY_KEY, ["run-2"], ["loss"]])).toBe(
      "run-2-loss",
    );
    expect(invalidateSpy.mock.invocationCallOrder[1]).toBeLessThan(
      removeSpy.mock.invocationCallOrder[0],
    );
  });

  it("invalidates only the log list queries", async () => {
    const { invalidateSpy, removeSpy, result } = renderLogQueryCache();

    await act(async () => {
      await result.current.invalidateLogLists();
    });

    expect(invalidateSpy).toHaveBeenCalledTimes(2);
    expect(invalidateSpy).toHaveBeenNthCalledWith(1, {
      queryKey: LOG_EXPERIMENTS_QUERY_KEY,
    });
    expect(invalidateSpy).toHaveBeenNthCalledWith(2, {
      queryKey: LOG_RUNS_QUERY_KEY,
    });
    expect(removeSpy).not.toHaveBeenCalled();
  });

  it("marks every Logs query family stale through one semantic refresh", async () => {
    const { invalidateSpy, removeSpy, result } = renderLogQueryCache();

    await act(async () => {
      await result.current.refreshLogs();
    });

    expect(invalidateSpy.mock.calls.map(([filters]) => filters)).toEqual([
      { queryKey: LOG_EXPERIMENTS_QUERY_KEY },
      { queryKey: LOG_RUNS_QUERY_KEY },
      { queryKey: LOG_TAGS_QUERY_KEY },
      { queryKey: LOG_SCALARS_QUERY_KEY },
      { queryKey: LOG_MEDIA_QUERY_KEY },
      { queryKey: LOG_CHECKPOINTS_QUERY_KEY },
      { queryKey: LOG_ARTIFACTS_QUERY_KEY },
    ]);
    expect(removeSpy).not.toHaveBeenCalled();
  });

  it("removes only affected caches for a mutation run-id set", async () => {
    const { client, removeSpy, result } = renderLogQueryCache();
    client.setQueryData([...LOG_SCALARS_QUERY_KEY, ["run-1"], ["loss"]], "loss");
    client.setQueryData([...LOG_SCALARS_QUERY_KEY, ["run-2"], ["accuracy"]], "accuracy");
    client.setQueryData([...LOG_SCALARS_QUERY_KEY, ["run-3"], ["loss"]], "kept");
    client.setQueryData([...LOG_TAGS_QUERY_KEY, ["run-3"]], "tags");

    await act(async () => {
      await result.current.refreshAfterMutation({ runIds: ["run-1", "run-2"] });
    });

    expect(removeSpy).toHaveBeenCalledTimes(5);
    expect(client.getQueryData([...LOG_SCALARS_QUERY_KEY, ["run-1"], ["loss"]])).toBeUndefined();
    expect(
      client.getQueryData([...LOG_SCALARS_QUERY_KEY, ["run-2"], ["accuracy"]]),
    ).toBeUndefined();
    expect(client.getQueryData([...LOG_SCALARS_QUERY_KEY, ["run-3"], ["loss"]])).toBe(
      "kept",
    );
    expect(client.getQueryData([...LOG_TAGS_QUERY_KEY, ["run-3"]])).toBe("tags");
  });
});
