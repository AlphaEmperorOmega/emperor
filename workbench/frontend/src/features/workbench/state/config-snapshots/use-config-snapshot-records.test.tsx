import { createElement, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  fetch: vi.fn(),
  create: vi.fn(),
  rename: vi.fn(),
  update: vi.fn(),
  remove: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  fetchConfigSnapshots: mocks.fetch,
  createConfigSnapshot: mocks.create,
  renameConfigSnapshot: mocks.rename,
  updateConfigSnapshot: mocks.update,
  deleteConfigSnapshot: mocks.remove,
}));

import { useConfigSnapshotRecords } from "@/features/workbench/state/config-snapshots/use-config-snapshot-records";

function createWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return createElement(QueryClientProvider, { client }, children);
  };
}

beforeEach(() => {
  Object.values(mocks).forEach((mock) => mock.mockReset());
  mocks.fetch.mockResolvedValue({
    modelType: "linears",
    model: "linear",
    snapshots: [],
  });
});

describe("Config Snapshot records protected access", () => {
  it("keeps the query and every mutation command inert until enabled", async () => {
    const rendered = renderHook(
      ({ enabled }) =>
        useConfigSnapshotRecords(
          { modelType: "linears", model: "linear" },
          { enabled },
        ),
      { wrapper: createWrapper(), initialProps: { enabled: false } },
    );

    act(() => {
      rendered.result.current.actions.create({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        name: "draft",
        overrides: {},
      });
      rendered.result.current.actions.rename({ id: "snapshot-1", name: "renamed" });
      rendered.result.current.actions.update({
        id: "snapshot-1",
        input: { name: "updated" },
      });
      rendered.result.current.actions.remove("snapshot-1");
    });

    expect(mocks.fetch).not.toHaveBeenCalled();
    expect(mocks.create).not.toHaveBeenCalled();
    expect(mocks.rename).not.toHaveBeenCalled();
    expect(mocks.update).not.toHaveBeenCalled();
    expect(mocks.remove).not.toHaveBeenCalled();

    rendered.rerender({ enabled: true });
    await waitFor(() => expect(mocks.fetch).toHaveBeenCalledTimes(1));
  });
});
