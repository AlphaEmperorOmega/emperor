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

vi.mock("@/lib/api/client", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api/client")>()),
  createMutationRequestOptions: () => ({ idempotencyKey: "snapshot-command" }),
}));
vi.mock("@/lib/api/config-snapshots", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/api/config-snapshots")>()),
  fetchConfigSnapshots: mocks.fetch,
  createConfigSnapshot: mocks.create,
  renameConfigSnapshot: mocks.rename,
  updateConfigSnapshot: mocks.update,
  deleteConfigSnapshot: mocks.remove,
}));

import {
  type ConfigSnapshotMutationOutcome,
  useConfigSnapshotRecords,
} from "@/features/workbench/state/config-snapshots/use-config-snapshot-records";

const identity = { modelType: "linears", model: "linear" };

function record(overrides: Record<string, unknown> = {}) {
  return {
    id: "snapshot-1",
    modelType: identity.modelType,
    model: identity.model,
    preset: "baseline",
    name: "Snapshot one",
    overrides: {},
    createdAt: "2026-06-01T00:00:00.000Z",
    updatedAt: "2026-06-01T00:00:00.000Z",
    ...overrides,
  };
}

function createHarness() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return {
    client,
    wrapper({ children }: { children: ReactNode }) {
      return createElement(QueryClientProvider, { client }, children);
    },
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

beforeEach(() => {
  Object.values(mocks).forEach((mock) => mock.mockReset());
  mocks.fetch.mockResolvedValue({
    ...identity,
    snapshots: [record()],
  });
  mocks.create.mockResolvedValue(record());
  mocks.rename.mockImplementation(async (id: string, name: string) =>
    record({ id, name }),
  );
  mocks.update.mockImplementation(
    async (id: string, input: { name?: string; overrides?: Record<string, string> }) =>
      record({ id, ...input }),
  );
  mocks.remove.mockResolvedValue({ ...identity, snapshots: [] });
});

describe("Config Snapshot records mutation lifecycle", () => {
  it("keeps the query and every mutation command inert until enabled", async () => {
    const harness = createHarness();
    const rendered = renderHook(
      ({ enabled }) => useConfigSnapshotRecords(identity, { enabled }),
      { wrapper: harness.wrapper, initialProps: { enabled: false } },
    );

    await act(async () => {
      await expect(
        rendered.result.current.actions.create({
          ...identity,
          preset: "baseline",
          name: "draft",
          overrides: {},
        }),
      ).resolves.toMatchObject({ ok: false, retryable: false });
      await rendered.result.current.actions.rename({
        id: "snapshot-1",
        name: "renamed",
      });
      await rendered.result.current.actions.update({
        id: "snapshot-1",
        input: { name: "updated" },
      });
      await rendered.result.current.actions.remove("snapshot-1");
    });

    expect(mocks.fetch).not.toHaveBeenCalled();
    expect(mocks.create).not.toHaveBeenCalled();
    expect(mocks.rename).not.toHaveBeenCalled();
    expect(mocks.update).not.toHaveBeenCalled();
    expect(mocks.remove).not.toHaveBeenCalled();
    expect(rendered.result.current.status.mutation.phase).toBe("idle");

    rendered.rerender({ enabled: true });
    await waitFor(() => expect(mocks.fetch).toHaveBeenCalledTimes(1));
  });

  it("publishes pending until persistence and invalidation both complete", async () => {
    const pendingCreate = deferred<ReturnType<typeof record>>();
    mocks.create.mockReturnValueOnce(pendingCreate.promise);
    const harness = createHarness();
    const { result } = renderHook(() => useConfigSnapshotRecords(identity), {
      wrapper: harness.wrapper,
    });
    await waitFor(() => expect(result.current.status.isReady).toBe(true));

    let createPromise!: Promise<ConfigSnapshotMutationOutcome>;
    act(() => {
      createPromise = result.current.actions.create({
        ...identity,
        preset: "baseline",
        name: "Pending",
        overrides: { hidden_dim: "128" },
      });
    });
    await waitFor(() => {
      expect(result.current.status.mutation).toMatchObject({
        phase: "pending",
        kind: "create",
        error: "",
        canRetry: false,
      });
    });
    expect(mocks.fetch).toHaveBeenCalledTimes(1);

    let outcome!: ConfigSnapshotMutationOutcome;
    await act(async () => {
      pendingCreate.resolve(record({ id: "created", name: "Pending" }));
      outcome = await createPromise;
    });

    expect(outcome).toMatchObject({
      ok: true,
      kind: "create",
      snapshotId: "created",
    });
    await waitFor(() => {
      expect(result.current.status.mutation).toMatchObject({
        phase: "succeeded",
        kind: "create",
        snapshotId: "created",
      });
    });
    expect(mocks.fetch).toHaveBeenCalledTimes(2);
  });

  it("retains and retries the exact failed create, rename, update, and removal commands", async () => {
    const harness = createHarness();
    const { result } = renderHook(() => useConfigSnapshotRecords(identity), {
      wrapper: harness.wrapper,
    });
    await waitFor(() => expect(result.current.status.isReady).toBe(true));

    const cases = [
      {
        kind: "create",
        api: mocks.create,
        success: record({ id: "created" }),
        run: () =>
          result.current.actions.create({
            ...identity,
            preset: "baseline",
            name: "Created",
            overrides: { hidden_dim: "128" },
          }),
      },
      {
        kind: "rename",
        api: mocks.rename,
        success: record({ name: "Renamed" }),
        run: () =>
          result.current.actions.rename({
            id: "snapshot-1",
            name: "Renamed",
          }),
      },
      {
        kind: "update",
        api: mocks.update,
        success: record({ name: "Updated", overrides: { hidden_dim: "192" } }),
        run: () =>
          result.current.actions.update({
            id: "snapshot-1",
            input: {
              name: "Updated",
              overrides: { hidden_dim: "192" },
            },
          }),
      },
      {
        kind: "remove",
        api: mocks.remove,
        success: { ...identity, snapshots: [] },
        run: () => result.current.actions.remove("snapshot-1"),
      },
    ] as const;

    for (const mutationCase of cases) {
      mutationCase.api
        .mockRejectedValueOnce(new Error(`${mutationCase.kind} rejected`))
        .mockResolvedValueOnce(mutationCase.success);
      const callCount = mutationCase.api.mock.calls.length;

      let failed!: ConfigSnapshotMutationOutcome;
      await act(async () => {
        failed = await mutationCase.run();
      });
      expect(failed).toMatchObject({
        ok: false,
        kind: mutationCase.kind,
        error: `${mutationCase.kind} rejected`,
        retryable: true,
      });
      await waitFor(() => {
        expect(result.current.status.mutation).toMatchObject({
          phase: "failed",
          kind: mutationCase.kind,
          error: `${mutationCase.kind} rejected`,
          canRetry: true,
        });
      });

      let retried!: ConfigSnapshotMutationOutcome | null;
      await act(async () => {
        retried = await result.current.actions.retry();
      });
      expect(retried).toMatchObject({ ok: true, kind: mutationCase.kind });
      expect(mutationCase.api).toHaveBeenCalledTimes(callCount + 2);
      expect(mutationCase.api.mock.calls.at(-1)).toEqual(
        mutationCase.api.mock.calls.at(-2),
      );
      await waitFor(() => {
        expect(result.current.status.mutation).toMatchObject({
          phase: "succeeded",
          kind: mutationCase.kind,
        });
      });

      act(() => result.current.actions.dismissMutation());
      await waitFor(() => {
        expect(result.current.status.mutation.phase).toBe("idle");
      });
    }

    expect(mocks.fetch).toHaveBeenCalledTimes(4);
  });

  it("quarantines late mutation completion after a connection change", async () => {
    const pendingRemoval = deferred<{ snapshots: never[] } & typeof identity>();
    mocks.remove.mockReturnValueOnce(pendingRemoval.promise);
    const harness = createHarness();
    const { result } = renderHook(() => useConfigSnapshotRecords(identity), {
      wrapper: harness.wrapper,
    });
    await waitFor(() => expect(result.current.status.isReady).toBe(true));

    let removePromise!: Promise<ConfigSnapshotMutationOutcome>;
    act(() => {
      removePromise = result.current.actions.remove("snapshot-1");
    });
    await waitFor(() => {
      expect(result.current.status.mutation.phase).toBe("pending");
    });

    act(() => result.current.actions.clearForConnectionChange());
    await waitFor(() => {
      expect(result.current.status.mutation.phase).toBe("idle");
    });

    let outcome!: ConfigSnapshotMutationOutcome;
    await act(async () => {
      pendingRemoval.resolve({ ...identity, snapshots: [] });
      outcome = await removePromise;
    });
    expect(outcome).toMatchObject({
      ok: false,
      kind: "remove",
      retryable: false,
    });
    expect(result.current.status.mutation.phase).toBe("idle");
    expect(mocks.fetch).toHaveBeenCalledTimes(1);
  });
});
