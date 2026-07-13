import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  useModelPackageMetadata: vi.fn(),
  useConfigSnapshotRecords: vi.fn(),
}));

vi.mock(
  "@/features/workbench/state/model-package/use-model-package-metadata",
  () => ({ useModelPackageMetadata: mocks.useModelPackageMetadata }),
);

vi.mock(
  "@/features/workbench/state/config-snapshots/use-config-snapshot-records",
  () => ({ useConfigSnapshotRecords: mocks.useConfigSnapshotRecords }),
);

import { type ConfigField } from "@/lib/api";
import { type ModelPackageMetadataSelection } from "@/features/workbench/state/model-package/use-model-package-metadata";
import { useTrainingDraftState } from "@/features/workbench/state/training/use-training-draft-state";
import { type WorkbenchWorkspace } from "@/types/workbench";

const ready = { isLoading: false, isReady: true, isError: false, error: null };

const hiddenDim: ConfigField = {
  key: "hidden_dim",
  configKey: "HIDDEN_DIM",
  flag: "--hidden-dim",
  label: "Hidden dim",
  section: "Model",
  sectionPath: ["Model"],
  type: "int",
  default: 256,
  nullable: false,
  choices: [],
  locked: false,
};

function installModelPackageMetadata() {
  mocks.useModelPackageMetadata.mockImplementation(
    ({ modelPackage }: ModelPackageMetadataSelection) => ({
      modelPackages: { records: [modelPackage], ...ready },
      presets: {
        records: [{ name: "baseline", label: "Baseline", description: "" }],
        ...ready,
      },
      datasetMetadata: {
        defaultExperimentTask: "image-classification",
        groups: [
          {
            experimentTask: "image-classification",
            label: "Image classification",
            datasets: [
              { name: "Mnist", label: "MNIST", inputDim: 784, outputDim: 10 },
            ],
          },
        ],
        ...ready,
      },
      monitorMetadata: {
        records: [
          {
            name: "loss",
            label: "Loss",
            description: "Training loss",
            kinds: ["scalar"],
            defaultEnabled: false,
          },
        ],
        ...ready,
      },
      runtimeDefaults: { fields: [hiddenDim], ...ready },
      searchMetadata: { axes: [], ...ready },
    }),
  );
}

const idleMutation = {
  phase: "idle" as const,
  kind: null,
  snapshotId: null,
  error: "",
  canRetry: false,
};

function installSnapshotRecords(records: Array<Record<string, unknown>> = []) {
  mocks.useConfigSnapshotRecords.mockReturnValue({
    records,
    status: {
      isLoading: false,
      isReady: true,
      isError: false,
      error: null,
      mutation: idleMutation,
    },
    actions: {
      create: vi.fn(),
      rename: vi.fn(),
      update: vi.fn(),
      remove: vi.fn(),
      retry: vi.fn(),
      dismissMutation: vi.fn(),
      clearForConnectionChange: vi.fn(),
    },
  });
}

describe("useTrainingDraftState Runtime Defaults", () => {
  it("seeds on first Training open and keeps the established draft independent", async () => {
    installModelPackageMetadata();
    installSnapshotRecords();
    const { result, rerender } = renderHook(
      ({ activeWorkspace, seed }) =>
        useTrainingDraftState({
          activeWorkspace,
          models: [
            { modelType: "linears", model: "linear" },
            { modelType: "experts", model: "expert" },
          ],
          seed,
        }),
      {
        initialProps: {
          activeWorkspace: "model" as WorkbenchWorkspace,
          seed: {
            modelType: "linears",
            model: "linear",
            preset: "baseline",
          },
        },
      },
    );

    expect(result.current.setup.model.selected).toBe("");

    rerender({
      activeWorkspace: "training",
      seed: {
        modelType: "linears",
        model: "linear",
        preset: "baseline",
      },
    });
    await waitFor(() => {
      expect(result.current.setup.model.selectedType).toBe("linears");
      expect(result.current.setup.model.selected).toBe("linear");
    });

    rerender({
      activeWorkspace: "training",
      seed: {
        modelType: "experts",
        model: "expert",
        preset: "baseline",
      },
    });
    expect(result.current.setup.model.selectedType).toBe("linears");
    expect(result.current.setup.model.selected).toBe("linear");
  });

  it("resets model-owned selections before reconciling the next package defaults", async () => {
    installModelPackageMetadata();
    installSnapshotRecords([
      {
        id: "snapshot-1",
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        name: "Snapshot one",
        overrides: { hidden_dim: "128" },
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      },
    ]);
    const { result } = renderHook(() =>
      useTrainingDraftState({
        activeWorkspace: "training",
        models: [
          { modelType: "linears", model: "linear" },
          { modelType: "experts", model: "expert" },
        ],
        seed: {
          modelType: "linears",
          model: "linear",
          preset: "baseline",
        },
      }),
    );
    await waitFor(() => {
      expect(result.current.setup.datasets.selected).toEqual(["Mnist"]);
      expect(result.current.setup.variants.snapshots).toHaveLength(1);
    });

    act(() => {
      result.current.setup.variants.selectSnapshots(["snapshot-1"]);
      result.current.setup.monitors.select(["loss"]);
      result.current.runtimeDefaults.edit("hidden_dim", "128");
    });
    expect(result.current.setup.variants.selectedSnapshotIds).toEqual([
      "snapshot-1",
    ]);
    expect(result.current.setup.monitors.selected).toEqual(["loss"]);
    expect(result.current.runtimeDefaults.active).toEqual({ hidden_dim: "128" });

    act(() => result.current.setup.model.select("expert", "experts"));
    await waitFor(() => {
      expect(result.current.setup.model.selectedType).toBe("experts");
      expect(result.current.setup.model.selected).toBe("expert");
      expect(result.current.setup.variants.selectedPresets).toEqual([
        "baseline",
      ]);
      expect(result.current.setup.datasets.selected).toEqual(["Mnist"]);
    });
    expect(result.current.setup.variants.selectedSnapshotIds).toEqual([]);
    expect(result.current.setup.monitors.selected).toEqual([]);
    expect(result.current.runtimeDefaults.active).toEqual({});
  });

  it("canonicalizes edits and suppresses a value returned to its Runtime Default", async () => {
    installModelPackageMetadata();
    installSnapshotRecords();

    const { result } = renderHook(() =>
      useTrainingDraftState({
        activeWorkspace: "training",
        models: [{ modelType: "linears", model: "linear" }],
        seed: {
          modelType: "linears",
          model: "linear",
          preset: "baseline",
        },
      }),
    );

    await waitFor(() => {
      expect(result.current.setup.model.selected).toBe("linear");
      expect(result.current.status.isSchemaReady).toBe(true);
    });

    act(() => result.current.runtimeDefaults.edit("HIDDEN-DIM", "128"));
    expect(result.current.runtimeDefaults.active).toEqual({ hidden_dim: "128" });

    act(() => result.current.runtimeDefaults.edit("hidden_dim", "256"));
    expect(result.current.runtimeDefaults.active).toEqual({});
  });

  it("retains Training selection after failed removal and reconciles it after retry", async () => {
    installModelPackageMetadata();
    const remove = vi.fn().mockResolvedValue({
      ok: false,
      kind: "remove",
      snapshotId: "snapshot-1",
      error: "Removal rejected.",
      retryable: true,
    });
    const retry = vi.fn().mockResolvedValue({
      ok: true,
      kind: "remove",
      snapshotId: "snapshot-1",
      record: null,
    });
    mocks.useConfigSnapshotRecords.mockReturnValue({
      records: [
        {
          id: "snapshot-1",
          modelType: "linears",
          model: "linear",
          preset: "baseline",
          name: "Snapshot one",
          overrides: { hidden_dim: "128" },
          createdAt: "2026-06-01T00:00:00.000Z",
          updatedAt: "2026-06-01T00:00:00.000Z",
        },
      ],
      status: {
        isLoading: false,
        isReady: true,
        isError: false,
        error: null,
        mutation: idleMutation,
      },
      actions: {
        create: vi.fn(),
        rename: vi.fn(),
        update: vi.fn(),
        remove,
        retry,
        dismissMutation: vi.fn(),
        clearForConnectionChange: vi.fn(),
      },
    });
    const { result } = renderHook(() =>
      useTrainingDraftState({
        activeWorkspace: "training",
        models: [{ modelType: "linears", model: "linear" }],
        seed: {
          modelType: "linears",
          model: "linear",
          preset: "baseline",
        },
      }),
    );
    await waitFor(() => expect(result.current.setup.model.selected).toBe("linear"));
    act(() => result.current.setup.variants.selectSnapshots(["snapshot-1"]));
    expect(result.current.setup.variants.selectedSnapshotIds).toEqual([
      "snapshot-1",
    ]);

    await act(async () => {
      await result.current.setup.variants.removeSnapshot("snapshot-1");
    });
    expect(result.current.setup.variants.selectedSnapshotIds).toEqual([
      "snapshot-1",
    ]);

    await act(async () => {
      await result.current.setup.variants.retrySnapshotMutation();
    });
    expect(result.current.setup.variants.selectedSnapshotIds).toEqual([]);
  });
});
