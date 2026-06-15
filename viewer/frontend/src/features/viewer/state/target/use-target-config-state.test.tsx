import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  useViewerQueries: vi.fn(),
  useConfigSnapshots: vi.fn(),
  useConfigSnapshotLibrary: vi.fn(),
  requestPreview: vi.fn(),
  clearPreview: vi.fn(),
  resetGraphSelectionAndExpansion: vi.fn(),
  resetGraphExpansion: vi.fn(),
}));

vi.mock("@/features/viewer/state/use-viewer-queries", () => ({
  LOCAL_DEFAULT_CAPABILITIES: {
    authMode: "none",
    trainingEnabled: true,
    logDeletionEnabled: true,
    configSnapshotsEnabled: true,
    historicalLogsEnabled: true,
    liveMonitorDataEnabled: true,
    historicalMonitorDataEnabled: true,
    uploadsEnabled: false,
    maxUploadSize: null,
    dataSourcesEnabled: false,
    dataSources: [],
  },
  useViewerQueries: mocks.useViewerQueries,
}));

vi.mock("@/features/viewer/state/target/use-config-snapshots", () => ({
  useConfigSnapshotLibrary: mocks.useConfigSnapshotLibrary,
  useConfigSnapshots: mocks.useConfigSnapshots,
}));

import { type ConfigSnapshotRecord, type LogRun } from "@/lib/api";
import {
  clearPersistedTargetSelection,
  writePersistedTargetSelection,
} from "@/features/viewer/state/target/target-selection-storage";
import {
  useTargetConfigState,
} from "@/features/viewer/state/target/use-target-config-state";

const capabilities = {
  authMode: "none",
  trainingEnabled: true,
  logDeletionEnabled: true,
  configSnapshotsEnabled: true,
  historicalLogsEnabled: true,
  liveMonitorDataEnabled: true,
  historicalMonitorDataEnabled: true,
  uploadsEnabled: false,
  maxUploadSize: null,
  dataSourcesEnabled: false,
  dataSources: [],
};
let snapshots: ConfigSnapshotRecord[] = [];
let librarySnapshots: ConfigSnapshotRecord[] = [];
let configSnapshotsLoading = false;

function query<TData>(data: TData) {
  return {
    data,
    isLoading: false,
    isSuccess: true,
    isError: false,
    error: null,
  };
}

function loadingQuery<TData>(data: TData) {
  return {
    data,
    isLoading: true,
    isSuccess: false,
    isError: false,
    error: null,
  };
}

function logRun(overrides: Partial<LogRun> & Pick<LogRun, "id">): LogRun {
  return {
    id: overrides.id,
    group: overrides.group ?? overrides.experiment ?? "exp_linear",
    experiment: overrides.experiment ?? "exp_linear",
    model: overrides.model ?? "linears/linear",
    preset: overrides.preset ?? "fast",
    dataset: overrides.dataset ?? "FashionMnist",
    runName: overrides.runName ?? `${overrides.id}_20260601_010203`,
    timestamp: overrides.timestamp ?? "2026-06-01 01:02:03",
    version: overrides.version ?? "version_0",
    relativePath:
      overrides.relativePath ??
      "exp_linear/linears/linear/fast/FashionMnist/run/version_0",
    hasResult: overrides.hasResult ?? false,
    eventFileCount: overrides.eventFileCount ?? 1,
    checkpointCount: overrides.checkpointCount ?? 0,
    hasHparams: overrides.hasHparams ?? true,
    metrics: overrides.metrics ?? {},
  };
}

function renderTargetState() {
  return renderHook(() =>
    useTargetConfigState({
      requestPreview: mocks.requestPreview,
      clearPreview: mocks.clearPreview,
      resetGraphSelectionAndExpansion: mocks.resetGraphSelectionAndExpansion,
      resetGraphExpansion: mocks.resetGraphExpansion,
    }),
  );
}

beforeEach(() => {
  clearPersistedTargetSelection();
  snapshots = [];
  librarySnapshots = [];
  configSnapshotsLoading = false;
  mocks.requestPreview.mockReset();
  mocks.clearPreview.mockReset();
  mocks.resetGraphSelectionAndExpansion.mockReset();
  mocks.resetGraphExpansion.mockReset();
  mocks.useViewerQueries.mockReset().mockImplementation(
    (selectedModel: string, selectedPreset: string) => {
      const presets =
        selectedModel === "experts/experts_linear"
          ? [{ name: "expert-baseline", label: "Expert baseline", description: "" }]
          : [
              { name: "baseline", label: "Baseline", description: "" },
              { name: "fast", label: "Fast", description: "" },
            ];
      const datasets =
        selectedModel === "experts/experts_linear"
          ? [{ name: "ExpertToy", label: "Expert Toy", inputDim: 64, outputDim: 4 }]
          : [
              { name: "Mnist", label: "MNIST", inputDim: 784, outputDim: 10 },
              {
                name: "FashionMnist",
                label: "Fashion MNIST",
                inputDim: 784,
                outputDim: 10,
              },
            ];

      return {
        healthQuery: query({ status: "ok" }),
        capabilitiesQuery: query(capabilities),
        modelsQuery: query({
          models: ["linears/linear", "experts/experts_linear"],
        }),
        presetsQuery: query({
          model: selectedModel,
          presets: selectedModel ? presets : [],
        }),
        datasetsQuery: query({
          model: selectedModel,
          datasets: selectedModel ? datasets : [],
        }),
        monitorsQuery: query({ model: selectedModel, monitors: [] }),
        schemaQuery: query({ model: selectedModel, preset: selectedPreset, fields: [] }),
        searchSpaceQuery: query({ model: selectedModel, preset: selectedPreset, axes: [] }),
      };
    },
  );
  mocks.useConfigSnapshots.mockReset().mockImplementation(() => ({
    query: configSnapshotsLoading
      ? loadingQuery({ model: "linears/linear", snapshots })
      : query({ model: "linears/linear", snapshots }),
    snapshots,
    createMutation: { mutate: vi.fn() },
    renameMutation: { mutate: vi.fn() },
    updateMutation: { mutate: vi.fn() },
    deleteMutation: { mutate: vi.fn() },
  }));
  mocks.useConfigSnapshotLibrary.mockReset().mockImplementation(() => ({
    query: query({ snapshots: librarySnapshots }),
    snapshots: librarySnapshots,
  }));
});

describe("useTargetConfigState", () => {
  it("auto-selects the first target and requests the initial preview", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linears/linear");
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedDatasets).toEqual(["Mnist"]);
    });

    expect(mocks.requestPreview).toHaveBeenCalledWith({
      model: "linears/linear",
      preset: "baseline",
      dataset: "Mnist",
      overrides: {},
    });
  });

  it("requests a new preview when the selected preview dataset changes", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linears/linear");
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedDatasets).toEqual(["Mnist"]);
    });

    mocks.requestPreview.mockClear();
    mocks.resetGraphSelectionAndExpansion.mockClear();

    act(() => {
      result.current.target.setDatasetSelection(["FashionMnist"]);
    });

    await waitFor(() => {
      expect(result.current.target.selectedDatasets).toEqual(["FashionMnist"]);
    });
    await waitFor(() => {
      expect(mocks.requestPreview).toHaveBeenCalledWith({
        model: "linears/linear",
        preset: "baseline",
        dataset: "FashionMnist",
        overrides: {},
      });
    });
    expect(mocks.resetGraphSelectionAndExpansion).toHaveBeenCalled();
  });

  it("syncs a selected historical run into the target without keeping overrides", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedPreset).toBe("baseline");
    });

    act(() => {
      result.current.target.updateOverride("hidden_size", "128");
    });
    act(() => {
      result.current.syncSelectedLogRun(logRun({ id: "run-fast" }));
    });

    expect(result.current.target.selectedPreset).toBe("fast");
    expect(result.current.target.selectedTargetMode).toBe("experiment");
    expect(result.current.target.selectedExperimentRunId).toBe("run-fast");
    expect(result.current.target.selectedSnapshotId).toBe("");
    expect(result.current.target.selectedTrainingPresets).toEqual(["fast"]);
    expect(result.current.target.selectedDatasets).toEqual(["FashionMnist"]);
    expect(result.current.target.overrides).toEqual({});
    expect(mocks.resetGraphSelectionAndExpansion).toHaveBeenCalled();
    expect(mocks.requestPreview).toHaveBeenLastCalledWith({
      model: "linears/linear",
      preset: "fast",
      dataset: "FashionMnist",
      overrides: {},
    });
  });

  it("selects another preset without auto-refreshing the preview", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linears/linear");
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedDatasets).toEqual(["Mnist"]);
    });

    mocks.requestPreview.mockClear();
    mocks.resetGraphSelectionAndExpansion.mockClear();

    act(() => {
      result.current.target.selectPreset("fast");
    });

    expect(result.current.target.selectedPreset).toBe("fast");
    expect(result.current.target.overrides).toEqual({});
    expect(mocks.requestPreview).not.toHaveBeenCalled();
    expect(mocks.resetGraphSelectionAndExpansion).not.toHaveBeenCalled();
  });

  it("includes a snapshot from an unselected preset without selecting its preset", async () => {
    snapshots = [
      {
        id: "snapshot-fast",
        name: "Fast tuned",
        model: "linears/linear",
        preset: "fast",
        overrides: { hidden_size: "256" },
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      },
    ];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
    });

    act(() => {
      result.current.target.includeConfigSnapshot("snapshot-fast");
    });

    expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
    expect(result.current.target.selectedTrainingSnapshotIds).toEqual([
      "snapshot-fast",
    ]);
  });

  it("excludes one snapshot without changing selected presets", async () => {
    snapshots = [
      {
        id: "snapshot-fast",
        name: "Fast tuned",
        model: "linears/linear",
        preset: "fast",
        overrides: { hidden_size: "256" },
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      },
    ];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
    });

    act(() => {
      result.current.target.includeConfigSnapshot("snapshot-fast");
    });
    act(() => {
      result.current.target.excludeConfigSnapshot("snapshot-fast");
    });

    expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
    expect(result.current.target.selectedTrainingSnapshotIds).toEqual([]);
  });

  it("keeps a selected snapshot when its source preset is removed", async () => {
    snapshots = [
      {
        id: "snapshot-fast",
        name: "Fast tuned",
        model: "linears/linear",
        preset: "fast",
        overrides: { hidden_size: "256" },
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      },
    ];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
    });

    act(() => {
      result.current.target.setTrainingPresetSelection(["baseline", "fast"]);
      result.current.target.includeConfigSnapshot("snapshot-fast");
    });
    act(() => {
      result.current.target.excludeDraftTrainingPreset("fast");
    });

    expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
    expect(result.current.target.selectedTrainingSnapshotIds).toEqual([
      "snapshot-fast",
    ]);
  });

  it("allows empty base preset selection after a snapshot is selected", async () => {
    snapshots = [
      {
        id: "snapshot-baseline",
        name: "Baseline tuned",
        model: "linears/linear",
        preset: "baseline",
        overrides: { hidden_size: "256" },
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      },
    ];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
    });

    act(() => {
      result.current.target.includeConfigSnapshot("snapshot-baseline");
    });
    act(() => {
      result.current.target.setTrainingPresetSelection([]);
    });

    expect(result.current.target.selectedPreset).toBe("baseline");
    expect(result.current.target.selectedTrainingPresets).toEqual([]);
    expect(result.current.target.selectedTrainingSnapshotIds).toEqual([
      "snapshot-baseline",
    ]);
  });

  it("allows the progress draft to deselect the primary preset", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
    });

    act(() => {
      result.current.target.toggleDraftTrainingPreset("baseline");
    });

    expect(result.current.target.selectedPreset).toBe("baseline");
    expect(result.current.target.selectedTrainingPresets).toEqual([]);

    act(() => {
      result.current.target.toggleDraftTrainingPreset("baseline");
    });

    expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
  });

  it("keeps restored snapshot targets snapshot-only while snapshots load", async () => {
    configSnapshotsLoading = true;
    writePersistedTargetSelection({
      selectedModel: "linears/linear",
      selectedPreset: "fast",
      selectedTargetMode: "snapshot",
      selectedSnapshotId: "snapshot-fast",
    });

    const { result, rerender } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linears/linear");
      expect(result.current.target.selectedPreset).toBe("fast");
    });
    expect(result.current.target.selectedTrainingPresets).toEqual([]);

    snapshots = [
      {
        id: "snapshot-fast",
        name: "Fast tuned",
        model: "linears/linear",
        preset: "fast",
        overrides: { hidden_size: "256" },
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      },
    ];
    configSnapshotsLoading = false;
    rerender();

    await waitFor(() => {
      expect(result.current.target.selectedTargetMode).toBe("snapshot");
      expect(result.current.target.selectedSnapshotId).toBe("snapshot-fast");
      expect(result.current.target.selectedTrainingSnapshotIds).toEqual([
        "snapshot-fast",
      ]);
    });
    expect(result.current.target.selectedTrainingPresets).toEqual([]);
    expect(result.current.target.overrides).toEqual({ hidden_size: "256" });
  });

  it("excludes a draft preset without changing the selected preview preset", async () => {
    snapshots = [
      {
        id: "snapshot-baseline",
        name: "Baseline tuned",
        model: "linears/linear",
        preset: "baseline",
        overrides: { hidden_size: "256" },
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      },
    ];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
      expect(result.current.target.configSnapshotCount).toBe(1);
    });

    act(() => {
      result.current.target.excludeDraftTrainingPreset("baseline");
    });

    expect(result.current.target.selectedPreset).toBe("baseline");
    expect(result.current.target.selectedTrainingPresets).toEqual([]);
    expect(result.current.target.configSnapshotCount).toBe(0);
    expect(result.current.target.allConfigSnapshotCount).toBe(1);
  });

  it("loads Config Snapshot overrides into the selected target preset", async () => {
    snapshots = [
      {
        id: "snapshot-1",
        name: "Wide",
        model: "linears/linear",
        preset: "fast",
        overrides: { hidden_size: "256" },
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      },
    ];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linears/linear");
    });

    expect(result.current.target.allConfigSnapshotCount).toBe(1);
    expect(result.current.target.configSnapshotCount).toBe(0);

    act(() => {
      expect(result.current.target.loadConfigSnapshot("snapshot-1")).toBe(true);
    });

    await waitFor(() => {
      expect(result.current.target.selectedPreset).toBe("fast");
      expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
      expect(result.current.target.selectedTrainingSnapshotIds).toEqual([
        "snapshot-1",
      ]);
      expect(result.current.target.overrides).toEqual({ hidden_size: "256" });
    });
  });

  it("loads a Config Snapshot from another model after switching target state", async () => {
    librarySnapshots = [
      {
        id: "expert-snapshot",
        name: "Expert tuned",
        model: "experts/experts_linear",
        preset: "expert-baseline",
        overrides: { expert_width: "4" },
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      },
    ];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linears/linear");
    });

    act(() => {
      expect(result.current.target.loadConfigSnapshot("expert-snapshot")).toBe(true);
    });

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("experts/experts_linear");
      expect(result.current.target.selectedPreset).toBe("expert-baseline");
      expect(result.current.target.selectedTrainingPresets).toEqual([]);
      expect(result.current.target.selectedDatasets).toEqual(["ExpertToy"]);
      expect(result.current.target.overrides).toEqual({ expert_width: "4" });
    });
  });
});
