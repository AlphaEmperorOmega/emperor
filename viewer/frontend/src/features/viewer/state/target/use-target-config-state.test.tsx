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
    trainingCancellationCapability: "unsupported",
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

import {
  type ConfigField,
  type ConfigSnapshotRecord,
  type LogRun,
  type MonitorOption,
} from "@/lib/api";
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
  trainingCancellationCapability: "unsupported",
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
let monitorOptions: MonitorOption[] = [];
let configSnapshotsLoading = false;
let schemaFieldsByPreset: Record<string, ConfigField[]> = {};

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
    modelType: overrides.modelType ?? "linears",
    model: overrides.model ?? "linear",
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

function configField(
  overrides: Partial<ConfigField> & Pick<ConfigField, "key">,
): ConfigField {
  const section = overrides.section ?? "Model";
  return {
    key: overrides.key,
    configKey: overrides.configKey ?? overrides.key.toUpperCase(),
    flag: overrides.flag ?? `--${overrides.key.replace(/_/g, "-")}`,
    label: overrides.label ?? overrides.key,
    section,
    sectionPath: overrides.sectionPath ?? [section || "General"],
    type: overrides.type ?? "int",
    default: "default" in overrides ? overrides.default ?? null : 64,
    nullable: overrides.nullable ?? false,
    choices: overrides.choices ?? [],
    locked: overrides.locked ?? false,
    lockedValue: overrides.lockedValue,
    lockedReason: overrides.lockedReason,
  };
}

const adaptiveConfigFields = [
  configField({
    key: "weight_option_flag",
    type: "bool",
    default: false,
    choices: [true, false],
    section: "Weight Generator Options",
  }),
  configField({
    key: "weight_option",
    type: "class",
    default: null,
    nullable: true,
    choices: ["SingleModelDynamicWeightConfig", "DualModelDynamicWeightConfig"],
    section: "Weight Generator Options",
  }),
  configField({
    key: "mask_option_flag",
    type: "bool",
    default: false,
    choices: [true, false],
    section: "Mask Options",
  }),
  configField({
    key: "row_mask_option",
    type: "class",
    default: null,
    nullable: true,
    choices: ["DiagonalAxisMaskConfig", "OuterProductMaskConfig"],
    section: "Mask Options",
  }),
];

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
  monitorOptions = [];
  configSnapshotsLoading = false;
  schemaFieldsByPreset = {};
  mocks.requestPreview.mockReset();
  mocks.clearPreview.mockReset();
  mocks.resetGraphSelectionAndExpansion.mockReset();
  mocks.resetGraphExpansion.mockReset();
  mocks.useViewerQueries.mockReset().mockImplementation(
    (
      selectedModelType: string,
      selectedModel: string,
      selectedPreset: string,
    ) => {
      const presets =
        selectedModelType === "experts" && selectedModel === "linear"
          ? [{ name: "expert-baseline", label: "Expert baseline", description: "" }]
          : [
              { name: "baseline", label: "Baseline", description: "" },
              { name: "fast", label: "Fast", description: "" },
            ];
      const datasets =
        selectedModelType === "experts" && selectedModel === "linear"
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
          models: [
            { modelType: "linears", model: "linear" },
            { modelType: "experts", model: "linear" },
          ],
        }),
        presetsQuery: query({
          modelType: selectedModelType,
          model: selectedModel,
          presets: selectedModel ? presets : [],
        }),
        datasetsQuery: query({
          modelType: selectedModelType,
          model: selectedModel,
          defaultExperimentTask: "image-classification",
          datasetGroups: selectedModel
            ? [
                {
                  experimentTask: "image-classification",
                  label: "Image Classification",
                  datasets,
                },
              ]
            : [],
        }),
        monitorsQuery: query({
          modelType: selectedModelType,
          model: selectedModel,
          monitors: selectedModel ? monitorOptions : [],
        }),
        schemaQuery: query({
          modelType: selectedModelType,
          model: selectedModel,
          preset: selectedPreset,
          fields: schemaFieldsByPreset[selectedPreset] ?? [],
        }),
        searchSpaceQuery: query({
          modelType: selectedModelType,
          model: selectedModel,
          preset: selectedPreset,
          axes: [],
        }),
      };
    },
  );
  mocks.useConfigSnapshots.mockReset().mockImplementation(() => ({
    query: configSnapshotsLoading
      ? loadingQuery({ modelType: "linears", model: "linear", snapshots })
      : query({ modelType: "linears", model: "linear", snapshots }),
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
      expect(result.current.target.selectedModelType).toBe("linears");
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedDatasets).toEqual(["Mnist"]);
    });

    expect(mocks.requestPreview).toHaveBeenCalledWith({
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      experimentTask: "image-classification",
      dataset: "Mnist",
      overrides: {},
      targetMode: "preset",
      targetId: "baseline",
    });
  });

  it("requests a new preview when the selected preview dataset changes", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedModelType).toBe("linears");
      expect(result.current.target.selectedModel).toBe("linear");
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
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        experimentTask: "image-classification",
        dataset: "FashionMnist",
        overrides: {},
        targetMode: "preset",
        targetId: "baseline",
      });
    });
    expect(mocks.resetGraphSelectionAndExpansion).toHaveBeenCalled();
  });

  it("sets, selects all, clears, and prunes monitor selections", async () => {
    monitorOptions = [
      {
        name: "linear",
        label: "Linear layers",
        description: "Layer activations",
        kinds: ["scalar"],
        defaultEnabled: false,
      },
      {
        name: "sampler",
        label: "Sampler usage",
        description: "Sampler activity",
        kinds: ["histogram"],
        defaultEnabled: false,
      },
    ];
    const { result, rerender } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
    });

    act(() => {
      result.current.target.setMonitorSelection([
        "sampler",
        "unknown",
        "linear",
        "linear",
      ]);
    });
    expect(result.current.target.selectedMonitors).toEqual([
      "sampler",
      "linear",
    ]);

    act(() => {
      result.current.target.selectAllMonitors();
    });
    expect(result.current.target.selectedMonitors).toEqual([
      "linear",
      "sampler",
    ]);

    act(() => {
      result.current.target.clearMonitors();
    });
    expect(result.current.target.selectedMonitors).toEqual([]);

    act(() => {
      result.current.target.setMonitorSelection(["linear", "sampler"]);
    });
    monitorOptions = monitorOptions.slice(0, 1);
    rerender();

    await waitFor(() => {
      expect(result.current.target.selectedMonitors).toEqual(["linear"]);
    });
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
    expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
    expect(result.current.target.selectedTrainingDatasets).toEqual(["Mnist"]);
    expect(result.current.target.selectedDatasets).toEqual(["FashionMnist"]);
    expect(result.current.target.overrides).toEqual({});
    expect(mocks.resetGraphSelectionAndExpansion).toHaveBeenCalled();
    expect(mocks.requestPreview).toHaveBeenLastCalledWith({
      modelType: "linears",
      model: "linear",
      preset: "fast",
      dataset: "FashionMnist",
      overrides: {},
      targetMode: "experiment",
      targetId: "run-fast",
      logRunId: "run-fast",
    });
  });

  it("syncs a selected historical run whose preset and dataset are absent from the catalog", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedDatasets).toEqual(["Mnist"]);
    });

    act(() => {
      result.current.target.updateOverride("hidden_size", "128");
    });
    act(() => {
      result.current.syncSelectedLogRun(
        logRun({
          id: "kaggle-run",
          experiment: "kaggle_linear_all",
          preset: "KAGGLE_LINEAR",
          dataset: "KaggleDigits",
        }),
      );
    });

    expect(result.current.target.selectedTargetMode).toBe("experiment");
    expect(result.current.target.selectedExperimentRunId).toBe("kaggle-run");
    expect(result.current.target.selectedExperimentName).toBe(
      "kaggle_linear_all",
    );
    expect(result.current.target.selectedExperimentPreset).toBe("KAGGLE_LINEAR");
    expect(result.current.target.selectedExperimentDataset).toBe("KaggleDigits");
    expect(result.current.target.selectedPreset).toBe("baseline");
    expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
    expect(result.current.target.selectedDatasets).toEqual(["Mnist"]);
    expect(result.current.target.overrides).toEqual({});
    expect(mocks.requestPreview).toHaveBeenLastCalledWith({
      modelType: "linears",
      model: "linear",
      preset: "KAGGLE_LINEAR",
      dataset: "KaggleDigits",
      overrides: {},
      targetMode: "experiment",
      targetId: "kaggle-run",
      logRunId: "kaggle-run",
    });
  });

  it("keeps the log run id on automatic experiment preview refreshes", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedPreset).toBe("baseline");
    });

    act(() => {
      result.current.syncSelectedLogRun(logRun({ id: "run-fast" }));
    });

    await waitFor(() => {
      expect(result.current.target.selectedTargetMode).toBe("experiment");
      expect(result.current.target.selectedExperimentRunId).toBe("run-fast");
    });

    mocks.requestPreview.mockClear();
    act(() => {
      result.current.target.updateOverride("hidden_size", "128", {
        preserveTargetSelection: true,
      });
    });

    await waitFor(() => {
      expect(mocks.requestPreview).toHaveBeenLastCalledWith({
        modelType: "linears",
        model: "linear",
        preset: "fast",
        dataset: "FashionMnist",
        overrides: { hidden_size: "128" },
        targetMode: "experiment",
        targetId: "run-fast",
        logRunId: "run-fast",
      });
    });
  });

  it("preserves the selected historical run when resetting overrides", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedPreset).toBe("baseline");
    });

    act(() => {
      result.current.syncSelectedLogRun(logRun({ id: "run-fast" }));
    });

    await waitFor(() => {
      expect(result.current.target.selectedTargetMode).toBe("experiment");
      expect(result.current.target.selectedExperimentRunId).toBe("run-fast");
    });

    act(() => {
      result.current.target.updateOverride("hidden_size", "128", {
        preserveTargetSelection: true,
      });
    });
    await waitFor(() => {
      expect(result.current.target.overrides).toEqual({ hidden_size: "128" });
    });

    mocks.requestPreview.mockClear();
    act(() => {
      result.current.target.resetOverridesPreservingTargetSelection();
    });

    await waitFor(() => {
      expect(result.current.target.selectedTargetMode).toBe("experiment");
      expect(result.current.target.selectedExperimentRunId).toBe("run-fast");
      expect(result.current.target.overrides).toEqual({});
    });
    expect(mocks.requestPreview).toHaveBeenLastCalledWith({
      modelType: "linears",
      model: "linear",
      preset: "fast",
      dataset: "FashionMnist",
      overrides: {},
      targetMode: "experiment",
      targetId: "run-fast",
      logRunId: "run-fast",
    });
  });

  it("selects another preset, preserves overrides, and refreshes the preview", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedModelType).toBe("linears");
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedDatasets).toEqual(["Mnist"]);
    });

    mocks.requestPreview.mockClear();
    mocks.resetGraphSelectionAndExpansion.mockClear();

    act(() => {
      result.current.target.updateOverride("hidden_size", "128");
    });
    mocks.requestPreview.mockClear();
    mocks.resetGraphSelectionAndExpansion.mockClear();

    act(() => {
      result.current.target.selectTargetPreset("fast");
    });

    expect(result.current.target.selectedPreset).toBe("fast");
    expect(result.current.target.overrides).toEqual({ hidden_size: "128" });
    expect(mocks.requestPreview).toHaveBeenCalledWith({
      modelType: "linears",
      model: "linear",
      preset: "fast",
      experimentTask: "image-classification",
      dataset: "Mnist",
      overrides: { hidden_size: "128" },
      targetMode: "preset",
      targetId: "fast",
    });
    expect(mocks.resetGraphSelectionAndExpansion).toHaveBeenCalled();
  });

  it("clears preset overrides when the model changes", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedModel).toBe("linear");
    });

    act(() => {
      result.current.target.updateOverride("hidden_size", "128");
    });
    expect(result.current.target.presetOverrides).toEqual({ hidden_size: "128" });

    act(() => {
      result.current.target.selectModel("linear", "experts");
    });

    await waitFor(() => {
      expect(result.current.target.selectedModelType).toBe("experts");
      expect(result.current.target.selectedModel).toBe("linear");
    });
    expect(result.current.target.presetOverrides).toEqual({});
    expect(result.current.target.overrides).toEqual({});
  });

  it("selects a footer training model type without changing the preview target", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedModelType).toBe("linears");
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedTrainingModelType).toBe("linears");
      expect(result.current.target.selectedTrainingModel).toBe("linear");
      expect(result.current.target.selectedTrainingPrimaryPreset).toBe("baseline");
    });

    mocks.requestPreview.mockClear();
    mocks.clearPreview.mockClear();
    mocks.resetGraphSelectionAndExpansion.mockClear();

    act(() => {
      result.current.target.selectTrainingModelType("experts");
    });

    await waitFor(() => {
      expect(result.current.target.selectedTrainingModelType).toBe("experts");
      expect(result.current.target.selectedTrainingModel).toBe("linear");
      expect(result.current.target.selectedTrainingPrimaryPreset).toBe(
        "expert-baseline",
      );
      expect(result.current.target.selectedTrainingPresets).toEqual([
        "expert-baseline",
      ]);
      expect(result.current.target.selectedTrainingDatasets).toEqual([
        "ExpertToy",
      ]);
    });
    expect(result.current.target.selectedModelType).toBe("linears");
    expect(result.current.target.selectedModel).toBe("linear");
    expect(result.current.target.selectedPreset).toBe("baseline");
    expect(result.current.target.selectedDatasets).toEqual(["Mnist"]);
    expect(mocks.requestPreview).not.toHaveBeenCalled();
    expect(mocks.clearPreview).not.toHaveBeenCalled();
    expect(mocks.resetGraphSelectionAndExpansion).not.toHaveBeenCalled();
  });

  it("keeps training overrides independent and resets them for a new training model", async () => {
    schemaFieldsByPreset = {
      baseline: [configField({ key: "hidden_dim" })],
      "expert-baseline": [configField({ key: "hidden_dim" })],
    };
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedTrainingModel).toBe("linear");
      expect(result.current.target.selectedTrainingPrimaryPreset).toBe("baseline");
    });

    act(() => {
      result.current.target.updateTrainingOverride("hidden_dim", "128");
    });

    expect(result.current.target.trainingOverrides).toEqual({
      hidden_dim: "128",
    });
    expect(result.current.target.presetOverrides).toEqual({});
    expect(result.current.target.overrides).toEqual({});

    act(() => {
      result.current.target.resetTrainingOverrides();
    });

    expect(result.current.target.trainingOverrides).toEqual({});

    act(() => {
      result.current.target.updateTrainingOverride("hidden_dim", "192");
    });
    expect(result.current.target.trainingOverrides).toEqual({
      hidden_dim: "192",
    });

    act(() => {
      result.current.target.selectTrainingModelType("experts");
    });

    await waitFor(() => {
      expect(result.current.target.selectedTrainingModelType).toBe("experts");
      expect(result.current.target.selectedTrainingModel).toBe("linear");
      expect(result.current.target.selectedTrainingPrimaryPreset).toBe(
        "expert-baseline",
      );
    });
    expect(result.current.target.trainingOverrides).toEqual({});
    expect(result.current.target.selectedModelType).toBe("linears");
    expect(result.current.target.selectedModel).toBe("linear");
    expect(result.current.target.presetOverrides).toEqual({});
  });

  it("does not rewrite the seeded footer training target when the preview model changes", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedTrainingModel).toBe("linear");
      expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
      expect(result.current.target.selectedTrainingDatasets).toEqual(["Mnist"]);
    });

    act(() => {
      result.current.target.selectModelType("experts");
    });

    await waitFor(() => {
      expect(result.current.target.selectedModelType).toBe("experts");
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("expert-baseline");
      expect(result.current.target.selectedDatasets).toEqual(["ExpertToy"]);
    });
    expect(result.current.target.selectedTrainingModelType).toBe("linears");
    expect(result.current.target.selectedTrainingModel).toBe("linear");
    expect(result.current.target.selectedTrainingPrimaryPreset).toBe("baseline");
    expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
    expect(result.current.target.selectedTrainingDatasets).toEqual(["Mnist"]);
  });

  it("keeps locked preset overrides in the draft but omits them from active preview overrides", async () => {
    schemaFieldsByPreset = {
      baseline: [configField({ key: "layer_width" })],
      fast: [
        configField({
          key: "layer_width",
          locked: true,
          lockedValue: 96,
          lockedReason: "Preset controlled",
        }),
      ],
    };
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedPreset).toBe("baseline");
    });

    act(() => {
      result.current.target.updateOverride("layer_width", "128");
    });
    expect(result.current.target.overrides).toEqual({ layer_width: "128" });

    mocks.requestPreview.mockClear();
    act(() => {
      result.current.target.selectPreset("fast");
    });

    await waitFor(() => {
      expect(result.current.target.selectedPreset).toBe("fast");
      expect(result.current.target.presetOverrides).toEqual({
        layer_width: "128",
      });
      expect(result.current.target.overrides).toEqual({});
      expect(result.current.target.inactiveLockedOverrideCount).toBe(1);
    });
    await waitFor(() => {
      expect(mocks.requestPreview).toHaveBeenLastCalledWith({
        modelType: "linears",
        model: "linear",
        preset: "fast",
        experimentTask: "image-classification",
        dataset: "Mnist",
        overrides: {},
        targetMode: "preset",
        targetId: "fast",
      });
    });
  });

  it("normalizes adaptive flag updates in preset override state", async () => {
    schemaFieldsByPreset = { baseline: adaptiveConfigFields };
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedPreset).toBe("baseline");
    });

    act(() => {
      result.current.target.updateOverride("weight_option_flag", "true");
    });

    expect(result.current.target.overrides).toEqual({
      weight_option_flag: "true",
      weight_option: "SingleModelDynamicWeightConfig",
    });

    act(() => {
      result.current.target.updateOverride("weight_option_flag", "false");
    });

    expect(result.current.target.overrides).toEqual({
      weight_option_flag: "false",
    });
  });

  it("normalizes adaptive flag updates in snapshot editor drafts", async () => {
    schemaFieldsByPreset = { baseline: adaptiveConfigFields };
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedPreset).toBe("baseline");
    });

    act(() => {
      result.current.target.updateSnapshotEditorDraftOverride(
        "mask_option_flag",
        "true",
      );
    });

    expect(result.current.target.snapshotEditorDraft).toEqual({
      mask_option_flag: "true",
      row_mask_option: "DiagonalAxisMaskConfig",
    });

    act(() => {
      result.current.target.updateSnapshotEditorDraftOverride(
        "mask_option_flag",
        "false",
      );
    });

    expect(result.current.target.snapshotEditorDraft).toEqual({
      mask_option_flag: "false",
    });
  });

  it("normalizes old adaptive snapshot overrides before preview", async () => {
    schemaFieldsByPreset = { baseline: adaptiveConfigFields };
    snapshots = [
      {
        id: "snapshot-old-adaptive",
        name: "Old adaptive",
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        overrides: { weight_option_flag: "true" },
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      },
    ];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedDatasets).toEqual(["Mnist"]);
    });

    mocks.requestPreview.mockClear();
    act(() => {
      expect(
        result.current.target.selectTargetSnapshot("snapshot-old-adaptive", {
          includeTrainingSnapshot: false,
        }),
      ).toBe(true);
    });

    expect(result.current.target.snapshotEditorDraft).toEqual({
      weight_option_flag: "true",
      weight_option: "SingleModelDynamicWeightConfig",
    });
    expect(mocks.requestPreview).toHaveBeenLastCalledWith({
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      experimentTask: "image-classification",
      dataset: "Mnist",
      overrides: {
        weight_option_flag: "true",
        weight_option: "SingleModelDynamicWeightConfig",
      },
      targetMode: "snapshot",
      targetId: "snapshot-old-adaptive",
    });
  });

  it("persists normalized adaptive overrides when updating a selected snapshot", async () => {
    schemaFieldsByPreset = { baseline: adaptiveConfigFields };
    snapshots = [
      {
        id: "snapshot-old-adaptive",
        name: "Old adaptive",
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        overrides: { weight_option_flag: "true" },
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      },
    ];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedPreset).toBe("baseline");
    });

    act(() => {
      result.current.target.selectTargetSnapshot("snapshot-old-adaptive", {
        includeTrainingSnapshot: false,
      });
    });

    await waitFor(() => {
      expect(result.current.target.snapshotEditorDraft).toEqual({
        weight_option_flag: "true",
        weight_option: "SingleModelDynamicWeightConfig",
      });
    });

    const updateResult = result.current.target.updateSelectedConfigSnapshot(
      "Old adaptive",
    );

    expect(updateResult).toMatchObject({
      ok: true,
      snapshot: {
        overrides: {
          weight_option_flag: "true",
          weight_option: "SingleModelDynamicWeightConfig",
        },
      },
    });
  });

  it("isolates snapshot editor drafts from preset overrides and restores preset overrides on return", async () => {
    snapshots = [
      {
        id: "snapshot-baseline",
        name: "Baseline tuned",
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        overrides: { hidden_size: "256" },
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      },
    ];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedPreset).toBe("baseline");
    });

    act(() => {
      result.current.target.updateOverride("hidden_size", "128");
    });
    act(() => {
      expect(
        result.current.target.prepareSelectedSnapshotEdit("snapshot-baseline", {
          includeTrainingSnapshot: false,
        }),
      ).toBe(true);
    });

    expect(result.current.target.presetOverrides).toEqual({ hidden_size: "128" });
    expect(result.current.target.snapshotEditorDraft).toEqual({
      hidden_size: "256",
    });
    expect(result.current.target.overrides).toEqual({ hidden_size: "256" });

    act(() => {
      result.current.target.updateSnapshotEditorDraftOverride("hidden_size", "384");
    });

    expect(result.current.target.presetOverrides).toEqual({ hidden_size: "128" });
    expect(result.current.target.snapshotEditorDraft).toEqual({
      hidden_size: "384",
    });

    act(() => {
      result.current.target.activateTargetPresetMode();
    });

    expect(result.current.target.selectedTargetMode).toBe("preset");
    expect(result.current.target.overrides).toEqual({ hidden_size: "128" });
  });

  it("preserves the selected snapshot when resetting overrides", async () => {
    snapshots = [
      {
        id: "snapshot-baseline",
        name: "Baseline tuned",
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        overrides: { hidden_size: "256" },
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      },
    ];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedPreset).toBe("baseline");
    });

    act(() => {
      expect(
        result.current.target.prepareSelectedSnapshotEdit("snapshot-baseline", {
          includeTrainingSnapshot: false,
        }),
      ).toBe(true);
    });

    await waitFor(() => {
      expect(result.current.target.selectedTargetMode).toBe("snapshot");
      expect(result.current.target.selectedSnapshotId).toBe("snapshot-baseline");
      expect(result.current.target.overrides).toEqual({ hidden_size: "256" });
    });

    mocks.requestPreview.mockClear();
    act(() => {
      result.current.target.resetOverridesPreservingTargetSelection();
    });

    await waitFor(() => {
      expect(result.current.target.selectedTargetMode).toBe("snapshot");
      expect(result.current.target.selectedSnapshotId).toBe("snapshot-baseline");
      expect(result.current.target.overrides).toEqual({});
    });
    expect(mocks.requestPreview).toHaveBeenLastCalledWith({
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      experimentTask: "image-classification",
      dataset: "Mnist",
      overrides: {},
      targetMode: "snapshot",
      targetId: "snapshot-baseline",
    });
    expect(mocks.requestPreview.mock.calls.at(-1)?.[0]).not.toHaveProperty(
      "logRunId",
    );
  });

  it("includes a snapshot from an unselected preset without selecting its preset", async () => {
    snapshots = [
      {
        id: "snapshot-fast",
        name: "Fast tuned",
        modelType: "linears",
        model: "linear",
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
        modelType: "linears",
        model: "linear",
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
        modelType: "linears",
        model: "linear",
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

  it("keeps the selected target preset when Training Job presets change", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedPreset).toBe("baseline");
      expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
    });

    act(() => {
      result.current.target.setTrainingPresetSelection(["fast"]);
    });

    expect(result.current.target.selectedPreset).toBe("baseline");
    expect(result.current.target.selectedTrainingPresets).toEqual(["fast"]);
  });

  it("allows empty base preset selection after a snapshot is selected", async () => {
    snapshots = [
      {
        id: "snapshot-baseline",
        name: "Baseline tuned",
        modelType: "linears",
        model: "linear",
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

  it("does not warn when training overrides and snapshots are selected", async () => {
    snapshots = [
      {
        id: "snapshot-baseline",
        name: "Baseline tuned",
        modelType: "linears",
        model: "linear",
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
      result.current.target.updateTrainingOverride("hidden_size", "128");
    });
    act(() => {
      result.current.target.includeConfigSnapshot("snapshot-baseline");
    });

    expect(result.current.target.snapshotOverrideWarning).toBe("");
  });

  it("prepares a preset snapshot draft without re-adding an empty training preset selection", async () => {
    snapshots = [
      {
        id: "snapshot-baseline",
        name: "Baseline tuned",
        modelType: "linears",
        model: "linear",
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
    act(() => {
      expect(
        result.current.target.preparePresetSnapshotDraft("fast", {
          includeTrainingPreset: false,
        }),
      ).toBe(true);
    });

    await waitFor(() => {
      expect(result.current.target.selectedPreset).toBe("fast");
      expect(result.current.target.selectedTrainingPresets).toEqual([]);
    });
    expect(result.current.target.selectedTrainingSnapshotIds).toEqual([
      "snapshot-baseline",
    ]);
    expect(result.current.target.overrides).toEqual({});
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
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedPreset: "fast",
      selectedTargetMode: "snapshot",
      selectedSnapshotId: "snapshot-fast",
    });

    const { result, rerender } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedModelType).toBe("linears");
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("fast");
    });
    expect(result.current.target.selectedTrainingPresets).toEqual([]);

    snapshots = [
      {
        id: "snapshot-fast",
        name: "Fast tuned",
        modelType: "linears",
        model: "linear",
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
        modelType: "linears",
        model: "linear",
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
    expect(result.current.target.configSnapshotCount).toBe(1);
    expect(result.current.target.trainingConfigSnapshotCount).toBe(0);
    expect(result.current.target.allConfigSnapshotCount).toBe(1);
  });

  it("loads Config Snapshot overrides into the selected target preset", async () => {
    snapshots = [
      {
        id: "snapshot-1",
        name: "Wide",
        modelType: "linears",
        model: "linear",
        preset: "fast",
        overrides: { hidden_size: "256" },
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      },
    ];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedModelType).toBe("linears");
      expect(result.current.target.selectedModel).toBe("linear");
    });

    expect(result.current.target.allConfigSnapshotCount).toBe(1);
    expect(result.current.target.configSnapshotCount).toBe(0);

    act(() => {
      expect(result.current.target.loadConfigSnapshot("snapshot-1")).toBe(true);
    });

    await waitFor(() => {
      expect(result.current.target.selectedPreset).toBe("fast");
      expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
      expect(result.current.target.selectedTrainingSnapshotIds).toEqual([]);
      expect(result.current.target.overrides).toEqual({ hidden_size: "256" });
    });
  });

  it("loads a Config Snapshot from another model after switching target state", async () => {
    librarySnapshots = [
      {
        id: "expert-snapshot",
        name: "Expert tuned",
        modelType: "experts",
        model: "linear",
        preset: "expert-baseline",
        overrides: { expert_width: "4" },
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      },
    ];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.target.selectedModelType).toBe("linears");
      expect(result.current.target.selectedModel).toBe("linear");
    });

    act(() => {
      expect(result.current.target.loadConfigSnapshot("expert-snapshot")).toBe(true);
    });

    await waitFor(() => {
      expect(result.current.target.selectedModelType).toBe("experts");
      expect(result.current.target.selectedModel).toBe("linear");
      expect(result.current.target.selectedPreset).toBe("expert-baseline");
      expect(result.current.target.selectedTrainingPresets).toEqual(["baseline"]);
      expect(result.current.target.selectedDatasets).toEqual(["ExpertToy"]);
      expect(result.current.target.selectedTrainingDatasets).toEqual(["Mnist"]);
      expect(result.current.target.overrides).toEqual({ expert_width: "4" });
    });
  });
});
