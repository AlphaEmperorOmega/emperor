import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  useWorkbenchQueries: vi.fn(),
  useConfigSnapshotRecords: vi.fn(),
  useInspectionPreviewState: vi.fn(),
  useLogRunsQuery: vi.fn(),
  useLogTagsQuery: vi.fn(),
  requestPreview: vi.fn(),
  clearPreview: vi.fn(),
  writeTargetSelection: vi.fn(),
  renameSnapshotRecord: vi.fn(),
  removeSnapshotRecord: vi.fn(),
  retrySnapshotRecordMutation: vi.fn(),
  dismissSnapshotRecordMutation: vi.fn(),
  clearSnapshotRecordsForConnectionChange: vi.fn(),
}));

vi.mock("@/features/workbench/state/use-workbench-queries", () => ({
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
  useWorkbenchQueries: mocks.useWorkbenchQueries,
}));

vi.mock(
  "@/features/workbench/state/config-snapshots/use-config-snapshot-records",
  () => ({
    useConfigSnapshotRecords: mocks.useConfigSnapshotRecords,
  }),
);

vi.mock(
  "@/features/workbench/state/target/_inspection-preview",
  async (importOriginal) => ({
    ...(await importOriginal<
      typeof import("@/features/workbench/state/target/_inspection-preview")
    >()),
    useInspectionPreviewState: mocks.useInspectionPreviewState,
  }),
);

vi.mock("@/features/workbench/state/logs/use-log-queries", () => ({
  useLogRunsQuery: mocks.useLogRunsQuery,
  useLogTagsQuery: mocks.useLogTagsQuery,
}));

vi.mock(
  "@/features/workbench/state/target/target-selection-storage",
  async (importOriginal) => {
    const actual = await importOriginal<
      typeof import("@/features/workbench/state/target/target-selection-storage")
    >();
    return {
      ...actual,
      writePersistedTargetSelection: (
        selection: Parameters<
          typeof actual.writePersistedTargetSelection
        >[0],
      ) => {
        mocks.writeTargetSelection(selection);
        actual.writePersistedTargetSelection(selection);
      },
    };
  },
);

import {
  type ConfigField,
  type ConfigSnapshotRecord,
  type DatasetGroup,
  type LogRun,
  type LogRunTags,
  type MonitorOption,
} from "@/lib/api";
import {
  readPersistedTargetSelection,
  writePersistedTargetSelection,
} from "@/features/workbench/state/target/target-selection-storage";
import {
  useModelPackageInspectionState,
} from "@/features/workbench/state/target/use-model-package-inspection-state";

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
let monitorOptions: MonitorOption[] = [];
let configSnapshotsLoading = false;
let configSnapshotsError = false;
let schemaFieldsByPreset: Record<string, ConfigField[]> = {};
let schemaQueryError = false;
let datasetGroupsOverride: DatasetGroup[] | undefined;
let historicalRuns: LogRun[] = [];
let historicalRunTags: LogRunTags[] = [];

function query<TData>(data: TData) {
  return {
    data,
    isLoading: false,
    isSuccess: true,
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
  return renderHook(() => useModelPackageInspectionState({}));
}

async function selectHistoricalRunThroughInspection(
  result: ReturnType<typeof renderTargetState>["result"],
  run: LogRun,
) {
  act(() => {
    result.current.contexts.model.actions.browseHistoricalRuns();
  });
  await waitFor(() => {
    expect(
      result.current.historical.browsing.historicalExperimentOptions.some(
        (option) => option.value === run.experiment,
      ),
    ).toBe(true);
  });
  act(() => {
    result.current.historical.browsing.setSelectedHistoricalExperimentFilter(
      run.experiment,
    );
  });
  await waitFor(() => {
    expect(
      result.current.historical.browsing.historicalDatasetOptions.some(
        (option) => option.value === run.dataset,
      ),
    ).toBe(true);
  });
  act(() => {
    result.current.historical.browsing.setSelectedHistoricalDatasetFilter(
      run.dataset,
    );
  });
  await waitFor(() => {
    expect(
      result.current.historical.browsing.historicalPresetOptions.some(
        (option) => option.value === run.preset,
      ),
    ).toBe(true);
  });
  act(() => {
    result.current.historical.browsing.setSelectedHistoricalPreset(run.preset);
  });
  await waitFor(() => {
    expect(result.current.historical.browsing.selectedLogRunId).toBe(run.id);
  });
}

beforeEach(() => {
  window.localStorage.clear();
  snapshots = [];
  monitorOptions = [];
  configSnapshotsLoading = false;
  configSnapshotsError = false;
  schemaFieldsByPreset = {};
  schemaQueryError = false;
  datasetGroupsOverride = undefined;
  historicalRuns = [];
  historicalRunTags = [];
  mocks.requestPreview.mockReset();
  mocks.clearPreview.mockReset();
  mocks.writeTargetSelection.mockReset();
  mocks.renameSnapshotRecord.mockReset().mockImplementation(async ({ id }) => ({
    ok: true,
    kind: "rename",
    snapshotId: id,
    record: null,
  }));
  mocks.removeSnapshotRecord.mockReset().mockImplementation(async (id) => {
    snapshots = snapshots.filter((snapshot) => snapshot.id !== id);
    return {
      ok: true,
      kind: "remove",
      snapshotId: id,
      record: null,
    };
  });
  mocks.retrySnapshotRecordMutation.mockReset().mockResolvedValue(null);
  mocks.dismissSnapshotRecordMutation.mockReset();
  mocks.clearSnapshotRecordsForConnectionChange.mockReset();
  mocks.useLogRunsQuery.mockReset().mockImplementation(({ enabled }) => ({
    data: enabled ? { runs: historicalRuns } : undefined,
    isLoading: false,
    isSuccess: Boolean(enabled),
    isError: false,
    error: null,
  }));
  mocks.useLogTagsQuery.mockReset().mockImplementation(({ enabled }) => ({
    data: enabled ? { runs: historicalRunTags } : undefined,
    isLoading: false,
    isSuccess: Boolean(enabled),
    isError: false,
    error: null,
  }));
  mocks.useInspectionPreviewState.mockReset().mockImplementation(() => ({
    response: undefined,
    request: null,
    status: { isBuilding: false, isError: false, error: null },
    clear: mocks.clearPreview,
    clearForConnectionChange: mocks.clearPreview,
    ensure: mocks.requestPreview,
    refresh: mocks.requestPreview,
  }));
  mocks.useWorkbenchQueries.mockReset().mockImplementation(
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
            { modelType: "linears", model: "linear_adaptive" },
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
          datasetGroups:
            datasetGroupsOverride ??
            (selectedModel
              ? [
                {
                  experimentTask: "image-classification",
                  label: "Image Classification",
                  datasets,
                },
                ...(selectedModelType === "linears"
                  ? [
                      {
                        experimentTask: "fashion-classification",
                        label: "Fashion Classification",
                        datasets: datasets.filter(
                          (dataset) => dataset.name === "FashionMnist",
                        ),
                      },
                    ]
                  : []),
                ]
              : []),
        }),
        monitorsQuery: query({
          modelType: selectedModelType,
          model: selectedModel,
          monitors: selectedModel ? monitorOptions : [],
        }),
        schemaQuery: schemaQueryError
          ? {
              data: undefined,
              isLoading: false,
              isSuccess: false,
              isError: true,
              error: new Error("schema read failed"),
            }
          : query({
              modelType: selectedModelType,
              model: selectedModel,
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
  mocks.useConfigSnapshotRecords.mockReset().mockImplementation(() => ({
    records: snapshots,
    status: configSnapshotsError
      ? {
          isLoading: false,
          isReady: false,
          isError: true,
          error: new Error("snapshot read failed"),
          mutation: {
            phase: "idle",
            kind: null,
            snapshotId: null,
            error: "",
            canRetry: false,
          },
        }
      : configSnapshotsLoading
        ? {
            isLoading: true,
            isReady: false,
            isError: false,
            error: null,
            mutation: {
              phase: "idle",
              kind: null,
              snapshotId: null,
              error: "",
              canRetry: false,
            },
          }
        : {
            isLoading: false,
            isReady: true,
            isError: false,
            error: null,
            mutation: {
              phase: "idle",
              kind: null,
              snapshotId: null,
              error: "",
              canRetry: false,
            },
          },
    actions: {
      create: vi.fn(),
      rename: mocks.renameSnapshotRecord,
      update: vi.fn(),
      remove: mocks.removeSnapshotRecord,
      retry: mocks.retrySnapshotRecordMutation,
      dismissMutation: mocks.dismissSnapshotRecordMutation,
      clearForConnectionChange: mocks.clearSnapshotRecordsForConnectionChange,
    },
  }));
});

describe("useModelPackageInspectionState", () => {
  it("restores a valid persisted preset target before requesting Inspection", async () => {
    writePersistedTargetSelection({
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedPreset: "fast",
      selectedTargetMode: "preset",
      selectedSnapshotId: "",
    });

    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.target.kind).toBe("preset");
      expect(result.current.contexts.model.browser.selectedModelType).toBe(
        "linears",
      );
      expect(result.current.contexts.model.browser.selectedModel).toBe(
        "linear",
      );
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "fast",
      );
      expect(
        result.current.contexts.model.browser.selectedSnapshotId,
      ).toBe("");
    });
    expect(mocks.requestPreview).toHaveBeenLastCalledWith(
      expect.objectContaining({
        modelType: "linears",
        model: "linear",
        preset: "fast",
        targetMode: "preset",
        targetId: "fast",
      }),
    );
  });

  it("normalizes an unavailable persisted Model Package to the catalog default", async () => {
    writePersistedTargetSelection({
      selectedModelType: "missing-type",
      selectedModel: "missing-model",
      selectedPreset: "missing-preset",
      selectedTargetMode: "preset",
      selectedSnapshotId: "",
    });

    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.target.kind).toBe("preset");
      expect(result.current.contexts.model.browser).toMatchObject({
        selectedModelType: "linears",
        selectedModel: "linear",
        selectedPreset: "baseline",
        selectedDatasets: ["Mnist"],
      });
    });
  });

  it("falls back to the persisted preset when its snapshot is unavailable", async () => {
    writePersistedTargetSelection({
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedPreset: "fast",
      selectedTargetMode: "snapshot",
      selectedSnapshotId: "deleted-snapshot",
    });

    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.target.kind).toBe("preset");
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "fast",
      );
      expect(
        result.current.contexts.model.browser.selectedSnapshotId,
      ).toBe("");
    });
    expect(mocks.requestPreview).toHaveBeenLastCalledWith(
      expect.objectContaining({
        preset: "fast",
        targetMode: "preset",
        targetId: "fast",
      }),
    );
  });

  it("auto-selects the first target and requests the initial preview", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser).toMatchObject({
        selectedModelType: "linears",
        selectedModel: "linear",
        selectedPreset: "baseline",
        selectedDatasets: ["Mnist"],
      });
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

  it("defers manual Inspection requests until protected access is ready", async () => {
    const rendered = renderHook(
      ({ protectedReadsEnabled }) =>
        useModelPackageInspectionState({ protectedReadsEnabled }),
      { initialProps: { protectedReadsEnabled: false } },
    );

    await waitFor(() => {
      expect(rendered.result.current.contexts.model.browser).toMatchObject({
        selectedModel: "linear",
        selectedPreset: "baseline",
        selectedDatasets: ["Mnist"],
      });
    });
    act(() => {
      expect(
        rendered.result.current.contexts.model.actions.selectPresetTarget("fast"),
      ).toBe(true);
    });
    expect(mocks.requestPreview).not.toHaveBeenCalled();

    rendered.rerender({ protectedReadsEnabled: true });
    await waitFor(() => {
      expect(mocks.requestPreview).toHaveBeenCalledWith(
        expect.objectContaining({ preset: "fast", targetId: "fast" }),
      );
    });
  });

  it("browses historical Runs without replacing the complete Inspection target", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.target.kind).toBe("preset");
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "baseline",
      );
    });
    mocks.requestPreview.mockClear();
    mocks.clearPreview.mockClear();

    act(() => {
      result.current.contexts.model.actions.browseHistoricalRuns();
    });

    expect(result.current.contexts.model.browser.mode).toBe("experiment");
    expect(result.current.contexts.model.target.kind).toBe("preset");
    expect(result.current.contexts.model.browser.selectedPreset).toBe(
      "baseline",
    );
    expect(mocks.requestPreview).not.toHaveBeenCalled();
    expect(mocks.clearPreview).not.toHaveBeenCalled();
  });

  it("requests a new preview when the Experiment Task changes its dataset", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser).toMatchObject({
        selectedModelType: "linears",
        selectedModel: "linear",
        selectedPreset: "baseline",
        selectedDatasets: ["Mnist"],
      });
    });

    mocks.requestPreview.mockClear();
    const transitionRevision = result.current.inspection.transition.revision;

    act(() => {
      result.current.contexts.model.actions.selectExperimentTask(
        "fashion-classification",
      );
    });

    await waitFor(() => {
      expect(result.current.contexts.model.browser).toMatchObject({
        selectedExperimentTask: "fashion-classification",
        selectedDatasets: ["FashionMnist"],
      });
    });
    await waitFor(() => {
      expect(mocks.requestPreview).toHaveBeenCalledWith({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        experimentTask: "fashion-classification",
        dataset: "FashionMnist",
        overrides: {},
        targetMode: "preset",
        targetId: "baseline",
      });
    });
    expect(result.current.inspection.transition).toEqual({
      revision: transitionRevision + 1,
      cause: "target-changed",
    });
  });

  it("syncs a selected historical run into the target without keeping overrides", async () => {
    const run = logRun({ id: "run-fast" });
    historicalRuns = [run];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "baseline",
      );
    });

    act(() => {
      result.current.contexts.model.actions.editRuntimeDefault(
        "hidden_size",
        "128",
      );
    });
    const transitionRevision = result.current.inspection.transition.revision;
    await selectHistoricalRunThroughInspection(result, run);

    expect(result.current.contexts.model.target).toMatchObject({
      kind: "historical-run",
      preset: "fast",
      datasets: ["FashionMnist"],
      run: { runId: "run-fast" },
    });
    expect(result.current.contexts.model.browser.selectedSnapshotId).toBe(
      "",
    );
    expect(result.current.contexts.model.runtimeDefaults.active).toEqual(
      {},
    );
    expect(result.current.inspection.transition).toEqual({
      revision: transitionRevision + 1,
      cause: "target-changed",
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

  it("syncs a selected historical run whose preset and dataset are absent from the catalog", async () => {
    const run = logRun({
      id: "kaggle-run",
      experiment: "kaggle_linear_all",
      preset: "KAGGLE_LINEAR",
      dataset: "KaggleDigits",
    });
    historicalRuns = [run];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser).toMatchObject({
        selectedPreset: "baseline",
        selectedDatasets: ["Mnist"],
      });
    });

    act(() => {
      result.current.contexts.model.actions.editRuntimeDefault(
        "hidden_size",
        "128",
      );
    });
    await selectHistoricalRunThroughInspection(result, run);

    expect(result.current.contexts.model.target).toMatchObject({
      kind: "historical-run",
      preset: "KAGGLE_LINEAR",
      datasets: ["KaggleDigits"],
      run: {
        runId: "kaggle-run",
        experiment: "kaggle_linear_all",
        preset: "KAGGLE_LINEAR",
        dataset: "KaggleDigits",
      },
    });
    expect(result.current.contexts.model.browser).toMatchObject({
      selectedPreset: "baseline",
      selectedDatasets: ["Mnist"],
    });
    expect(result.current.contexts.model.runtimeDefaults.active).toEqual(
      {},
    );
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

  it("exits a historical target before editing preset Runtime Defaults", async () => {
    const run = logRun({ id: "run-fast" });
    historicalRuns = [run];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "baseline",
      );
    });

    await selectHistoricalRunThroughInspection(result, run);

    await waitFor(() => {
      expect(result.current.contexts.model.target).toMatchObject({
        kind: "historical-run",
        run: { runId: "run-fast" },
      });
    });

    mocks.requestPreview.mockClear();
    act(() => {
      result.current.contexts.model.actions.editRuntimeDefault(
        "hidden_size",
        "128",
      );
    });

    await waitFor(() => {
      expect(result.current.contexts.model.target.kind).toBe("preset");
      expect(mocks.requestPreview).toHaveBeenLastCalledWith({
        modelType: "linears",
        model: "linear",
        preset: "fast",
        experimentTask: "image-classification",
        dataset: "FashionMnist",
        overrides: { hidden_size: "128" },
        targetMode: "preset",
        targetId: "fast",
      });
    });
  });

  it("resets a historical target to its compatible catalog preset", async () => {
    const run = logRun({ id: "run-fast" });
    historicalRuns = [run];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "baseline",
      );
    });

    await selectHistoricalRunThroughInspection(result, run);

    await waitFor(() => {
      expect(result.current.contexts.model.target).toMatchObject({
        kind: "historical-run",
        run: { runId: "run-fast" },
      });
    });

    mocks.requestPreview.mockClear();
    act(() => {
      result.current.contexts.model.actions.resetRuntimeDefaults();
    });

    await waitFor(() => {
      expect(result.current.contexts.model.target.kind).toBe("preset");
      expect(result.current.contexts.model.runtimeDefaults.active).toEqual(
        {},
      );
    });
    expect(mocks.requestPreview).toHaveBeenLastCalledWith({
      modelType: "linears",
      model: "linear",
      preset: "fast",
      experimentTask: "image-classification",
      dataset: "FashionMnist",
      overrides: {},
      targetMode: "preset",
      targetId: "fast",
    });
  });

  it("selects another preset, preserves overrides, and refreshes the preview", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser).toMatchObject({
        selectedModelType: "linears",
        selectedModel: "linear",
        selectedPreset: "baseline",
        selectedDatasets: ["Mnist"],
      });
    });

    mocks.requestPreview.mockClear();

    act(() => {
      result.current.contexts.model.actions.editRuntimeDefault(
        "hidden_size",
        "128",
      );
    });
    mocks.requestPreview.mockClear();
    const transitionRevision = result.current.inspection.transition.revision;

    act(() => {
      result.current.contexts.model.actions.selectPresetTarget("fast");
    });

    expect(result.current.contexts.model.browser.selectedPreset).toBe(
      "fast",
    );
    expect(result.current.contexts.model.runtimeDefaults.active).toEqual({
      hidden_size: "128",
    });
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
    expect(result.current.inspection.transition).toEqual({
      revision: transitionRevision + 1,
      cause: "target-changed",
    });
  });

  it("clears preset overrides when the model changes", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedModel).toBe(
        "linear",
      );
    });

    act(() => {
      result.current.contexts.model.actions.editRuntimeDefault(
        "hidden_size",
        "128",
      );
    });
    expect(result.current.contexts.model.runtimeDefaults.active).toEqual({
      hidden_size: "128",
    });

    act(() => {
      result.current.contexts.model.actions.selectModelPackage(
        "linear",
        "experts",
      );
    });

    await waitFor(() => {
      expect(result.current.contexts.model.browser).toMatchObject({
        selectedModelType: "experts",
        selectedModel: "linear",
      });
    });
    expect(result.current.contexts.model.runtimeDefaults.active).toEqual({});
  });

  it("omits locked preset overrides from preview and restores them for an unlocked preset", async () => {
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
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "baseline",
      );
    });

    act(() => {
      result.current.contexts.model.actions.editRuntimeDefault(
        "layer_width",
        "128",
      );
    });
    expect(result.current.contexts.model.runtimeDefaults.active).toEqual({
      layer_width: "128",
    });

    mocks.requestPreview.mockClear();
    act(() => {
      result.current.contexts.model.actions.selectPresetTarget("fast");
    });

    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "fast",
      );
      expect(result.current.contexts.model.runtimeDefaults.active).toEqual(
        {},
      );
      expect(
        result.current.contexts.model.runtimeDefaults.inactiveLockedCount,
      ).toBe(1);
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

    act(() => {
      result.current.contexts.model.actions.selectPresetTarget("baseline");
    });

    await waitFor(() => {
      expect(result.current.contexts.model.runtimeDefaults.active).toEqual({
        layer_width: "128",
      });
    });
  });

  it("repairs adaptive edits and suppresses a flag returned to its Runtime Default", async () => {
    schemaFieldsByPreset = { baseline: adaptiveConfigFields };
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "baseline",
      );
    });

    act(() => {
      result.current.contexts.model.actions.editRuntimeDefault(
        "weight_option_flag",
        "true",
      );
    });

    expect(result.current.contexts.model.runtimeDefaults.active).toEqual({
      weight_option_flag: "true",
      weight_option: "SingleModelDynamicWeightConfig",
    });

    act(() => {
      result.current.contexts.model.actions.editRuntimeDefault(
        "weight_option_flag",
        "false",
      );
    });

    expect(result.current.contexts.model.runtimeDefaults.active).toEqual({});
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
      expect(result.current.contexts.model.browser).toMatchObject({
        selectedPreset: "baseline",
        selectedDatasets: ["Mnist"],
      });
    });

    mocks.requestPreview.mockClear();
    act(() => {
      expect(
        result.current.contexts.model.actions.selectSnapshotTarget(
          "snapshot-old-adaptive",
        ),
      ).toBe(true);
    });

    expect(result.current.contexts.model.runtimeDefaults.active).toEqual({
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

  it("selects a different-preset snapshot after its schema settles with one transition", async () => {
    schemaFieldsByPreset = {
      baseline: [configField({ key: "baseline_only" })],
      fast: [configField({ key: "fast_only" })],
    };
    snapshots = [
      {
        id: "snapshot-fast",
        name: "Fast tuned",
        modelType: "linears",
        model: "linear",
        preset: "fast",
        overrides: { fast_only: "7" },
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      },
    ];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "baseline",
      );
    });
    mocks.requestPreview.mockClear();
    const transitionRevision = result.current.inspection.transition.revision;
    act(() => {
      result.current.contexts.model.actions.selectSnapshotTarget(
        "snapshot-fast",
      );
    });

    await waitFor(() => {
      expect(result.current.contexts.model.target).toMatchObject({
        kind: "snapshot",
        preset: "fast",
      });
      expect(result.current.contexts.model.runtimeDefaults.active).toEqual({
        fast_only: "7",
      });
      expect(mocks.requestPreview).toHaveBeenLastCalledWith(
        expect.objectContaining({
          preset: "fast",
          targetMode: "snapshot",
          targetId: "snapshot-fast",
          overrides: { fast_only: "7" },
        }),
      );
    });
    expect(result.current.inspection.transition).toEqual({
      revision: transitionRevision + 1,
      cause: "target-changed",
    });
  });

  it("keeps snapshot target values isolated from the preset override draft", async () => {
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
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "baseline",
      );
    });

    act(() => {
      result.current.contexts.model.actions.editRuntimeDefault(
        "hidden_size",
        "128",
      );
    });
    act(() => {
      expect(
        result.current.contexts.model.actions.selectSnapshotTarget(
          "snapshot-baseline",
        ),
      ).toBe(true);
    });

    expect(result.current.contexts.model.runtimeDefaults.active).toEqual({
      hidden_size: "256",
    });

    act(() => {
      result.current.contexts.model.actions.showPresetTarget();
    });

    expect(result.current.contexts.model.browser.mode).toBe("preset");
    expect(result.current.contexts.model.target.kind).toBe("snapshot");
    expect(result.current.contexts.model.runtimeDefaults.active).toEqual({
      hidden_size: "256",
    });

    act(() => {
      result.current.contexts.model.actions.selectPresetTarget(
        "baseline",
      );
    });

    expect(result.current.contexts.model.target.kind).toBe("preset");
    expect(result.current.contexts.model.runtimeDefaults.active).toEqual({
      hidden_size: "128",
    });
  });

  it("resets a snapshot target to its preset with empty Runtime Defaults", async () => {
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
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "baseline",
      );
    });

    act(() => {
      expect(
        result.current.contexts.model.actions.selectSnapshotTarget(
          "snapshot-baseline",
        ),
      ).toBe(true);
    });

    await waitFor(() => {
      expect(result.current.contexts.model.target.kind).toBe("snapshot");
      expect(
        result.current.contexts.model.browser.selectedSnapshotId,
      ).toBe("snapshot-baseline");
      expect(result.current.contexts.model.runtimeDefaults.active).toEqual({
        hidden_size: "256",
      });
    });

    mocks.requestPreview.mockClear();
    act(() => {
      result.current.contexts.model.actions.resetRuntimeDefaults();
    });

    await waitFor(() => {
      expect(result.current.contexts.model.target.kind).toBe("preset");
      expect(
        result.current.contexts.model.browser.selectedSnapshotId,
      ).toBe("");
      expect(result.current.contexts.model.runtimeDefaults.active).toEqual(
        {},
      );
    });
    expect(mocks.requestPreview).toHaveBeenLastCalledWith({
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      experimentTask: "image-classification",
      dataset: "Mnist",
      overrides: {},
      targetMode: "preset",
      targetId: "baseline",
    });
    expect(mocks.requestPreview.mock.calls.at(-1)?.[0]).not.toHaveProperty(
      "logRunId",
    );
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
      expect(result.current.contexts.model.browser).toMatchObject({
        selectedModelType: "linears",
        selectedModel: "linear",
        selectedPreset: "fast",
      });
    });

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
      expect(result.current.contexts.model.target.kind).toBe("snapshot");
      expect(
        result.current.contexts.model.browser.selectedSnapshotId,
      ).toBe("snapshot-fast");
    });
    expect(result.current.contexts.model.runtimeDefaults.active).toEqual({
      hidden_size: "256",
    });
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
      expect(result.current.contexts.model.browser).toMatchObject({
        selectedModelType: "linears",
        selectedModel: "linear",
      });
    });

    expect(result.current.contexts.snapshots.records.allCount).toBe(1);

    act(() => {
      expect(
        result.current.contexts.snapshots.actions.selectTarget("snapshot-1"),
      ).toBe(true);
    });

    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "fast",
      );
      expect(result.current.contexts.model.runtimeDefaults.active).toEqual({
        hidden_size: "256",
      });
    });
  });

  it("settles preset restoration when Dataset Metadata is successfully empty", async () => {
    datasetGroupsOverride = [];
    writePersistedTargetSelection({
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedPreset: "fast",
      selectedTargetMode: "preset",
      selectedSnapshotId: "",
    });
    mocks.writeTargetSelection.mockClear();

    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.target).toMatchObject({
        kind: "preset",
        preset: "fast",
        datasets: [],
      });
      expect(mocks.writeTargetSelection).toHaveBeenCalled();
    });
    expect(mocks.requestPreview).not.toHaveBeenCalled();
    expect(readPersistedTargetSelection()).toMatchObject({
      selectedPreset: "fast",
      selectedTargetMode: "preset",
    });
  });

  it("lets explicit preset intent win while snapshot restoration is waiting", async () => {
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
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "fast",
      );
    });
    act(() => {
      expect(
        result.current.contexts.model.actions.selectPresetTarget(
          "baseline",
        ),
      ).toBe(true);
    });

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
      expect(result.current.contexts.model.target).toMatchObject({
        kind: "preset",
        preset: "baseline",
      });
      expect(
        result.current.contexts.model.browser.selectedSnapshotId,
      ).toBe("");
    });
  });

  it("falls back when a persisted snapshot declares an unavailable preset", async () => {
    snapshots = [
      {
        id: "snapshot-incompatible",
        name: "Removed preset",
        modelType: "linears",
        model: "linear",
        preset: "removed-preset",
        overrides: { hidden_size: "256" },
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      },
    ];
    writePersistedTargetSelection({
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedPreset: "fast",
      selectedTargetMode: "snapshot",
      selectedSnapshotId: "snapshot-incompatible",
    });

    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.target).toMatchObject({
        kind: "preset",
        preset: "fast",
      });
      expect(
        result.current.contexts.model.browser.selectedSnapshotId,
      ).toBe("");
    });
  });

  it("falls back when persisted snapshot records fail to load", async () => {
    configSnapshotsError = true;
    writePersistedTargetSelection({
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedPreset: "fast",
      selectedTargetMode: "snapshot",
      selectedSnapshotId: "snapshot-fast",
    });

    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.target).toMatchObject({
        kind: "preset",
        preset: "fast",
      });
      expect(
        result.current.contexts.model.browser.selectedSnapshotId,
      ).toBe("");
    });
  });

  it("falls back when the persisted snapshot schema fails to load", async () => {
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
    schemaQueryError = true;
    writePersistedTargetSelection({
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedPreset: "fast",
      selectedTargetMode: "snapshot",
      selectedSnapshotId: "snapshot-fast",
    });

    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.target).toMatchObject({
        kind: "preset",
        preset: "fast",
      });
      expect(
        result.current.contexts.model.browser.selectedSnapshotId,
      ).toBe("");
    });
  });

  it("reconciles deletion of the active snapshot to its preset target", async () => {
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
    const { result, rerender } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "baseline",
      );
    });
    act(() => {
      result.current.contexts.model.actions.selectSnapshotTarget(
        "snapshot-baseline",
      );
    });
    const transitionRevision = result.current.inspection.transition.revision;

    await act(async () => {
      await result.current.contexts.snapshots.actions.remove(
        "snapshot-baseline",
      );
    });
    rerender();

    expect(result.current.contexts.model.target.kind).toBe("preset");
    expect(result.current.contexts.model.browser.selectedSnapshotId).toBe(
      "",
    );
    expect(result.current.contexts.model.runtimeDefaults.active).toEqual(
      {},
    );
    expect(result.current.inspection.transition).toEqual({
      revision: transitionRevision + 1,
      cause: "target-changed",
    });
  });

  it("retains the active snapshot after failed removal and reconciles it after retry", async () => {
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
    mocks.removeSnapshotRecord.mockResolvedValueOnce({
      ok: false,
      kind: "remove",
      snapshotId: "snapshot-baseline",
      error: "Removal rejected.",
      retryable: true,
    });
    mocks.retrySnapshotRecordMutation.mockImplementationOnce(async () => {
      snapshots = [];
      return {
        ok: true,
        kind: "remove",
        snapshotId: "snapshot-baseline",
        record: null,
      };
    });
    const { result, rerender } = renderTargetState();
    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "baseline",
      );
    });
    act(() => {
      result.current.contexts.model.actions.selectSnapshotTarget(
        "snapshot-baseline",
      );
    });
    const transitionRevision = result.current.inspection.transition.revision;

    await act(async () => {
      await result.current.contexts.snapshots.actions.remove(
        "snapshot-baseline",
      );
    });
    expect(result.current.contexts.model.target.kind).toBe("snapshot");
    expect(result.current.contexts.model.browser.selectedSnapshotId).toBe(
      "snapshot-baseline",
    );
    expect(result.current.inspection.transition.revision).toBe(
      transitionRevision,
    );

    await act(async () => {
      await result.current.contexts.snapshots.actions.retryMutation();
    });
    rerender();
    expect(result.current.contexts.model.target.kind).toBe("preset");
    expect(result.current.contexts.model.browser.selectedSnapshotId).toBe("");
    expect(result.current.inspection.transition).toEqual({
      revision: transitionRevision + 1,
      cause: "target-changed",
    });
  });

  it("reconciles an active snapshot that disappears during record refresh", async () => {
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
    const { result, rerender } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "baseline",
      );
    });
    act(() => {
      result.current.contexts.model.actions.selectSnapshotTarget(
        "snapshot-baseline",
      );
    });
    const transitionRevision = result.current.inspection.transition.revision;
    snapshots = [];
    rerender();

    await waitFor(() => {
      expect(result.current.contexts.model.target.kind).toBe("preset");
      expect(
        result.current.contexts.model.browser.selectedSnapshotId,
      ).toBe("");
    });
    expect(result.current.inspection.transition).toEqual({
      revision: transitionRevision + 1,
      cause: "target-changed",
    });
  });

  it("reconciles an active snapshot that becomes preset-incompatible", async () => {
    const snapshot: ConfigSnapshotRecord = {
      id: "snapshot-baseline",
      name: "Baseline tuned",
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      overrides: { hidden_size: "256" },
      createdAt: "2026-06-01T00:00:00.000Z",
      updatedAt: "2026-06-01T00:00:00.000Z",
    };
    snapshots = [snapshot];
    const { result, rerender } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "baseline",
      );
    });
    act(() => {
      result.current.contexts.model.actions.selectSnapshotTarget(
        snapshot.id,
      );
    });
    snapshots = [{ ...snapshot, preset: "removed-preset" }];
    rerender();

    await waitFor(() => {
      expect(result.current.contexts.model.target.kind).toBe("preset");
      expect(
        result.current.contexts.model.browser.selectedSnapshotId,
      ).toBe("");
    });
  });

  it("replaces Inspection when the active snapshot record changes semantically", async () => {
    const snapshot: ConfigSnapshotRecord = {
      id: "snapshot-baseline",
      name: "Baseline tuned",
      modelType: "linears",
      model: "linear",
      preset: "baseline",
      overrides: { hidden_size: "256" },
      createdAt: "2026-06-01T00:00:00.000Z",
      updatedAt: "2026-06-01T00:00:00.000Z",
    };
    snapshots = [snapshot];
    const { result, rerender } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "baseline",
      );
    });
    act(() => {
      result.current.contexts.model.actions.selectSnapshotTarget(snapshot.id);
    });
    const transitionRevision = result.current.inspection.transition.revision;
    mocks.requestPreview.mockClear();
    snapshots = [
      {
        ...snapshot,
        overrides: { hidden_size: "512" },
        updatedAt: "2026-06-02T00:00:00.000Z",
      },
    ];
    rerender();

    await waitFor(() => {
      expect(result.current.contexts.model.runtimeDefaults.active).toEqual({
        hidden_size: "512",
      });
      expect(result.current.inspection.transition).toEqual({
        revision: transitionRevision + 1,
        cause: "target-changed",
      });
    });
    expect(mocks.requestPreview).toHaveBeenLastCalledWith(
      expect.objectContaining({
        targetMode: "snapshot",
        targetId: snapshot.id,
        overrides: { hidden_size: "512" },
      }),
    );
  });

  it("emits complete-target transitions even when no preview dataset exists", async () => {
    datasetGroupsOverride = [];
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
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "baseline",
      );
    });
    mocks.requestPreview.mockClear();
    const presetRevision = result.current.inspection.transition.revision;
    act(() => {
      result.current.contexts.model.actions.selectSnapshotTarget(
        "snapshot-baseline",
      );
    });
    expect(result.current.contexts.model.target.kind).toBe("snapshot");
    expect(result.current.inspection.transition.revision).toBe(
      presetRevision + 1,
    );

    const snapshotRevision = result.current.inspection.transition.revision;
    act(() => {
      result.current.contexts.model.actions.selectPresetTarget("fast");
    });
    expect(result.current.contexts.model.target).toMatchObject({
      kind: "preset",
      preset: "fast",
    });
    expect(result.current.inspection.transition.revision).toBe(
      snapshotRevision + 1,
    );
    expect(mocks.requestPreview).not.toHaveBeenCalled();
  });

  it("rejects a historical Run from another Model Package atomically", async () => {
    historicalRuns = [
      logRun({
        id: "wrong-package",
        modelType: "experts",
        model: "linear",
      }),
    ];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedModel).toBe(
        "linear",
      );
    });
    mocks.requestPreview.mockClear();
    const transition = result.current.inspection.transition;

    act(() => {
      result.current.contexts.model.actions.browseHistoricalRuns();
    });

    expect(result.current.contexts.model.target.kind).toBe("preset");
    expect(
      result.current.historical.browsing.historicalExperimentOptions,
    ).toEqual([]);
    expect(result.current.inspection.transition).toEqual(transition);
    expect(mocks.requestPreview).not.toHaveBeenCalled();
  });

  it("rejects invalid catalog transition intents atomically", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "baseline",
      );
    });
    const browser = result.current.contexts.model.browser;
    const target = result.current.contexts.model.target;
    const transition = result.current.inspection.transition;
    mocks.requestPreview.mockClear();

    act(() => {
      expect(
        result.current.contexts.model.actions.selectModelPackage(
          "missing-model",
          "linears",
        ),
      ).toBe(false);
      expect(
        result.current.contexts.model.actions.selectPresetTarget(
          "missing-preset",
        ),
      ).toBe(false);
      expect(
        result.current.contexts.model.actions.selectSnapshotTarget(
          "missing-snapshot",
        ),
      ).toBe(false);
      expect(
        result.current.contexts.model.actions.selectExperimentTask(
          "missing-task",
        ),
      ).toBe(false);
    });

    expect(result.current.contexts.model.browser).toBe(browser);
    expect(result.current.contexts.model.target).toBe(target);
    expect(result.current.inspection.transition).toEqual(transition);
    expect(mocks.requestPreview).not.toHaveBeenCalled();
  });

  it("treats an Experiment Task change as browsing while a historical target remains complete", async () => {
    const run = logRun({ id: "run-fast" });
    historicalRuns = [run];
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "baseline",
      );
    });
    await selectHistoricalRunThroughInspection(result, run);
    const transition = result.current.inspection.transition;
    mocks.requestPreview.mockClear();

    act(() => {
      result.current.contexts.model.actions.selectExperimentTask(
        "fashion-classification",
      );
    });

    expect(result.current.contexts.model.browser).toMatchObject({
      selectedExperimentTask: "fashion-classification",
      selectedDatasets: ["FashionMnist"],
    });
    expect(result.current.contexts.model.target).toMatchObject({
      kind: "historical-run",
      run: { runId: "run-fast" },
    });
    expect(result.current.inspection.transition).toEqual(transition);
    expect(mocks.requestPreview).not.toHaveBeenCalled();
  });

  it("clears incompatible state on a same-type Model Package switch", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedModel).toBe(
        "linear",
      );
    });
    act(() => {
      result.current.contexts.model.actions.editRuntimeDefault(
        "hidden_size",
        "128",
      );
    });
    act(() => {
      expect(
        result.current.contexts.model.actions.selectModelPackage(
          "linear_adaptive",
          "linears",
        ),
      ).toBe(true);
    });

    await waitFor(() => {
      expect(result.current.contexts.model.browser).toMatchObject({
        selectedModelType: "linears",
        selectedModel: "linear_adaptive",
        selectedPreset: "baseline",
        selectedDatasets: ["Mnist"],
      });
      expect(result.current.contexts.model.target.kind).toBe("preset");
      expect(result.current.contexts.model.runtimeDefaults.active).toEqual(
        {},
      );
    });
  });

  it("emits one semantic event for edit, refresh, reset, and Model Package transitions", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedModel).toBe("linear");
    });
    let revision = result.current.inspection.transition.revision;

    act(() => {
      result.current.contexts.model.actions.editRuntimeDefault(
        "hidden_size",
        "128",
      );
    });
    expect(result.current.inspection.transition).toEqual({
      revision: ++revision,
      cause: "target-changed",
    });

    act(() => {
      result.current.contexts.model.actions.refreshInspection();
    });
    expect(result.current.inspection.transition).toEqual({
      revision: ++revision,
      cause: "inspection-refreshed",
    });

    act(() => {
      result.current.contexts.model.actions.resetRuntimeDefaults();
    });
    expect(result.current.inspection.transition).toEqual({
      revision: ++revision,
      cause: "target-changed",
    });

    act(() => {
      result.current.contexts.model.actions.selectModelPackage(
        "linear_adaptive",
        "linears",
      );
    });
    expect(result.current.inspection.transition).toEqual({
      revision: ++revision,
      cause: "target-changed",
    });
  });

  it("suppresses a semantic transition for a token-equivalent no-op edit", async () => {
    const { result } = renderTargetState();

    await waitFor(() => {
      expect(result.current.contexts.model.browser.selectedModel).toBe("linear");
    });
    act(() => {
      result.current.contexts.model.actions.editRuntimeDefault(
        "hidden_size",
        "128",
      );
    });
    const transition = result.current.inspection.transition;

    act(() => {
      result.current.contexts.model.actions.editRuntimeDefault(
        "HIDDEN-SIZE",
        "128",
      );
    });

    expect(result.current.contexts.model.runtimeDefaults.active).toEqual({
      hidden_size: "128",
    });
    expect(result.current.inspection.transition).toBe(transition);
  });

  it("exits a snapshot target when clearing an already-empty preset draft", async () => {
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
      expect(result.current.contexts.model.browser.selectedPreset).toBe(
        "baseline",
      );
    });
    act(() => {
      result.current.contexts.model.actions.selectSnapshotTarget(
        "snapshot-baseline",
      );
    });
    const transitionRevision = result.current.inspection.transition.revision;

    act(() => {
      result.current.contexts.model.actions.clearRuntimeDefault(
        "hidden_size",
      );
    });

    expect(result.current.contexts.model.target.kind).toBe("preset");
    expect(result.current.contexts.model.runtimeDefaults.active).toEqual(
      {},
    );
    expect(result.current.inspection.transition).toEqual({
      revision: transitionRevision + 1,
      cause: "target-changed",
    });
  });

});
