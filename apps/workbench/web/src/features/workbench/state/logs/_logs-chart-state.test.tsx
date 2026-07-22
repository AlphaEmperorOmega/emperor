import { createElement, type ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  fetchLogCheckpoints: vi.fn(),
  fetchLogMedia: vi.fn(),
  fetchLogScalars: vi.fn(),
}));

vi.mock("@/lib/api/log-query-client", () => ({
  fetchLogCheckpoints: mocks.fetchLogCheckpoints,
  fetchLogMedia: mocks.fetchLogMedia,
  fetchLogScalars: mocks.fetchLogScalars,
}));

import {
  defaultLogBestRunMetricTag,
  deriveLogsChartEmptyState,
  groupLogScalarSeriesByTag,
  groupSelectedLogMetrics,
  useLogsChartViewModel,
} from "@/features/workbench/state/logs/_logs-chart-state";
import {
  buildLogScalarTagOptions,
  buildTrainValidationScalarPairs,
  defaultTrainValidationScalarPairSuffixes,
  isDefaultScalarTag,
  groupLogPlotSelectorTags,
  isLogPlotSelectorScalarTag,
} from "@/features/workbench/state/logs/logs-selectors";
import { buildConfusionMatrixHeatmaps } from "@/features/workbench/state/logs/log-diagnostics";
import type { LogRun, LogRunTags, LogScalarSeries } from "@/lib/api/logs";

type ChartSource = Parameters<typeof useLogsChartViewModel>[0];

function deferred<Value>() {
  let resolve!: (value: Value) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<Value>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, reject, resolve };
}

function scalarSeries({
  points,
  runId,
  tag,
}: Partial<LogScalarSeries> & Pick<LogScalarSeries, "runId" | "tag">): LogScalarSeries {
  return {
    runId,
    tag,
    points: points ?? [{ step: 1, wallTime: 100, value: 0.25 }],
  };
}

function logRun(overrides: Partial<LogRun> & Pick<LogRun, "id">): LogRun {
  const experiment = overrides.experiment ?? "exp_a";
  const dataset = overrides.dataset ?? "Cifar10";
  const modelType = overrides.modelType ?? "linears";
  const model = overrides.model ?? "linear";
  const preset = overrides.preset ?? "baseline";
  return {
    id: overrides.id,
    group: overrides.group ?? null,
    experiment,
    modelType,
    model,
    preset,
    dataset,
    runName: overrides.runName ?? overrides.id,
    timestamp: overrides.timestamp ?? null,
    version: overrides.version ?? "version_0",
    relativePath:
      overrides.relativePath ??
      `${experiment}/${modelType}/${model}/${preset}/${dataset}/${overrides.id}/version_0`,
    hasResult: overrides.hasResult ?? true,
    eventFileCount: overrides.eventFileCount ?? 1,
    checkpointCount: overrides.checkpointCount ?? 0,
    hasHparams: overrides.hasHparams ?? true,
    metrics: overrides.metrics ?? {},
  };
}

function logRunTags(runId: string, scalarTags: string[]): LogRunTags {
  return {
    runId,
    scalarTags,
    histogramTags: [],
    imageTags: [],
    textTags: [],
  };
}

function chartSource({
  collapsedMetricGroups = new Set<string>(),
  refreshLogLists = vi.fn(() => Promise.resolve()),
  runs = [logRun({ id: "run-1" })],
  selectedTagList = ["train/loss"],
  tagOptions = [
    { value: "validation/accuracy", label: "validation/accuracy", count: 1 },
    { value: "train/loss", label: "train/loss", count: 1 },
  ],
  tagRecords,
  tagsFetching = false,
  tagsRefreshing = false,
}: {
  collapsedMetricGroups?: Set<string>;
  refreshLogLists?: () => Promise<void>;
  runs?: LogRun[];
  selectedTagList?: string[];
  tagOptions?: Array<{ value: string; label: string; count?: number }>;
  tagRecords?: LogRunTags[];
  tagsFetching?: boolean;
  tagsRefreshing?: boolean;
} = {}): ChartSource {
  return {
    enabled: true,
    collapsedMetricGroups,
    confusionMatrixRateTags: [],
    hasMoreRuns: false,
    loadedScalarTagRunCount: runs.length,
    runsLoading: false,
    selectedTagList,
    tagOptions,
    tagRecords:
      tagRecords ??
      runs.map((run) =>
          logRunTags(
            run.id,
            tagOptions.map((option) => option.value),
          ),
        ),
    tagsFetching,
    tagsLoading: false,
    tagsRefreshing,
    commands: {
      refresh: () => {
        void refreshLogLists();
      },
      openRunDetail: vi.fn(),
      setMetricGroupExpanded: vi.fn(),
      selectMetricTags: vi.fn(),
    },
    visibleRunIds: runs.map((run) => run.id),
    visibleRuns: runs,
  } as unknown as ChartSource;
}

function renderChartState(source: ChartSource) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return renderHook(
    ({ currentSource }) => useLogsChartViewModel(currentSource),
    {
      initialProps: { currentSource: source },
      wrapper: ({ children }: { children: ReactNode }) =>
        createElement(QueryClientProvider, { client }, children),
    },
  );
}

function scalarRequestsFor(tag: string) {
  return mocks.fetchLogScalars.mock.calls
    .map(([input]) => input as { runIds: string[]; tags: string[] })
    .filter((input) => input.tags.includes(tag));
}

beforeEach(() => {
  vi.clearAllMocks();
  mocks.fetchLogCheckpoints.mockResolvedValue({ checkpoints: [] });
  mocks.fetchLogMedia.mockResolvedValue({ images: [], texts: [] });
  mocks.fetchLogScalars.mockImplementation(
    ({ runIds, tags }: { runIds: string[]; tags: string[] }) =>
      Promise.resolve({
        series: runIds.flatMap((runId) =>
          tags.map((tag) => scalarSeries({ runId, tag })),
        ),
      }),
  );
});

describe("logs chart view model", () => {
  it("activates chart reads on visibility and refreshes through the public projection", async () => {
    const refreshLogLists = vi.fn(() => Promise.resolve());
    const { result } = renderChartState(chartSource({ refreshLogLists }));

    await waitFor(() =>
      expect(scalarRequestsFor("validation/accuracy")).toHaveLength(1),
    );
    expect(result.current.content.bestRun.selectedMetricTag).toBe(
      "validation/accuracy",
    );
    expect(scalarRequestsFor("train/loss")).toHaveLength(0);
    expect(result.current.status.scalarTagQueryStates.has("train/loss"))
      .toBe(false);

    act(() => result.current.actions.onScalarChartVisible("train/loss"));

    await waitFor(() => expect(scalarRequestsFor("train/loss")).toHaveLength(1));
    await waitFor(() =>
      expect(
        result.current.content.metricsByGroup.train[0]?.series.map(
          (series) => series.runId,
        ),
      ).toEqual(["run-1"]),
    );
    expect(result.current.status.scalarTagQueryStates.get("train/loss"))
      .toMatchObject({ hasRequested: true, isError: false });
    expect(result.current.status.isRefreshDisabled).toBe(false);

    act(() => result.current.actions.onRefresh());
    expect(refreshLogLists).toHaveBeenCalledTimes(1);
  });

  it("fans visible charts out progressively within the ten-Run and six-tag limits", async () => {
    const runs = Array.from({ length: 24 }, (_, index) =>
      logRun({ id: `run-${index}` }),
    );
    const trainTags = Array.from(
      { length: 7 },
      (_, index) => `train/metric-${index}`,
    );
    const { result } = renderChartState(
      chartSource({
        runs,
        selectedTagList: trainTags,
        tagOptions: [
          {
            value: "validation/accuracy",
            label: "validation/accuracy",
            count: runs.length,
          },
          ...trainTags.map((tag) => ({
            value: tag,
            label: tag,
            count: runs.length,
          })),
        ],
      }),
    );

    await waitFor(() =>
      expect(scalarRequestsFor("validation/accuracy")).toHaveLength(1),
    );
    expect(trainTags.flatMap(scalarRequestsFor)).toHaveLength(0);

    act(() => {
      for (const tag of trainTags) {
        result.current.actions.onScalarChartVisible(tag);
      }
    });

    await waitFor(() => {
      const requests = mocks.fetchLogScalars.mock.calls
        .map(([input]) => input as { runIds: string[]; tags: string[] })
        .filter((input) => input.tags.some((tag) => trainTags.includes(tag)));
      expect(requests).toHaveLength(6);
      expect(
        requests.every(
          (request) => request.runIds.length <= 10 && request.tags.length <= 6,
        ),
      ).toBe(true);
      const requestedPairs = requests.flatMap((request) =>
        request.runIds.flatMap((runId) =>
          request.tags.map((tag) => `${runId}\u0000${tag}`),
        ),
      );
      expect(requestedPairs).toHaveLength(runs.length * trainTags.length);
      expect(new Set(requestedPairs).size).toBe(requestedPairs.length);
    });
  });

  it("scopes requests and status to visible metric groups", async () => {
    const trainResponse = deferred<{ series: LogScalarSeries[] }>();
    const otherResponse = deferred<{ series: LogScalarSeries[] }>();
    mocks.fetchLogScalars.mockImplementation(
      ({ runIds, tags }: { runIds: string[]; tags: string[] }) => {
        if (tags.includes("train/loss")) {
          return trainResponse.promise;
        }
        if (tags.includes("custom/metric")) {
          return otherResponse.promise;
        }
        return Promise.resolve({
          series: runIds.flatMap((runId) =>
            tags.map((tag) => scalarSeries({ runId, tag })),
          ),
        });
      },
    );
    const sourceInput = {
      collapsedMetricGroups: new Set(["other"]),
      selectedTagList: ["train/loss", "custom/metric"],
      tagOptions: [
        {
          value: "validation/accuracy",
          label: "validation/accuracy",
          count: 1,
        },
        { value: "train/loss", label: "train/loss", count: 1 },
        { value: "custom/metric", label: "custom/metric", count: 1 },
      ],
    };
    const rendered = renderChartState(chartSource(sourceInput));

    act(() => {
      rendered.result.current.actions.onScalarChartVisible("train/loss");
      rendered.result.current.actions.onScalarChartVisible("custom/metric");
    });
    await waitFor(() => expect(scalarRequestsFor("train/loss")).toHaveLength(1));

    expect(scalarRequestsFor("custom/metric")).toHaveLength(0);
    expect(rendered.result.current.status.scalarQueryStates.train).toMatchObject({
      isInitialLoading: true,
      isFetching: true,
      isError: false,
    });
    expect(rendered.result.current.status.scalarQueryStates.other).toEqual({
      isInitialLoading: false,
      isFetching: false,
      isError: false,
      error: null,
    });

    rendered.rerender({
      currentSource: chartSource({
        ...sourceInput,
        collapsedMetricGroups: new Set(),
      }),
    });
    await waitFor(() =>
      expect(scalarRequestsFor("custom/metric")).toHaveLength(1),
    );
    expect(rendered.result.current.status.scalarQueryStates.other).toMatchObject({
      isInitialLoading: true,
      isFetching: true,
      isError: false,
    });

    const otherError = new Error("other metrics unavailable");
    act(() => {
      trainResponse.resolve({
        series: [scalarSeries({ runId: "run-1", tag: "train/loss" })],
      });
      otherResponse.reject(otherError);
    });
    await waitFor(() =>
      expect(rendered.result.current.status.scalarQueryStates.other).toMatchObject({
        isFetching: false,
        isError: true,
        error: otherError,
      }),
    );
  });

  it("retains only compatible scalar series while tags and replacement reads settle", async () => {
    const replacement = deferred<{ series: LogScalarSeries[] }>();
    mocks.fetchLogScalars.mockImplementation(
      ({ runIds, tags }: { runIds: string[]; tags: string[] }) => {
        if (tags.includes("train/loss") && runIds.includes("run-2")) {
          return replacement.promise;
        }
        return Promise.resolve({
          series: runIds.flatMap((runId) =>
            tags.map((tag) => scalarSeries({ runId, tag })),
          ),
        });
      },
    );
    const run1 = logRun({ id: "run-1" });
    const run2 = logRun({ id: "run-2" });
    const rendered = renderChartState(chartSource({ runs: [run1] }));

    act(() =>
      rendered.result.current.actions.onScalarChartVisible("train/loss"),
    );
    await waitFor(() =>
      expect(
        rendered.result.current.content.metricsByGroup.train[0]?.series.map(
          (series) => series.runId,
        ),
      ).toEqual(["run-1"]),
    );

    rendered.rerender({
      currentSource: chartSource({
        runs: [run1, run2],
        tagRecords: [logRunTags("run-1", ["train/loss"])],
        tagsFetching: true,
        tagsRefreshing: true,
      }),
    });
    expect(scalarRequestsFor("train/loss")).toHaveLength(1);
    expect(
      rendered.result.current.content.metricsByGroup.train[0]?.series.map(
        (series) => series.runId,
      ),
    ).toEqual(["run-1"]);

    rendered.rerender({
      currentSource: chartSource({ runs: [run1, run2] }),
    });
    await waitFor(() => expect(scalarRequestsFor("train/loss")).toHaveLength(2));
    expect(rendered.result.current.status.scalarQueryStates.train).toMatchObject({
      isInitialLoading: true,
      isFetching: true,
    });
    expect(
      rendered.result.current.content.metricsByGroup.train[0]?.series.map(
        (series) => series.runId,
      ),
    ).toEqual(["run-1"]);

    act(() => {
      replacement.resolve({
        series: [
          scalarSeries({ runId: "run-1", tag: "train/loss" }),
          scalarSeries({ runId: "run-2", tag: "train/loss" }),
        ],
      });
    });
    await waitFor(() =>
      expect(
        rendered.result.current.content.metricsByGroup.train[0]?.series.map(
          (series) => series.runId,
        ),
      ).toEqual(["run-1", "run-2"]),
    );

    rendered.rerender({
      currentSource: chartSource({
        runs: [run2],
        tagRecords: [
          logRunTags("run-1", ["train/loss"]),
          logRunTags("run-2", ["train/loss"]),
        ],
        tagsFetching: true,
        tagsRefreshing: true,
      }),
    });
    await waitFor(() =>
      expect(
        rendered.result.current.content.metricsByGroup.train[0]?.series.map(
          (series) => series.runId,
        ),
      ).toEqual(["run-2"]),
    );
  });

  it("groups non-empty scalar series by tag", () => {
    const seriesByTag = groupLogScalarSeriesByTag([
      scalarSeries({ runId: "run-1", tag: "train/loss" }),
      scalarSeries({ runId: "run-2", tag: "train/loss" }),
      scalarSeries({ runId: "run-1", tag: "validation/loss", points: [] }),
      scalarSeries({ runId: "run-1", tag: "test/accuracy" }),
    ]);

    expect(seriesByTag.get("train/loss")?.map((series) => series.runId)).toEqual([
      "run-1",
      "run-2",
    ]);
    expect(seriesByTag.has("validation/loss")).toBe(false);
    expect(seriesByTag.get("test/accuracy")?.map((series) => series.runId)).toEqual([
      "run-1",
    ]);
  });

  it("groups selected log metrics even before scalar series load", () => {
    const seriesByTag = groupLogScalarSeriesByTag([
      scalarSeries({ runId: "run-1", tag: "train/loss" }),
    ]);

    const groups = groupSelectedLogMetrics({
      selectedTagList: ["train/loss", "validation/accuracy", "custom/metric"],
      seriesByTag,
    });

    expect(groups.train).toEqual([
      { tag: "train/loss", series: seriesByTag.get("train/loss") },
    ]);
    expect(groups.validation).toEqual([
      { tag: "validation/accuracy", series: [] },
    ]);
    expect(groups.other).toEqual([{ tag: "custom/metric", series: [] }]);
  });

  it("defaults to the core epoch scalar tags without diagnostic metrics", () => {
    expect(isDefaultScalarTag("validation/accuracy_epoch")).toBe(true);
    expect(isDefaultScalarTag("validation/loss_epoch")).toBe(true);
    expect(isDefaultScalarTag("train/loss_epoch")).toBe(true);
    expect(isDefaultScalarTag("train/accuracy_epoch")).toBe(true);
    expect(isDefaultScalarTag("train/loss")).toBe(false);
    expect(isDefaultScalarTag("train/accuracy")).toBe(false);
    expect(isDefaultScalarTag("gap/accuracy")).toBe(false);
    expect(isDefaultScalarTag("gap/loss")).toBe(false);
    expect(isDefaultScalarTag("best_validation/accuracy")).toBe(false);
    expect(isDefaultScalarTag("gradients/global_norm")).toBe(false);
    expect(isDefaultScalarTag("validation/confidence/mean")).toBe(false);
    expect(isDefaultScalarTag("validation/per_class/class_3/f1")).toBe(false);
    expect(
      isDefaultScalarTag(
        "validation/confusion_matrix/true_class_3/predicted_class_5/rate",
      ),
    ).toBe(false);
    expect(
      isDefaultScalarTag(
        "validation/confusion_matrix/true_class_3/predicted_class_5/count",
      ),
    ).toBe(false);
    expect(isDefaultScalarTag("main_model.layers.0.model/weights/mean")).toBe(false);
  });

  it("limits accordion plot selectors to compact train and validation metrics", () => {
    expect(isLogPlotSelectorScalarTag("validation/accuracy")).toBe(true);
    expect(isLogPlotSelectorScalarTag("validation/f1_score")).toBe(true);
    expect(isLogPlotSelectorScalarTag("train/f1_score")).toBe(true);
    expect(isLogPlotSelectorScalarTag("validation/per_class/class_3/f1_score"))
      .toBe(false);
    expect(isLogPlotSelectorScalarTag("validation/kaggle_auc")).toBe(false);
    expect(isLogPlotSelectorScalarTag("gap/accuracy")).toBe(false);

    expect(
      groupLogPlotSelectorTags([
        "validation/loss",
        "validation/per_class/class_3/f1_score",
        "validation/f1_score",
        "train/accuracy",
        "train/kaggle_logloss",
        "gap/accuracy",
      ]),
    ).toEqual({
      train: ["train/accuracy"],
      validation: ["validation/loss", "validation/f1_score"],
      test: [],
      other: [],
    });
  });

  it("builds complete train-validation scalar pairs by suffix", () => {
    const pairs = buildTrainValidationScalarPairs([
      { value: "train/loss_epoch" },
      { value: "validation/loss_epoch" },
      { value: "train/accuracy_epoch" },
      { value: "validation/accuracy_epoch" },
      { value: "train/only" },
      { value: "validation/other_only" },
      { value: "test/accuracy" },
    ]);

    expect(pairs).toEqual([
      {
        suffix: "loss_epoch",
        trainTag: "train/loss_epoch",
        validationTag: "validation/loss_epoch",
      },
      {
        suffix: "accuracy_epoch",
        trainTag: "train/accuracy_epoch",
        validationTag: "validation/accuracy_epoch",
      },
    ]);
  });

  it("defaults train-validation pairs only when both tags are default scalars", () => {
    const pairs = buildTrainValidationScalarPairs([
      { value: "train/loss" },
      { value: "validation/loss" },
      { value: "train/loss_epoch" },
      { value: "validation/loss_epoch" },
      { value: "train/accuracy_epoch" },
      { value: "validation/accuracy_epoch" },
      { value: "train/confidence/mean" },
      { value: "validation/confidence/mean" },
    ]);

    expect(defaultTrainValidationScalarPairSuffixes(pairs)).toEqual([
      "loss_epoch",
      "accuracy_epoch",
    ]);
  });

  it("splits confusion-matrix tags out of normal scalar tag options", () => {
    const confusionRateTag =
      "validation/confusion_matrix/true_class_0/predicted_class_1/rate";
    const confusionCountTag =
      "validation/confusion_matrix/true_class_0/predicted_class_1/count";

    const tagOptions = buildLogScalarTagOptions([
      {
        scalarTags: ["validation/accuracy", confusionRateTag, confusionCountTag],
      },
      {
        scalarTags: ["train/loss", confusionRateTag],
      },
    ]);

    expect(tagOptions.tagOptions.map((option) => option.value)).toEqual([
      "train/loss",
      "validation/accuracy",
    ]);
    expect(tagOptions.confusionMatrixRateTags).toEqual([confusionRateTag]);
  });

  it("prefers validation accuracy for the best-run default metric", () => {
    expect(
      defaultLogBestRunMetricTag([
        { value: "train/loss", label: "train/loss", count: 2 },
        { value: "test/accuracy", label: "test/accuracy", count: 2 },
        { value: "validation/accuracy", label: "validation/accuracy", count: 2 },
      ]),
    ).toBe("validation/accuracy");
    expect(
      defaultLogBestRunMetricTag([
        { value: "test/accuracy", label: "test/accuracy", count: 2 },
        { value: "train/accuracy", label: "train/accuracy", count: 2 },
      ]),
    ).toBe("test/accuracy");
    expect(defaultLogBestRunMetricTag([{ value: "custom", label: "custom" }]))
      .toBe("custom");
    expect(defaultLogBestRunMetricTag([])).toBeNull();
  });

  it("builds heatmaps from confusion-matrix rates", () => {
    const confusionRateTag =
      "validation/confusion_matrix/true_class_0/predicted_class_1/rate";
    const confusionCountTag =
      "validation/confusion_matrix/true_class_0/predicted_class_1/count";
    const seriesByTag = groupLogScalarSeriesByTag([
      scalarSeries({ runId: "run-1", tag: "train/loss" }),
      scalarSeries({
        runId: "run-1",
        tag: confusionRateTag,
        points: [{ step: 2, wallTime: 200, value: 0.42 }],
      }),
      scalarSeries({
        runId: "run-1",
        tag: confusionCountTag,
        points: [{ step: 2, wallTime: 200, value: 5 }],
      }),
    ]);
    const heatmaps = buildConfusionMatrixHeatmaps({
      matrixTagList: [confusionRateTag],
      seriesByTag,
      runsById: new Map([["run-1", logRun({ id: "run-1" })]]),
      runOrder: ["run-1"],
    });

    expect(heatmaps).toHaveLength(1);
    expect(heatmaps[0]).toMatchObject({
      split: "validation",
      runId: "run-1",
      classCount: 2,
      cells: [{ trueClass: 0, predictedClass: 1, value: 0.42 }],
    });
  });

  it("derives empty states for selected tag and scalar point gaps", () => {
    expect(
      deriveLogsChartEmptyState({
        expandedSelectedTagCount: 0,
        confusionMatrixTagCount: 0,
        hasEventFiles: true,
        runsLoading: false,
        scalarLoading: false,
        selectedSeriesCount: 0,
        selectedTagCount: 0,
        tagOptionCount: 2,
        tagsLoading: false,
        tagsRefreshing: false,
        visibleRunCount: 1,
      }),
    ).toMatchObject({ title: "No scalar tags selected", busy: false });

    expect(
      deriveLogsChartEmptyState({
        expandedSelectedTagCount: 1,
        confusionMatrixTagCount: 0,
        hasEventFiles: true,
        runsLoading: false,
        scalarLoading: false,
        selectedSeriesCount: 0,
        selectedTagCount: 1,
        tagOptionCount: 2,
        tagsLoading: false,
        tagsRefreshing: false,
        visibleRunCount: 1,
      }),
    ).toBeNull();

    expect(
      deriveLogsChartEmptyState({
        expandedSelectedTagCount: 1,
        confusionMatrixTagCount: 0,
        hasEventFiles: true,
        runsLoading: false,
        scalarLoading: true,
        selectedSeriesCount: 0,
        selectedTagCount: 1,
        tagOptionCount: 2,
        tagsLoading: false,
        tagsRefreshing: true,
        visibleRunCount: 1,
      }),
    ).toMatchObject({ title: "Refreshing TensorBoard tags…", busy: true });

    expect(
      deriveLogsChartEmptyState({
        expandedSelectedTagCount: 1,
        confusionMatrixTagCount: 0,
        hasEventFiles: true,
        runsLoading: false,
        scalarLoading: true,
        selectedSeriesCount: 0,
        selectedTagCount: 1,
        tagOptionCount: 2,
        tagsLoading: false,
        tagsRefreshing: false,
        visibleRunCount: 1,
      }),
    ).toBeNull();
  });

  it("allows the confusion matrix accordion when no normal scalar tags are selected", () => {
    expect(
      deriveLogsChartEmptyState({
        expandedSelectedTagCount: 0,
        confusionMatrixTagCount: 4,
        hasEventFiles: true,
        runsLoading: false,
        scalarLoading: false,
        selectedSeriesCount: 0,
        selectedTagCount: 0,
        tagOptionCount: 0,
        tagsLoading: false,
        tagsRefreshing: false,
        visibleRunCount: 1,
      }),
    ).toBeNull();
  });

  it("does not show a global scalar loading state when loaded chart data remains visible", () => {
    expect(
      deriveLogsChartEmptyState({
        expandedSelectedTagCount: 2,
        confusionMatrixTagCount: 0,
        hasEventFiles: true,
        runsLoading: false,
        scalarLoading: true,
        selectedSeriesCount: 1,
        selectedTagCount: 2,
        tagOptionCount: 2,
        tagsLoading: false,
        tagsRefreshing: false,
        visibleRunCount: 1,
      }),
    ).toBeNull();
  });

  it("does not return an empty state when selected scalar series are renderable", () => {
    expect(
      deriveLogsChartEmptyState({
        expandedSelectedTagCount: 1,
        confusionMatrixTagCount: 0,
        hasEventFiles: true,
        runsLoading: false,
        scalarLoading: false,
        selectedSeriesCount: 1,
        selectedTagCount: 1,
        tagOptionCount: 2,
        tagsLoading: false,
        tagsRefreshing: false,
        visibleRunCount: 1,
      }),
    ).toBeNull();
  });

});
