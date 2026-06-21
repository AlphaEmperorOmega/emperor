import { describe, expect, it } from "vitest";
import {
  deriveDatasetSelectionState,
  deriveMonitorSource,
  deriveParameterActivityByNodePath,
} from "@/features/viewer/state/graph-monitor/graph-monitor-selectors";
import {
  type GraphNode,
  type InspectResponse,
  type LogRun,
  type LogRunTags,
  type ParameterStatus,
  type TrainingJob,
} from "@/lib/api";

function run(overrides: Partial<LogRun> & Pick<LogRun, "id">): LogRun {
  return {
    id: overrides.id,
    group: overrides.group ?? overrides.experiment ?? "exp_a",
    experiment: overrides.experiment ?? "exp_a",
    modelType: overrides.modelType ?? "linears",
    model: overrides.model ?? "linear",
    preset: overrides.preset ?? "baseline",
    dataset: overrides.dataset ?? "Mnist",
    runName: overrides.runName ?? `${overrides.id}_20260601_010203`,
    timestamp: overrides.timestamp ?? "2026-06-01 01:02:03",
    version: overrides.version ?? "version_0",
    relativePath:
      overrides.relativePath ?? "exp_a/linear/baseline/Mnist/run/version_0",
    hasResult: overrides.hasResult ?? false,
    eventFileCount: overrides.eventFileCount ?? 1,
    checkpointCount: overrides.checkpointCount ?? 0,
    hasHparams: overrides.hasHparams ?? true,
    metrics: overrides.metrics ?? {},
  };
}

function layerTags(runId: string): LogRunTags {
  return {
    runId,
    scalarTags: ["main_model.0.model/weights/mean"],
    histogramTags: [],
    imageTags: [],
    textTags: [],
  };
}

function performanceTags(runId: string): LogRunTags {
  return {
    runId,
    scalarTags: ["epoch", "train/loss", "test/accuracy"],
    histogramTags: [],
    imageTags: [],
    textTags: [],
  };
}

function node(
  id: string,
  path: string,
  typeName: string,
  overrides: Partial<GraphNode> = {},
): GraphNode {
  return {
    id,
    label: id,
    typeName,
    path,
    graphRole: overrides.graphRole ?? "architecture",
    parameterCount: overrides.parameterCount ?? 0,
    parameterSizeBytes: overrides.parameterSizeBytes ?? (overrides.parameterCount ?? 0) * 4,
    details: overrides.details ?? {},
    config: overrides.config ?? null,
  };
}

function monitorGraph(): InspectResponse {
  const root = node("root", "main_model", "LayerStack");
  const layer0 = node("layer-0", "main_model.0", "Layer");
  const linear0 = node("linear-0", "main_model.0.model", "LinearLayer", {
    details: { weightShape: "2 x 2", biasShape: "2" },
  });
  const layer1 = node("layer-1", "main_model.1", "Layer");
  const linear1 = node("linear-1", "main_model.1.model", "LinearLayer", {
    details: { weightShape: "2 x 2", biasShape: "2" },
  });

  return {
    modelType: "linears",
    model: "linear",
    preset: "baseline",
    parameterCount: 0,
    parameterSizeBytes: 0,
    nodes: [root, layer0, linear0, layer1, linear1],
    edges: [
      { id: "root-layer-0", source: root.id, target: layer0.id },
      { id: "layer-0-linear-0", source: layer0.id, target: linear0.id },
      { id: "root-layer-1", source: root.id, target: layer1.id },
      { id: "layer-1-linear-1", source: layer1.id, target: linear1.id },
    ],
  };
}

function monitorGraphWithoutBias(): InspectResponse {
  const graph = monitorGraph();
  return {
    ...graph,
    nodes: graph.nodes.map((graphNode) =>
      graphNode.typeName === "LinearLayer"
        ? { ...graphNode, details: { weightShape: "2 x 2" } }
        : graphNode,
    ),
  };
}

function trainingJob(overrides: Partial<TrainingJob> = {}): TrainingJob {
  return {
    id: overrides.id ?? "job-1",
    status: overrides.status ?? "running",
    modelType: overrides.modelType ?? "linears",
    model: overrides.model ?? "linear",
    preset: overrides.preset ?? "baseline",
    presets: overrides.presets ?? ["baseline"],
    datasets: overrides.datasets ?? ["Mnist"],
    overrides: overrides.overrides ?? {},
    search: overrides.search ?? null,
    plannedRunCount: overrides.plannedRunCount ?? 1,
    runPlan: overrides.runPlan ?? null,
    monitors: overrides.monitors ?? [],
    logFolder: overrides.logFolder ?? "runs",
    createdAt: overrides.createdAt ?? "2026-06-01T00:00:00.000Z",
    updatedAt: overrides.updatedAt ?? "2026-06-01T00:00:00.000Z",
    exitCode: overrides.exitCode ?? null,
    pid: overrides.pid ?? 1,
    currentPreset: overrides.currentPreset ?? "baseline",
    currentDataset: overrides.currentDataset ?? "Mnist",
    epoch: overrides.epoch ?? 1,
    step: overrides.step ?? 10,
    metrics: overrides.metrics ?? {},
    logDir: overrides.logDir ?? "runs/job-1",
    events: overrides.events ?? [],
    eventCount: overrides.eventCount ?? 0,
    eventCounts: overrides.eventCounts ?? {},
    eventsTruncated: overrides.eventsTruncated ?? false,
    clusterGrowth: overrides.clusterGrowth ?? [],
    logTail: overrides.logTail ?? [],
    resultLinks: overrides.resultLinks ?? [],
  };
}

function parameterStatus(
  overrides: Partial<ParameterStatus> & Pick<ParameterStatus, "sourceId">,
): ParameterStatus {
  return {
    sourceId: overrides.sourceId,
    preset: overrides.preset ?? "baseline",
    dataset: overrides.dataset ?? "Mnist",
    logDir: overrides.logDir ?? "logs/run",
    nodes:
      overrides.nodes ?? [
        {
          nodePath: "main_model.0.model",
          weights: {
            status: "updated",
            metric: "main_model.0.model/weights/relative_delta_norm",
            lastStep: 12,
            observedPoints: 2,
          },
          bias: {
            status: "unchanged",
            metric: "main_model.0.model/bias/delta_norm",
            lastStep: 12,
            observedPoints: 1,
          },
        },
      ],
  };
}

describe("viewer state selectors", () => {
  it("derives preset options, visible runs, and the selected run's monitor group", () => {
    const state = deriveDatasetSelectionState({
      logRuns: [
        run({
          id: "old-mnist",
          experiment: "exp_a",
          dataset: "Mnist",
          preset: "baseline",
          timestamp: "2026-06-01 01:00:00",
        }),
        run({
          id: "new-mnist",
          experiment: "exp_a",
          dataset: "Mnist",
          preset: "baseline",
          timestamp: "2026-06-03 01:00:00",
        }),
        run({
          id: "fashion",
          experiment: "exp_a",
          dataset: "FashionMnist",
          preset: "fast",
          timestamp: "2026-06-02 12:00:00",
        }),
        run({
          id: "fast-mnist",
          experiment: "exp_a",
          dataset: "Mnist",
          preset: "fast",
          timestamp: "2026-06-02 06:00:00",
        }),
        run({
          id: "other-experiment",
          experiment: "exp_b",
          dataset: "Cifar10",
          preset: "baseline",
          timestamp: "2026-06-02 01:00:00",
        }),
        run({ id: "other-model", model: "bert", timestamp: "2026-06-04 01:00:00" }),
      ],
      modelRunTags: [
        layerTags("old-mnist"),
        layerTags("new-mnist"),
        layerTags("fashion"),
        layerTags("fast-mnist"),
        layerTags("other-experiment"),
      ],
      selectedModel: "linear",
      selectedHistoricalPreset: "",
      selectedLogRunId: "old-mnist",
    });

    expect(state.modelLogRuns.map((item) => item.id)).toEqual([
      "new-mnist",
      "fashion",
      "fast-mnist",
      "other-experiment",
      "old-mnist",
    ]);
    expect(state.historicalPresetOptions).toEqual([
      { value: "baseline", label: "baseline", count: 3 },
      { value: "fast", label: "fast", count: 2 },
    ]);
    expect(state.visibleHistoricalRuns.map((item) => item.id)).toEqual([
      "new-mnist",
      "fashion",
      "fast-mnist",
      "other-experiment",
      "old-mnist",
    ]);
    // experiment/dataset are derived from the selected run, not separate filters.
    expect(state.selectedHistoricalExperiment).toBe("exp_a");
    expect(state.selectedHistoricalDataset).toBe("Mnist");
    expect(state.selectedHistoricalRunPreset).toBe("baseline");
    expect(state.filteredHistoricalRuns.map((item) => item.id)).toEqual([
      "new-mnist",
      "old-mnist",
    ]);
    expect(state.historicalMonitorRuns.map((item) => item.id)).toEqual([
      "new-mnist",
      "old-mnist",
    ]);
    expect(state.filteredHistoricalRunIds).toEqual(["new-mnist", "old-mnist"]);
    expect(state.selectedLogRun?.id).toBe("old-mnist");
  });

  it("narrows visible runs by preset and clears monitor data when nothing is selected", () => {
    const state = deriveDatasetSelectionState({
      logRuns: [
        run({ id: "baseline-run", experiment: "exp_a", dataset: "Mnist", preset: "baseline" }),
        run({ id: "fast-run", experiment: "exp_a", dataset: "Mnist", preset: "fast" }),
      ],
      modelRunTags: [layerTags("baseline-run"), layerTags("fast-run")],
      selectedModel: "linear",
      selectedHistoricalPreset: "fast",
      selectedLogRunId: null,
    });

    expect(state.visibleHistoricalRuns.map((item) => item.id)).toEqual(["fast-run"]);
    expect(state.selectedLogRun).toBeUndefined();
    expect(state.selectedHistoricalExperiment).toBe("");
    expect(state.selectedHistoricalDataset).toBe("");
    expect(state.selectedHistoricalRunPreset).toBe("");
    expect(state.filteredHistoricalRuns).toEqual([]);
    expect(state.historicalMonitorRuns).toEqual([]);
    expect(state.filteredHistoricalRunIds).toEqual([]);
  });

  it("keeps only presets/runs from experiments with per-layer monitor data", () => {
    const state = deriveDatasetSelectionState({
      logRuns: [
        run({ id: "layer-run", experiment: "exp_a", dataset: "Mnist", preset: "baseline" }),
        run({ id: "perf-run", experiment: "exp_b", dataset: "Mnist", preset: "fast" }),
      ],
      modelRunTags: [layerTags("layer-run"), performanceTags("perf-run")],
      selectedModel: "linear",
      selectedHistoricalPreset: "",
      selectedLogRunId: null,
    });

    expect(state.historicalPresetOptions).toEqual([
      { value: "baseline", label: "baseline", count: 1 },
    ]);
    expect(state.visibleHistoricalRuns.map((item) => item.id)).toEqual([
      "layer-run",
    ]);
  });

  it("shows no presets or runs until model run tags have loaded", () => {
    const state = deriveDatasetSelectionState({
      logRuns: [run({ id: "layer-run", experiment: "exp_a", dataset: "Mnist" })],
      modelRunTags: undefined,
      selectedModel: "linear",
      selectedHistoricalPreset: "",
      selectedLogRunId: null,
    });

    expect(state.modelLogRuns.map((item) => item.id)).toEqual(["layer-run"]);
    expect(state.historicalPresetOptions).toEqual([]);
    expect(state.visibleHistoricalRuns).toEqual([]);
  });

  it("keeps the selected model run visible while monitor tags are loading", () => {
    const state = deriveDatasetSelectionState({
      logRuns: [
        run({
          id: "selected-run",
          experiment: "exp_a",
          dataset: "Mnist",
          preset: "baseline",
        }),
      ],
      modelRunTags: undefined,
      selectedModel: "linear",
      selectedHistoricalPreset: "",
      selectedLogRunId: "selected-run",
    });

    expect(state.historicalPresetOptions).toEqual([]);
    expect(state.visibleHistoricalRuns.map((item) => item.id)).toEqual([
      "selected-run",
    ]);
    expect(state.selectedLogRun?.id).toBe("selected-run");
    expect(state.selectedHistoricalExperiment).toBe("exp_a");
    expect(state.selectedHistoricalDataset).toBe("Mnist");
    expect(state.selectedHistoricalRunPreset).toBe("baseline");
    expect(state.filteredHistoricalRuns).toEqual([]);
    expect(state.filteredHistoricalRunIds).toEqual([]);
  });

  it("derives graph monitor targets, comparison groups, and tag availability", () => {
    const graph = monitorGraph();
    const layer0 = graph.nodes.find((item) => item.id === "layer-0");
    const linear0 = graph.nodes.find((item) => item.id === "linear-0");
    const linear1 = graph.nodes.find((item) => item.id === "linear-1");
    const logRunTags: LogRunTags[] = [
      {
        runId: "new-mnist",
        scalarTags: ["main_model.0.model/weights/mean"],
        histogramTags: [],
        imageTags: [],
        textTags: [],
      },
    ];

    const state = deriveMonitorSource({
      graph,
      selectedNode: layer0,
      graphMonitorNode: linear0,
      activeTrainingJob: undefined,
      historicalMonitorRuns: [run({ id: "new-mnist" })],
      selectedHistoricalExperiment: "exp_a",
      selectedHistoricalDataset: "Mnist",
      selectedHistoricalPreset: "baseline",
      logRunTags,
      filteredHistoricalRunIds: ["new-mnist"],
    });

    expect(state.activeJobHasMonitorSource).toBe(false);
    expect(state.selectedMonitorNode?.id).toBe("linear-0");
    expect(state.selectedLogRunHasMonitorTags).toBe(true);
    expect(state.selectedMonitorComparisonCandidateGroups["same-stack"]).toEqual([
      linear1,
    ]);
    expect(state.graphMonitorComparisonCandidateGroups["same-stack"]).toEqual([
      linear1,
    ]);
  });

  it("prefers active linear jobs over historical monitor groups", () => {
    const activeJob = trainingJob({ monitors: ["linear"] });
    const historicalRun = run({ id: "new-mnist" });

    const activeState = deriveMonitorSource({
      activeTrainingJob: activeJob,
      historicalMonitorRuns: [historicalRun],
      selectedHistoricalExperiment: "exp_a",
      selectedHistoricalDataset: "Mnist",
      selectedHistoricalPreset: "baseline",
    });

    expect(activeState.graphMonitorSource).toEqual({
      kind: "active-job",
      job: activeJob,
    });

    const historicalState = deriveMonitorSource({
      activeTrainingJob: trainingJob({ monitors: ["loss"] }),
      historicalMonitorRuns: [historicalRun],
      selectedHistoricalExperiment: "exp_a",
      selectedHistoricalDataset: "Mnist",
      selectedHistoricalPreset: "baseline",
    });

    expect(historicalState.graphMonitorSource).toEqual({
      kind: "historical-run-group",
      runs: [historicalRun],
      experiment: "exp_a",
      dataset: "Mnist",
      preset: "baseline",
    });
  });

  it("uses the resolved monitor name for active-job chart source selection", () => {
    const attention = node("attention-0", "encoder.0.attention", "SelfAttention");
    const graph: InspectResponse = {
      modelType: "transformer_encoder",
      model: "bert",
      preset: "baseline",
      parameterCount: 0,
      parameterSizeBytes: 0,
      nodes: [node("root", "encoder", "LayerStack"), attention],
      edges: [{ id: "root-attention", source: "root", target: attention.id }],
    };
    const activeJob = trainingJob({ monitors: ["attention"] });

    const state = deriveMonitorSource({
      graph,
      selectedNode: attention,
      graphMonitorNode: attention,
      activeTrainingJob: activeJob,
      historicalMonitorRuns: [run({ id: "historical" })],
    });

    expect(state.selectedMonitorNode).toBe(attention);
    expect(state.activeJobHasMonitorSource).toBe(true);
    expect(state.graphMonitorSource).toEqual({
      kind: "active-job",
      job: activeJob,
    });

    const fallbackState = deriveMonitorSource({
      graph,
      graphMonitorNode: attention,
      activeTrainingJob: trainingJob({ monitors: ["linear"] }),
      historicalMonitorRuns: [run({ id: "historical" })],
      selectedHistoricalExperiment: "exp_a",
      selectedHistoricalDataset: "Mnist",
      selectedHistoricalPreset: "baseline",
    });

    expect(fallbackState.graphMonitorSource).toEqual({
      kind: "historical-run-group",
      runs: [run({ id: "historical" })],
      experiment: "exp_a",
      dataset: "Mnist",
      preset: "baseline",
    });
  });

  it("maps active-job parameter status through linear monitor targets", () => {
    const graph = monitorGraph();
    const activeJob = trainingJob({ monitors: ["linear"] });
    const source = deriveMonitorSource({
      graph,
      activeTrainingJob: activeJob,
      historicalMonitorRuns: [run({ id: "historical" })],
    }).graphMonitorSource;

    const activityByPath = deriveParameterActivityByNodePath({
      graph,
      source,
      status: parameterStatus({ sourceId: activeJob.id }),
    });

    const activity = activityByPath?.get("main_model.0.model");
    expect(activity?.weights.status).toBe("updated");
    expect(activity?.weights.source).toBe("active-job");
    expect(activity?.weights.sourceLabel).toBe("active job job-1");
    expect(activity?.weights.metric).toBe(
      "main_model.0.model/weights/relative_delta_norm",
    );
    expect(activity?.bias?.status).toBe("unchanged");
  });

  it("aggregates historical parameter status with mixed amber states", () => {
    const graph = monitorGraph();
    const historicalRuns = [run({ id: "run-updated" }), run({ id: "run-static" })];
    const source = deriveMonitorSource({
      graph,
      activeTrainingJob: undefined,
      historicalMonitorRuns: historicalRuns,
      selectedHistoricalExperiment: "exp_a",
      selectedHistoricalDataset: "Mnist",
      selectedHistoricalPreset: "baseline",
    }).graphMonitorSource;

    const activityByPath = deriveParameterActivityByNodePath({
      graph,
      source,
      status: {
        runs: [
          parameterStatus({ sourceId: "run-updated" }),
          parameterStatus({
            sourceId: "run-static",
            nodes: [
              {
                nodePath: "main_model.0.model",
                weights: {
                  status: "unchanged",
                  metric: "main_model.0.model/weights/delta_norm",
                  lastStep: 10,
                  observedPoints: 1,
                },
                bias: {
                  status: "missing",
                  metric: null,
                  lastStep: null,
                  observedPoints: 0,
                },
              },
            ],
          }),
        ],
      },
    });

    const activity = activityByPath?.get("main_model.0.model");
    expect(activity?.weights.status).toBe("mixed");
    expect(activity?.weights.source).toBe("historical");
    expect(activity?.weights.updatedRuns).toBe(1);
    expect(activity?.weights.unchangedRuns).toBe(1);
    expect(activity?.weights.totalRuns).toBe(2);
    expect(activity?.bias?.status).toBe("unchanged");
    expect(activity?.bias?.missingRuns).toBe(1);
  });

  it("does not create graph bias activity when the inspected target has no bias shape", () => {
    const graph = monitorGraphWithoutBias();
    const activeJob = trainingJob({ monitors: ["linear"] });
    const source = deriveMonitorSource({
      graph,
      activeTrainingJob: activeJob,
    }).graphMonitorSource;

    const activityByPath = deriveParameterActivityByNodePath({
      graph,
      source,
      status: parameterStatus({ sourceId: activeJob.id }),
    });

    const activity = activityByPath?.get("main_model.0.model");
    expect(activity?.weights.status).toBe("updated");
    expect(activity?.bias).toBeUndefined();
  });
});
