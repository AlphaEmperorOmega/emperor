import { describe, expect, it } from "vitest";
import {
  deriveDatasetSelectionState,
  deriveMonitorSource,
  deriveTargetSelectionState,
} from "@/components/features/viewer/state/viewer-state-selectors";
import {
  type ConfigField,
  type Dataset,
  type GraphNode,
  type InspectResponse,
  type LogRun,
  type LogRunTags,
  type Preset,
  type TrainingJob,
} from "@/lib/api";
import { type ConfigSnapshot } from "@/lib/config-snapshots";

function field(overrides: Partial<ConfigField> & Pick<ConfigField, "key">): ConfigField {
  return {
    key: overrides.key,
    configKey: overrides.configKey ?? overrides.key.toUpperCase(),
    flag: overrides.flag ?? `--${overrides.key.replace(/_/g, "-")}`,
    label: overrides.label ?? overrides.key,
    section: overrides.section ?? "General",
    type: overrides.type ?? "int",
    default: overrides.default ?? 0,
    nullable: overrides.nullable ?? false,
    choices: overrides.choices ?? [],
    locked: overrides.locked ?? false,
    lockedValue: overrides.lockedValue,
    lockedReason: overrides.lockedReason,
  };
}

function dataset(name: string): Dataset {
  return {
    name,
    label: name,
    inputDim: 8,
    outputDim: 2,
  };
}

function preset(name: string, description = ""): Preset {
  return {
    name,
    label: name,
    description,
  };
}

function snapshot(overrides: Partial<ConfigSnapshot> & Pick<ConfigSnapshot, "id">): ConfigSnapshot {
  return {
    id: overrides.id,
    name: overrides.name ?? overrides.id,
    model: overrides.model ?? "linear",
    preset: overrides.preset ?? "baseline",
    overrides: overrides.overrides ?? {},
    createdAt: overrides.createdAt ?? "2026-06-01T00:00:00.000Z",
  };
}

function run(overrides: Partial<LogRun> & Pick<LogRun, "id">): LogRun {
  return {
    id: overrides.id,
    group: overrides.group ?? overrides.experiment ?? "exp_a",
    experiment: overrides.experiment ?? "exp_a",
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
    details: overrides.details ?? {},
    config: overrides.config ?? null,
  };
}

function monitorGraph(): InspectResponse {
  const root = node("root", "main_model", "LayerStack");
  const layer0 = node("layer-0", "main_model.0", "Layer");
  const linear0 = node("linear-0", "main_model.0.model", "LinearLayer");
  const layer1 = node("layer-1", "main_model.1", "Layer");
  const linear1 = node("linear-1", "main_model.1.model", "LinearLayer");

  return {
    model: "linear",
    preset: "baseline",
    parameterCount: 0,
    nodes: [root, layer0, linear0, layer1, linear1],
    edges: [
      { id: "root-layer-0", source: root.id, target: layer0.id },
      { id: "layer-0-linear-0", source: layer0.id, target: linear0.id },
      { id: "root-layer-1", source: root.id, target: layer1.id },
      { id: "layer-1-linear-1", source: layer1.id, target: linear1.id },
    ],
  };
}

function trainingJob(overrides: Partial<TrainingJob> = {}): TrainingJob {
  return {
    id: overrides.id ?? "job-1",
    status: overrides.status ?? "Running",
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
    logTail: overrides.logTail ?? [],
    resultLinks: overrides.resultLinks ?? [],
  };
}

describe("viewer state selectors", () => {
  it("derives target, preset, dataset, and config counts", () => {
    const state = deriveTargetSelectionState({
      datasets: [dataset("Mnist"), dataset("Cifar10")],
      presets: [preset("baseline"), preset("fast", "Fast preset")],
      schemaFields: [
        field({ key: "hidden_dim", section: "Model" }),
        field({
          key: "layer_norm",
          label: "Layer Norm",
          section: "Preset",
          locked: true,
          lockedValue: true,
        }),
        field({ key: "dropout", section: "" }),
      ],
      configSnapshots: [
        snapshot({ id: "fast", preset: "fast" }),
        snapshot({ id: "baseline", preset: "baseline" }),
        snapshot({ id: "other-model", model: "bert", preset: "fast" }),
      ],
      selectedModel: "linear",
      selectedPreset: "fast",
      selectedTrainingPresets: ["fast", "baseline"],
      overrides: { hidden_dim: "128", dropout: "0.1" },
    });

    expect(state.datasetNames).toEqual(["Mnist", "Cifar10"]);
    expect(state.presetNames).toEqual(["baseline", "fast"]);
    expect(state.selectedPresetMeta?.description).toBe("Fast preset");
    expect(state.configSections.map((section) => section.title)).toEqual([
      "Model",
      "Preset",
      "General",
    ]);
    expect(state.configFields.map((configField) => configField.key)).toEqual([
      "hidden_dim",
      "layer_norm",
      "dropout",
    ]);
    expect(state.overrideCount).toBe(2);
    expect(state.presetOwnedFieldCount).toBe(1);
    expect(state.fieldCount).toBe(3);
    expect(state.visibleConfigSnapshots.map((item) => item.id)).toEqual([
      "fast",
      "baseline",
    ]);
    expect(state.configSnapshotGroups).toEqual([
      { preset: "fast", snapshots: [snapshot({ id: "fast", preset: "fast" })] },
      {
        preset: "baseline",
        snapshots: [snapshot({ id: "baseline", preset: "baseline" })],
      },
    ]);
  });

  it("derives historical run options, filters, and selected run", () => {
    const state = deriveDatasetSelectionState({
      logRuns: [
        run({
          id: "old-mnist",
          experiment: "exp_a",
          dataset: "Mnist",
          timestamp: "2026-06-01 01:00:00",
        }),
        run({
          id: "new-mnist",
          experiment: "exp_a",
          dataset: "Mnist",
          timestamp: "2026-06-03 01:00:00",
        }),
        run({
          id: "fashion",
          experiment: "exp_a",
          dataset: "FashionMnist",
          timestamp: "2026-06-02 12:00:00",
        }),
        run({
          id: "other-experiment",
          experiment: "exp_b",
          dataset: "Cifar10",
          timestamp: "2026-06-02 01:00:00",
        }),
        run({ id: "other-model", model: "bert", timestamp: "2026-06-04 01:00:00" }),
      ],
      selectedModel: "linear",
      selectedHistoricalExperiment: "exp_a",
      selectedHistoricalDataset: "Mnist",
      selectedLogRunId: "old-mnist",
    });

    expect(state.modelLogRuns.map((item) => item.id)).toEqual([
      "new-mnist",
      "fashion",
      "other-experiment",
      "old-mnist",
    ]);
    expect(state.historicalExperimentOptions).toEqual([
      { value: "exp_a", label: "exp_a", count: 3 },
      { value: "exp_b", label: "exp_b", count: 1 },
    ]);
    expect(state.historicalDatasetOptions).toEqual([
      { value: "Mnist", label: "Mnist", count: 2 },
      { value: "FashionMnist", label: "FashionMnist", count: 1 },
    ]);
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
    });

    expect(historicalState.graphMonitorSource).toEqual({
      kind: "historical-run-group",
      runs: [historicalRun],
      experiment: "exp_a",
      dataset: "Mnist",
    });
  });
});
