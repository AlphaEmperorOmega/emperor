import { describe, expect, it } from "vitest";
import {
  expectedLinearParameterChannels,
  summarizeHistoricalParameterStatus,
} from "@/lib/parameter-summary";
import {
  type GraphNode,
  type InspectResponse,
  type LogRun,
  type ParameterStatus,
} from "@/lib/api";

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
    relativePath: overrides.relativePath ?? "exp_a/linear/baseline/Mnist/run/version_0",
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

function graph({ withBias = true }: { withBias?: boolean } = {}): InspectResponse {
  const root = node("root", "main_model", "LayerStack");
  const layers = Array.from({ length: 4 }, (_, index) => {
    const wrapper = node(`layer-${index}`, `main_model.${index}`, "Layer");
    const linear = node(`linear-${index}`, `main_model.${index}.model`, "LinearLayer", {
      details: withBias
        ? { weightShape: "2 x 2", biasShape: "2" }
        : { weightShape: "2 x 2" },
    });
    return { wrapper, linear };
  });

  return {
    model: "linear",
    preset: "baseline",
    parameterCount: 0,
    nodes: [root, ...layers.flatMap((layer) => [layer.wrapper, layer.linear])],
    edges: layers.flatMap((layer, index) => [
      { id: `root-layer-${index}`, source: root.id, target: layer.wrapper.id },
      {
        id: `layer-${index}-linear-${index}`,
        source: layer.wrapper.id,
        target: layer.linear.id,
      },
    ]),
  };
}

function parameterStatus(
  sourceId: string,
  overrides: Partial<ParameterStatus> = {},
): ParameterStatus {
  return {
    sourceId,
    preset: overrides.preset ?? "baseline",
    dataset: overrides.dataset ?? "Mnist",
    logDir: overrides.logDir ?? "logs/run",
    nodes:
      overrides.nodes ??
      Array.from({ length: 4 }, (_, index) => ({
        nodePath: `main_model.${index}.model`,
        weights: {
          status: "updated" as const,
          metric: `main_model.${index}.model/weights/relative_delta_norm`,
          lastStep: 12,
          observedPoints: 2,
        },
        bias: {
          status: "updated" as const,
          metric: `main_model.${index}.model/bias/relative_delta_norm`,
          lastStep: 12,
          observedPoints: 2,
        },
      })),
  };
}

describe("parameter summaries", () => {
  it("counts four layers with weights and bias as eight updated parameters", () => {
    const summary = summarizeHistoricalParameterStatus({
      graph: graph(),
      status: { runs: [parameterStatus("run-1")] },
      runs: [run({ id: "run-1" })],
    });

    expect(summary.total).toBe(8);
    expect(summary.counts).toEqual({
      updated: 8,
      unchanged: 0,
      mixed: 0,
      notTracked: 0,
    });
    expect(summary.severity).toBe("success");
  });

  it("makes any unchanged expected parameter a danger summary", () => {
    const summary = summarizeHistoricalParameterStatus({
      graph: graph(),
      status: {
        runs: [
          parameterStatus("run-1", {
            nodes: [
              {
                nodePath: "main_model.0.model",
                weights: {
                  status: "unchanged",
                  metric: "main_model.0.model/weights/delta_norm",
                  lastStep: 10,
                  observedPoints: 2,
                },
                bias: {
                  status: "updated",
                  metric: "main_model.0.model/bias/delta_norm",
                  lastStep: 10,
                  observedPoints: 2,
                },
              },
              ...parameterStatus("run-1").nodes.slice(1),
            ],
          }),
        ],
      },
      runs: [run({ id: "run-1" })],
    });

    expect(summary.counts.unchanged).toBe(1);
    expect(summary.severity).toBe("danger");
  });

  it("counts mixed expected parameters across historical runs", () => {
    const summary = summarizeHistoricalParameterStatus({
      graph: graph(),
      status: {
        runs: [
          parameterStatus("run-updated"),
          parameterStatus("run-static", {
            nodes: [
              {
                nodePath: "main_model.0.model",
                weights: {
                  status: "unchanged",
                  metric: "main_model.0.model/weights/delta_norm",
                  lastStep: 10,
                  observedPoints: 2,
                },
                bias: {
                  status: "updated",
                  metric: "main_model.0.model/bias/delta_norm",
                  lastStep: 10,
                  observedPoints: 2,
                },
              },
              ...parameterStatus("run-static").nodes.slice(1),
            ],
          }),
        ],
      },
      runs: [run({ id: "run-updated" }), run({ id: "run-static" })],
    });

    expect(summary.counts.mixed).toBe(1);
    expect(summary.severity).toBe("warning");
  });

  it("combines missing and unknown parameters into not tracked", () => {
    const summary = summarizeHistoricalParameterStatus({
      graph: graph(),
      status: {
        runs: [
          parameterStatus("run-1", {
            nodes: [
              {
                nodePath: "main_model.0.model",
                weights: {
                  status: "missing",
                  metric: null,
                  lastStep: null,
                  observedPoints: 0,
                },
                bias: {
                  status: "unknown",
                  metric: null,
                  lastStep: null,
                  observedPoints: 0,
                },
              },
            ],
          }),
        ],
      },
      runs: [run({ id: "run-1" })],
    });

    expect(summary.counts.notTracked).toBe(8);
    expect(summary.breakdown.missing).toBe(1);
    expect(summary.breakdown.unknown).toBe(7);
    expect(summary.severity).toBe("not-tracked");
  });

  it("excludes bias channels when the inspected graph has no bias shape", () => {
    expect(expectedLinearParameterChannels(graph({ withBias: false }))).toHaveLength(4);

    const summary = summarizeHistoricalParameterStatus({
      graph: graph({ withBias: false }),
      status: { runs: [parameterStatus("run-1")] },
      runs: [run({ id: "run-1" })],
    });

    expect(summary.total).toBe(4);
    expect(summary.counts.updated).toBe(4);
  });
});
