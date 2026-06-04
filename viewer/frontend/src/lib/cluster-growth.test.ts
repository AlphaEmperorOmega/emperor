import { describe, expect, it } from "vitest";
import { buildClusterGrowth } from "@/lib/cluster-growth";
import { type TrainingJob } from "@/lib/api";

function job(events: Array<Record<string, unknown>>): TrainingJob {
  return {
    id: "job-1",
    status: "running",
    model: "neuron_linear",
    preset: "baseline",
    datasets: ["Mnist"],
    overrides: {},
    monitors: [],
    logFolder: "test_model",
    createdAt: "",
    updatedAt: "",
    exitCode: null,
    pid: 1,
    currentDataset: "Mnist",
    epoch: 0,
    step: 0,
    metrics: {},
    logDir: null,
    events,
    logTail: [],
    resultLinks: [],
  };
}

describe("buildClusterGrowth", () => {
  it("tracks count, capacity, and additions per cluster node", () => {
    const summaries = buildClusterGrowth(
      job([
        {
          type: "cluster_initialized",
          node: "neuron_cluster",
          count: 1,
          capacity: [3, 1, 1],
        },
        { type: "step", metrics: { loss: 0.5 } },
        {
          type: "neuron_added",
          node: "neuron_cluster",
          coord: [2, 1, 1],
          count: 2,
          capacity: [3, 1, 1],
          step: 140,
        },
        {
          type: "neuron_added",
          node: "neuron_cluster",
          coord: [3, 1, 1],
          count: 3,
          capacity: [3, 1, 1],
          step: 280,
        },
      ]),
    );

    expect(summaries).toHaveLength(1);
    const summary = summaries[0];
    expect(summary.node).toBe("neuron_cluster");
    expect(summary.count).toBe(3);
    expect(summary.capacityTotal).toBe(3);
    expect(summary.additions).toEqual([
      { coord: [2, 1, 1], step: 140, epoch: null },
      { coord: [3, 1, 1], step: 280, epoch: null },
    ]);
  });

  it("returns an empty list when there are no cluster events", () => {
    expect(buildClusterGrowth(job([{ type: "step", metrics: {} }]))).toEqual([]);
    expect(buildClusterGrowth(undefined)).toEqual([]);
  });
});
