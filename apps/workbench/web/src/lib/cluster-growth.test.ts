import { describe, expect, it } from "vitest";
import { buildClusterGrowth } from "@/lib/cluster-growth";
import type { TrainingJob, TrainingProgressEvent } from "@/lib/api/training-jobs";

function job(events: TrainingProgressEvent[]): TrainingJob {
  return {
    id: "job-1",
    status: "running",
    modelType: "neuron",
    model: "linear",
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
    eventCount: events.length,
    eventCounts: {},
    eventsTruncated: false,
    clusterGrowth: [],
    logTail: [],
    logTailTruncated: false,
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
          coordinates: [[1, 1, 1]],
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

  it("counts summarized same-step neuron growth bursts", () => {
    const summaries = buildClusterGrowth(
      job([
        {
          type: "neurons_added",
          node: "neuron_cluster",
          coordinates: [
            [2, 1, 1],
            [3, 1, 1],
          ],
          coordinateCount: 125,
          coordinatesTruncated: true,
          count: 126,
          capacity: [200, 1, 1],
          step: 320,
          epoch: 2,
        },
      ]),
    );

    expect(summaries).toHaveLength(1);
    expect(summaries[0]).toMatchObject({
      node: "neuron_cluster",
      count: 126,
      capacityTotal: 200,
      additionCount: 125,
      additions: [
        { coord: [2, 1, 1], step: 320, epoch: 2 },
        { coord: [3, 1, 1], step: 320, epoch: 2 },
      ],
    });
  });

  it("uses server-projected cluster growth before scanning event history", () => {
    const clusterGrowth: TrainingJob["clusterGrowth"] = [
      {
        node: "root.cluster",
        count: 12,
        capacityTotal: 27,
        additionCount: 10,
        additions: [{ coord: [1, 2, 3], step: 40, epoch: 2 }],
      },
    ];
    const withSummary = {
      ...job([
        {
          type: "cluster_initialized",
          node: "ignored",
          count: 1,
          capacity: [1],
          coordinates: [],
        },
      ]),
      clusterGrowth,
    };

    expect(buildClusterGrowth(withSummary)).toEqual(clusterGrowth);
  });
});
