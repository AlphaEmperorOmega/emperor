import type { TrainingJob } from "@/lib/api/training-jobs";

export type ClusterGrowthAddition = {
  coord: [number, number, number];
  step: number | null;
  epoch: number | null;
};

export type ClusterGrowthSummary = {
  node: string;
  count: number;
  capacityTotal: number;
  additionCount: number;
  additions: ClusterGrowthAddition[];
};

function toCoordinate(value: unknown): [number, number, number] | null {
  if (
    Array.isArray(value) &&
    value.length === 3 &&
    value.every((item) => typeof item === "number")
  ) {
    return [value[0], value[1], value[2]] as [number, number, number];
  }
  return null;
}

function calculateCapacityTotal(value: unknown): number {
  if (!Array.isArray(value)) {
    return 0;
  }
  return value.reduce(
    (product: number, axis) => product * (typeof axis === "number" ? axis : 1),
    1,
  );
}

function numberOrNull(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}

// Reconstructs neuron cluster growth from training progress events so the UI
// can show when (and where) neurons were added over the course of a run.
export function buildClusterGrowth(
  job: TrainingJob | undefined,
): ClusterGrowthSummary[] {
  if (!job) {
    return [];
  }
  if (job.clusterGrowth.length > 0) {
    return job.clusterGrowth.map((summary) => ({
      node: summary.node,
      count: summary.count,
      capacityTotal: summary.capacityTotal,
      additionCount: summary.additionCount,
      additions: summary.additions,
    }));
  }

  const summariesByNode = new Map<string, ClusterGrowthSummary>();
  const getOrCreateSummary = (node: string): ClusterGrowthSummary => {
    const existingSummary = summariesByNode.get(node);
    if (existingSummary) {
      return existingSummary;
    }
    const newSummary: ClusterGrowthSummary = {
      node,
      count: 0,
      capacityTotal: 0,
      additionCount: 0,
      additions: [],
    };
    summariesByNode.set(node, newSummary);
    return newSummary;
  };

  for (const event of job.events) {
    const node = typeof event.node === "string" ? event.node : null;
    if (!node) {
      continue;
    }

    if (event.type === "cluster_initialized") {
      const summary = getOrCreateSummary(node);
      summary.count = typeof event.count === "number" ? event.count : summary.count;
      summary.capacityTotal =
        calculateCapacityTotal(event.capacity) || summary.capacityTotal;
    } else if (event.type === "neuron_added" || event.type === "neurons_added") {
      const summary = getOrCreateSummary(node);
      if (typeof event.count === "number") {
        summary.count = event.count;
      }
      const eventCapacityTotal = calculateCapacityTotal(event.capacity);
      if (eventCapacityTotal) {
        summary.capacityTotal = eventCapacityTotal;
      }
      if (event.type === "neurons_added") {
        const coordinateCount =
          typeof event.coordinateCount === "number" ? event.coordinateCount : 0;
        const coordinates = Array.isArray(event.coordinates)
          ? event.coordinates
          : [];
        summary.additionCount += Math.max(0, coordinateCount);
        for (const coordinate of coordinates.slice(-50)) {
          const parsedCoordinate = toCoordinate(coordinate);
          if (parsedCoordinate) {
            summary.additions.push({
              coord: parsedCoordinate,
              step: numberOrNull(event.step),
              epoch: numberOrNull(event.epoch),
            });
          }
        }
        continue;
      }
      const coordinate = toCoordinate(event.coord);
      if (coordinate) {
        summary.additionCount += 1;
        summary.additions.push({
          coord: coordinate,
          step: numberOrNull(event.step),
          epoch: numberOrNull(event.epoch),
        });
      }
    }
  }

  return [...summariesByNode.values()];
}
