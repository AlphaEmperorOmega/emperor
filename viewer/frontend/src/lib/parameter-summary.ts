import {
  type GraphNode,
  type InspectResponse,
  type LogParameterStatusResponse,
  type LogRun,
  type ParameterActivityStatus,
  type ParameterChannelStatus,
} from "@/lib/api";
import {
  createLinearMonitorTargetResolver,
  type LinearMonitorTargetResolver,
} from "@/lib/graph/monitor-targets";

export type ExpectedParameterChannelName = "weights" | "bias";

export type ExpectedParameterChannel = {
  nodePath: string;
  channel: ExpectedParameterChannelName;
  node: GraphNode;
};

export type ParameterSummarySeverity =
  | "success"
  | "danger"
  | "warning"
  | "not-tracked";

export type ParameterSummaryCounts = {
  updated: number;
  unchanged: number;
  mixed: number;
  notTracked: number;
};

export type ParameterSummaryBreakdown = {
  missing: number;
  unknown: number;
};

export type ParameterSummary = {
  counts: ParameterSummaryCounts;
  breakdown: ParameterSummaryBreakdown;
  total: number;
  severity: ParameterSummarySeverity;
};

export type HistoricalParameterSummaryState = {
  summary?: ParameterSummary;
  isLoading: boolean;
  isError: boolean;
  error?: unknown;
};

function hasShape(value: unknown) {
  if (typeof value === "string") {
    return value.trim().length > 0;
  }
  return (
    Array.isArray(value) &&
    value.every((item) => typeof item === "number" || typeof item === "string")
  );
}

export function expectedLinearParameterChannels(
  graph: InspectResponse | undefined,
  resolver: LinearMonitorTargetResolver = createLinearMonitorTargetResolver(graph),
): ExpectedParameterChannel[] {
  const targets = new Map<string, GraphNode>();
  for (const node of graph?.nodes ?? []) {
    const target = resolver(node);
    if (target) {
      targets.set(target.path, target);
    }
  }

  return [...targets.values()]
    .sort((left, right) => left.path.localeCompare(right.path))
    .flatMap((node) => {
      const channels: ExpectedParameterChannel[] = [
        { nodePath: node.path, channel: "weights", node },
      ];
      if (hasShape(node.details.biasShape)) {
        channels.push({ nodePath: node.path, channel: "bias", node });
      }
      return channels;
    });
}

function statusNodeByPath(status: LogParameterStatusResponse["runs"][number]) {
  return new Map((status.nodes ?? []).map((node) => [node.nodePath, node]));
}

function classifyChannel(statuses: ParameterActivityStatus[]) {
  const updatedRuns = statuses.filter((status) => status === "updated").length;
  const unchangedRuns = statuses.filter((status) => status === "unchanged").length;
  const missingRuns = statuses.filter((status) => status === "missing").length;
  const unknownRuns = statuses.filter((status) => status === "unknown").length;
  const totalRuns = statuses.length;

  if (updatedRuns > 0 && unchangedRuns > 0) {
    return { status: "mixed" as const, missingRuns, unknownRuns };
  }
  if (unchangedRuns > 0) {
    return { status: "unchanged" as const, missingRuns, unknownRuns };
  }
  if (updatedRuns === totalRuns && totalRuns > 0) {
    return { status: "updated" as const, missingRuns, unknownRuns };
  }
  if (updatedRuns > 0) {
    return { status: "mixed" as const, missingRuns, unknownRuns };
  }
  return { status: "notTracked" as const, missingRuns, unknownRuns };
}

function channelStatus(
  channel: ExpectedParameterChannelName,
  status: ParameterChannelStatus | undefined,
) {
  if (!status) {
    return "unknown" as const;
  }
  return status.status;
}

export function summarizeHistoricalParameterStatus({
  graph,
  status,
  runs,
  resolver,
}: {
  graph: InspectResponse | undefined;
  status: LogParameterStatusResponse | undefined;
  runs: LogRun[];
  resolver?: LinearMonitorTargetResolver;
}): ParameterSummary {
  const expectedChannels = expectedLinearParameterChannels(graph, resolver);
  const counts: ParameterSummaryCounts = {
    updated: 0,
    unchanged: 0,
    mixed: 0,
    notTracked: 0,
  };
  const breakdown: ParameterSummaryBreakdown = {
    missing: 0,
    unknown: 0,
  };
  const nodesByRunId = new Map(
    (status?.runs ?? []).map((runStatus) => [
      runStatus.sourceId,
      statusNodeByPath(runStatus),
    ]),
  );

  for (const expected of expectedChannels) {
    const statuses = runs.map((run) => {
      const nodeStatus = nodesByRunId.get(run.id)?.get(expected.nodePath);
      return channelStatus(expected.channel, nodeStatus?.[expected.channel]);
    });
    const classification = classifyChannel(statuses);
    counts[classification.status] += 1;
    if (classification.status === "notTracked") {
      if (classification.missingRuns > 0 && classification.unknownRuns === 0) {
        breakdown.missing += 1;
      } else {
        breakdown.unknown += 1;
      }
    }
  }

  const total = expectedChannels.length;
  const severity: ParameterSummarySeverity =
    counts.unchanged > 0
      ? "danger"
      : counts.mixed > 0
        ? "warning"
        : counts.notTracked > 0 || total === 0
          ? "not-tracked"
          : "success";

  return {
    counts,
    breakdown,
    total,
    severity,
  };
}
