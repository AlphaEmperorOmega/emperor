import {
  type GraphNode,
  type InspectResponse,
  type LogParameterStatusResponse,
  type LogRun,
  type LogRunTags,
  type ParameterChannelStatus,
  type ParameterStatus,
} from "@/lib/api";
import {
  anyLogRunTagsMatchNodePath,
  experimentsWithLayerMonitorData,
  filterHistoricalRuns,
  historicalPresetOptions as buildHistoricalPresetOptions,
  latestHistoricalMonitorRuns,
  sortLogRunsNewestFirst,
  type HistoricalRunOption,
} from "@/lib/historical-monitor-runs";
import {
  buildMonitorComparisonCandidateGroups,
  createMonitorTargetResolver,
  createLinearMonitorTargetResolver,
  type LinearMonitorComparisonCandidateGroups,
  type LinearMonitorTargetResolver,
  type MonitorTargetResolver,
} from "@/lib/graph/monitor-targets";
import {
  type GraphParameterActivity,
  type GraphParameterActivityChannel,
  type GraphParameterActivitySource,
} from "@/lib/graph/types";
import { expectedLinearParameterChannels } from "@/lib/parameter-summary";
import { type ActiveMonitorJob, type MonitorChartsSource } from "@/types/monitor";

export type DatasetSelectionInput = {
  logRuns?: LogRun[];
  modelRunTags?: LogRunTags[];
  selectedModel: string;
  selectedHistoricalPreset: string;
  selectedLogRunId: string | null;
};

export type DatasetSelectionState = {
  modelLogRuns: LogRun[];
  historicalPresetOptions: HistoricalRunOption[];
  visibleHistoricalRuns: LogRun[];
  selectedHistoricalExperiment: string;
  selectedHistoricalDataset: string;
  selectedHistoricalRunPreset: string;
  filteredHistoricalRuns: LogRun[];
  historicalMonitorRuns: LogRun[];
  filteredHistoricalRunIds: string[];
  selectedLogRun: LogRun | undefined;
};

export type MonitorSourceInput = {
  graph?: InspectResponse;
  selectedNode?: GraphNode;
  graphMonitorNode?: GraphNode;
  activeTrainingJob?: ActiveMonitorJob;
  historicalMonitorRuns?: LogRun[];
  selectedHistoricalExperiment?: string;
  selectedHistoricalDataset?: string;
  selectedHistoricalPreset?: string;
  logRunTags?: LogRunTags[];
  filteredHistoricalRunIds?: string[];
  monitorTargetResolver?: MonitorTargetResolver;
  linearMonitorTargetResolver?: LinearMonitorTargetResolver;
};

export type MonitorSourceState = {
  monitorTargetResolver: MonitorTargetResolver;
  linearMonitorTargetResolver: LinearMonitorTargetResolver;
  activeJobHasMonitorSource: boolean;
  selectedMonitorNode: GraphNode | undefined;
  selectedMonitorComparisonCandidateGroups: LinearMonitorComparisonCandidateGroups;
  selectedLogRunHasMonitorTags: boolean;
  graphMonitorComparisonCandidateGroups: LinearMonitorComparisonCandidateGroups;
  graphMonitorSource: MonitorChartsSource | undefined;
};

export type ParameterActivityInput = {
  graph?: InspectResponse;
  source?: MonitorChartsSource;
  status?: ParameterStatus | LogParameterStatusResponse;
  linearMonitorTargetResolver?: LinearMonitorTargetResolver;
};

export function deriveDatasetSelectionState(
  input: DatasetSelectionInput,
): DatasetSelectionState {
  const modelLogRuns = sortLogRunsNewestFirst(
    (input.logRuns ?? []).filter((run) => run.model === input.selectedModel),
  );
  // Only surface experiments that carry per-layer monitor data; model-performance
  // metrics remain available in the Logs workspace. Until tags load this set is
  // empty, so the panel shows a loading state instead of an unfiltered list.
  const layerMonitorExperiments = experimentsWithLayerMonitorData(
    modelLogRuns,
    input.modelRunTags,
  );
  const eligibleRuns = modelLogRuns.filter((run) =>
    layerMonitorExperiments.has(run.experiment),
  );
  // The preset filter ("" = all presets) only controls which runs are listed; it is
  // independent of the build/training preset selected under the model.
  const historicalPresetOptions = buildHistoricalPresetOptions(eligibleRuns);
  const visibleHistoricalRuns = eligibleRuns.filter(
    (run) =>
      !input.selectedHistoricalPreset || run.preset === input.selectedHistoricalPreset,
  );
  // Selection drives everything downstream: the experiment/dataset labels and the
  // monitor run group are derived from the picked run, so with nothing selected the
  // charts stay empty.
  const selectedLogRun = visibleHistoricalRuns.find(
    (run) => run.id === input.selectedLogRunId,
  );
  const selectedHistoricalExperiment = selectedLogRun?.experiment ?? "";
  const selectedHistoricalDataset = selectedLogRun?.dataset ?? "";
  const selectedHistoricalRunPreset = selectedLogRun?.preset ?? "";
  const filteredHistoricalRuns = selectedLogRun
    ? filterHistoricalRuns(
        eligibleRuns,
        selectedHistoricalExperiment,
        selectedHistoricalDataset,
        selectedHistoricalRunPreset,
      )
    : [];
  const historicalMonitorRuns = latestHistoricalMonitorRuns(filteredHistoricalRuns);
  const filteredHistoricalRunIds = filteredHistoricalRuns.map((run) => run.id);

  return {
    modelLogRuns,
    historicalPresetOptions,
    visibleHistoricalRuns,
    selectedHistoricalExperiment,
    selectedHistoricalDataset,
    selectedHistoricalRunPreset,
    filteredHistoricalRuns,
    historicalMonitorRuns,
    filteredHistoricalRunIds,
    selectedLogRun,
  };
}

export function deriveMonitorSource(input: MonitorSourceInput): MonitorSourceState {
  const activeTrainingJob = input.activeTrainingJob;
  const filteredHistoricalRunIds = input.filteredHistoricalRunIds ?? [];
  const targetIsAvailable = (target: {
    monitorName: string;
    node: GraphNode;
  }) => {
    if (activeTrainingJob?.monitors.includes(target.monitorName)) {
      return true;
    }
    return anyLogRunTagsMatchNodePath(
      input.logRunTags,
      filteredHistoricalRunIds,
      target.node.path,
    );
  };
  const monitorTargetResolver =
    input.monitorTargetResolver ??
    createMonitorTargetResolver(input.graph, targetIsAvailable);
  const linearMonitorTargetResolver =
    input.linearMonitorTargetResolver ??
    createLinearMonitorTargetResolver(input.graph);
  const selectedMonitorTarget = monitorTargetResolver(input.selectedNode);
  const graphMonitorTarget = monitorTargetResolver(input.graphMonitorNode);
  const selectedMonitorName = selectedMonitorTarget?.monitorName;
  const graphMonitorName = graphMonitorTarget?.monitorName;
  const activeSelectedTrainingJob =
    selectedMonitorName &&
    activeTrainingJob?.monitors.includes(selectedMonitorName)
      ? activeTrainingJob
      : undefined;
  const activeGraphTrainingJob =
    graphMonitorName && activeTrainingJob?.monitors.includes(graphMonitorName)
      ? activeTrainingJob
      : undefined;
  const activeLinearTrainingJob = activeTrainingJob?.monitors.includes("linear")
    ? input.activeTrainingJob
    : undefined;
  const activeJobHasMonitorSource = Boolean(
    activeSelectedTrainingJob ?? activeLinearTrainingJob,
  );
  const selectedMonitorNode = selectedMonitorTarget?.node;
  const selectedMonitorComparisonCandidateGroups =
    buildMonitorComparisonCandidateGroups(
      input.graph,
      selectedMonitorNode,
      selectedMonitorName,
    );
  const selectedLogRunHasMonitorTags = anyLogRunTagsMatchNodePath(
    input.logRunTags,
    filteredHistoricalRunIds,
    selectedMonitorNode?.path,
  );
  const graphMonitorComparisonCandidateGroups =
    buildMonitorComparisonCandidateGroups(
      input.graph,
      graphMonitorTarget?.node ?? input.graphMonitorNode,
      graphMonitorName,
    );
  const historicalMonitorRuns = input.historicalMonitorRuns ?? [];
  const graphMonitorSource: MonitorChartsSource | undefined = activeGraphTrainingJob
    ? { kind: "active-job", job: activeGraphTrainingJob }
    : !input.graphMonitorNode && activeLinearTrainingJob
      ? { kind: "active-job", job: activeLinearTrainingJob }
    : historicalMonitorRuns.length > 0
      ? {
          kind: "historical-run-group",
          runs: historicalMonitorRuns,
          experiment: input.selectedHistoricalExperiment ?? "",
          dataset: input.selectedHistoricalDataset ?? "",
          preset: input.selectedHistoricalPreset ?? "",
        }
      : undefined;

  return {
    monitorTargetResolver,
    linearMonitorTargetResolver,
    activeJobHasMonitorSource,
    selectedMonitorNode,
    selectedMonitorComparisonCandidateGroups,
    selectedLogRunHasMonitorTags,
    graphMonitorComparisonCandidateGroups,
    graphMonitorSource,
  };
}

function unknownParameterChannel(
  source: GraphParameterActivitySource,
  sourceLabel: string,
): GraphParameterActivityChannel {
  return {
    status: "unknown",
    source,
    sourceLabel,
    observedPoints: 0,
  };
}

function activeParameterChannel(
  channel: ParameterChannelStatus | undefined,
  sourceLabel: string,
): GraphParameterActivityChannel {
  if (!channel) {
    return unknownParameterChannel("active-job", sourceLabel);
  }
  return {
    status: channel.status,
    source: "active-job",
    sourceLabel,
    metric: channel.metric,
    lastStep: channel.lastStep,
    observedPoints: channel.observedPoints,
  };
}

function statusNodeByPath(status: ParameterStatus | undefined) {
  return new Map((status?.nodes ?? []).map((node) => [node.nodePath, node]));
}

function firstDefinedMetric(
  statuses: Array<ParameterChannelStatus | undefined>,
  status: ParameterChannelStatus["status"],
) {
  return statuses.find((item) => item?.status === status && item.metric)?.metric;
}

function maxLastStep(statuses: Array<ParameterChannelStatus | undefined>) {
  const steps = statuses
    .map((item) => item?.lastStep)
    .filter((step): step is number => typeof step === "number");
  return steps.length > 0 ? Math.max(...steps) : undefined;
}

function historicalParameterChannel(
  statuses: Array<ParameterChannelStatus | undefined>,
  sourceLabel: string,
): GraphParameterActivityChannel {
  const normalizedStatuses: ParameterChannelStatus[] = statuses.map(
    (status) => status ?? {
      status: "unknown" as const,
      metric: null,
      lastStep: null,
      observedPoints: 0,
    },
  );
  const updatedRuns = normalizedStatuses.filter(
    (status) => status.status === "updated",
  ).length;
  const unchangedRuns = normalizedStatuses.filter(
    (status) => status.status === "unchanged",
  ).length;
  const missingRuns = normalizedStatuses.filter(
    (status) => status.status === "missing",
  ).length;
  const unknownRuns = normalizedStatuses.filter(
    (status) => status.status === "unknown",
  ).length;
  const observedRuns = updatedRuns + unchangedRuns;
  const totalRuns = normalizedStatuses.length;
  const status =
    observedRuns === 0
      ? "unknown"
      : updatedRuns === totalRuns
        ? "updated"
        : updatedRuns > 0
          ? "mixed"
          : "unchanged";

  return {
    status,
    source: "historical",
    sourceLabel,
    metric:
      firstDefinedMetric(normalizedStatuses, "updated") ??
      firstDefinedMetric(normalizedStatuses, "unchanged") ??
      firstDefinedMetric(normalizedStatuses, "missing") ??
      firstDefinedMetric(normalizedStatuses, "unknown") ??
      null,
    lastStep: maxLastStep(normalizedStatuses),
    observedPoints: normalizedStatuses.reduce(
      (total, item) => total + item.observedPoints,
      0,
    ),
    updatedRuns,
    unchangedRuns,
    missingRuns,
    unknownRuns,
    totalRuns,
  };
}

function isLogParameterStatusResponse(
  status: ParameterStatus | LogParameterStatusResponse | undefined,
): status is LogParameterStatusResponse {
  return Boolean(status && "runs" in status);
}

export function deriveParameterActivityByNodePath(
  input: ParameterActivityInput,
): Map<string, GraphParameterActivity> | undefined {
  if (!input.graph || !input.source) {
    return undefined;
  }

  const resolver =
    input.linearMonitorTargetResolver ??
    createLinearMonitorTargetResolver(input.graph);
  const expectedChannels = expectedLinearParameterChannels(input.graph, resolver);
  if (expectedChannels.length === 0) {
    return undefined;
  }
  const targetPaths = [
    ...new Set(expectedChannels.map((channel) => channel.nodePath)),
  ].sort();
  const biasTargetPaths = new Set(
    expectedChannels
      .filter((channel) => channel.channel === "bias")
      .map((channel) => channel.nodePath),
  );

  if (input.source.kind === "active-job") {
    const status = isLogParameterStatusResponse(input.status)
      ? undefined
      : input.status;
    const nodesByPath = statusNodeByPath(status);
    const sourceLabel = `active job ${input.source.job.id}`;
    return new Map(
      targetPaths.map((targetPath) => {
        const node = nodesByPath.get(targetPath);
        const activity: GraphParameterActivity = {
          targetPath,
          weights: activeParameterChannel(node?.weights, sourceLabel),
        };
        if (biasTargetPaths.has(targetPath)) {
          activity.bias = activeParameterChannel(node?.bias, sourceLabel);
        }
        return [
          targetPath,
          activity,
        ];
      }),
    );
  }

  const runs =
    input.source.kind === "historical-run-group"
      ? input.source.runs
      : [input.source.run];
  const statusRuns = isLogParameterStatusResponse(input.status)
    ? input.status.runs
    : [];
  const nodesByRunId = new Map(
    statusRuns.map((runStatus) => [
      runStatus.sourceId,
      statusNodeByPath(runStatus),
    ]),
  );
  const sourceLabel =
    runs.length === 1 ? "1 historical run" : `${runs.length} historical runs`;

  return new Map(
    targetPaths.map((targetPath) => {
      const nodeStatuses = runs.map((run) =>
        nodesByRunId.get(run.id)?.get(targetPath),
      );
      const activity: GraphParameterActivity = {
        targetPath,
        weights: historicalParameterChannel(
          nodeStatuses.map((node) => node?.weights),
          sourceLabel,
        ),
      };
      if (biasTargetPaths.has(targetPath)) {
        activity.bias = historicalParameterChannel(
          nodeStatuses.map((node) => node?.bias),
          sourceLabel,
        );
      }
      return [
        targetPath,
        activity,
      ];
    }),
  );
}
