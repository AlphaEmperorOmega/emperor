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
import { presetOwnedCount, type ConfigSection, type OverrideValues } from "@/lib/config";
import {
  groupConfigSnapshotsByPreset,
  selectedConfigSnapshots,
  type ConfigSnapshot,
  type ConfigSnapshotGroup,
} from "@/lib/config-snapshots";
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
  buildLinearMonitorComparisonCandidateGroups,
  createLinearMonitorTargetResolver,
  type LinearMonitorComparisonCandidateGroups,
} from "@/lib/graph/monitor-targets";
import { type MonitorChartsSource } from "@/types/monitor";

export type TargetSelectionInput = {
  datasets?: Dataset[];
  presets?: Preset[];
  schemaFields?: ConfigField[];
  configSnapshots: ConfigSnapshot[];
  selectedModel: string;
  selectedPreset: string;
  selectedTrainingPresets: string[];
  overrides: OverrideValues;
};

export type TargetSelectionState = {
  datasetNames: string[];
  presetNames: string[];
  selectedPresetMeta: Preset | undefined;
  configSections: ConfigSection[];
  configFields: ConfigField[];
  visibleConfigSnapshots: ConfigSnapshot[];
  configSnapshotGroups: ConfigSnapshotGroup[];
  overrideCount: number;
  presetOwnedFieldCount: number;
  fieldCount: number;
};

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
  filteredHistoricalRuns: LogRun[];
  historicalMonitorRuns: LogRun[];
  filteredHistoricalRunIds: string[];
  selectedLogRun: LogRun | undefined;
};

export type LinearMonitorTargetResolver = (
  node: GraphNode | undefined,
) => GraphNode | undefined;

export type MonitorSourceInput = {
  graph?: InspectResponse;
  selectedNode?: GraphNode;
  graphMonitorNode?: GraphNode;
  activeTrainingJob?: TrainingJob;
  historicalMonitorRuns?: LogRun[];
  selectedHistoricalExperiment?: string;
  selectedHistoricalDataset?: string;
  logRunTags?: LogRunTags[];
  filteredHistoricalRunIds?: string[];
  linearMonitorTargetResolver?: LinearMonitorTargetResolver;
};

export type MonitorSourceState = {
  linearMonitorTargetResolver: LinearMonitorTargetResolver;
  activeJobHasMonitorSource: boolean;
  selectedMonitorNode: GraphNode | undefined;
  selectedMonitorComparisonCandidateGroups: LinearMonitorComparisonCandidateGroups;
  selectedLogRunHasMonitorTags: boolean;
  graphMonitorComparisonCandidateGroups: LinearMonitorComparisonCandidateGroups;
  graphMonitorSource: MonitorChartsSource | undefined;
};

export function deriveTargetSelectionState(
  input: TargetSelectionInput,
): TargetSelectionState {
  const datasetNames = (input.datasets ?? []).map((dataset) => dataset.name);
  const presetNames = (input.presets ?? []).map((preset) => preset.name);
  const selectedPresetMeta = (input.presets ?? []).find(
    (preset) => preset.name === input.selectedPreset,
  );

  const groups = new Map<string, ConfigField[]>();
  for (const field of input.schemaFields ?? []) {
    const section = field.section || "General";
    groups.set(section, [...(groups.get(section) ?? []), field]);
  }
  const configSections = Array.from(groups, ([title, fields]) => ({ title, fields }));
  const configFields = configSections.flatMap((section) => section.fields);
  const visibleConfigSnapshots = selectedConfigSnapshots(
    input.configSnapshots,
    input.selectedModel,
    input.selectedTrainingPresets,
  );
  const configSnapshotGroups = groupConfigSnapshotsByPreset(
    visibleConfigSnapshots,
    input.selectedTrainingPresets,
  );
  const presetOwnedFieldCount = configSections.reduce(
    (total, section) => total + presetOwnedCount(section.fields),
    0,
  );
  const fieldCount = configSections.reduce(
    (total, section) => total + section.fields.length,
    0,
  );

  return {
    datasetNames,
    presetNames,
    selectedPresetMeta,
    configSections,
    configFields,
    visibleConfigSnapshots,
    configSnapshotGroups,
    overrideCount: Object.keys(input.overrides).length,
    presetOwnedFieldCount,
    fieldCount,
  };
}

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
  const filteredHistoricalRuns = selectedLogRun
    ? filterHistoricalRuns(
        eligibleRuns,
        selectedHistoricalExperiment,
        selectedHistoricalDataset,
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
    filteredHistoricalRuns,
    historicalMonitorRuns,
    filteredHistoricalRunIds,
    selectedLogRun,
  };
}

export function deriveMonitorSource(input: MonitorSourceInput): MonitorSourceState {
  const linearMonitorTargetResolver =
    input.linearMonitorTargetResolver ??
    createLinearMonitorTargetResolver(input.graph);
  const activeLinearTrainingJob = input.activeTrainingJob?.monitors.includes("linear")
    ? input.activeTrainingJob
    : undefined;
  const activeJobHasMonitorSource = Boolean(activeLinearTrainingJob);
  const selectedMonitorNode = linearMonitorTargetResolver(input.selectedNode);
  const selectedMonitorComparisonCandidateGroups =
    buildLinearMonitorComparisonCandidateGroups(input.graph, selectedMonitorNode);
  const selectedLogRunHasMonitorTags = anyLogRunTagsMatchNodePath(
    input.logRunTags,
    input.filteredHistoricalRunIds ?? [],
    selectedMonitorNode?.path,
  );
  const graphMonitorComparisonCandidateGroups =
    buildLinearMonitorComparisonCandidateGroups(input.graph, input.graphMonitorNode);
  const historicalMonitorRuns = input.historicalMonitorRuns ?? [];
  const graphMonitorSource: MonitorChartsSource | undefined = activeLinearTrainingJob
    ? { kind: "active-job", job: activeLinearTrainingJob }
    : historicalMonitorRuns.length > 0
      ? {
          kind: "historical-run-group",
          runs: historicalMonitorRuns,
          experiment: input.selectedHistoricalExperiment ?? "",
          dataset: input.selectedHistoricalDataset ?? "",
        }
      : undefined;

  return {
    linearMonitorTargetResolver,
    activeJobHasMonitorSource,
    selectedMonitorNode,
    selectedMonitorComparisonCandidateGroups,
    selectedLogRunHasMonitorTags,
    graphMonitorComparisonCandidateGroups,
    graphMonitorSource,
  };
}
