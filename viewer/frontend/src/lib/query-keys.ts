import {
  type TrainingRunPlan,
  type TrainingSearchCreateInput,
} from "@/lib/api";
import { type OverrideValues } from "@/lib/config";

type StringList = readonly string[];
type NormalizedQueryObject = {
  readonly [key: string]: NormalizedQueryValue;
};
type NormalizedQueryValue =
  | string
  | number
  | boolean
  | null
  | readonly NormalizedQueryValue[]
  | NormalizedQueryObject;

export type TrainingRunPlanQueryKeyInput = {
  model: string;
  preset: string;
  presets: StringList;
  datasets: StringList;
  overrides: OverrideValues;
  logFolder: string;
  search?: TrainingSearchCreateInput;
  submittedRunPlan?: TrainingRunPlan;
};

function normalizedStringSet(values: StringList) {
  return Array.from(new Set(values)).sort();
}

function normalizedQueryValue(value: unknown): NormalizedQueryValue {
  if (value === undefined || value === null) {
    return null;
  }
  if (
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((item) => normalizedQueryValue(item));
  }
  if (typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([, entryValue]) => entryValue !== undefined)
        .sort(([leftKey], [rightKey]) => leftKey.localeCompare(rightKey))
        .map(([entryKey, entryValue]) => [
          entryKey,
          normalizedQueryValue(entryValue),
        ]),
    );
  }
  return null;
}

function trainingRunPlanInputKey({
  model,
  preset,
  presets,
  datasets,
  overrides,
  logFolder,
  search,
  submittedRunPlan,
}: TrainingRunPlanQueryKeyInput): NormalizedQueryObject {
  return {
    datasets: [...datasets],
    logFolder,
    model,
    overrides: normalizedQueryValue(overrides),
    preset,
    presets: [...presets],
    search: normalizedQueryValue(search ?? null),
    submittedRunPlan: normalizedQueryValue(submittedRunPlan ?? null),
  };
}

export const logQueryKeys = {
  runs: () => ["log-runs"] as const,
  experiments: () => ["log-experiments"] as const,
  tags: () => ["log-tags"] as const,
  tagsForRuns: (runIds: StringList) =>
    ["log-tags", normalizedStringSet(runIds)] as const,
  filteredHistoricalRunTags: (runIds: StringList) =>
    ["log-tags", "filtered-historical-runs", normalizedStringSet(runIds)] as const,
  modelRunTags: (runIds: StringList) =>
    ["log-tags", "model-runs", normalizedStringSet(runIds)] as const,
  scalars: () => ["log-scalars"] as const,
  scalarsForRunsAndTags: (runIds: StringList, tags: StringList) =>
    [
      "log-scalars",
      normalizedStringSet(runIds),
      normalizedStringSet(tags),
    ] as const,
  media: () => ["log-media"] as const,
  mediaForRunsAndTags: (
    runIds: StringList,
    imageTags: StringList,
    textTags: StringList,
  ) =>
    [
      "log-media",
      normalizedStringSet(runIds),
      normalizedStringSet(imageTags),
      normalizedStringSet(textTags),
    ] as const,
  checkpoints: () => ["log-checkpoints"] as const,
  checkpointsForRuns: (runIds: StringList) =>
    ["log-checkpoints", normalizedStringSet(runIds)] as const,
  artifacts: () => ["log-artifacts"] as const,
  artifactsForRun: (runId: string | null | undefined) =>
    ["log-artifacts", runId ?? null] as const,
};

export const viewerQueryKeys = {
  health: () => ["health"] as const,
  capabilities: () => ["capabilities"] as const,
  models: () => ["models"] as const,
  presets: (selectedModel: string) => ["presets", selectedModel] as const,
  datasets: (selectedModel: string) => ["datasets", selectedModel] as const,
  monitors: (selectedModel: string) => ["monitors", selectedModel] as const,
  configSchema: (selectedModel: string, selectedPreset: string) =>
    ["config-schema", selectedModel, selectedPreset] as const,
  searchSpace: (selectedModel: string, selectedPreset: string) =>
    ["search-space", selectedModel, selectedPreset] as const,
  historicalSummaryInspection: (
    model: string,
    preset: string,
    dataset: string,
  ) => ["inspect", "historical-summary", model, preset, dataset] as const,
  comparisonInspection: (model: string, preset: string, dataset: string) =>
    ["comparison-inspection", model, preset, dataset] as const,
  configSnapshots: (selectedModel: string) =>
    ["config-snapshots", selectedModel] as const,
  configSnapshotLibrary: () => ["config-snapshot-library"] as const,
};

export const trainingQueryKeys = {
  job: (activeJobId: string | null) => ["training-job", activeJobId] as const,
  jobEvents: (jobId: string, offset: number, limit: number) =>
    ["training-job-events", jobId, offset, limit] as const,
  runPlan: (planNonce: number, input: TrainingRunPlanQueryKeyInput) =>
    ["training-run-plan", planNonce, trainingRunPlanInputKey(input)] as const,
};

export const monitorQueryKeys = {
  activeJob: (
    jobId: string,
    nodePath: string | undefined,
    preset: string | undefined,
    dataset: string | undefined,
  ) => ["monitor-data", "active-job", jobId, nodePath, preset, dataset] as const,
  historicalRun: (runId: string | undefined, nodePath: string | undefined) =>
    ["monitor-data", "historical-run", runId, nodePath] as const,
  historicalRunGroup: (runIds: StringList, nodePath: string | undefined) =>
    [
      "monitor-data",
      "historical-run-group",
      normalizedStringSet(runIds),
      nodePath,
    ] as const,
  activeJobParameterStatus: (
    jobId: string,
    preset: string | undefined,
    dataset: string | undefined,
  ) => ["monitor-parameter-status", "active-job", jobId, preset, dataset] as const,
  historicalParameterStatus: (runIds: StringList) =>
    [
      "monitor-parameter-status",
      "historical-run-group",
      normalizedStringSet(runIds),
    ] as const,
  historicalParameterSummary: (
    model: string,
    preset: string,
    dataset: string,
    runIds: StringList,
  ) =>
    [
      "monitor-parameter-summary",
      "historical-run-group",
      model,
      preset,
      dataset,
      normalizedStringSet(runIds),
    ] as const,
};

export const LOG_RUNS_QUERY_KEY = logQueryKeys.runs();
export const LOG_EXPERIMENTS_QUERY_KEY = logQueryKeys.experiments();
export const LOG_TAGS_QUERY_KEY = logQueryKeys.tags();
export const LOG_SCALARS_QUERY_KEY = logQueryKeys.scalars();
export const LOG_MEDIA_QUERY_KEY = logQueryKeys.media();
export const LOG_CHECKPOINTS_QUERY_KEY = logQueryKeys.checkpoints();
export const LOG_ARTIFACTS_QUERY_KEY = logQueryKeys.artifacts();
