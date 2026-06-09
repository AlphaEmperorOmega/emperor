type StringList = readonly string[];

export const logQueryKeys = {
  runs: () => ["log-runs"] as const,
  experiments: () => ["log-experiments"] as const,
  tags: () => ["log-tags"] as const,
  tagsForRuns: (runIds: StringList) => ["log-tags", runIds] as const,
  filteredHistoricalRunTags: (runIds: StringList) =>
    ["log-tags", "filtered-historical-runs", runIds] as const,
  modelRunTags: (runIds: StringList) =>
    ["log-tags", "model-runs", runIds] as const,
  scalars: () => ["log-scalars"] as const,
  scalarsForRunsAndTags: (runIds: StringList, tags: StringList) =>
    ["log-scalars", runIds, tags] as const,
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
};

export const trainingQueryKeys = {
  job: (activeJobId: string | null) => ["training-job", activeJobId] as const,
  runPlan: (planNonce: number, planInputKey: string) =>
    ["training-run-plan", planNonce, planInputKey] as const,
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
    ["monitor-data", "historical-run-group", runIds, nodePath] as const,
  activeJobParameterStatus: (
    jobId: string,
    preset: string | undefined,
    dataset: string | undefined,
  ) => ["monitor-parameter-status", "active-job", jobId, preset, dataset] as const,
  historicalParameterStatus: (runIds: StringList) =>
    ["monitor-parameter-status", "historical-run-group", runIds] as const,
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
      runIds,
    ] as const,
};

export const LOG_RUNS_QUERY_KEY = logQueryKeys.runs();
export const LOG_EXPERIMENTS_QUERY_KEY = logQueryKeys.experiments();
export const LOG_TAGS_QUERY_KEY = logQueryKeys.tags();
export const LOG_SCALARS_QUERY_KEY = logQueryKeys.scalars();
