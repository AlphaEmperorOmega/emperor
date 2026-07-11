import { type ReactNode, useCallback, useEffect, useRef } from "react";
import { createWorkbenchContext } from "@/features/workbench/providers/create-context";
import {
  useModelPackageInspection,
} from "@/features/workbench/providers/workbench-providers";
import {
  useRegisterWorkbenchConnectionReset,
  useWorkbenchCapabilities,
  isWorkbenchProtectedAccessReady,
  useWorkbenchConnection,
} from "@/features/workbench/providers/workbench-connection-provider";
import { useActiveTrainingJob } from "@/features/workbench/providers/training-provider";
import {
  useLogsWorkspaceImplementation,
  type LogsWorkspaceImplementation,
  type LogsScopeMode,
  type LogsTargetScope,
} from "@/features/workbench/state/logs/_use-logs-workspace-state";
import {
  useLogsChartViewModel,
  type LogsChartSource,
} from "@/features/workbench/state/logs/_logs-chart-state";
import { useLogRunArtifactsQuery } from "@/features/workbench/state/logs/use-log-queries";
import { type ChecklistOption } from "@/features/workbench/state/logs/logs-selectors";

export type LogsBrowserFilterKey =
  | "experiments"
  | "datasets"
  | "models"
  | "presets"
  | "tags";

export type LogsBrowserFilter = {
  options: ChecklistOption[];
  selectedValues: string[];
};

export type LogsBrowser = {
  scope: {
    mode: LogsScopeMode;
    target: LogsTargetScope;
    canUseCurrentTarget: boolean;
    allRunsSelected: boolean;
    useCurrentTarget: () => void;
    showAllRuns: () => void;
  };
  filters: Record<LogsBrowserFilterKey, LogsBrowserFilter>;
  status: {
    isScanning: boolean;
    isRefreshing: boolean;
    runsError: unknown;
    experimentsError: unknown;
    tagsError: unknown;
  };
  results: {
    hasExperiments: boolean;
    hasRuns: boolean;
  };
  pagination: {
    runs: {
      loaded: number;
      total: number;
      canLoadMore: boolean;
      isLoadingMore: boolean;
    };
    scalarTags: {
      loadedRuns: number;
      totalRuns: number;
      canLoadMore: boolean;
      isLoadingMore: boolean;
    };
  };
  actions: {
    toggleFilter: (filter: LogsBrowserFilterKey, value: string) => void;
    selectAll: (filter: LogsBrowserFilterKey) => void;
    selectNone: (filter: LogsBrowserFilterKey) => void;
    refresh: () => Promise<void>;
    loadMoreRuns: () => void;
    loadMoreScalarTags: () => void;
  };
};

export type LogsDeletion = Pick<
  LogsWorkspaceImplementation["deletion"],
  "enabled" | "presetTargetExperiment" | "operation" | "actions"
>;
export type LogsCharts = ReturnType<typeof useLogsChartViewModel>;
export type {
  LogBestRunViewModel,
  LogMetricChartLayoutGroupKey,
  LogMetricGroupScalarQueryState,
  LogsChartEmptyState,
  ScalarChartGridMode,
} from "@/features/workbench/state/logs/_logs-chart-state";

type LogsDetailSource = Pick<
  LogsWorkspaceImplementation,
  "enabled" | "selectedRun"
>;

const [LogsBrowserProvider, useLogsBrowser] =
  createWorkbenchContext<LogsBrowser>("LogsBrowserContext");
const [LogsChartSourceProvider, useLogsChartSource] =
  createWorkbenchContext<LogsChartSource>("LogsChartSourceContext");
const [LogsDetailSourceProvider, useLogsDetailSource] =
  createWorkbenchContext<LogsDetailSource>("LogsDetailSourceContext");
const [LogsDeletionProvider, useLogsDeletion] =
  createWorkbenchContext<LogsDeletion>("LogsDeletionContext");

export { useLogsBrowser, useLogsDeletion };

const noStartedExperiments: readonly string[] = [];

function selectedValues(
  selected: Set<string>,
  options: ChecklistOption[],
) {
  return options
    .filter((option) => selected.has(option.value))
    .map((option) => option.value);
}

function hasCompleteTarget(target: LogsTargetScope) {
  return Boolean(
    target.modelType &&
      target.model &&
      target.preset &&
      target.datasets.length > 0,
  );
}

function logsBrowserProjection(state: LogsWorkspaceImplementation): LogsBrowser {
  const filters: LogsBrowser["filters"] = {
    experiments: {
      options: state.experimentOptions,
      selectedValues: selectedValues(
        state.selectedExperiments,
        state.experimentOptions,
      ),
    },
    datasets: {
      options: state.datasetOptions,
      selectedValues: selectedValues(state.selectedDatasets, state.datasetOptions),
    },
    models: {
      options: state.modelOptions,
      selectedValues: selectedValues(state.selectedModels, state.modelOptions),
    },
    presets: {
      options: state.presetOptions,
      selectedValues: selectedValues(state.selectedPresets, state.presetOptions),
    },
    tags: {
      options: state.tagOptions,
      selectedValues: selectedValues(state.selectedTags, state.tagOptions),
    },
  };
  const toggleActions: Record<
    LogsBrowserFilterKey,
    (value: string) => void
  > = {
    experiments: state.toggleExperiment,
    datasets: state.toggleDataset,
    models: state.toggleModel,
    presets: state.togglePreset,
    tags: state.toggleTag,
  };
  const allActions: Record<LogsBrowserFilterKey, () => void> = {
    experiments: state.selectAllExperiments,
    datasets: state.selectAllDatasets,
    models: state.selectAllModels,
    presets: state.selectAllPresets,
    tags: state.selectAllTags,
  };
  const noneActions: Record<LogsBrowserFilterKey, () => void> = {
    experiments: state.selectNoExperiments,
    datasets: state.selectNoDatasets,
    models: state.selectNoModels,
    presets: state.selectNoPresets,
    tags: state.selectNoTags,
  };
  const allExperimentValues = filters.experiments.options.map(
    (option) => option.value,
  );

  return {
    scope: {
      mode: state.scopeMode,
      target: state.targetScope,
      canUseCurrentTarget: hasCompleteTarget(state.targetScope),
      allRunsSelected:
        state.scopeMode === "custom" &&
        filters.experiments.selectedValues.length === allExperimentValues.length &&
        allExperimentValues.every((value) =>
          filters.experiments.selectedValues.includes(value),
        ),
      useCurrentTarget: state.useCurrentTargetScope,
      showAllRuns: state.showAllRuns,
    },
    filters,
    status: {
      isScanning: state.runsQuery.isLoading || state.experimentsQuery.isLoading,
      isRefreshing: state.runsQuery.isFetching || state.experimentsQuery.isFetching,
      runsError: state.runsQuery.error,
      experimentsError: state.experimentsQuery.error,
      tagsError: state.tagsQuery.error,
    },
    results: {
      hasExperiments: state.experimentOptions.length > 0,
      hasRuns: state.runs.length > 0,
    },
    pagination: {
      runs: {
        loaded: state.loadedRunCount,
        total: state.totalRunCount,
        canLoadMore: state.canLoadMoreRuns,
        isLoadingMore: state.isLoadingMoreRuns,
      },
      scalarTags: {
        loadedRuns: state.loadedScalarTagRunCount,
        totalRuns: state.totalScalarTagRunCount,
        canLoadMore: state.canLoadMoreScalarTags,
        isLoadingMore: state.isLoadingMoreScalarTags,
      },
    },
    actions: {
      toggleFilter: (filter, value) => toggleActions[filter](value),
      selectAll: (filter) => allActions[filter](),
      selectNone: (filter) => noneActions[filter](),
      refresh: state.refreshLogLists,
      loadMoreRuns: state.loadMoreRuns,
      loadMoreScalarTags: state.loadMoreScalarTags,
    },
  };
}

function logsChartSource(state: LogsWorkspaceImplementation): LogsChartSource {
  return {
    collapsedMetricGroups: state.collapsedMetricGroups,
    confusionMatrixRateTags: state.confusionMatrixRateTags,
    enabled: state.enabled,
    loadedScalarTagRunCount: state.loadedScalarTagRunCount,
    refreshLogLists: state.refreshLogLists,
    runsQuery: state.runsQuery,
    selectedTagList: state.selectedTagList,
    setSelectedDetailRunId: state.setSelectedDetailRunId,
    tagOptions: state.tagOptions,
    tagsQuery: state.tagsQuery,
    toggleMetricGroup: state.toggleMetricGroup,
    toggleTag: state.toggleTag,
    visibleRunIds: state.visibleRunIds,
    visibleRuns: state.visibleRuns,
  };
}

function logsDeletionProjection(state: LogsWorkspaceImplementation): LogsDeletion {
  return {
    enabled: state.deletion.enabled,
    presetTargetExperiment: state.deletion.presetTargetExperiment,
    operation: state.deletion.operation,
    actions: state.deletion.actions,
  };
}

export function useLogsCharts() {
  return useLogsChartViewModel(useLogsChartSource());
}

export function useLogRunDetail() {
  const source = useLogsDetailSource();
  const artifactsQuery = useLogRunArtifactsQuery({
    runId: source.selectedRun?.id,
    enabled: source.enabled,
  });
  return {
    run: source.selectedRun,
    artifacts: artifactsQuery.data,
    status: {
      isLoading: artifactsQuery.isLoading,
      error: artifactsQuery.error,
    },
  };
}

export function LogsWorkspaceProvider({
  enabled,
  startedExperiments = noStartedExperiments,
  children,
}: {
  enabled: boolean;
  startedExperiments?: readonly string[];
  children: ReactNode;
}) {
  const { capabilities } = useWorkbenchCapabilities();
  const workbenchConnection = useWorkbenchConnection();
  const protectedAccessReady = isWorkbenchProtectedAccessReady(
    workbenchConnection,
  );
  const { target, options } = useModelPackageInspection();
  const { activeTrainingJob } = useActiveTrainingJob();
  const targetPreset =
    target.kind === "historical-run"
      ? target.preset
      : options.presets.find((preset) => preset.name === target.preset)?.label ??
        target.preset;
  const state = useLogsWorkspaceImplementation({
    enabled: enabled && protectedAccessReady,
    logDeletionEnabled:
      capabilities.logDeletionEnabled && protectedAccessReady,
    targetScope: {
      modelType: target.modelPackage.modelType,
      model: target.modelPackage.model,
      preset: targetPreset,
      datasets: target.datasets,
    },
  });
  const includeStartedExperiment = state.includeStartedExperiment;
  const deliveredStartedExperimentsRef = useRef(new Set<string>());
  const clearLogsForConnectionChange = state.clearForConnectionChange;
  const clearForConnectionChange = useCallback(() => {
    deliveredStartedExperimentsRef.current.clear();
    clearLogsForConnectionChange();
  }, [clearLogsForConnectionChange]);
  useRegisterWorkbenchConnectionReset(clearForConnectionChange);

  useEffect(() => {
    const observedExperiments = activeTrainingJob?.logFolder
      ? [...startedExperiments, activeTrainingJob.logFolder]
      : startedExperiments;
    for (const experiment of observedExperiments) {
      if (deliveredStartedExperimentsRef.current.has(experiment)) {
        continue;
      }
      deliveredStartedExperimentsRef.current.add(experiment);
      includeStartedExperiment(experiment);
    }
  }, [activeTrainingJob?.logFolder, includeStartedExperiment, startedExperiments]);

  return (
    <LogsBrowserProvider value={logsBrowserProjection(state)}>
      <LogsChartSourceProvider value={logsChartSource(state)}>
        <LogsDetailSourceProvider
          value={{ enabled: state.enabled, selectedRun: state.selectedRun }}
        >
          <LogsDeletionProvider value={logsDeletionProjection(state)}>
            {children}
          </LogsDeletionProvider>
        </LogsDetailSourceProvider>
      </LogsChartSourceProvider>
    </LogsBrowserProvider>
  );
}
