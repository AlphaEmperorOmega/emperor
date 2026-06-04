import { useCallback, useEffect, useMemo, useState } from "react";
import {
  type ConfigField,
  type GraphNode,
  type LogRun,
  type TrainingJob,
} from "@/lib/api";
import { useLogRunsQuery, useLogTagsQuery } from "@/hooks/use-log-queries";
import { presetOwnedCount, type OverrideValues } from "@/lib/config";
import {
  DEFAULT_TRAINING_SEARCH_STATE,
  type TrainingSearchState,
} from "@/lib/training-search";
import {
  createConfigSnapshot,
  groupConfigSnapshotsByPreset,
  selectedConfigSnapshots,
  type ConfigSnapshot,
  type ConfigSnapshotCreateResult,
} from "@/lib/config-snapshots";
import {
  anyLogRunTagsMatchNodePath,
  filterHistoricalRuns,
  historicalDatasetOptions as buildHistoricalDatasetOptions,
  historicalExperimentOptions as buildHistoricalExperimentOptions,
  latestHistoricalMonitorRuns,
  sortLogRunsNewestFirst,
} from "@/lib/historical-monitor-runs";
import { useGraphViewState } from "@/components/features/viewer/state/use-graph-view-state";
import { useLockedOverrideSync } from "@/components/features/viewer/state/use-locked-override-sync";
import { usePreviewInspectionState } from "@/components/features/viewer/state/use-preview-inspection";
import { useTargetOverridesState } from "@/components/features/viewer/state/use-target-overrides";
import { useViewerQueries } from "@/components/features/viewer/state/use-viewer-queries";
import {
  buildLinearMonitorComparisonCandidateGroups,
  createLinearMonitorTargetResolver,
} from "@/lib/graph/monitor-targets";
import { type MonitorChartsSource } from "@/types/monitor";

function resolveRunPresetName(
  run: LogRun,
  presets: Array<{ name: string; label: string }>,
) {
  const normalizedRunPreset = run.preset.toLowerCase();
  return (
    presets.find(
      (preset) =>
        preset.name === run.preset ||
        preset.label === run.preset ||
        preset.name.toLowerCase() === normalizedRunPreset ||
        preset.label.toLowerCase() === normalizedRunPreset,
    )?.name ?? ""
  );
}

function uniqueValidValues(values: string[], validValues: string[]) {
  const validValueSet = new Set(validValues);
  const seen = new Set<string>();
  return values.filter((value) => {
    if (!validValueSet.has(value) || seen.has(value)) {
      return false;
    }
    seen.add(value);
    return true;
  });
}

function normalizeTrainingPresetSelection(
  values: string[],
  presetNames: string[],
  primaryPreset: string,
) {
  if (presetNames.length === 0) {
    return [];
  }
  const validPresets = uniqueValidValues(values, presetNames);
  const nextValues =
    validPresets.length > 0
      ? validPresets
      : primaryPreset && presetNames.includes(primaryPreset)
        ? [primaryPreset]
        : [];
  if (!primaryPreset || !nextValues.includes(primaryPreset)) {
    return nextValues;
  }
  return [
    primaryPreset,
    ...nextValues.filter((preset) => preset !== primaryPreset),
  ];
}

function normalizeDatasetSelection(
  values: string[],
  datasetNames: string[],
  fallbackValues: string[] = [],
) {
  if (datasetNames.length === 0) {
    return [];
  }
  const validDatasets = uniqueValidValues(values, datasetNames);
  if (validDatasets.length > 0) {
    return validDatasets;
  }
  const fallbackDataset = fallbackValues.find((dataset) =>
    datasetNames.includes(dataset),
  );
  return [fallbackDataset ?? datasetNames[0]];
}

function createSnapshotId() {
  return globalThis.crypto?.randomUUID?.() ?? `snapshot-${Date.now()}`;
}

export type ViewerStateOptions = {
  /** Notifies the logs workspace when a training job starts writing to a folder. */
  onJobStarted?: (logFolder: string) => void;
};

/**
 * Single orchestration engine behind the viewer context providers.
 *
 * The four domains it returns (target/config, graph view, historical runs,
 * training) are genuinely coupled — selecting a model resets the graph, picking
 * a historical run re-runs the preview, opening a monitor reads the active job —
 * so the cascade effects and composite actions live here in one place and the
 * result is split into per-domain slices that the providers distribute.
 */
export function useViewerState(options: ViewerStateOptions = {}) {
  const { onJobStarted } = options;

  // --- Target + overrides + queries -------------------------------------
  const targetState = useTargetOverridesState();
  const {
    selectedModel,
    setSelectedModel,
    selectedPreset,
    setSelectedPreset,
    overrides,
    setOverrides,
    selectPreset,
    updateOverride,
    clearOverride,
    clearOverrides,
  } = targetState;
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [selectedTrainingPresets, setSelectedTrainingPresets] = useState<string[]>([]);
  const [selectedMonitors, setSelectedMonitors] = useState<string[]>([]);
  const [configSnapshots, setConfigSnapshots] = useState<ConfigSnapshot[]>([]);
  const [trainingSearch, setTrainingSearch] = useState<TrainingSearchState>(
    DEFAULT_TRAINING_SEARCH_STATE,
  );
  const {
    healthQuery,
    modelsQuery,
    presetsQuery,
    datasetsQuery,
    monitorsQuery,
    schemaQuery,
    searchSpaceQuery,
  } = useViewerQueries(selectedModel, selectedPreset);

  // --- Preview inspection (graph data) ----------------------------------
  const { graph, requestPreview, previewInspection } = usePreviewInspectionState();

  // --- Historical runs ---------------------------------------------------
  const [selectedLogRunId, setSelectedLogRunId] = useState<string | null>(null);
  const [selectedHistoricalExperiment, setSelectedHistoricalExperiment] =
    useState("");
  const [selectedHistoricalDataset, setSelectedHistoricalDataset] = useState("");
  const logRunsQuery = useLogRunsQuery();

  // --- Training + monitor charts ----------------------------------------
  const [activeJobId, setActiveJobId] = useState<string | null>(null);
  const [activeTrainingJob, setActiveTrainingJob] = useState<TrainingJob | undefined>();
  const [graphMonitorNode, setGraphMonitorNode] = useState<GraphNode | undefined>();

  const datasetNames = useMemo(
    () => (datasetsQuery.data?.datasets ?? []).map((dataset) => dataset.name),
    [datasetsQuery.data],
  );
  const presetNames = useMemo(
    () => (presetsQuery.data?.presets ?? []).map((preset) => preset.name),
    [presetsQuery.data],
  );
  const modelLogRuns = useMemo(
    () =>
      sortLogRunsNewestFirst(
        (logRunsQuery.data?.runs ?? []).filter((run) => run.model === selectedModel),
      ),
    [logRunsQuery.data, selectedModel],
  );
  const historicalExperimentOptions = useMemo(
    () => buildHistoricalExperimentOptions(modelLogRuns),
    [modelLogRuns],
  );
  const historicalDatasetOptions = useMemo(
    () =>
      buildHistoricalDatasetOptions(
        modelLogRuns,
        selectedHistoricalExperiment,
      ),
    [modelLogRuns, selectedHistoricalExperiment],
  );
  const filteredHistoricalRuns = useMemo(
    () =>
      filterHistoricalRuns(
        modelLogRuns,
        selectedHistoricalExperiment,
        selectedHistoricalDataset,
      ),
    [modelLogRuns, selectedHistoricalDataset, selectedHistoricalExperiment],
  );
  const historicalMonitorRuns = useMemo(
    () => latestHistoricalMonitorRuns(filteredHistoricalRuns),
    [filteredHistoricalRuns],
  );
  const filteredHistoricalRunIds = useMemo(
    () => filteredHistoricalRuns.map((run) => run.id),
    [filteredHistoricalRuns],
  );
  const selectedLogRun = useMemo(
    () => filteredHistoricalRuns.find((run) => run.id === selectedLogRunId),
    [filteredHistoricalRuns, selectedLogRunId],
  );
  const logRunTagsQuery = useLogTagsQuery({
    runIds: filteredHistoricalRunIds,
    queryKey: ["log-tags", "filtered-historical-runs", filteredHistoricalRunIds],
  });
  const linearMonitorTargetResolver = useMemo(
    () => createLinearMonitorTargetResolver(graph),
    [graph],
  );
  const activeJobHasMonitorSource = Boolean(
    activeTrainingJob?.monitors.includes("linear"),
  );
  const canOpenGraphNodeMonitor = useCallback((monitorTarget: GraphNode) => {
    if (activeJobHasMonitorSource) {
      return true;
    }
    return anyLogRunTagsMatchNodePath(
      logRunTagsQuery.data?.runs,
      filteredHistoricalRunIds,
      monitorTarget.path,
    );
  }, [activeJobHasMonitorSource, filteredHistoricalRunIds, logRunTagsQuery.data]);
  const graphState = useGraphViewState(graph, {
    canOpenMonitor: canOpenGraphNodeMonitor,
    onOpenMonitor: setGraphMonitorNode,
    resolveMonitorTarget: linearMonitorTargetResolver,
  });
  const resetGraphSelectionAndExpansion = graphState.resetGraphSelectionAndExpansion;

  // Selection cascade: models load → first model auto-selected → presets/datasets
  // load → first preset + dataset auto-selected and the initial preview is
  // requested → dataset/monitor lists are pruned to what the model supports.
  useEffect(() => {
    if (!selectedModel && modelsQuery.data?.models.length) {
      setSelectedModel(modelsQuery.data.models[0]);
    }
  }, [modelsQuery.data, selectedModel, setSelectedModel]);

  useEffect(() => {
    if (!selectedModel) {
      setSelectedHistoricalExperiment("");
      setSelectedHistoricalDataset("");
      setSelectedLogRunId(null);
      return;
    }
    setSelectedHistoricalExperiment((current) => {
      if (
        current &&
        historicalExperimentOptions.some((option) => option.value === current)
      ) {
        return current;
      }
      return historicalExperimentOptions[0]?.value ?? "";
    });
  }, [historicalExperimentOptions, selectedModel]);

  useEffect(() => {
    setSelectedHistoricalDataset((current) => {
      if (
        current &&
        historicalDatasetOptions.some((option) => option.value === current)
      ) {
        return current;
      }
      return historicalDatasetOptions[0]?.value ?? "";
    });
  }, [historicalDatasetOptions]);

  useEffect(() => {
    if (!selectedModel) {
      setSelectedLogRunId(null);
      return;
    }
    setSelectedLogRunId((current) => {
      if (current && filteredHistoricalRuns.some((run) => run.id === current)) {
        return current;
      }
      return filteredHistoricalRuns[0]?.id ?? null;
    });
  }, [filteredHistoricalRuns, selectedModel]);

  useEffect(() => {
    const firstPreset = presetsQuery.data?.presets[0]?.name;
    const firstDataset = datasetNames[0];
    if (firstPreset && firstDataset && !selectedPreset) {
      setSelectedPreset(firstPreset);
      setSelectedTrainingPresets([firstPreset]);
      setOverrides({});
      requestPreview({
        model: selectedModel,
        preset: firstPreset,
        dataset: firstDataset,
        overrides: {},
      });
    }
  }, [
    datasetNames,
    presetsQuery.data,
    requestPreview,
    selectedModel,
    selectedPreset,
    setOverrides,
    setSelectedPreset,
  ]);

  useEffect(() => {
    setTrainingSearch(DEFAULT_TRAINING_SEARCH_STATE);
  }, [selectedModel, selectedPreset]);

  useEffect(() => {
    if (configSnapshots.length > 0) {
      setTrainingSearch(DEFAULT_TRAINING_SEARCH_STATE);
    }
  }, [configSnapshots.length]);

  useEffect(() => {
    if (!selectedPreset || presetNames.length === 0) {
      setSelectedTrainingPresets([]);
      return;
    }
    setSelectedTrainingPresets((current) => {
      return normalizeTrainingPresetSelection(current, presetNames, selectedPreset);
    });
  }, [presetNames, selectedPreset]);

  useEffect(() => {
    setConfigSnapshots((current) =>
      current.filter(
        (snapshot) =>
          snapshot.model === selectedModel &&
          selectedTrainingPresets.includes(snapshot.preset),
      ),
    );
  }, [selectedModel, selectedTrainingPresets]);

  useEffect(() => {
    if (!selectedModel || !selectedLogRun) {
      return;
    }
    const preset = resolveRunPresetName(
      selectedLogRun,
      presetsQuery.data?.presets ?? [],
    );
    const dataset = datasetNames.includes(selectedLogRun.dataset)
      ? selectedLogRun.dataset
      : "";
    if (!preset || !dataset) {
      return;
    }
    setSelectedPreset(preset);
    setSelectedTrainingPresets([preset]);
    setSelectedDatasets([dataset]);
    setOverrides({});
    resetGraphSelectionAndExpansion();
    requestPreview({
      model: selectedModel,
      preset,
      dataset,
      overrides: {},
    });
  }, [
    datasetNames,
    presetsQuery.data,
    requestPreview,
    resetGraphSelectionAndExpansion,
    selectedLogRun,
    selectedModel,
    setOverrides,
    setSelectedPreset,
  ]);

  useEffect(() => {
    if (datasetNames.length === 0) {
      setSelectedDatasets([]);
      return;
    }
    setSelectedDatasets((current) => {
      return normalizeDatasetSelection(current, datasetNames);
    });
  }, [datasetNames]);

  useEffect(() => {
    const monitorNames = (monitorsQuery.data?.monitors ?? []).map((monitor) => monitor.name);
    setSelectedMonitors((current) =>
      current.filter((monitor) => monitorNames.includes(monitor)),
    );
  }, [monitorsQuery.data]);

  useLockedOverrideSync(schemaQuery.data, setOverrides);

  const selectedPresetMeta = presetsQuery.data?.presets.find(
    (preset) => preset.name === selectedPreset,
  );
  const configSections = useMemo(() => {
    const groups = new Map<string, ConfigField[]>();
    for (const field of schemaQuery.data?.fields ?? []) {
      const section = field.section || "General";
      groups.set(section, [...(groups.get(section) ?? []), field]);
    }
    return Array.from(groups, ([title, fields]) => ({ title, fields }));
  }, [schemaQuery.data]);
  const configFields = useMemo(
    () => configSections.flatMap((section) => section.fields),
    [configSections],
  );
  const visibleConfigSnapshots = useMemo(
    () =>
      selectedConfigSnapshots(
        configSnapshots,
        selectedModel,
        selectedTrainingPresets,
      ),
    [configSnapshots, selectedModel, selectedTrainingPresets],
  );
  const configSnapshotGroups = useMemo(
    () =>
      groupConfigSnapshotsByPreset(
        visibleConfigSnapshots,
        selectedTrainingPresets,
      ),
    [selectedTrainingPresets, visibleConfigSnapshots],
  );

  const selectModel = useCallback(
    (model: string) => {
      targetState.selectModel(model);
      setSelectedDatasets([]);
      setSelectedTrainingPresets([]);
      setSelectedMonitors([]);
      setConfigSnapshots([]);
      setSelectedHistoricalExperiment("");
      setSelectedHistoricalDataset("");
      setSelectedLogRunId(null);
      graphState.resetGraphSelectionAndExpansion();
    },
    [graphState, targetState],
  );

  const addConfigSnapshot = useCallback(
    (name: string): ConfigSnapshotCreateResult => {
      const result = createConfigSnapshot({
        id: createSnapshotId(),
        name,
        model: selectedModel,
        preset: selectedPreset,
        fields: configFields,
        overrides,
        snapshots: configSnapshots,
        createdAt: new Date().toISOString(),
      });
      if (result.ok) {
        setConfigSnapshots((current) => [...current, result.snapshot]);
      }
      return result;
    },
    [configFields, configSnapshots, overrides, selectedModel, selectedPreset],
  );

  const removeConfigSnapshot = useCallback((snapshotId: string) => {
    setConfigSnapshots((current) =>
      current.filter((snapshot) => snapshot.id !== snapshotId),
    );
  }, []);

  const renameConfigSnapshot = useCallback((snapshotId: string, name: string) => {
    const nextName = name.trim();
    setConfigSnapshots((current) =>
      current.map((snapshot) =>
        snapshot.id === snapshotId && nextName
          ? { ...snapshot, name: nextName }
          : snapshot,
      ),
    );
  }, []);

  const loadConfigSnapshot = useCallback(
    (snapshotId: string) => {
      const snapshot = configSnapshots.find((candidate) => candidate.id === snapshotId);
      if (!snapshot || snapshot.model !== selectedModel) {
        return false;
      }
      setSelectedPreset(snapshot.preset);
      setSelectedTrainingPresets((current) =>
        normalizeTrainingPresetSelection(
          [...current, snapshot.preset],
          presetNames,
          snapshot.preset,
        ),
      );
      setOverrides({ ...snapshot.overrides });
      return true;
    },
    [configSnapshots, presetNames, selectedModel, setOverrides, setSelectedPreset],
  );

  const selectTrainingPrimaryPreset = useCallback(
    (preset: string) => {
      selectPreset(preset);
      setSelectedTrainingPresets([preset]);
    },
    [selectPreset],
  );

  const setTrainingPresetSelection = useCallback(
    (presets: string[]) => {
      const validPresets = uniqueValidValues(presets, presetNames);
      const fallbackPreset =
        selectedPreset && presetNames.includes(selectedPreset)
          ? selectedPreset
          : presetNames[0] ?? "";
      const nextPrimary = validPresets.includes(selectedPreset)
        ? selectedPreset
        : validPresets[0] ?? fallbackPreset;

      if (nextPrimary && nextPrimary !== selectedPreset) {
        selectPreset(nextPrimary);
      }

      setSelectedTrainingPresets(
        normalizeTrainingPresetSelection(
          validPresets.length > 0 ? validPresets : fallbackPreset ? [fallbackPreset] : [],
          presetNames,
          nextPrimary,
        ),
      );
    },
    [presetNames, selectPreset, selectedPreset],
  );

  const toggleTrainingPreset = useCallback(
    (preset: string) => {
      const next = selectedTrainingPresets.includes(preset)
        ? selectedTrainingPresets.filter((item) => item !== preset)
        : [...selectedTrainingPresets, preset];
      setTrainingPresetSelection(next);
    },
    [selectedTrainingPresets, setTrainingPresetSelection],
  );

  const makeTrainingPresetPrimary = useCallback(
    (preset: string) => {
      if (!presetNames.includes(preset)) {
        return;
      }
      selectPreset(preset);
      setSelectedTrainingPresets((current) =>
        normalizeTrainingPresetSelection([...current, preset], presetNames, preset),
      );
    },
    [presetNames, selectPreset],
  );

  const selectAllTrainingPresets = useCallback(() => {
    if (!selectedPreset) {
      setSelectedTrainingPresets([]);
      return;
    }
    setSelectedTrainingPresets([
      selectedPreset,
      ...presetNames.filter((preset) => preset !== selectedPreset),
    ]);
  }, [presetNames, selectedPreset]);

  const selectPrimaryTrainingPreset = useCallback(() => {
    setSelectedTrainingPresets(selectedPreset ? [selectedPreset] : []);
  }, [selectedPreset]);

  const setDatasetSelection = useCallback(
    (datasets: string[]) => {
      setSelectedDatasets((current) =>
        normalizeDatasetSelection(datasets, datasetNames, current),
      );
    },
    [datasetNames],
  );

  const toggleDataset = useCallback((dataset: string) => {
    setSelectedDatasets((current) => {
      const next = current.includes(dataset)
        ? current.filter((item) => item !== dataset)
        : [...current, dataset];
      return normalizeDatasetSelection(next, datasetNames, current);
    });
  }, [datasetNames]);

  const selectAllDatasets = useCallback(() => {
    setSelectedDatasets(datasetNames);
  }, [datasetNames]);

  const selectFirstDataset = useCallback(() => {
    setSelectedDatasets(datasetNames[0] ? [datasetNames[0]] : []);
  }, [datasetNames]);

  const toggleMonitor = useCallback((monitor: string) => {
    setSelectedMonitors((current) =>
      current.includes(monitor)
        ? current.filter((item) => item !== monitor)
        : [...current, monitor],
    );
  }, []);

  const selectLogRun = useCallback((runId: string) => {
    setSelectedLogRunId(runId);
  }, []);

  const updatePreview = useCallback(() => {
    const previewDataset = selectedDatasets[0];
    if (!selectedModel || !selectedPreset || !previewDataset) {
      return;
    }
    graphState.resetGraphSelectionAndExpansion();
    requestPreview({
      model: selectedModel,
      preset: selectedPreset,
      dataset: previewDataset,
      overrides: { ...overrides },
    });
  }, [graphState, overrides, requestPreview, selectedDatasets, selectedModel, selectedPreset]);

  const resetOverrides = useCallback(() => {
    clearOverrides();
    graphState.resetGraphExpansion();
    const previewDataset = selectedDatasets[0];
    if (selectedModel && selectedPreset && previewDataset) {
      requestPreview({
        model: selectedModel,
        preset: selectedPreset,
        dataset: previewDataset,
        overrides: {},
      });
    }
  }, [clearOverrides, graphState, requestPreview, selectedDatasets, selectedModel, selectedPreset]);

  const handleTrainingJobChange = useCallback(
    (job: TrainingJob | undefined) => {
      setActiveTrainingJob(job);
      if (job?.logFolder) {
        onJobStarted?.(job.logFolder);
      }
    },
    [onJobStarted],
  );

  const closeGraphNodeMonitor = useCallback(() => {
    setGraphMonitorNode(undefined);
  }, []);

  const apiOnline = healthQuery.data?.status === "ok";
  const overrideCount = Object.keys(overrides).length;
  const presetOwnedFieldCount = configSections.reduce(
    (total, section) => total + presetOwnedCount(section.fields),
    0,
  );
  const fieldCount = configSections.reduce((total, section) => total + section.fields.length, 0);
  const selectedMonitorNode = linearMonitorTargetResolver(graphState.selectedNode);
  const selectedMonitorComparisonCandidateGroups = useMemo(
    () => buildLinearMonitorComparisonCandidateGroups(graph, selectedMonitorNode),
    [graph, selectedMonitorNode],
  );
  const selectedLogRunHasMonitorTags = anyLogRunTagsMatchNodePath(
    logRunTagsQuery.data?.runs,
    filteredHistoricalRunIds,
    selectedMonitorNode?.path,
  );

  const graphMonitorComparisonCandidateGroups = useMemo(
    () => buildLinearMonitorComparisonCandidateGroups(graph, graphMonitorNode),
    [graph, graphMonitorNode],
  );
  const graphMonitorSource = useMemo<MonitorChartsSource | undefined>(() => {
    const activeLinearTrainingJob = activeTrainingJob?.monitors.includes("linear")
      ? activeTrainingJob
      : undefined;
    if (activeLinearTrainingJob) {
      return { kind: "active-job", job: activeLinearTrainingJob };
    }
    return historicalMonitorRuns.length > 0
      ? {
          kind: "historical-run-group",
          runs: historicalMonitorRuns,
          experiment: selectedHistoricalExperiment,
          dataset: selectedHistoricalDataset,
        }
      : undefined;
  }, [
    activeTrainingJob,
    historicalMonitorRuns,
    selectedHistoricalDataset,
    selectedHistoricalExperiment,
  ]);

  return {
    target: {
      selectedModel,
      selectModel,
      selectedPreset,
      selectPreset: selectTrainingPrimaryPreset,
      selectedPresetMeta,
      selectedTrainingPresets,
      setTrainingPresetSelection,
      toggleTrainingPreset,
      makeTrainingPresetPrimary,
      selectAllTrainingPresets,
      selectPrimaryTrainingPreset,
      selectedDatasets,
      setDatasetSelection,
      toggleDataset,
      selectAllDatasets,
      selectFirstDataset,
      selectedMonitors,
      toggleMonitor,
      overrides: overrides as OverrideValues,
      configSections,
      overrideCount,
      presetOwnedFieldCount,
      fieldCount,
      apiOnline,
      trainingSearch,
      setTrainingSearch,
      configSnapshots: visibleConfigSnapshots,
      configSnapshotGroups,
      configSnapshotCount: visibleConfigSnapshots.length,
      addConfigSnapshot,
      removeConfigSnapshot,
      renameConfigSnapshot,
      loadConfigSnapshot,
      updateOverride,
      clearOverride,
      updatePreview,
      resetOverrides,
      modelsQuery,
      presetsQuery,
      datasetsQuery,
      monitorsQuery,
      schemaQuery,
      searchSpaceQuery,
    },
    graph: {
      graph,
      graphForDetail: graphState.graphForDetail,
      nodes: graphState.nodes,
      edges: graphState.edges,
      graphDetailMode: graphState.graphDetailMode,
      setGraphDetailMode: graphState.setGraphDetailMode,
      graphScope: graphState.graphScope,
      setGraphScope: graphState.setGraphScope,
      expandedGraphNodeIds: graphState.expandedGraphNodeIds,
      selectedNodeId: graphState.selectedNodeId,
      setSelectedNodeId: graphState.setSelectedNodeId,
      selectedNode: graphState.selectedNode,
      selectedMonitorNode,
      selectedMonitorComparisonCandidateGroups,
      collapseGraphNodes: graphState.collapseGraphNodes,
      revealGraphNode: graphState.revealGraphNode,
      previewInspection,
    },
    history: {
      filteredHistoricalRuns,
      historicalMonitorRuns,
      historicalExperimentOptions,
      historicalDatasetOptions,
      selectedHistoricalExperiment,
      setSelectedHistoricalExperiment,
      selectedHistoricalDataset,
      setSelectedHistoricalDataset,
      selectedLogRunId,
      selectLogRun,
      logRunsQuery,
      logRunTagsQuery,
      selectedLogRunHasMonitorTags,
    },
    training: {
      activeJobId,
      setActiveJobId,
      activeTrainingJob,
      onJobChange: handleTrainingJobChange,
      graphMonitorNode,
      openGraphNodeMonitor: setGraphMonitorNode,
      closeGraphNodeMonitor,
      graphMonitorSource,
      graphMonitorComparisonCandidateGroups,
    },
  };
}

export type ViewerState = ReturnType<typeof useViewerState>;
export type TargetConfigContextValue = ViewerState["target"];
export type GraphViewContextValue = ViewerState["graph"];
export type HistoricalRunsContextValue = ViewerState["history"];
export type TrainingContextValue = ViewerState["training"];
