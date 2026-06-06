import { useCallback, useEffect, useMemo, useState } from "react";
import {
  type GraphNode,
  type LogRun,
  type TrainingJob,
} from "@/lib/api";
import { useLogRunsQuery, useLogTagsQuery } from "@/hooks/use-log-queries";
import { type OverrideValues } from "@/lib/config";
import {
  DEFAULT_TRAINING_SEARCH_STATE,
  type TrainingSearchState,
} from "@/lib/training-search";
import {
  createConfigSnapshot,
  type ConfigSnapshot,
  type ConfigSnapshotCreateResult,
} from "@/lib/config-snapshots";
import {
  normalizePrimarySelection,
  normalizeSelection,
  selectionValuesEqual,
  uniqueValidValues,
} from "@/lib/selection";
import { anyLogRunTagsMatchNodePath } from "@/lib/historical-monitor-runs";
import { useGraphViewState } from "@/components/features/viewer/state/use-graph-view-state";
import { useLockedOverrideSync } from "@/components/features/viewer/state/use-locked-override-sync";
import { usePreviewInspectionState } from "@/components/features/viewer/state/use-preview-inspection";
import { useTargetOverridesState } from "@/components/features/viewer/state/use-target-overrides";
import {
  LOCAL_DEFAULT_CAPABILITIES,
  useViewerQueries,
} from "@/components/features/viewer/state/use-viewer-queries";
import {
  deriveDatasetSelectionState,
  deriveMonitorSource,
  deriveTargetSelectionState,
} from "@/components/features/viewer/state/viewer-state-selectors";
import { logQueryKeys } from "@/lib/query-keys";

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

function overridesAreEmpty(overrides: OverrideValues) {
  return Object.keys(overrides).length === 0;
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
    selectModel: selectTargetModel,
  } = useTargetOverridesState();
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [selectedTrainingPresets, setSelectedTrainingPresets] = useState<string[]>([]);
  const [selectedMonitors, setSelectedMonitors] = useState<string[]>([]);
  const [configSnapshots, setConfigSnapshots] = useState<ConfigSnapshot[]>([]);
  const [trainingSearch, setTrainingSearch] = useState<TrainingSearchState>(
    DEFAULT_TRAINING_SEARCH_STATE,
  );
  const {
    healthQuery,
    capabilitiesQuery,
    modelsQuery,
    presetsQuery,
    datasetsQuery,
    monitorsQuery,
    schemaQuery,
    searchSpaceQuery,
  } = useViewerQueries(selectedModel, selectedPreset);
  const capabilities = capabilitiesQuery.data ?? LOCAL_DEFAULT_CAPABILITIES;

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

  const targetSelectionState = useMemo(
    () =>
      deriveTargetSelectionState({
        datasets: datasetsQuery.data?.datasets,
        presets: presetsQuery.data?.presets,
        schemaFields: schemaQuery.data?.fields,
        configSnapshots,
        selectedModel,
        selectedPreset,
        selectedTrainingPresets,
        overrides: overrides as OverrideValues,
      }),
    [
      configSnapshots,
      datasetsQuery.data?.datasets,
      overrides,
      presetsQuery.data?.presets,
      schemaQuery.data?.fields,
      selectedModel,
      selectedPreset,
      selectedTrainingPresets,
    ],
  );
  const {
    datasetNames,
    presetNames,
    selectedPresetMeta,
    configSections,
    configFields,
    visibleConfigSnapshots,
    configSnapshotGroups,
    overrideCount,
    presetOwnedFieldCount,
    fieldCount,
  } = targetSelectionState;

  const datasetSelectionState = useMemo(
    () =>
      deriveDatasetSelectionState({
        logRuns: logRunsQuery.data?.runs,
        selectedModel,
        selectedHistoricalExperiment,
        selectedHistoricalDataset,
        selectedLogRunId,
      }),
    [
      logRunsQuery.data?.runs,
      selectedHistoricalDataset,
      selectedHistoricalExperiment,
      selectedLogRunId,
      selectedModel,
    ],
  );
  const {
    historicalExperimentOptions,
    historicalDatasetOptions,
    filteredHistoricalRuns,
    historicalMonitorRuns,
    filteredHistoricalRunIds,
    selectedLogRun,
  } = datasetSelectionState;
  const logRunTagsQuery = useLogTagsQuery({
    runIds: filteredHistoricalRunIds,
    queryKey: logQueryKeys.filteredHistoricalRunTags(filteredHistoricalRunIds),
  });
  const monitorSourceBeforeGraph = useMemo(
    () =>
      deriveMonitorSource({
        graph,
        activeTrainingJob,
      }),
    [activeTrainingJob, graph],
  );
  const { activeJobHasMonitorSource, linearMonitorTargetResolver } =
    monitorSourceBeforeGraph;
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
  const resetGraphExpansion = graphState.resetGraphExpansion;

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
      setSelectedTrainingPresets((current) =>
        current.length === 0 ? current : [],
      );
      return;
    }
    setSelectedTrainingPresets((current) => {
      const next = normalizePrimarySelection(
        current,
        presetNames,
        selectedPreset || undefined,
      );
      return selectionValuesEqual(current, next) ? current : next;
    });
  }, [presetNames, selectedPreset]);

  useEffect(() => {
    setConfigSnapshots((current) => {
      const next = current.filter(
        (snapshot) =>
          snapshot.model === selectedModel &&
          selectedTrainingPresets.includes(snapshot.preset),
      );
      return next.length === current.length ? current : next;
    });
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
    const desiredTrainingPresets = [preset];
    const desiredDatasets = [dataset];
    const overridesAlreadyEmpty = overridesAreEmpty(overrides);
    const alreadySynced =
      selectedPreset === preset &&
      selectionValuesEqual(selectedTrainingPresets, desiredTrainingPresets) &&
      selectionValuesEqual(selectedDatasets, desiredDatasets) &&
      overridesAlreadyEmpty;
    if (alreadySynced) {
      return;
    }

    if (selectedPreset !== preset) {
      setSelectedPreset(preset);
    }
    setSelectedTrainingPresets((current) =>
      selectionValuesEqual(current, desiredTrainingPresets)
        ? current
        : desiredTrainingPresets,
    );
    setSelectedDatasets((current) =>
      selectionValuesEqual(current, desiredDatasets) ? current : desiredDatasets,
    );
    if (!overridesAlreadyEmpty) {
      setOverrides({});
    }
    resetGraphSelectionAndExpansion();
    requestPreview({
      model: selectedModel,
      preset,
      dataset,
      overrides: {},
    });
  }, [
    datasetNames,
    overrides,
    presetsQuery.data,
    requestPreview,
    resetGraphSelectionAndExpansion,
    selectedDatasets,
    selectedLogRun,
    selectedModel,
    selectedPreset,
    selectedTrainingPresets,
    setOverrides,
    setSelectedPreset,
  ]);

  useEffect(() => {
    if (datasetNames.length === 0) {
      setSelectedDatasets((current) => (current.length === 0 ? current : []));
      return;
    }
    setSelectedDatasets((current) => {
      const next = normalizeSelection(current, datasetNames);
      return selectionValuesEqual(current, next) ? current : next;
    });
  }, [datasetNames]);

  useEffect(() => {
    const monitorNames = (monitorsQuery.data?.monitors ?? []).map((monitor) => monitor.name);
    setSelectedMonitors((current) => {
      const next = current.filter((monitor) => monitorNames.includes(monitor));
      return next.length === current.length ? current : next;
    });
  }, [monitorsQuery.data]);

  useLockedOverrideSync(schemaQuery.data, setOverrides);

  const selectModel = useCallback(
    (model: string) => {
      selectTargetModel(model);
      setSelectedDatasets([]);
      setSelectedTrainingPresets([]);
      setSelectedMonitors([]);
      setConfigSnapshots([]);
      setSelectedHistoricalExperiment("");
      setSelectedHistoricalDataset("");
      setSelectedLogRunId(null);
      resetGraphSelectionAndExpansion();
    },
    [resetGraphSelectionAndExpansion, selectTargetModel],
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
        normalizePrimarySelection(
          [...current, snapshot.preset],
          presetNames,
          snapshot.preset || undefined,
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
        normalizePrimarySelection(
          validPresets,
          presetNames,
          nextPrimary || undefined,
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
        normalizePrimarySelection(
          [...current, preset],
          presetNames,
          preset || undefined,
        ),
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
        normalizeSelection(datasets, datasetNames, current),
      );
    },
    [datasetNames],
  );

  const toggleDataset = useCallback((dataset: string) => {
    setSelectedDatasets((current) => {
      const next = current.includes(dataset)
        ? current.filter((item) => item !== dataset)
        : [...current, dataset];
      return normalizeSelection(next, datasetNames, current);
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
    resetGraphSelectionAndExpansion();
    requestPreview({
      model: selectedModel,
      preset: selectedPreset,
      dataset: previewDataset,
      overrides: { ...overrides },
    });
  }, [
    overrides,
    requestPreview,
    resetGraphSelectionAndExpansion,
    selectedDatasets,
    selectedModel,
    selectedPreset,
  ]);

  const resetOverrides = useCallback(() => {
    clearOverrides();
    resetGraphExpansion();
    const previewDataset = selectedDatasets[0];
    if (selectedModel && selectedPreset && previewDataset) {
      requestPreview({
        model: selectedModel,
        preset: selectedPreset,
        dataset: previewDataset,
        overrides: {},
      });
    }
  }, [
    clearOverrides,
    requestPreview,
    resetGraphExpansion,
    selectedDatasets,
    selectedModel,
    selectedPreset,
  ]);

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
  const monitorSourceState = useMemo(
    () =>
      deriveMonitorSource({
        graph,
        selectedNode: graphState.selectedNode,
        graphMonitorNode,
        activeTrainingJob,
        historicalMonitorRuns,
        selectedHistoricalExperiment,
        selectedHistoricalDataset,
        logRunTags: logRunTagsQuery.data?.runs,
        filteredHistoricalRunIds,
        linearMonitorTargetResolver,
      }),
    [
      activeTrainingJob,
      filteredHistoricalRunIds,
      graph,
      graphMonitorNode,
      graphState.selectedNode,
      historicalMonitorRuns,
      linearMonitorTargetResolver,
      logRunTagsQuery.data?.runs,
      selectedHistoricalDataset,
      selectedHistoricalExperiment,
    ],
  );
  const {
    selectedMonitorNode,
    selectedMonitorComparisonCandidateGroups,
    selectedLogRunHasMonitorTags,
    graphMonitorComparisonCandidateGroups,
    graphMonitorSource,
  } = monitorSourceState;

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
      capabilities,
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
      capabilitiesQuery,
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
