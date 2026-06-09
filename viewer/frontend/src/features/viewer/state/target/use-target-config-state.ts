import { useCallback, useEffect, useMemo, useState } from "react";
import { type LogRun } from "@/lib/api";
import { type OverrideValues } from "@/lib/config";
import {
  createConfigSnapshot,
  type ConfigSnapshotCreateResult,
} from "@/lib/config-snapshots";
import {
  resolveRunPresetName,
} from "@/lib/historical-monitor-runs";
import {
  modelsForType,
  modelTypeForId,
  modelTypeOptions,
  normalizePrimarySelection,
  normalizeSelection,
  selectionValuesEqual,
  uniqueValidValues,
} from "@/lib/selection";
import {
  DEFAULT_TRAINING_SEARCH_STATE,
  type TrainingSearchState,
} from "@/lib/training-search";
import {
  type PreviewInspectionRequest,
} from "@/features/viewer/state/graph-monitor/use-preview-inspection";
import { useConfigSnapshots } from "@/features/viewer/state/target/use-config-snapshots";
import { useLockedOverrideSync } from "@/features/viewer/state/target/use-locked-override-sync";
import { useTargetOverridesState } from "@/features/viewer/state/target/use-target-overrides";
import {
  LOCAL_DEFAULT_CAPABILITIES,
  useViewerQueries,
} from "@/features/viewer/state/use-viewer-queries";
import {
  deriveTargetSelectionState,
} from "@/features/viewer/state/target/target-selection";

const EMPTY_MODEL_IDS: string[] = [];

type TargetConfigStateOptions = {
  requestPreview: (request: PreviewInspectionRequest) => void;
  resetGraphSelectionAndExpansion: () => void;
  resetGraphExpansion: () => void;
  onModelSelected?: () => void;
};

function overridesAreEmpty(overrides: OverrideValues) {
  return Object.keys(overrides).length === 0;
}

function createSnapshotId() {
  return globalThis.crypto?.randomUUID?.() ?? `snapshot-${Date.now()}`;
}

export function useTargetConfigState({
  requestPreview,
  resetGraphSelectionAndExpansion,
  resetGraphExpansion,
  onModelSelected,
}: TargetConfigStateOptions) {
  const {
    selectedModel,
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
  const [selectedModelType, setSelectedModelType] = useState("");
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [selectedTrainingPresets, setSelectedTrainingPresets] = useState<string[]>([]);
  const [selectedMonitors, setSelectedMonitors] = useState<string[]>([]);
  const {
    snapshots: configSnapshots,
    createMutation: createSnapshotMutation,
    renameMutation: renameSnapshotMutation,
    deleteMutation: deleteSnapshotMutation,
  } = useConfigSnapshots(selectedModel);
  const [deselectedSnapshotIds, setDeselectedSnapshotIds] = useState<string[]>([]);
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
  const catalogModels = modelsQuery.data?.models ?? EMPTY_MODEL_IDS;
  const availableModelTypeOptions = useMemo(
    () => modelTypeOptions(catalogModels),
    [catalogModels],
  );

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

  const selectModel = useCallback(
    (model: string) => {
      setSelectedModelType(model ? modelTypeForId(model) : "");
      selectTargetModel(model);
      setSelectedDatasets([]);
      setSelectedTrainingPresets([]);
      setSelectedMonitors([]);
      onModelSelected?.();
      resetGraphSelectionAndExpansion();
    },
    [onModelSelected, resetGraphSelectionAndExpansion, selectTargetModel],
  );

  const selectModelType = useCallback(
    (modelType: string) => {
      setSelectedModelType(modelType);
      const firstModel = modelsForType(catalogModels, modelType)[0] ?? "";
      if (firstModel && firstModel !== selectedModel) {
        selectModel(firstModel);
      }
    },
    [catalogModels, selectModel, selectedModel],
  );

  // Selection cascade: model types/models load -> first type/model auto-selected -> presets/datasets
  // load -> first preset + dataset auto-selected and the initial preview is
  // requested -> dataset/monitor lists are pruned to what the model supports.
  useEffect(() => {
    if (catalogModels.length === 0) {
      if (selectedModelType) {
        setSelectedModelType("");
      }
      return;
    }

    const availableModelTypes = availableModelTypeOptions.map((option) => option.value);
    const selectedTypeIsValid =
      selectedModelType.length > 0 && availableModelTypes.includes(selectedModelType);
    const nextModelType = selectedTypeIsValid
      ? selectedModelType
      : availableModelTypes[0] ?? "";
    if (!nextModelType) {
      return;
    }
    if (nextModelType !== selectedModelType) {
      setSelectedModelType(nextModelType);
    }

    const modelsInSelectedType = modelsForType(catalogModels, nextModelType);
    if (!selectedModel || !modelsInSelectedType.includes(selectedModel)) {
      const firstModel = modelsInSelectedType[0];
      if (firstModel) {
        selectModel(firstModel);
      }
    }
  }, [
    availableModelTypeOptions,
    catalogModels,
    selectModel,
    selectedModel,
    selectedModelType,
  ]);

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

  const syncSelectedLogRun = useCallback(
    (selectedLogRun: LogRun) => {
      if (!selectedModel) {
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
    },
    [
      datasetNames,
      overrides,
      presetsQuery.data,
      requestPreview,
      resetGraphSelectionAndExpansion,
      selectedDatasets,
      selectedModel,
      selectedPreset,
      selectedTrainingPresets,
      setOverrides,
      setSelectedPreset,
    ],
  );

  const addConfigSnapshot = useCallback(
    (name: string): ConfigSnapshotCreateResult => {
      // Validate client-side for instant dialog feedback; the server re-validates
      // and is the source of truth. The client-generated id is discarded: the
      // persisted snapshot, with its server id, arrives via query invalidation.
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
        createSnapshotMutation.mutate({
          model: selectedModel,
          preset: selectedPreset,
          name: result.snapshot.name,
          overrides: result.snapshot.overrides,
        });
      }
      return result;
    },
    [
      configFields,
      configSnapshots,
      createSnapshotMutation,
      overrides,
      selectedModel,
      selectedPreset,
    ],
  );

  const removeConfigSnapshot = useCallback(
    (snapshotId: string) => {
      deleteSnapshotMutation.mutate(snapshotId);
    },
    [deleteSnapshotMutation],
  );

  const renameConfigSnapshot = useCallback(
    (snapshotId: string, name: string) => {
      const nextName = name.trim();
      if (!nextName) {
        return;
      }
      renameSnapshotMutation.mutate({ id: snapshotId, name: nextName });
    },
    [renameSnapshotMutation],
  );

  const toggleConfigSnapshotRunSelection = useCallback((snapshotId: string) => {
    setDeselectedSnapshotIds((current) =>
      current.includes(snapshotId)
        ? current.filter((id) => id !== snapshotId)
        : [...current, snapshotId],
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

  const toggleDataset = useCallback(
    (dataset: string) => {
      setSelectedDatasets((current) => {
        const next = current.includes(dataset)
          ? current.filter((item) => item !== dataset)
          : [...current, dataset];
        return normalizeSelection(next, datasetNames, current);
      });
    },
    [datasetNames],
  );

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

  const apiOnline = healthQuery.data?.status === "ok";

  return {
    target: {
      selectedModelType,
      selectModelType,
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
      deselectedSnapshotIds,
      addConfigSnapshot,
      removeConfigSnapshot,
      renameConfigSnapshot,
      loadConfigSnapshot,
      toggleConfigSnapshotRunSelection,
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
    selection: {
      selectedModel,
      selectedPreset,
      selectedDatasets,
    },
    queries: {
      presetsQuery,
    },
    syncSelectedLogRun,
  };
}
