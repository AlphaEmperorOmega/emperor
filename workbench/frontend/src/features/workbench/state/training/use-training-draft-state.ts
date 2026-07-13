import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type ModelIdentity,
} from "@/lib/api";
import {
  effectivePresetOverrides,
  lockedOverrideKeys,
  runtimeDefaultsEditor,
  type OverrideValues,
} from "@/lib/config";
import {
  modelNameForId,
  modelTypeForId,
  modelTypeOptions as createModelTypeOptions,
  modelsForType,
  normalizePrimarySelection,
  normalizeSelection,
  selectionValuesEqual,
  uniqueValidValues,
} from "@/lib/selection";
import {
  DEFAULT_TRAINING_SEARCH_STATE,
  type TrainingSearchState,
} from "@/lib/training-search-state";
import {
  useConfigSnapshotRecords,
} from "@/features/workbench/state/config-snapshots/use-config-snapshot-records";
import {
  deriveModelPackageSelection,
  datasetsForExperimentTask,
  experimentTaskOptions,
  normalizeExperimentTask,
} from "@/features/workbench/state/model-package/model-package-selection";
import { useModelPackageMetadata } from "@/features/workbench/state/model-package/use-model-package-metadata";
import { type WorkbenchWorkspace } from "@/types/workbench";

export type TrainingDraftSeed = {
  modelType: string;
  model: string;
  preset: string;
};

type TrainingDraftStateOptions = {
  activeWorkspace: WorkbenchWorkspace;
  models: ModelIdentity[];
  seed: TrainingDraftSeed;
  protectedReadsEnabled?: boolean;
};

export function useTrainingDraftState({
  activeWorkspace,
  models,
  seed,
  protectedReadsEnabled = true,
}: TrainingDraftStateOptions) {
  const [isInitialized, setIsInitialized] = useState(false);
  const [selectedModelType, setSelectedModelType] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedPrimaryPreset, setSelectedPrimaryPreset] = useState("");
  const [selectedPresets, setSelectedPresets] = useState<string[]>([]);
  const [selectedExperimentTask, setSelectedExperimentTask] = useState("");
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [selectedMonitors, setSelectedMonitors] = useState<string[]>([]);
  const [bulkOverrides, setBulkOverrides] = useState<OverrideValues>({});
  const [selectedSnapshotIds, setSelectedSnapshotIds] = useState<string[]>([]);
  const [search, setSearch] = useState<TrainingSearchState>(
    DEFAULT_TRAINING_SEARCH_STATE,
  );
  const allowEmptyPresetDraftRef = useRef(false);

  useEffect(() => {
    if (
      isInitialized ||
      activeWorkspace !== "training" ||
      !seed.modelType ||
      !seed.model ||
      !seed.preset
    ) {
      return;
    }
    setIsInitialized(true);
    setSelectedModelType(seed.modelType);
    setSelectedModel(seed.model);
    setSelectedPrimaryPreset(seed.preset);
    setSelectedPresets([seed.preset]);
  }, [activeWorkspace, isInitialized, seed.model, seed.modelType, seed.preset]);

  const selectedIdentity = useMemo(
    () => ({ modelType: selectedModelType, model: selectedModel }),
    [selectedModel, selectedModelType],
  );
  const metadataSelection = useMemo(
    () => ({
      modelPackage: selectedIdentity,
      preset: selectedPrimaryPreset,
      searchPresets: selectedPresets,
    }),
    [selectedIdentity, selectedPresets, selectedPrimaryPreset],
  );
  const metadata = useModelPackageMetadata(metadataSelection, {
    includeSearchMetadata: activeWorkspace === "training",
    protectedReadsEnabled,
  });
  const {
    records: configSnapshots,
    status: configSnapshotsStatus,
    actions: configSnapshotActions,
  } = useConfigSnapshotRecords(selectedIdentity, {
    enabled: protectedReadsEnabled,
  });
  const deleteSnapshotRecord = configSnapshotActions.remove;
  const retrySnapshotRecordMutation = configSnapshotActions.retry;
  const dismissSnapshotRecordMutation = configSnapshotActions.dismissMutation;
  const clearSnapshotRecordsForConnectionChange =
    configSnapshotActions.clearForConnectionChange;

  const presets = metadata.presets.records;
  const datasetGroups = metadata.datasetMetadata.groups;
  const defaultExperimentTask = metadata.datasetMetadata.defaultExperimentTask;
  const activeExperimentTask = normalizeExperimentTask(
    selectedExperimentTask,
    defaultExperimentTask,
    datasetGroups,
  );
  const experimentTaskOptionList = useMemo(
    () => experimentTaskOptions(datasetGroups),
    [datasetGroups],
  );
  const datasets = datasetsForExperimentTask(
    datasetGroups,
    activeExperimentTask,
  );
  const monitors = metadata.monitorMetadata.records;
  const searchAxes = metadata.searchMetadata.axes;

  const selectionState = useMemo(
    () =>
      deriveModelPackageSelection({
        datasets,
        presets,
        schemaFields: metadata.runtimeDefaults.fields,
        configSnapshots,
        selectedModelType,
        selectedModel,
        selectedPreset: selectedPrimaryPreset,
      }),
    [
      configSnapshots,
      datasets,
      presets,
      metadata.runtimeDefaults.fields,
      selectedModel,
      selectedModelType,
      selectedPrimaryPreset,
    ],
  );
  const {
    datasetNames,
    presetNames,
    configSections,
    configFields,
    fieldCount,
    modelConfigSnapshots,
  } = selectionState;
  const effectiveBulkOverrides = useMemo(
    () => effectivePresetOverrides(configFields, bulkOverrides),
    [bulkOverrides, configFields],
  );
  const inactiveLockedOverrideCount = useMemo(
    () => lockedOverrideKeys(configFields, bulkOverrides).length,
    [bulkOverrides, configFields],
  );

  useEffect(() => {
    if (configFields.length === 0) {
      return;
    }
    setBulkOverrides((current) =>
      runtimeDefaultsEditor.normalize(configFields, current),
    );
  }, [configFields]);

  useEffect(() => {
    setSelectedExperimentTask((current) => {
      const validTasks = datasetGroups.map((group) => group.experimentTask);
      const next = current && !validTasks.includes(current) ? "" : current;
      return current === next ? current : next;
    });
  }, [datasetGroups]);

  useEffect(() => {
    setSearch(DEFAULT_TRAINING_SEARCH_STATE);
  }, [selectedModel, selectedPrimaryPreset]);

  useEffect(() => {
    const selectedSnapshotPreset = selectedSnapshotIds
      .map(
        (snapshotId) =>
          modelConfigSnapshots.find((snapshot) => snapshot.id === snapshotId)
            ?.preset,
      )
      .find((preset): preset is string =>
        Boolean(preset && presetNames.includes(preset)),
      );
    if (presetNames.length === 0) {
      setSelectedPrimaryPreset((current) => (current ? "" : current));
      setSelectedPresets((current) => {
        if (current.length === 0 && allowEmptyPresetDraftRef.current) {
          return current;
        }
        return current.length === 0 ? current : [];
      });
      return;
    }
    if (
      !selectedPrimaryPreset ||
      !presetNames.includes(selectedPrimaryPreset)
    ) {
      setSelectedPrimaryPreset(
        selectedSnapshotPreset ||
          selectedPresets.find((preset) => presetNames.includes(preset)) ||
          presetNames[0] ||
          "",
      );
      return;
    }
    setSelectedPresets((current) => {
      if (current.length === 0 && allowEmptyPresetDraftRef.current) {
        return current;
      }
      const next = normalizePrimarySelection(
        current,
        presetNames,
        selectedPrimaryPreset || undefined,
      );
      return selectionValuesEqual(current, next) ? current : next;
    });
  }, [
    modelConfigSnapshots,
    presetNames,
    selectedPresets,
    selectedPrimaryPreset,
    selectedSnapshotIds,
  ]);

  useEffect(() => {
    const snapshotIds = modelConfigSnapshots.map((snapshot) => snapshot.id);
    setSelectedSnapshotIds((current) => {
      const next = uniqueValidValues(current, snapshotIds);
      return selectionValuesEqual(current, next) ? current : next;
    });
  }, [modelConfigSnapshots]);

  useEffect(() => {
    if (datasetNames.length === 0) {
      setSelectedDatasets((current) =>
        current.length === 0 ? current : [],
      );
      return;
    }
    setSelectedDatasets((current) => {
      const next = normalizeSelection(current, datasetNames);
      return selectionValuesEqual(current, next) ? current : next;
    });
  }, [datasetNames]);

  useEffect(() => {
    const monitorNames = monitors.map((monitor) => monitor.name);
    setSelectedMonitors((current) => {
      const next = current.filter((monitor) => monitorNames.includes(monitor));
      return next.length === current.length ? current : next;
    });
  }, [monitors]);

  const resetSelectionsForModel = useCallback(() => {
    allowEmptyPresetDraftRef.current = false;
    setSelectedPrimaryPreset("");
    setSelectedPresets([]);
    setSelectedSnapshotIds([]);
    setSelectedExperimentTask("");
    setSelectedDatasets([]);
    setSelectedMonitors([]);
    setBulkOverrides({});
    setSearch(DEFAULT_TRAINING_SEARCH_STATE);
  }, []);

  const selectModel = useCallback(
    (model: string, modelType = selectedModelType) => {
      const nextModel = modelNameForId(model);
      const nextModelType = model.includes("/")
        ? modelTypeForId(model)
        : modelType;
      setIsInitialized(true);
      setSelectedModelType(nextModel ? nextModelType : "");
      setSelectedModel(nextModel);
      resetSelectionsForModel();
    },
    [resetSelectionsForModel, selectedModelType],
  );

  const selectModelType = useCallback(
    (modelType: string) => {
      setSelectedModelType(modelType);
      const firstModel = modelsForType(models, modelType)[0];
      if (firstModel) {
        selectModel(firstModel.model, firstModel.modelType);
        return;
      }
      setSelectedModel("");
      resetSelectionsForModel();
    },
    [models, resetSelectionsForModel, selectModel],
  );

  const selectPrimaryPreset = useCallback(
    (preset: string) => {
      if (!presetNames.includes(preset)) {
        return;
      }
      allowEmptyPresetDraftRef.current = false;
      setSelectedPrimaryPreset(preset);
      setSelectedPresets((current) =>
        normalizePrimarySelection(
          current.includes(preset) ? current : [...current, preset],
          presetNames,
          preset,
        ),
      );
    },
    [presetNames],
  );

  const setPresetSelection = useCallback(
    (nextPresets: string[]) => {
      const validPresets = uniqueValidValues(nextPresets, presetNames);
      if (validPresets.length === 0 && selectedSnapshotIds.length > 0) {
        allowEmptyPresetDraftRef.current = true;
        setSelectedPresets([]);
        return;
      }
      const fallbackPreset =
        selectedPrimaryPreset && presetNames.includes(selectedPrimaryPreset)
          ? selectedPrimaryPreset
          : presetNames[0] ?? "";
      const nextPrimary = validPresets.includes(selectedPrimaryPreset)
        ? selectedPrimaryPreset
        : validPresets.length > 0
          ? validPresets[0]
          : fallbackPreset;

      allowEmptyPresetDraftRef.current = false;
      setSelectedPrimaryPreset(nextPrimary);
      setSelectedPresets(
        normalizePrimarySelection(
          validPresets,
          presetNames,
          nextPrimary || undefined,
        ),
      );
    },
    [presetNames, selectedPrimaryPreset, selectedSnapshotIds.length],
  );

  const togglePreset = useCallback(
    (preset: string) => {
      const next = selectedPresets.includes(preset)
        ? selectedPresets.filter((item) => item !== preset)
        : [...selectedPresets, preset];
      setPresetSelection(next);
    },
    [selectedPresets, setPresetSelection],
  );

  const excludeDraftPreset = useCallback(
    (preset: string) => {
      if (!presetNames.includes(preset)) {
        return;
      }
      setSelectedPresets((current) => {
        if (!current.includes(preset)) {
          return current;
        }
        const next = current.filter((item) => item !== preset);
        allowEmptyPresetDraftRef.current = next.length === 0;
        return next;
      });
    },
    [presetNames],
  );

  const makePresetPrimary = useCallback(
    (preset: string) => {
      if (!presetNames.includes(preset)) {
        return;
      }
      allowEmptyPresetDraftRef.current = false;
      setSelectedPrimaryPreset(preset);
      setSelectedPresets((current) =>
        normalizePrimarySelection(
          [...current, preset],
          presetNames,
          preset,
        ),
      );
    },
    [presetNames],
  );

  const selectAllPresets = useCallback(() => {
    allowEmptyPresetDraftRef.current = false;
    if (!selectedPrimaryPreset) {
      setSelectedPresets([]);
      return;
    }
    setSelectedPresets([
      selectedPrimaryPreset,
      ...presetNames.filter((preset) => preset !== selectedPrimaryPreset),
    ]);
  }, [presetNames, selectedPrimaryPreset]);

  const selectOnlyPrimaryPreset = useCallback(() => {
    allowEmptyPresetDraftRef.current = false;
    setSelectedPresets(
      selectedPrimaryPreset ? [selectedPrimaryPreset] : [],
    );
  }, [selectedPrimaryPreset]);

  const setSnapshotSelection = useCallback(
    (snapshotIds: string[]) => {
      const validIds = modelConfigSnapshots.map((snapshot) => snapshot.id);
      setSelectedSnapshotIds(uniqueValidValues(snapshotIds, validIds));
    },
    [modelConfigSnapshots],
  );

  const includeSnapshot = useCallback(
    (snapshotId: string) => {
      if (!modelConfigSnapshots.some((snapshot) => snapshot.id === snapshotId)) {
        return;
      }
      setSelectedSnapshotIds((current) =>
        current.includes(snapshotId) ? current : [...current, snapshotId],
      );
    },
    [modelConfigSnapshots],
  );

  const excludeSnapshot = useCallback(
    (snapshotId: string) => {
      if (!modelConfigSnapshots.some((snapshot) => snapshot.id === snapshotId)) {
        return;
      }
      setSelectedSnapshotIds((current) =>
        current.includes(snapshotId)
          ? current.filter((id) => id !== snapshotId)
          : current,
      );
    },
    [modelConfigSnapshots],
  );

  const removeSnapshot = useCallback(
    async (snapshotId: string) => {
      const outcome = await deleteSnapshotRecord(snapshotId);
      if (outcome.ok) {
        setSelectedSnapshotIds((current) =>
          current.filter((id) => id !== snapshotId),
        );
      }
      return outcome;
    },
    [deleteSnapshotRecord],
  );
  const retrySnapshotMutation = useCallback(async () => {
    const outcome = await retrySnapshotRecordMutation();
    if (outcome?.ok && outcome.kind === "remove" && outcome.snapshotId) {
      setSelectedSnapshotIds((current) =>
        current.filter((id) => id !== outcome.snapshotId),
      );
    }
    return outcome;
  }, [retrySnapshotRecordMutation]);

  const selectExperimentTask = useCallback(
    (experimentTask: string) => {
      const nextTask = normalizeExperimentTask(
        experimentTask,
        defaultExperimentTask,
        datasetGroups,
      );
      const nextDatasetNames = datasetsForExperimentTask(
        datasetGroups,
        nextTask,
      ).map((dataset) => dataset.name);
      setSelectedExperimentTask(nextTask);
      setSelectedDatasets((current) =>
        normalizeSelection(current, nextDatasetNames),
      );
    },
    [datasetGroups, defaultExperimentTask],
  );

  const setDatasetSelection = useCallback(
    (nextDatasets: string[]) => {
      setSelectedDatasets((current) =>
        normalizeSelection(nextDatasets, datasetNames, current),
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

  const setMonitorSelection = useCallback(
    (nextMonitors: string[]) => {
      const monitorNames = monitors.map((monitor) => monitor.name);
      setSelectedMonitors(uniqueValidValues(nextMonitors, monitorNames));
    },
    [monitors],
  );

  const selectAllMonitors = useCallback(() => {
    setSelectedMonitors(monitors.map((monitor) => monitor.name));
  }, [monitors]);

  const clearMonitors = useCallback(() => {
    setSelectedMonitors([]);
  }, []);

  const updateOverride = useCallback(
    (key: string, value: string) => {
      setBulkOverrides((current) =>
        runtimeDefaultsEditor.edit(configFields, current, key, value),
      );
    },
    [configFields],
  );

  const clearOverride = useCallback(
    (key: string) => {
      setBulkOverrides((current) =>
        runtimeDefaultsEditor.clear(configFields, current, key),
      );
    },
    [configFields],
  );

  const resetOverrides = useCallback(() => {
    setBulkOverrides({});
  }, []);

  const updateSearch = useCallback((nextSearch: TrainingSearchState) => {
    setSearch(nextSearch);
  }, []);

  const clearForConnectionChange = useCallback(() => {
    clearSnapshotRecordsForConnectionChange();
    setIsInitialized(false);
    setSelectedModelType("");
    setSelectedModel("");
    resetSelectionsForModel();
  }, [clearSnapshotRecordsForConnectionChange, resetSelectionsForModel]);

  const modelTypeOptionList = useMemo(
    () => createModelTypeOptions(models),
    [models],
  );
  const modelOptionList = useMemo(
    () =>
      modelsForType(models, selectedModelType).map((model) => ({
        value: model.model,
        label: modelNameForId(model),
      })),
    [models, selectedModelType],
  );
  const presetOptionList = useMemo(
    () => presets.map((preset) => ({ value: preset.name, label: preset.name })),
    [presets],
  );
  const setup = useMemo(
    () => ({
      model: {
        selectedType: selectedModelType,
        selected: selectedModel,
        typeOptions: modelTypeOptionList,
        options: modelOptionList,
        selectType: selectModelType,
        select: selectModel,
      },
      variants: {
        primaryPreset: selectedPrimaryPreset,
        selectedPresets,
        selectedSnapshotIds,
        presetOptions: presetOptionList,
        snapshots: modelConfigSnapshots,
        snapshotMutation: configSnapshotsStatus.mutation,
        selectPrimaryPreset,
        selectPresets: setPresetSelection,
        togglePreset,
        excludePreset: excludeDraftPreset,
        makePresetPrimary,
        selectAllPresets,
        selectOnlyPrimaryPreset,
        selectSnapshots: setSnapshotSelection,
        includeSnapshot,
        excludeSnapshot,
        removeSnapshot,
        retrySnapshotMutation,
        dismissSnapshotMutation: dismissSnapshotRecordMutation,
      },
      experimentTask: {
        selected: activeExperimentTask,
        options: experimentTaskOptionList,
        select: selectExperimentTask,
      },
      datasets: {
        selected: selectedDatasets,
        options: datasets,
        select: setDatasetSelection,
        toggle: toggleDataset,
        selectAll: selectAllDatasets,
        selectFirst: selectFirstDataset,
      },
      monitors: {
        selected: selectedMonitors,
        options: monitors,
        isLoading: metadata.monitorMetadata.isLoading,
        select: setMonitorSelection,
        selectAll: selectAllMonitors,
        clear: clearMonitors,
      },
    }),
    [
      activeExperimentTask,
      clearMonitors,
      configSnapshotsStatus.mutation,
      datasets,
      dismissSnapshotRecordMutation,
      excludeDraftPreset,
      excludeSnapshot,
      experimentTaskOptionList,
      includeSnapshot,
      makePresetPrimary,
      modelConfigSnapshots,
      modelOptionList,
      modelTypeOptionList,
      monitors,
      metadata.monitorMetadata.isLoading,
      presetOptionList,
      removeSnapshot,
      retrySnapshotMutation,
      selectAllDatasets,
      selectAllMonitors,
      selectAllPresets,
      selectExperimentTask,
      selectFirstDataset,
      selectModel,
      selectModelType,
      selectOnlyPrimaryPreset,
      selectPrimaryPreset,
      selectedDatasets,
      selectedModel,
      selectedModelType,
      selectedMonitors,
      selectedPresets,
      selectedPrimaryPreset,
      selectedSnapshotIds,
      setDatasetSelection,
      setMonitorSelection,
      setPresetSelection,
      setSnapshotSelection,
      toggleDataset,
      togglePreset,
    ],
  );
  const runtimeDefaults = useMemo(
    () => ({
      active: effectiveBulkOverrides,
      sections: configSections,
      fieldCount,
      inactiveLockedCount: inactiveLockedOverrideCount,
      edit: updateOverride,
      clear: clearOverride,
      reset: resetOverrides,
    }),
    [
      clearOverride,
      configSections,
      effectiveBulkOverrides,
      fieldCount,
      inactiveLockedOverrideCount,
      resetOverrides,
      updateOverride,
    ],
  );
  const searchMetadata = useMemo(
    () => ({
      value: search,
      axes: searchAxes,
      isLoading: metadata.searchMetadata.isLoading,
      update: updateSearch,
    }),
    [search, searchAxes, metadata.searchMetadata.isLoading, updateSearch],
  );
  const status = useMemo(
    () => ({
      schemaLoading: metadata.runtimeDefaults.isLoading,
      isSchemaReady: metadata.runtimeDefaults.isReady,
    }),
    [metadata.runtimeDefaults.isLoading, metadata.runtimeDefaults.isReady],
  );

  return useMemo(
    () => ({
      setup,
      runtimeDefaults,
      searchMetadata,
      status,
      clearForConnectionChange,
    }),
    [
      clearForConnectionChange,
      runtimeDefaults,
      searchMetadata,
      setup,
      status,
    ],
  );
}

export type TrainingDraftState = ReturnType<typeof useTrainingDraftState>;
