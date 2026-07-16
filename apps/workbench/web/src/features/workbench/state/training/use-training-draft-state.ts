import { useCallback, useMemo, useState } from "react";
import type { ModelIdentity } from "@/lib/api/model-catalog";
import {
  type OverrideValues,
} from "@/lib/config";
import {
  effectivePresetOverrides,
  inactivePresetOwnedOverrideKeys,
  runtimeDefaultsEditor,
} from "@/features/workbench/state/runtime-defaults/runtime-defaults";
import {
  modelNameForId,
  modelTypeForId,
  modelTypeOptions as createModelTypeOptions,
  modelsForType,
  normalizePrimarySelection,
  normalizeSelection,
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
  const [selectedModelType, setSelectedModelType] = useState(seed.modelType);
  const [selectedModel, setSelectedModel] = useState(seed.model);
  const [requestedPrimaryPreset, setSelectedPrimaryPreset] = useState(
    seed.preset,
  );
  const [requestedPresets, setSelectedPresets] = useState<string[]>(
    seed.preset ? [seed.preset] : [],
  );
  const [selectedExperimentTask, setSelectedExperimentTask] = useState("");
  const [requestedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [requestedMonitors, setSelectedMonitors] = useState<string[]>([]);
  const [bulkOverrideDraft, setBulkOverrides] = useState<OverrideValues>({});
  const [requestedSnapshotIds, setSelectedSnapshotIds] = useState<string[]>([]);
  const [searchDraft, setSearch] = useState<{
    model: string;
    preset: string;
    value: TrainingSearchState;
  }>(() => ({
    model: seed.model,
    preset: seed.preset,
    value: DEFAULT_TRAINING_SEARCH_STATE,
  }));
  const [allowEmptyPresetDraft, setAllowEmptyPresetDraft] = useState(false);

  const selectedIdentity = useMemo(
    () => ({ modelType: selectedModelType, model: selectedModel }),
    [selectedModel, selectedModelType],
  );
  const metadataSelection = useMemo(
    () => ({
      modelPackage: selectedIdentity,
      preset: requestedPrimaryPreset,
      searchPresets: requestedPresets,
    }),
    [requestedPresets, requestedPrimaryPreset, selectedIdentity],
  );
  const metadata = useModelPackageMetadata(metadataSelection, {
    includeSearchMetadata: activeWorkspace === "training",
    normalizePresetSelection: true,
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
  const presetNames = useMemo(
    () => presets.map((preset) => preset.name),
    [presets],
  );
  const modelConfigSnapshotCandidates = useMemo(
    () =>
      configSnapshots.filter(
        (snapshot) =>
          snapshot.modelType === selectedModelType &&
          snapshot.model === selectedModel,
      ),
    [configSnapshots, selectedModel, selectedModelType],
  );
  const selectedSnapshotIds = useMemo(
    () =>
      uniqueValidValues(
        requestedSnapshotIds,
        modelConfigSnapshotCandidates.map((snapshot) => snapshot.id),
      ),
    [modelConfigSnapshotCandidates, requestedSnapshotIds],
  );
  const selectedSnapshotPreset = selectedSnapshotIds
    .map(
      (snapshotId) =>
        modelConfigSnapshotCandidates.find(
          (snapshot) => snapshot.id === snapshotId,
        )?.preset,
    )
    .find((preset): preset is string =>
      Boolean(preset && presetNames.includes(preset)),
    );
  const selectedPrimaryPreset =
    presetNames.length === 0
      ? ""
      : presetNames.includes(requestedPrimaryPreset)
        ? requestedPrimaryPreset
        : selectedSnapshotPreset ??
          requestedPresets.find((preset) => presetNames.includes(preset)) ??
          presetNames[0] ??
          "";
  const selectedPresets = useMemo(
    () =>
      presetNames.length === 0
        ? []
        : requestedPresets.length === 0 && allowEmptyPresetDraft
          ? []
          : normalizePrimarySelection(
              requestedPresets,
              presetNames,
              selectedPrimaryPreset || undefined,
            ),
    [
      allowEmptyPresetDraft,
      presetNames,
      requestedPresets,
      selectedPrimaryPreset,
    ],
  );
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
  const selectedMonitors = useMemo(() => {
    const monitorNames = monitors.map((monitor) => monitor.name);
    return uniqueValidValues(requestedMonitors, monitorNames);
  }, [monitors, requestedMonitors]);
  const search =
    searchDraft.model === selectedModel &&
    searchDraft.preset === selectedPrimaryPreset
      ? searchDraft.value
      : DEFAULT_TRAINING_SEARCH_STATE;

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
    configSections,
    configFields,
    fieldCount,
    modelConfigSnapshots,
  } = selectionState;
  const selectedDatasets = useMemo(
    () => normalizeSelection(requestedDatasets, datasetNames),
    [datasetNames, requestedDatasets],
  );
  const bulkOverrides =
    configFields.length > 0
      ? runtimeDefaultsEditor.normalize(configFields, bulkOverrideDraft)
      : bulkOverrideDraft;
  const effectiveBulkOverrides = useMemo(
    () => effectivePresetOverrides(configFields, bulkOverrides),
    [bulkOverrides, configFields],
  );
  const inactiveLockedOverrideCount = useMemo(
    () => inactivePresetOwnedOverrideKeys(configFields, bulkOverrides).length,
    [bulkOverrides, configFields],
  );

  const resetSelectionsForModel = useCallback(() => {
    setAllowEmptyPresetDraft(false);
    setSelectedPrimaryPreset("");
    setSelectedPresets([]);
    setSelectedSnapshotIds([]);
    setSelectedExperimentTask("");
    setSelectedDatasets([]);
    setSelectedMonitors([]);
    setBulkOverrides({});
    setSearch({
      model: "",
      preset: "",
      value: DEFAULT_TRAINING_SEARCH_STATE,
    });
  }, []);

  const selectModel = useCallback(
    (model: string, modelType = selectedModelType) => {
      const nextModel = modelNameForId(model);
      const nextModelType = model.includes("/")
        ? modelTypeForId(model)
        : modelType;
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
      setAllowEmptyPresetDraft(false);
      setSelectedPrimaryPreset(preset);
      setSelectedPresets(
        normalizePrimarySelection(
          selectedPresets.includes(preset)
            ? selectedPresets
            : [...selectedPresets, preset],
          presetNames,
          preset,
        ),
      );
    },
    [presetNames, selectedPresets],
  );

  const setPresetSelection = useCallback(
    (nextPresets: string[]) => {
      const validPresets = uniqueValidValues(nextPresets, presetNames);
      if (validPresets.length === 0 && selectedSnapshotIds.length > 0) {
        setAllowEmptyPresetDraft(true);
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

      setAllowEmptyPresetDraft(false);
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
      if (!selectedPresets.includes(preset)) {
        return;
      }
      const next = selectedPresets.filter((item) => item !== preset);
      setAllowEmptyPresetDraft(next.length === 0);
      setSelectedPresets(next);
    },
    [presetNames, selectedPresets],
  );

  const makePresetPrimary = useCallback(
    (preset: string) => {
      if (!presetNames.includes(preset)) {
        return;
      }
      setAllowEmptyPresetDraft(false);
      setSelectedPrimaryPreset(preset);
      setSelectedPresets(
        normalizePrimarySelection(
          [...selectedPresets, preset],
          presetNames,
          preset,
        ),
      );
    },
    [presetNames, selectedPresets],
  );

  const selectAllPresets = useCallback(() => {
    setAllowEmptyPresetDraft(false);
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
    setAllowEmptyPresetDraft(false);
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
      setSelectedSnapshotIds(
        selectedSnapshotIds.includes(snapshotId)
          ? selectedSnapshotIds
          : [...selectedSnapshotIds, snapshotId],
      );
    },
    [modelConfigSnapshots, selectedSnapshotIds],
  );

  const excludeSnapshot = useCallback(
    (snapshotId: string) => {
      if (!modelConfigSnapshots.some((snapshot) => snapshot.id === snapshotId)) {
        return;
      }
      setSelectedSnapshotIds(
        selectedSnapshotIds.includes(snapshotId)
          ? selectedSnapshotIds.filter((id) => id !== snapshotId)
          : selectedSnapshotIds,
      );
    },
    [modelConfigSnapshots, selectedSnapshotIds],
  );

  const removeSnapshot = useCallback(
    async (snapshotId: string) => {
      const outcome = await deleteSnapshotRecord(snapshotId);
      if (outcome.ok) {
        setSelectedSnapshotIds(
          selectedSnapshotIds.filter((id) => id !== snapshotId),
        );
      }
      return outcome;
    },
    [deleteSnapshotRecord, selectedSnapshotIds],
  );
  const retrySnapshotMutation = useCallback(async () => {
    const outcome = await retrySnapshotRecordMutation();
    if (outcome?.ok && outcome.kind === "remove" && outcome.snapshotId) {
      setSelectedSnapshotIds(
        selectedSnapshotIds.filter((id) => id !== outcome.snapshotId),
      );
    }
    return outcome;
  }, [retrySnapshotRecordMutation, selectedSnapshotIds]);

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
      setSelectedDatasets(
        normalizeSelection(selectedDatasets, nextDatasetNames),
      );
    },
    [datasetGroups, defaultExperimentTask, selectedDatasets],
  );

  const setDatasetSelection = useCallback(
    (nextDatasets: string[]) => {
      setSelectedDatasets(
        normalizeSelection(nextDatasets, datasetNames, selectedDatasets),
      );
    },
    [datasetNames, selectedDatasets],
  );

  const toggleDataset = useCallback(
    (dataset: string) => {
      const next = selectedDatasets.includes(dataset)
        ? selectedDatasets.filter((item) => item !== dataset)
        : [...selectedDatasets, dataset];
      setSelectedDatasets(
        normalizeSelection(next, datasetNames, selectedDatasets),
      );
    },
    [datasetNames, selectedDatasets],
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
      setBulkOverrides(
        runtimeDefaultsEditor.edit(configFields, bulkOverrides, key, value),
      );
    },
    [bulkOverrides, configFields],
  );

  const clearOverride = useCallback(
    (key: string) => {
      setBulkOverrides(
        runtimeDefaultsEditor.clear(configFields, bulkOverrides, key),
      );
    },
    [bulkOverrides, configFields],
  );

  const resetOverrides = useCallback(() => {
    setBulkOverrides({});
  }, []);

  const updateSearch = useCallback(
    (nextSearch: TrainingSearchState) => {
      setSearch({
        model: selectedModel,
        preset: selectedPrimaryPreset,
        value: nextSearch,
      });
    },
    [selectedModel, selectedPrimaryPreset],
  );

  const clearForConnectionChange = useCallback(() => {
    clearSnapshotRecordsForConnectionChange();
    resetSelectionsForModel();
    setSelectedModelType(seed.modelType);
    setSelectedModel(seed.model);
    setSelectedPrimaryPreset(seed.preset);
    setSelectedPresets(seed.preset ? [seed.preset] : []);
  }, [
    clearSnapshotRecordsForConnectionChange,
    resetSelectionsForModel,
    seed.model,
    seed.modelType,
    seed.preset,
  ]);

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
