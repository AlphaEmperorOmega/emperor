import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type DatasetGroup,
  type ModelIdentity,
  type MonitorOption,
  type Preset,
  type SearchAxis,
} from "@/lib/api";
import {
  configKeyToken,
  effectivePresetOverrides,
  lockedOverrideKeys,
  normalizeAdaptiveOptionOverrides,
  normalizeConfigOverrides,
  type OverrideValues,
} from "@/lib/config";
import {
  modelNameForId,
  modelTypeForId,
  modelsForType,
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
  useConfigSnapshotRecords,
} from "@/features/workbench/state/config-snapshots/use-config-snapshot-records";
import {
  deriveModelPackageSelection,
  datasetsForExperimentTask,
  experimentTaskOptions,
  normalizeExperimentTask,
} from "@/features/workbench/state/model-package/model-package-selection";
import { useWorkbenchQueries } from "@/features/workbench/state/use-workbench-queries";
import { type WorkbenchWorkspace } from "@/types/workbench";

const EMPTY_PRESETS: Preset[] = [];
const EMPTY_DATASET_GROUPS: DatasetGroup[] = [];
const EMPTY_MONITORS: MonitorOption[] = [];
const EMPTY_SEARCH_AXES: SearchAxis[] = [];

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

function overrideValuesEqual(left: OverrideValues, right: OverrideValues) {
  const leftEntries = Object.entries(left);
  const rightEntries = Object.entries(right);
  if (leftEntries.length !== rightEntries.length) {
    return false;
  }
  return leftEntries.every(([key, value]) => right[key] === value);
}

function withoutOverride(overrides: OverrideValues, key: string): OverrideValues {
  const token = configKeyToken(key);
  return Object.fromEntries(
    Object.entries(overrides).filter(
      ([overrideKey]) => configKeyToken(overrideKey) !== token,
    ),
  );
}

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
  const {
    presetsQuery,
    datasetsQuery,
    monitorsQuery,
    schemaQuery,
    searchSpaceQuery,
  } = useWorkbenchQueries(
    selectedModelType,
    selectedModel,
    selectedPrimaryPreset,
    selectedPresets,
    {
      includeSearchSpace: activeWorkspace === "training",
      protectedReadsEnabled,
    },
  );
  const {
    records: configSnapshots,
    actions: configSnapshotActions,
  } = useConfigSnapshotRecords(selectedIdentity, {
    enabled: protectedReadsEnabled,
  });
  const deleteSnapshotRecord = configSnapshotActions.remove;

  const presets = presetsQuery.data?.presets ?? EMPTY_PRESETS;
  const datasetGroups =
    datasetsQuery.data?.datasetGroups ?? EMPTY_DATASET_GROUPS;
  const defaultExperimentTask =
    datasetsQuery.data?.defaultExperimentTask ?? "";
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
  const monitors = monitorsQuery.data?.monitors ?? EMPTY_MONITORS;
  const searchAxes = searchSpaceQuery.data?.axes ?? EMPTY_SEARCH_AXES;

  const selectionState = useMemo(
    () =>
      deriveModelPackageSelection({
        datasets,
        presets,
        schemaFields: schemaQuery.data?.fields,
        configSnapshots,
        selectedModelType,
        selectedModel,
        selectedPreset: selectedPrimaryPreset,
      }),
    [
      configSnapshots,
      datasets,
      presets,
      schemaQuery.data?.fields,
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
    setBulkOverrides((current) => {
      const next = normalizeAdaptiveOptionOverrides(
        configFields,
        normalizeConfigOverrides(configFields, current),
      );
      return overrideValuesEqual(current, next) ? current : next;
    });
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
    (snapshotId: string) => {
      setSelectedSnapshotIds((current) =>
        current.filter((id) => id !== snapshotId),
      );
      deleteSnapshotRecord(snapshotId);
    },
    [deleteSnapshotRecord],
  );

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
        normalizeAdaptiveOptionOverrides(
          configFields,
          normalizeConfigOverrides(configFields, {
            ...current,
            [key]: value,
          }),
        ),
      );
    },
    [configFields],
  );

  const clearOverride = useCallback(
    (key: string) => {
      setBulkOverrides((current) =>
        normalizeAdaptiveOptionOverrides(
          configFields,
          normalizeConfigOverrides(
            configFields,
            withoutOverride(current, key),
          ),
        ),
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
    setIsInitialized(false);
    setSelectedModelType("");
    setSelectedModel("");
    resetSelectionsForModel();
  }, [resetSelectionsForModel]);

  return {
    selectedModelType,
    selectedModel,
    selectedPrimaryPreset,
    selectedPresets,
    selectedExperimentTask: activeExperimentTask,
    selectedDatasets,
    selectedMonitors,
    bulkOverrides: effectiveBulkOverrides,
    selectedSnapshotIds,
    search,
    models,
    presets,
    datasets,
    experimentTaskOptions: experimentTaskOptionList,
    monitors,
    configSections,
    configSnapshots: modelConfigSnapshots,
    searchAxes,
    monitorsLoading: monitorsQuery.isLoading,
    schemaLoading: schemaQuery.isLoading,
    isSchemaReady: schemaQuery.isSuccess,
    searchLoading: searchSpaceQuery.isLoading,
    fieldCount,
    inactiveLockedOverrideCount,
    snapshotOverrideWarning: "",
    selectModelType,
    selectModel,
    selectPrimaryPreset,
    setPresetSelection,
    togglePreset,
    excludeDraftPreset,
    makePresetPrimary,
    selectAllPresets,
    selectOnlyPrimaryPreset,
    setSnapshotSelection,
    includeSnapshot,
    excludeSnapshot,
    removeSnapshot,
    selectExperimentTask,
    setDatasetSelection,
    toggleDataset,
    selectAllDatasets,
    selectFirstDataset,
    setMonitorSelection,
    selectAllMonitors,
    clearMonitors,
    updateSearch,
    updateOverride,
    clearOverride,
    resetOverrides,
    clearForConnectionChange,
  };
}

export type TrainingDraftState = ReturnType<typeof useTrainingDraftState>;
