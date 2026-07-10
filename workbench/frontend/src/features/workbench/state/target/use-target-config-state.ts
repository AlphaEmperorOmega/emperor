import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type ConfigSnapshotRecord,
  type DatasetGroup,
  type LogRun,
  type ModelIdentity,
  type MonitorOption,
  type Preset,
  type SearchAxis,
} from "@/lib/api";
import {
  activeOverrideScopeLabel,
  configKeyToken,
  effectivePresetOverrides,
  lockedOverrideKeys,
  normalizeAdaptiveOptionOverrides,
  normalizeConfigOverrides,
  overrideValue,
  type ActiveOverrideScope,
  type OverrideValues,
} from "@/lib/config";
import {
  createConfigSnapshot,
  type ConfigSnapshotCreateResult,
  validateConfigSnapshotCandidate,
  validateConfigSnapshotName,
} from "@/lib/config-snapshots";
import {
  resolveRunPresetName,
} from "@/lib/historical-monitor-runs";
import {
  modelNameForId,
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
} from "@/features/workbench/state/graph-monitor/use-preview-inspection";
import {
  useConfigSnapshotLibrary,
  useConfigSnapshots,
} from "@/features/workbench/state/target/use-config-snapshots";
import { useTargetOverridesState } from "@/features/workbench/state/target/use-target-overrides";
import {
  readPersistedTargetSelection,
  writePersistedTargetSelection,
} from "@/features/workbench/state/target/target-selection-storage";
import {
  LOCAL_DEFAULT_CAPABILITIES,
  useWorkbenchQueries,
} from "@/features/workbench/state/use-workbench-queries";
import {
  deriveTargetSelectionState,
} from "@/features/workbench/state/target/target-selection";
import {
  datasetsForExperimentTask,
  experimentTaskOptions,
  normalizeExperimentTask,
} from "@/features/workbench/state/target/target-dataset-catalog";
import {
  previewTargetKey,
  resolvePreviewTarget,
  type HistoricalExperimentTarget,
  type TargetMode,
} from "@/features/workbench/state/target/target-preview";
import { type WorkbenchWorkspace } from "@/types/workbench";

const EMPTY_MODEL_IDS: ModelIdentity[] = [];
const EMPTY_PRESETS: Preset[] = [];
const EMPTY_DATASET_GROUPS: DatasetGroup[] = [];
const EMPTY_MONITORS: MonitorOption[] = [];
const EMPTY_SEARCH_AXES: SearchAxis[] = [];

type TargetConfigStateOptions = {
  activeWorkspace?: WorkbenchWorkspace;
  snapshotLibraryEnabled?: boolean;
  requestPreview: (request: PreviewInspectionRequest) => void;
  clearPreview: () => void;
  resetGraphSelectionAndExpansion: () => void;
  resetGraphExpansion: () => void;
  onModelSelected?: () => void;
  onTargetPresetSelected?: () => void;
  onTargetSnapshotSelected?: () => void;
};

function overridesAreEmpty(overrides: OverrideValues) {
  return Object.keys(overrides).length === 0;
}

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

function createSnapshotId() {
  return globalThis.crypto?.randomUUID?.() ?? `snapshot-${Date.now()}`;
}

type TargetSnapshotSelectionOptions = {
  includeTrainingSnapshot?: boolean;
};

type PresetSnapshotDraftOptions = {
  includeTrainingPreset?: boolean;
};

export function useTargetConfigState({
  activeWorkspace = "model",
  snapshotLibraryEnabled = false,
  requestPreview,
  clearPreview,
  resetGraphSelectionAndExpansion,
  resetGraphExpansion,
  onModelSelected,
  onTargetPresetSelected,
  onTargetSnapshotSelected,
}: TargetConfigStateOptions) {
  const initialTargetSelection = useMemo(readPersistedTargetSelection, []);
  const {
    selectedModel,
    selectedPreset,
    setSelectedPreset,
    presetOverrides,
    setPresetOverrides,
    selectPreset,
    clearPresetOverrides,
    selectModel: selectTargetModel,
  } = useTargetOverridesState({
    selectedModel: initialTargetSelection?.selectedModel,
    selectedPreset: initialTargetSelection?.selectedPreset,
  });
  const [snapshotEditorDraft, setSnapshotEditorDraft] =
    useState<OverrideValues>({});
  const [selectedModelType, setSelectedModelType] = useState(
    initialTargetSelection?.selectedModelType ?? "",
  );
  const [selectedTargetMode, setSelectedTargetMode] = useState<TargetMode>(
    initialTargetSelection?.selectedTargetMode ?? "preset",
  );
  const [selectedSnapshotId, setSelectedSnapshotId] = useState(
    initialTargetSelection?.selectedTargetMode === "snapshot"
      ? initialTargetSelection.selectedSnapshotId
      : "",
  );
  const [selectedExperimentTarget, setSelectedExperimentTarget] =
    useState<HistoricalExperimentTarget | null>(null);
  const selectedExperimentRunId = selectedExperimentTarget?.runId ?? "";
  const selectedExperimentName = selectedExperimentTarget?.experiment ?? "";
  const selectedExperimentPreset = selectedExperimentTarget?.preset ?? "";
  const selectedExperimentDataset = selectedExperimentTarget?.dataset ?? "";
  const clearExperimentTarget = useCallback(() => {
    setSelectedExperimentTarget((current) => (current === null ? current : null));
  }, []);
  const [selectedExperimentTask, setSelectedExperimentTask] = useState("");
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const initialTrainingPrimaryPreset =
    initialTargetSelection?.selectedPreset &&
    initialTargetSelection.selectedTargetMode !== "snapshot"
      ? initialTargetSelection.selectedPreset
      : "";
  const [selectedTrainingModelType, setSelectedTrainingModelType] = useState(
    initialTargetSelection?.selectedModelType ?? "",
  );
  const [selectedTrainingModel, setSelectedTrainingModel] = useState(
    initialTargetSelection?.selectedModel ?? "",
  );
  const [selectedTrainingPrimaryPreset, setSelectedTrainingPrimaryPreset] =
    useState(initialTrainingPrimaryPreset);
  const [selectedTrainingPresets, setSelectedTrainingPresets] = useState<string[]>(
    initialTrainingPrimaryPreset ? [initialTrainingPrimaryPreset] : [],
  );
  const [selectedTrainingExperimentTask, setSelectedTrainingExperimentTask] =
    useState("");
  const [selectedTrainingDatasets, setSelectedTrainingDatasets] = useState<string[]>([]);
  const [selectedTrainingMonitors, setSelectedTrainingMonitors] = useState<string[]>([]);
  const [trainingBulkOverrides, setTrainingBulkOverrides] =
    useState<OverrideValues>({});
  const lastRequestedPreviewTargetKeyRef = useRef("");
  const suppressedAutomaticPreviewTargetKeyRef = useRef("");
  const [isRestoringTargetSelection, setIsRestoringTargetSelection] =
    useState(Boolean(initialTargetSelection));
  const hasSeededTrainingTargetRef = useRef(
    Boolean(initialTargetSelection?.selectedModel),
  );
  const allowEmptyTrainingPresetDraftRef = useRef(false);
  const selectedModelIdentity = useMemo(
    () => ({ modelType: selectedModelType, model: selectedModel }),
    [selectedModel, selectedModelType],
  );
  const selectedTrainingModelIdentity = useMemo(
    () => ({
      modelType: selectedTrainingModelType,
      model: selectedTrainingModel,
    }),
    [selectedTrainingModel, selectedTrainingModelType],
  );
  const {
    query: configSnapshotsQuery,
    snapshots: configSnapshots,
    createMutation: createSnapshotMutation,
    renameMutation: renameSnapshotMutation,
    updateMutation: updateSnapshotMutation,
    deleteMutation: deleteSnapshotMutation,
  } = useConfigSnapshots(selectedModelIdentity);
  const { snapshots: trainingConfigSnapshots } = useConfigSnapshots(
    selectedTrainingModelIdentity,
  );
  const {
    query: configSnapshotLibraryQuery,
    snapshots: configSnapshotLibrary,
  } = useConfigSnapshotLibrary();
  const createSnapshotRecord = createSnapshotMutation.mutate;
  const renameSnapshotRecord = renameSnapshotMutation.mutate;
  const updateSnapshotRecord = updateSnapshotMutation.mutate;
  const deleteSnapshotRecord = deleteSnapshotMutation.mutate;
  const [pendingConfigSnapshot, setPendingConfigSnapshot] =
    useState<ConfigSnapshotRecord | null>(null);
  const [selectedTrainingSnapshotIds, setSelectedTrainingSnapshotIds] =
    useState<string[]>([]);
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
  } = useWorkbenchQueries(
    selectedModelType,
    selectedModel,
    selectedPreset,
    [],
    { includeSearchSpace: false },
  );
  const {
    presetsQuery: trainingPresetsQuery,
    datasetsQuery: trainingDatasetsQuery,
    monitorsQuery: trainingMonitorsQuery,
    schemaQuery: trainingSchemaQuery,
    searchSpaceQuery: trainingSearchSpaceQuery,
  } = useWorkbenchQueries(
    selectedTrainingModelType,
    selectedTrainingModel,
    selectedTrainingPrimaryPreset,
    selectedTrainingPresets,
  );
  const capabilities = capabilitiesQuery.data ?? LOCAL_DEFAULT_CAPABILITIES;
  const models = modelsQuery.data?.models ?? EMPTY_MODEL_IDS;
  const modelsLoading = modelsQuery.isLoading;
  const isModelsError = modelsQuery.isError;
  const modelsError = modelsQuery.error;
  const presets = presetsQuery.data?.presets ?? EMPTY_PRESETS;
  const presetsReady = presetsQuery.isSuccess;
  const isPresetsError = presetsQuery.isError;
  const presetsError = presetsQuery.error;
  const datasetGroups = datasetsQuery.data?.datasetGroups ?? EMPTY_DATASET_GROUPS;
  const defaultExperimentTask = datasetsQuery.data?.defaultExperimentTask ?? "";
  const activeExperimentTask = normalizeExperimentTask(
    selectedExperimentTask,
    defaultExperimentTask,
    datasetGroups,
  );
  const experimentTaskOptionsList = useMemo(
    () => experimentTaskOptions(datasetGroups),
    [datasetGroups],
  );
  const datasets = datasetsForExperimentTask(datasetGroups, activeExperimentTask);
  const isDatasetsError = datasetsQuery.isError;
  const datasetsError = datasetsQuery.error;
  const targetMonitors = monitorsQuery.data?.monitors ?? EMPTY_MONITORS;
  const targetMonitorsLoading = monitorsQuery.isLoading;
  const isSchemaReady = schemaQuery.isSuccess;
  const schemaLoading = schemaQuery.isLoading;
  const isSchemaError = schemaQuery.isError;
  const schemaError = schemaQuery.error;
  const trainingPresets = trainingPresetsQuery.data?.presets ?? EMPTY_PRESETS;
  const trainingDatasetGroups =
    trainingDatasetsQuery.data?.datasetGroups ?? EMPTY_DATASET_GROUPS;
  const trainingDefaultExperimentTask =
    trainingDatasetsQuery.data?.defaultExperimentTask ?? "";
  const activeTrainingExperimentTask = normalizeExperimentTask(
    selectedTrainingExperimentTask,
    trainingDefaultExperimentTask,
    trainingDatasetGroups,
  );
  const trainingExperimentTaskOptions = useMemo(
    () => experimentTaskOptions(trainingDatasetGroups),
    [trainingDatasetGroups],
  );
  const trainingDatasets = datasetsForExperimentTask(
    trainingDatasetGroups,
    activeTrainingExperimentTask,
  );
  const trainingMonitors = trainingMonitorsQuery.data?.monitors ?? EMPTY_MONITORS;
  const trainingMonitorsLoading = trainingMonitorsQuery.isLoading;
  const trainingSearchAxes = trainingSearchSpaceQuery.data?.axes ?? EMPTY_SEARCH_AXES;
  const trainingSearchAxesLoading = trainingSearchSpaceQuery.isLoading;
  const libraryLoading = configSnapshotLibraryQuery.isLoading;
  const isLibraryError = configSnapshotLibraryQuery.isError;
  const libraryError = configSnapshotLibraryQuery.error;
  const catalogModels = models;
  const availableModelTypeOptions = useMemo(
    () => modelTypeOptions(catalogModels),
    [catalogModels],
  );

  const targetSelectionState = useMemo(
    () =>
      deriveTargetSelectionState({
        datasets,
        presets,
        schemaFields: schemaQuery.data?.fields,
        configSnapshots,
        selectedModelType,
        selectedModel,
        selectedPreset,
        selectedTrainingPresets: selectedPreset ? [selectedPreset] : [],
        overrides: presetOverrides,
      }),
    [
      configSnapshots,
      datasets,
      presetOverrides,
      presets,
      schemaQuery.data?.fields,
      selectedModelType,
      selectedModel,
      selectedPreset,
    ],
  );
  const {
    datasetNames,
    presetNames,
    selectedPresetMeta,
    configSections,
    configFields,
    modelConfigSnapshots,
    modelConfigSnapshotGroups,
    visibleConfigSnapshots,
    configSnapshotGroups,
    presetOwnedFieldCount,
    fieldCount,
  } = targetSelectionState;
  const trainingTargetSelectionState = useMemo(
    () =>
      deriveTargetSelectionState({
        datasets: trainingDatasets,
        presets: trainingPresets,
        schemaFields: trainingSchemaQuery.data?.fields,
        configSnapshots: trainingConfigSnapshots,
        selectedModelType: selectedTrainingModelType,
        selectedModel: selectedTrainingModel,
        selectedPreset: selectedTrainingPrimaryPreset,
        selectedTrainingPresets,
        overrides: trainingBulkOverrides,
      }),
    [
      selectedTrainingModel,
      selectedTrainingModelType,
      selectedTrainingPresets,
      selectedTrainingPrimaryPreset,
      trainingBulkOverrides,
      trainingConfigSnapshots,
      trainingDatasets,
      trainingPresets,
      trainingSchemaQuery.data?.fields,
    ],
  );
  const {
    datasetNames: trainingDatasetNames,
    presetNames: trainingPresetNames,
    configSections: trainingConfigSections,
    configFields: trainingConfigFields,
    presetOwnedFieldCount: trainingPresetOwnedFieldCount,
    fieldCount: trainingFieldCount,
    modelConfigSnapshots: trainingModelConfigSnapshots,
    modelConfigSnapshotGroups: trainingModelConfigSnapshotGroups,
    visibleConfigSnapshots: visibleTrainingConfigSnapshots,
    configSnapshotGroups: trainingConfigSnapshotGroups,
  } = trainingTargetSelectionState;
  useEffect(() => {
    if (configFields.length === 0) {
      return;
    }
    setPresetOverrides((current) => {
      const next = normalizeAdaptiveOptionOverrides(
        configFields,
        normalizeConfigOverrides(configFields, current),
      );
      return overrideValuesEqual(current, next) ? current : next;
    });
    setSnapshotEditorDraft((current) => {
      const next = normalizeAdaptiveOptionOverrides(
        configFields,
        normalizeConfigOverrides(configFields, current),
      );
      return overrideValuesEqual(current, next) ? current : next;
    });
  }, [configFields, setPresetOverrides]);
  useEffect(() => {
    if (trainingConfigFields.length === 0) {
      return;
    }
    setTrainingBulkOverrides((current) => {
      const next = normalizeAdaptiveOptionOverrides(
        trainingConfigFields,
        normalizeConfigOverrides(trainingConfigFields, current),
      );
      return overrideValuesEqual(current, next) ? current : next;
    });
  }, [trainingConfigFields]);
  const selectedConfigSnapshot = useMemo(
    () =>
      modelConfigSnapshots.find(
        (snapshot) => snapshot.id === selectedSnapshotId,
      ),
    [modelConfigSnapshots, selectedSnapshotId],
  );
  const selectedTrainingSnapshots = useMemo(() => {
    const selectedSnapshotIds = new Set(selectedTrainingSnapshotIds);
    return trainingModelConfigSnapshots.filter((snapshot) =>
      selectedSnapshotIds.has(snapshot.id),
    );
  }, [selectedTrainingSnapshotIds, trainingModelConfigSnapshots]);
  const effectivePresetOverrideValues = useMemo(
    () => effectivePresetOverrides(configFields, presetOverrides),
    [configFields, presetOverrides],
  );
  const trainingBulkOverrideValues = useMemo(
    () => effectivePresetOverrides(trainingConfigFields, trainingBulkOverrides),
    [trainingBulkOverrides, trainingConfigFields],
  );
  const trainingInactiveLockedOverrideKeys = useMemo(
    () => lockedOverrideKeys(trainingConfigFields, trainingBulkOverrides),
    [trainingBulkOverrides, trainingConfigFields],
  );
  const inactiveLockedOverrideKeys = useMemo(
    () => lockedOverrideKeys(configFields, presetOverrides),
    [configFields, presetOverrides],
  );
  const inactiveLockedOverrides = useMemo(
    () =>
      Object.fromEntries(
        inactiveLockedOverrideKeys.map((key) => [
          key,
          overrideValue(presetOverrides, key) ?? "",
        ]),
      ) as OverrideValues,
    [inactiveLockedOverrideKeys, presetOverrides],
  );
  const activeOverrideScope: ActiveOverrideScope =
    selectedTargetMode === "snapshot" && selectedSnapshotId
      ? "snapshot"
      : "preset";
  const activeOverrides =
    activeOverrideScope === "snapshot"
      ? snapshotEditorDraft
      : effectivePresetOverrideValues;
  const overrideCount = Object.keys(activeOverrides).length;
  const inactiveLockedOverrideCount = inactiveLockedOverrideKeys.length;
  const trainingInactiveLockedOverrideCount =
    trainingInactiveLockedOverrideKeys.length;
  const snapshotOverrideWarning = "";

  const selectModel = useCallback(
    (model: string, modelType = selectedModelType) => {
      const nextModel = modelNameForId(model);
      const nextModelType = model.includes("/") ? modelTypeForId(model) : modelType;
      lastRequestedPreviewTargetKeyRef.current = "";
      clearPreview();
      setSelectedModelType(nextModel ? nextModelType : "");
      setSelectedTargetMode("preset");
      setSelectedSnapshotId("");
      clearExperimentTarget();
      selectTargetModel(nextModel);
      setSnapshotEditorDraft({});
      setSelectedExperimentTask("");
      setSelectedDatasets([]);
      onModelSelected?.();
      resetGraphSelectionAndExpansion();
    },
    [
      clearPreview,
      clearExperimentTarget,
      onModelSelected,
      resetGraphSelectionAndExpansion,
      selectTargetModel,
      selectedModelType,
    ],
  );

  const selectModelType = useCallback(
    (modelType: string) => {
      setSelectedModelType(modelType);
      const firstModel = modelsForType(catalogModels, modelType)[0];
      if (
        firstModel &&
        (firstModel.modelType !== selectedModelType ||
          firstModel.model !== selectedModel)
      ) {
        selectModel(firstModel.model, firstModel.modelType);
      }
    },
    [catalogModels, selectModel, selectedModel, selectedModelType],
  );

  const resetTrainingSelectionsForModel = useCallback(() => {
    allowEmptyTrainingPresetDraftRef.current = false;
    setSelectedTrainingPrimaryPreset("");
    setSelectedTrainingPresets([]);
    setSelectedTrainingSnapshotIds([]);
    setSelectedTrainingExperimentTask("");
    setSelectedTrainingDatasets([]);
    setSelectedTrainingMonitors([]);
    setTrainingBulkOverrides({});
    setTrainingSearch(DEFAULT_TRAINING_SEARCH_STATE);
  }, []);

  const selectTrainingModel = useCallback(
    (model: string, modelType = selectedTrainingModelType) => {
      const nextModel = modelNameForId(model);
      const nextModelType = model.includes("/")
        ? modelTypeForId(model)
        : modelType;
      setSelectedTrainingModelType(nextModel ? nextModelType : "");
      setSelectedTrainingModel(nextModel);
      resetTrainingSelectionsForModel();
    },
    [resetTrainingSelectionsForModel, selectedTrainingModelType],
  );

  const selectTrainingModelType = useCallback(
    (modelType: string) => {
      setSelectedTrainingModelType(modelType);
      const firstModel = modelsForType(catalogModels, modelType)[0];
      if (firstModel) {
        selectTrainingModel(firstModel.model, firstModel.modelType);
        return;
      }
      setSelectedTrainingModel("");
      resetTrainingSelectionsForModel();
    },
    [catalogModels, resetTrainingSelectionsForModel, selectTrainingModel],
  );

  // Selection cascade: model types/models load -> first type/model auto-selected -> presets/datasets
  // load -> first preset + dataset auto-selected and the initial preview is
  // requested -> dataset/monitor lists are pruned to what the model supports.
  useEffect(() => {
    setSelectedExperimentTask((current) => {
      const validTaskNames = datasetGroups.map((group) => group.experimentTask);
      const next = current && !validTaskNames.includes(current) ? "" : current;
      return current === next ? current : next;
    });
  }, [datasetGroups]);

  useEffect(() => {
    setSelectedTrainingExperimentTask((current) => {
      const validTaskNames = trainingDatasetGroups.map(
        (group) => group.experimentTask,
      );
      const next = current && !validTaskNames.includes(current) ? "" : current;
      return current === next ? current : next;
    });
  }, [trainingDatasetGroups]);

  useEffect(() => {
    if (catalogModels.length === 0) {
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
    if (
      !selectedModel ||
      selectedModelType !== nextModelType ||
      !modelsInSelectedType.some((model) => model.model === selectedModel)
    ) {
      const firstModel = modelsInSelectedType[0];
      if (firstModel) {
        selectModel(firstModel.model, firstModel.modelType);
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
    if (hasSeededTrainingTargetRef.current) {
      return;
    }
    if (!selectedModelType || !selectedModel || !selectedPreset) {
      return;
    }
    hasSeededTrainingTargetRef.current = true;
    setSelectedTrainingModelType(selectedModelType);
    setSelectedTrainingModel(selectedModel);
    setSelectedTrainingPrimaryPreset(selectedPreset);
    allowEmptyTrainingPresetDraftRef.current = false;
    setSelectedTrainingPresets([selectedPreset]);
  }, [selectedModel, selectedModelType, selectedPreset]);

  useEffect(() => {
    const firstPreset = presetNames[0];
    const firstDataset = datasetNames[0];
    if (!firstPreset || !firstDataset) {
      return;
    }
    if (!selectedPreset || !presetNames.includes(selectedPreset)) {
      const pendingSnapshotForModel =
        pendingConfigSnapshot &&
        selectedModelType === pendingConfigSnapshot.modelType &&
        selectedModel === pendingConfigSnapshot.model &&
        presetNames.includes(pendingConfigSnapshot.preset);
      const nextPreset = pendingSnapshotForModel
        ? pendingConfigSnapshot.preset
        : firstPreset;
      const shouldKeepSnapshotTarget =
        selectedTargetMode === "snapshot" && selectedSnapshotId.length > 0;
      if (!shouldKeepSnapshotTarget) {
        setSelectedTargetMode("preset");
        setSelectedSnapshotId("");
        clearExperimentTarget();
      }
      setSelectedPreset(nextPreset);
    }
  }, [
    datasetNames,
    clearExperimentTarget,
    pendingConfigSnapshot,
    presetNames,
    selectedPreset,
    selectedModel,
    selectedModelType,
    selectedSnapshotId,
    selectedTargetMode,
    setSelectedPreset,
  ]);

  useEffect(() => {
    if (!pendingConfigSnapshot) {
      return;
    }
    if (
      selectedModelType !== pendingConfigSnapshot.modelType ||
      selectedModel !== pendingConfigSnapshot.model
    ) {
      return;
    }
    if (!presetsQuery.isSuccess || !configSnapshotsQuery.isSuccess) {
      return;
    }
    if (!presetNames.includes(pendingConfigSnapshot.preset)) {
      setPendingConfigSnapshot(null);
      return;
    }
    if (selectedPreset !== pendingConfigSnapshot.preset) {
      setSelectedPreset(pendingConfigSnapshot.preset);
      return;
    }
    if (!schemaQuery.isSuccess) {
      return;
    }
    setSelectedTargetMode("snapshot");
    setSelectedSnapshotId(pendingConfigSnapshot.id);
    clearExperimentTarget();
    setSnapshotEditorDraft(
      normalizeAdaptiveOptionOverrides(
        configFields,
        normalizeConfigOverrides(configFields, pendingConfigSnapshot.overrides),
      ),
    );
    lastRequestedPreviewTargetKeyRef.current = "";
    setPendingConfigSnapshot(null);
  }, [
    configSnapshotsQuery.isSuccess,
    clearExperimentTarget,
    configFields,
    pendingConfigSnapshot,
    presetNames,
    presetsQuery.isSuccess,
    schemaQuery.isSuccess,
    selectedModel,
    selectedModelType,
    selectedPreset,
    setSelectedPreset,
  ]);

  useEffect(() => {
    if (!isRestoringTargetSelection) {
      return;
    }
    if (!initialTargetSelection) {
      setIsRestoringTargetSelection(false);
      return;
    }
    if (!selectedModel) {
      return;
    }
    if (selectedModel !== initialTargetSelection.selectedModel) {
      setIsRestoringTargetSelection(false);
      return;
    }
    if (
      !presetsQuery.isSuccess ||
      presetNames.length === 0 ||
      datasetNames.length === 0
    ) {
      return;
    }
    if (!selectedPreset || !presetNames.includes(selectedPreset)) {
      return;
    }
    if (initialTargetSelection.selectedTargetMode !== "snapshot") {
      setIsRestoringTargetSelection(false);
      return;
    }
    if (!initialTargetSelection.selectedSnapshotId) {
      setSelectedTargetMode("preset");
      setSelectedSnapshotId("");
      setSnapshotEditorDraft({});
      setIsRestoringTargetSelection(false);
      return;
    }
    if (configSnapshotsQuery.isError || schemaQuery.isError) {
      setSelectedTargetMode("preset");
      setSelectedSnapshotId("");
      clearExperimentTarget();
      setSnapshotEditorDraft({});
      setIsRestoringTargetSelection(false);
      return;
    }
    if (!configSnapshotsQuery.isSuccess) {
      return;
    }

    const snapshot = modelConfigSnapshots.find(
      (candidate) => candidate.id === initialTargetSelection.selectedSnapshotId,
    );
    if (!snapshot || !presetNames.includes(snapshot.preset)) {
      setSelectedTargetMode("preset");
      setSelectedSnapshotId("");
      clearExperimentTarget();
      setSnapshotEditorDraft({});
      setIsRestoringTargetSelection(false);
      return;
    }
    if (selectedPreset !== snapshot.preset) {
      setSelectedPreset(snapshot.preset);
      return;
    }
    if (!schemaQuery.isSuccess) {
      return;
    }

    setSelectedTargetMode("snapshot");
    setSelectedSnapshotId(snapshot.id);
    clearExperimentTarget();
    setSelectedTrainingPrimaryPreset(snapshot.preset);
    allowEmptyTrainingPresetDraftRef.current = true;
    setSelectedTrainingPresets([]);
    setSelectedTrainingSnapshotIds((current) =>
      current.includes(snapshot.id) ? current : [...current, snapshot.id],
    );
    const normalizedSnapshotOverrides = normalizeAdaptiveOptionOverrides(
      configFields,
      normalizeConfigOverrides(configFields, snapshot.overrides),
    );
    setSnapshotEditorDraft((current) =>
      overrideValuesEqual(current, normalizedSnapshotOverrides)
        ? current
        : normalizedSnapshotOverrides,
    );
    lastRequestedPreviewTargetKeyRef.current = "";
    setIsRestoringTargetSelection(false);
  }, [
    configSnapshotsQuery.isSuccess,
    configSnapshotsQuery.isError,
    clearExperimentTarget,
    configFields,
    datasetNames.length,
    initialTargetSelection,
    isRestoringTargetSelection,
    modelConfigSnapshots,
    presetNames,
    presetsQuery.isSuccess,
    schemaQuery.isSuccess,
    schemaQuery.isError,
    selectedModel,
    selectedPreset,
    setSelectedPreset,
  ]);

  useEffect(() => {
    if (isRestoringTargetSelection || !selectedModel || !selectedPreset) {
      return;
    }
    const persistedTargetMode =
      selectedTargetMode === "snapshot" && selectedSnapshotId
        ? "snapshot"
        : "preset";
    writePersistedTargetSelection({
      selectedModelType,
      selectedModel,
      selectedPreset,
      selectedTargetMode: persistedTargetMode,
      selectedSnapshotId:
        persistedTargetMode === "snapshot" ? selectedSnapshotId : "",
    });
  }, [
    isRestoringTargetSelection,
    selectedModel,
    selectedModelType,
    selectedPreset,
    selectedSnapshotId,
    selectedTargetMode,
  ]);

  useEffect(() => {
    if (isRestoringTargetSelection) {
      return;
    }
    if (
      pendingConfigSnapshot &&
      selectedModelType === pendingConfigSnapshot.modelType &&
      selectedModel === pendingConfigSnapshot.model
    ) {
      return;
    }
    if (selectedTargetMode === "snapshot" && selectedSnapshotId) {
      if (!selectedConfigSnapshot) {
        return;
      }
    }

    const {
      targetMode,
      targetId,
      preset: previewPreset,
      experimentTask: previewExperimentTask,
      dataset: previewDataset,
    } = resolvePreviewTarget({
      selectedTargetMode,
      selectedSnapshotId,
      selectedExperimentTarget,
      selectedPreset,
      selectedExperimentTask: activeExperimentTask,
      selectedDatasets,
    });
    if (!selectedModel) {
      return;
    }
    if (targetMode === "experiment" && !targetId) {
      const pendingExperimentTargetKey = previewTargetKey({
        modelType: selectedModelType,
        model: selectedModel,
        preset: previewPreset,
        experimentTask: previewExperimentTask,
        dataset: previewDataset,
        mode: targetMode,
        target: targetId,
        overrides: activeOverrides,
      });
      if (lastRequestedPreviewTargetKeyRef.current === pendingExperimentTargetKey) {
        return;
      }
      lastRequestedPreviewTargetKeyRef.current = pendingExperimentTargetKey;
      resetGraphSelectionAndExpansion();
      clearPreview();
      return;
    }
    if (!previewPreset || !previewDataset) {
      return;
    }
    const targetKey = previewTargetKey({
      modelType: selectedModelType,
      model: selectedModel,
      preset: previewPreset,
      experimentTask: previewExperimentTask,
      dataset: previewDataset,
      mode: targetMode,
      target: targetId,
      overrides: activeOverrides,
    });
    if (suppressedAutomaticPreviewTargetKeyRef.current) {
      if (suppressedAutomaticPreviewTargetKeyRef.current === targetKey) {
        return;
      }
      suppressedAutomaticPreviewTargetKeyRef.current = "";
    }
    if (lastRequestedPreviewTargetKeyRef.current === targetKey) {
      return;
    }

    lastRequestedPreviewTargetKeyRef.current = targetKey;
    resetGraphSelectionAndExpansion();
    const previewRequest: PreviewInspectionRequest = {
      modelType: selectedModelType,
      model: selectedModel,
      preset: previewPreset,
      experimentTask: previewExperimentTask || undefined,
      dataset: previewDataset,
      overrides: { ...activeOverrides },
      targetMode,
      targetId,
    };
    requestPreview(
      targetMode === "experiment"
        ? { ...previewRequest, logRunId: targetId }
        : previewRequest,
    );
  }, [
    activeOverrides,
    activeExperimentTask,
    clearPreview,
    isRestoringTargetSelection,
    pendingConfigSnapshot,
    requestPreview,
    resetGraphSelectionAndExpansion,
    selectedConfigSnapshot,
    selectedDatasets,
    selectedExperimentTarget,
    selectedModel,
    selectedModelType,
    selectedPreset,
    selectedSnapshotId,
    selectedTargetMode,
  ]);

  useEffect(() => {
    setTrainingSearch(DEFAULT_TRAINING_SEARCH_STATE);
  }, [selectedTrainingModel, selectedTrainingPrimaryPreset]);

  useEffect(() => {
    if (trainingConfigSnapshots.length > 0) {
      setTrainingSearch(DEFAULT_TRAINING_SEARCH_STATE);
    }
  }, [trainingConfigSnapshots.length]);

  useEffect(() => {
    if (
      isRestoringTargetSelection &&
      initialTargetSelection?.selectedTargetMode === "snapshot"
    ) {
      return;
    }
    const selectedSnapshotPreset =
      selectedTrainingSnapshotIds
        .map(
          (snapshotId) =>
            trainingModelConfigSnapshots.find(
              (snapshot) => snapshot.id === snapshotId,
            )?.preset,
        )
        .find((preset): preset is string =>
          Boolean(preset && trainingPresetNames.includes(preset)),
        ) ?? "";
    if (trainingPresetNames.length === 0) {
      setSelectedTrainingPrimaryPreset((current) => (current ? "" : current));
      setSelectedTrainingPresets((current) => {
        if (current.length === 0 && allowEmptyTrainingPresetDraftRef.current) {
          return current;
        }
        return current.length === 0 ? current : [];
      });
      return;
    }
    if (
      !selectedTrainingPrimaryPreset ||
      !trainingPresetNames.includes(selectedTrainingPrimaryPreset)
    ) {
      setSelectedTrainingPrimaryPreset(
        selectedSnapshotPreset ||
          selectedTrainingPresets.find((preset) =>
            trainingPresetNames.includes(preset),
          ) ||
          trainingPresetNames[0] ||
          "",
      );
      return;
    }
    setSelectedTrainingPresets((current) => {
      if (current.length === 0 && allowEmptyTrainingPresetDraftRef.current) {
        return current;
      }
      const next = normalizePrimarySelection(
        current,
        trainingPresetNames,
        selectedTrainingPrimaryPreset || undefined,
      );
      return selectionValuesEqual(current, next) ? current : next;
    });
  }, [
    initialTargetSelection?.selectedTargetMode,
    isRestoringTargetSelection,
    selectedTrainingPrimaryPreset,
    selectedTrainingPresets,
    selectedTrainingSnapshotIds,
    trainingModelConfigSnapshots,
    trainingPresetNames,
  ]);

  useEffect(() => {
    const snapshotIds = trainingModelConfigSnapshots.map((snapshot) => snapshot.id);
    setSelectedTrainingSnapshotIds((current) => {
      const next = uniqueValidValues(current, snapshotIds);
      return selectionValuesEqual(current, next) ? current : next;
    });
  }, [trainingModelConfigSnapshots]);

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
    if (trainingDatasetNames.length === 0) {
      setSelectedTrainingDatasets((current) =>
        current.length === 0 ? current : [],
      );
      return;
    }
    setSelectedTrainingDatasets((current) => {
      const next = normalizeSelection(current, trainingDatasetNames);
      return selectionValuesEqual(current, next) ? current : next;
    });
  }, [trainingDatasetNames]);

  useEffect(() => {
    const monitorNames = trainingMonitors.map((monitor) => monitor.name);
    setSelectedTrainingMonitors((current) => {
      const next = current.filter((monitor) => monitorNames.includes(monitor));
      return next.length === current.length ? current : next;
    });
  }, [trainingMonitors]);

  const syncSelectedLogRun = useCallback(
    (selectedLogRun: LogRun) => {
      if (!selectedModel) {
        return;
      }
      const rawPreset = selectedLogRun.preset;
      const rawDataset = selectedLogRun.dataset;
      const runExperimentTask = selectedLogRun.experimentTask ?? null;
      const catalogPreset = resolveRunPresetName(
        selectedLogRun,
        presets,
      );
      const catalogDataset = datasetNames.includes(rawDataset)
        ? rawDataset
        : "";
      const overridesAlreadyEmpty = overridesAreEmpty(presetOverrides);
      const catalogPresetSynced =
        !catalogPreset || selectedPreset === catalogPreset;
      const catalogDatasetSynced =
        !catalogDataset ||
        selectionValuesEqual(selectedDatasets, [catalogDataset]);
      const alreadySynced =
        selectedTargetMode === "experiment" &&
        selectedExperimentRunId === selectedLogRun.id &&
        selectedExperimentName === selectedLogRun.experiment &&
        selectedExperimentPreset === rawPreset &&
        selectedExperimentDataset === rawDataset &&
        (selectedExperimentTarget?.experimentTask ?? null) === runExperimentTask &&
        selectedSnapshotId === "" &&
        catalogPresetSynced &&
        catalogDatasetSynced &&
        overridesAlreadyEmpty;
      if (alreadySynced) {
        return;
      }

      setSelectedTargetMode("experiment");
      setSelectedSnapshotId("");
      setSelectedExperimentTarget({
        runId: selectedLogRun.id,
        experiment: selectedLogRun.experiment,
        preset: rawPreset,
        dataset: rawDataset,
        experimentTask: runExperimentTask,
      });
      if (runExperimentTask) {
        setSelectedExperimentTask(runExperimentTask);
      }
      if (catalogPreset && selectedPreset !== catalogPreset) {
        setSelectedPreset(catalogPreset);
      }
      if (catalogDataset) {
        setSelectedDatasets((current) =>
          selectionValuesEqual(current, [catalogDataset])
            ? current
            : [catalogDataset],
        );
      }
      if (!overridesAlreadyEmpty) {
        clearPresetOverrides();
      }
      lastRequestedPreviewTargetKeyRef.current = previewTargetKey({
        modelType: selectedModelType,
        model: selectedModel,
        preset: rawPreset,
        experimentTask: runExperimentTask,
        dataset: rawDataset,
        mode: "experiment",
        target: selectedLogRun.id,
        overrides: {},
      });
      resetGraphSelectionAndExpansion();
      requestPreview({
        modelType: selectedModelType,
        model: selectedModel,
        preset: rawPreset,
        experimentTask: runExperimentTask || undefined,
        dataset: rawDataset,
        overrides: {},
        targetMode: "experiment",
        targetId: selectedLogRun.id,
        logRunId: selectedLogRun.id,
      });
    },
    [
      clearPresetOverrides,
      datasetNames,
      presets,
      presetOverrides,
      requestPreview,
      resetGraphSelectionAndExpansion,
      selectedDatasets,
      selectedExperimentDataset,
      selectedExperimentName,
      selectedExperimentPreset,
      selectedExperimentRunId,
      selectedExperimentTarget,
      selectedModel,
      selectedModelType,
      selectedPreset,
      selectedSnapshotId,
      selectedTargetMode,
      setSelectedPreset,
    ],
  );

  const addConfigSnapshot = useCallback(
    (
      name: string,
      draftOverrides: OverrideValues = activeOverrides,
    ): ConfigSnapshotCreateResult => {
      const normalizedDraftOverrides = normalizeAdaptiveOptionOverrides(
        configFields,
        normalizeConfigOverrides(configFields, draftOverrides),
      );
      // Validate client-side for instant dialog feedback; the server re-validates
      // and is the source of truth. The client-generated id is discarded: the
      // persisted snapshot, with its server id, arrives via query invalidation.
      const result = createConfigSnapshot({
        id: createSnapshotId(),
        name,
        modelType: selectedModelType,
        model: selectedModel,
        preset: selectedPreset,
        fields: configFields,
        overrides: normalizedDraftOverrides,
        snapshots: configSnapshots,
        createdAt: new Date().toISOString(),
      });
      if (result.ok) {
        createSnapshotRecord({
          modelType: selectedModelType,
          model: selectedModel,
          preset: selectedPreset,
          name: result.snapshot.name,
          overrides: result.snapshot.overrides,
        });
      }
      return result;
    },
    [
      activeOverrides,
      configFields,
      configSnapshots,
      createSnapshotRecord,
      selectedModel,
      selectedModelType,
      selectedPreset,
    ],
  );

  const removeConfigSnapshot = useCallback(
    (snapshotId: string) => {
      setSelectedTrainingSnapshotIds((current) =>
        current.filter((id) => id !== snapshotId),
      );
      deleteSnapshotRecord(snapshotId);
    },
    [deleteSnapshotRecord],
  );

  const renameConfigSnapshot = useCallback(
    (snapshotId: string, name: string) => {
      const snapshot = configSnapshots.find(
        (candidate) => candidate.id === snapshotId,
      );
      if (!snapshot) {
        return;
      }
      const validation = validateConfigSnapshotName({
        modelType: snapshot.modelType,
        model: snapshot.model,
        preset: snapshot.preset,
        name,
        snapshots: configSnapshots,
        excludeSnapshotId: snapshotId,
      });
      if (!validation.ok) {
        return;
      }
      renameSnapshotRecord({
        id: snapshotId,
        name: validation.name,
      });
    },
    [configSnapshots, renameSnapshotRecord],
  );

  const updateSelectedConfigSnapshot = useCallback(
    (
      name: string,
      draftOverrides: OverrideValues = snapshotEditorDraft,
    ): ConfigSnapshotCreateResult => {
      if (!selectedConfigSnapshot) {
        return { ok: false, error: "Select a snapshot first." };
      }
      const normalizedDraftOverrides = normalizeAdaptiveOptionOverrides(
        configFields,
        normalizeConfigOverrides(configFields, draftOverrides),
      );
      const nameValidation = validateConfigSnapshotName({
        modelType: selectedConfigSnapshot.modelType,
        model: selectedConfigSnapshot.model,
        preset: selectedConfigSnapshot.preset,
        name,
        snapshots: configSnapshots,
        excludeSnapshotId: selectedConfigSnapshot.id,
      });
      if (!nameValidation.ok) {
        return nameValidation;
      }
      const validation = validateConfigSnapshotCandidate({
        modelType: selectedConfigSnapshot.modelType,
        model: selectedConfigSnapshot.model,
        preset: selectedConfigSnapshot.preset,
        fields: configFields,
        overrides: normalizedDraftOverrides,
        snapshots: configSnapshots,
        excludeSnapshotId: selectedConfigSnapshot.id,
      });
      if (!validation.ok) {
        return validation;
      }
      const snapshot = {
        ...selectedConfigSnapshot,
        name: nameValidation.name,
        overrides: validation.overrides,
      };
      updateSnapshotRecord({
        id: selectedConfigSnapshot.id,
        input: {
          name: nameValidation.name,
          overrides: validation.overrides,
        },
      });
      return { ok: true, snapshot };
    },
    [
      configFields,
      configSnapshots,
      selectedConfigSnapshot,
      snapshotEditorDraft,
      updateSnapshotRecord,
    ],
  );

  const includeConfigSnapshot = useCallback(
    (snapshotId: string) => {
      if (
        !trainingModelConfigSnapshots.some(
          (snapshot) => snapshot.id === snapshotId,
        )
      ) {
        return;
      }
      setSelectedTrainingSnapshotIds((current) =>
        current.includes(snapshotId) ? current : [...current, snapshotId],
      );
    },
    [trainingModelConfigSnapshots],
  );

  const excludeConfigSnapshot = useCallback(
    (snapshotId: string) => {
      if (
        !trainingModelConfigSnapshots.some(
          (snapshot) => snapshot.id === snapshotId,
        )
      ) {
        return;
      }
      setSelectedTrainingSnapshotIds((current) =>
        current.includes(snapshotId)
          ? current.filter((id) => id !== snapshotId)
          : current,
      );
    },
    [trainingModelConfigSnapshots],
  );

  const setTrainingSnapshotSelection = useCallback(
    (snapshotIds: string[]) => {
      const validSnapshotIds = trainingModelConfigSnapshots.map(
        (snapshot) => snapshot.id,
      );
      setSelectedTrainingSnapshotIds(uniqueValidValues(snapshotIds, validSnapshotIds));
    },
    [trainingModelConfigSnapshots],
  );

  const toggleConfigSnapshotRunSelection = useCallback(
    (snapshotId: string) => {
      if (
        !trainingModelConfigSnapshots.some(
          (snapshot) => snapshot.id === snapshotId,
        )
      ) {
        return;
      }
      if (selectedTrainingSnapshotIds.includes(snapshotId)) {
        excludeConfigSnapshot(snapshotId);
        return;
      }
      includeConfigSnapshot(snapshotId);
    },
    [
      excludeConfigSnapshot,
      includeConfigSnapshot,
      selectedTrainingSnapshotIds,
      trainingModelConfigSnapshots,
    ],
  );

  const loadConfigSnapshot = useCallback(
    (snapshotId: string) => {
      const snapshot =
        configSnapshots.find((candidate) => candidate.id === snapshotId) ??
        configSnapshotLibrary.find((candidate) => candidate.id === snapshotId);
      if (!snapshot) {
        return false;
      }
      setPendingConfigSnapshot(snapshot);
      if (
        snapshot.modelType !== selectedModelType ||
        snapshot.model !== selectedModel
      ) {
        selectModel(snapshot.model, snapshot.modelType);
      }
      return true;
    },
    [
      configSnapshotLibrary,
      configSnapshots,
      selectModel,
      selectedModel,
      selectedModelType,
    ],
  );

  const suppressAutomaticPreviewForPreset = useCallback(
    (preset: string) => {
      const previewDataset = selectedDatasets[0] ?? selectedExperimentDataset;
      suppressedAutomaticPreviewTargetKeyRef.current =
        selectedModel && preset && previewDataset
          ? previewTargetKey({
              modelType: selectedModelType,
              model: selectedModel,
              preset,
              experimentTask: activeExperimentTask,
              dataset: previewDataset,
              mode: "preset",
              target: preset,
              overrides: effectivePresetOverrideValues,
            })
          : "";
    },
    [
      effectivePresetOverrideValues,
      activeExperimentTask,
      selectedDatasets,
      selectedExperimentDataset,
      selectedModel,
      selectedModelType,
    ],
  );

  const selectPreviewPreset = useCallback(
    (preset: string) => {
      suppressAutomaticPreviewForPreset(preset);
      setSelectedTargetMode("preset");
      setSelectedSnapshotId("");
      clearExperimentTarget();
      onTargetPresetSelected?.();
      selectPreset(preset);
    },
    [
      clearExperimentTarget,
      onTargetPresetSelected,
      selectPreset,
      suppressAutomaticPreviewForPreset,
    ],
  );

  const selectTrainingPrimaryPreset = useCallback(
    (preset: string) => {
      if (!trainingPresetNames.includes(preset)) {
        return;
      }
      allowEmptyTrainingPresetDraftRef.current = false;
      setSelectedTrainingPrimaryPreset(preset);
      setSelectedTrainingPresets((current) =>
        normalizePrimarySelection(
          current.includes(preset) ? current : [...current, preset],
          trainingPresetNames,
          preset,
        ),
      );
    },
    [trainingPresetNames],
  );

  const selectTargetPreset = useCallback(
    (preset: string) => {
      const shouldRefreshPreview =
        selectedTargetMode !== "preset" ||
        selectedSnapshotId !== "" ||
        selectedExperimentRunId !== "" ||
        selectedPreset !== preset ||
        !overridesAreEmpty(effectivePresetOverrideValues);
      selectPreviewPreset(preset);
      if (!shouldRefreshPreview) {
        return;
      }

      const previewDataset = selectedDatasets[0] ?? selectedExperimentDataset;
      if (!selectedModel || !preset || !previewDataset) {
        lastRequestedPreviewTargetKeyRef.current = "";
        return;
      }

      lastRequestedPreviewTargetKeyRef.current = previewTargetKey({
        modelType: selectedModelType,
        model: selectedModel,
        preset,
        experimentTask: activeExperimentTask || undefined,
        dataset: previewDataset,
        mode: "preset",
        target: preset,
        overrides: effectivePresetOverrideValues,
      });
      resetGraphSelectionAndExpansion();
      requestPreview({
        modelType: selectedModelType,
        model: selectedModel,
        preset,
        experimentTask: activeExperimentTask,
        dataset: previewDataset,
        overrides: { ...effectivePresetOverrideValues },
        targetMode: "preset",
        targetId: preset,
      });
    },
    [
      effectivePresetOverrideValues,
      activeExperimentTask,
      requestPreview,
      resetGraphSelectionAndExpansion,
      selectPreviewPreset,
      selectedDatasets,
      selectedExperimentDataset,
      selectedExperimentRunId,
      selectedModel,
      selectedModelType,
      selectedPreset,
      selectedSnapshotId,
      selectedTargetMode,
    ],
  );

  const selectTargetSnapshot = useCallback(
    (
      snapshotId: string,
      { includeTrainingSnapshot = false }: TargetSnapshotSelectionOptions = {},
    ) => {
      const snapshot = modelConfigSnapshots.find(
        (candidate) => candidate.id === snapshotId,
      );
      if (!snapshot || !presetNames.includes(snapshot.preset)) {
        return false;
      }

      setSelectedTargetMode("snapshot");
      setSelectedSnapshotId(snapshot.id);
      clearExperimentTarget();
      onTargetSnapshotSelected?.();
      setSelectedPreset(snapshot.preset);
      if (includeTrainingSnapshot) {
        setSelectedTrainingSnapshotIds((current) =>
          current.includes(snapshot.id) ? current : [...current, snapshot.id],
        );
      }
      const normalizedSnapshotOverrides = normalizeAdaptiveOptionOverrides(
        configFields,
        normalizeConfigOverrides(configFields, snapshot.overrides),
      );
      setSnapshotEditorDraft(normalizedSnapshotOverrides);

      const previewDataset = selectedDatasets[0] ?? selectedExperimentDataset;
      if (!selectedModel || !snapshot.preset || !previewDataset) {
        lastRequestedPreviewTargetKeyRef.current = "";
        return true;
      }

      lastRequestedPreviewTargetKeyRef.current = previewTargetKey({
        modelType: selectedModelType,
        model: selectedModel,
        preset: snapshot.preset,
        experimentTask: activeExperimentTask || undefined,
        dataset: previewDataset,
        mode: "snapshot",
        target: snapshot.id,
        overrides: normalizedSnapshotOverrides,
      });
      resetGraphSelectionAndExpansion();
      requestPreview({
        modelType: selectedModelType,
        model: selectedModel,
        preset: snapshot.preset,
        experimentTask: activeExperimentTask,
        dataset: previewDataset,
        overrides: { ...normalizedSnapshotOverrides },
        targetMode: "snapshot",
        targetId: snapshot.id,
      });
      return true;
    },
    [
      modelConfigSnapshots,
      activeExperimentTask,
      clearExperimentTarget,
      configFields,
      presetNames,
      requestPreview,
      resetGraphSelectionAndExpansion,
      selectedDatasets,
      selectedExperimentDataset,
      selectedModel,
      selectedModelType,
      onTargetSnapshotSelected,
      setSelectedPreset,
    ],
  );

  const prepareSelectedSnapshotEdit = useCallback(
    (
      snapshotId: string,
      options?: TargetSnapshotSelectionOptions,
    ) => selectTargetSnapshot(snapshotId, options),
    [selectTargetSnapshot],
  );

  const activateTargetPresetMode = useCallback(() => {
    if (!selectedPreset) {
      setSelectedTargetMode("preset");
      setSelectedSnapshotId("");
      clearExperimentTarget();
      onTargetPresetSelected?.();
      lastRequestedPreviewTargetKeyRef.current = "";
      return;
    }
    selectPreviewPreset(selectedPreset);

    const previewDataset = selectedDatasets[0];
    if (!selectedModel || !previewDataset) {
      lastRequestedPreviewTargetKeyRef.current = "";
      return;
    }

    lastRequestedPreviewTargetKeyRef.current = previewTargetKey({
      modelType: selectedModelType,
      model: selectedModel,
      preset: selectedPreset,
      experimentTask: activeExperimentTask || undefined,
      dataset: previewDataset,
      mode: "preset",
      target: selectedPreset,
      overrides: effectivePresetOverrideValues,
    });
    resetGraphSelectionAndExpansion();
    requestPreview({
      modelType: selectedModelType,
      model: selectedModel,
      preset: selectedPreset,
      experimentTask: activeExperimentTask,
      dataset: previewDataset,
      overrides: { ...effectivePresetOverrideValues },
      targetMode: "preset",
      targetId: selectedPreset,
    });
  }, [
    effectivePresetOverrideValues,
    activeExperimentTask,
    clearExperimentTarget,
    onTargetPresetSelected,
    requestPreview,
    resetGraphSelectionAndExpansion,
    selectPreviewPreset,
    selectedDatasets,
    selectedModel,
    selectedModelType,
    selectedPreset,
  ]);

  const activateTargetSnapshotMode = useCallback(() => {
    const selectedSnapshot = modelConfigSnapshots.find(
      (snapshot) => snapshot.id === selectedSnapshotId,
    );
    const snapshot = selectedSnapshot ?? modelConfigSnapshots[0];
    if (!snapshot) {
      return false;
    }
    return selectTargetSnapshot(snapshot.id);
  }, [modelConfigSnapshots, selectTargetSnapshot, selectedSnapshotId]);

  const activateTargetExperimentMode = useCallback(() => {
    setSelectedTargetMode("experiment");
  }, []);

  const clearSelectedExperimentRun = useCallback(() => {
    clearExperimentTarget();
    lastRequestedPreviewTargetKeyRef.current = "";
  }, [clearExperimentTarget]);

  const setTrainingPresetSelection = useCallback(
    (presets: string[]) => {
      const validPresets = uniqueValidValues(presets, trainingPresetNames);
      if (validPresets.length === 0 && selectedTrainingSnapshotIds.length > 0) {
        allowEmptyTrainingPresetDraftRef.current = true;
        setSelectedTrainingPresets([]);
        return;
      }
      const fallbackPreset =
        selectedTrainingPrimaryPreset &&
        trainingPresetNames.includes(selectedTrainingPrimaryPreset)
          ? selectedTrainingPrimaryPreset
          : trainingPresetNames[0] ?? "";
      const nextPrimary = validPresets.includes(selectedTrainingPrimaryPreset)
        ? selectedTrainingPrimaryPreset
        : validPresets.length > 0
          ? validPresets[0]
          : fallbackPreset;

      allowEmptyTrainingPresetDraftRef.current = false;
      setSelectedTrainingPrimaryPreset(nextPrimary);
      setSelectedTrainingPresets(
        normalizePrimarySelection(
          validPresets,
          trainingPresetNames,
          nextPrimary || undefined,
        ),
      );
    },
    [
      selectedTrainingPrimaryPreset,
      selectedTrainingSnapshotIds.length,
      trainingPresetNames,
    ],
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

  const toggleDraftTrainingPreset = useCallback(
    (preset: string) => {
      if (!trainingPresetNames.includes(preset)) {
        return;
      }
      setSelectedTrainingPresets((current) => {
        if (current.includes(preset)) {
          const next = current.filter((item) => item !== preset);
          allowEmptyTrainingPresetDraftRef.current = next.length === 0;
          return next;
        }
        allowEmptyTrainingPresetDraftRef.current = false;
        return normalizePrimarySelection(
          [...current, preset],
          trainingPresetNames,
          preset === selectedTrainingPrimaryPreset ||
            current.includes(selectedTrainingPrimaryPreset)
            ? selectedTrainingPrimaryPreset
            : undefined,
        );
      });
    },
    [selectedTrainingPrimaryPreset, trainingPresetNames],
  );

  const excludeDraftTrainingPreset = useCallback(
    (preset: string) => {
      if (!trainingPresetNames.includes(preset)) {
        return;
      }
      setSelectedTrainingPresets((current) => {
        if (!current.includes(preset)) {
          return current;
        }
        const next = current.filter((item) => item !== preset);
        allowEmptyTrainingPresetDraftRef.current = next.length === 0;
        return next;
      });
    },
    [trainingPresetNames],
  );

  const makeTrainingPresetPrimary = useCallback(
    (preset: string) => {
      if (!trainingPresetNames.includes(preset)) {
        return;
      }
      allowEmptyTrainingPresetDraftRef.current = false;
      setSelectedTrainingPrimaryPreset(preset);
      setSelectedTrainingPresets((current) =>
        normalizePrimarySelection(
          [...current, preset],
          trainingPresetNames,
          preset || undefined,
        ),
      );
    },
    [trainingPresetNames],
  );

  const preparePresetSnapshotDraft = useCallback(
    (
      preset: string,
      options: PresetSnapshotDraftOptions = {},
    ) => {
      void options;
      if (!presetNames.includes(preset)) {
        return false;
      }
      suppressAutomaticPreviewForPreset(preset);
      selectPreset(preset);
      setSelectedTargetMode("preset");
      setSelectedSnapshotId("");
      clearExperimentTarget();
      setSnapshotEditorDraft({});
      return true;
    },
    [
      presetNames,
      clearExperimentTarget,
      selectPreset,
      suppressAutomaticPreviewForPreset,
    ],
  );

  const prepareTrainingPresetSnapshotDraft = useCallback(
    (preset: string) => {
      if (
        !selectedTrainingModelType ||
        !selectedTrainingModel ||
        !trainingPresetNames.includes(preset)
      ) {
        return false;
      }
      lastRequestedPreviewTargetKeyRef.current = "";
      if (
        selectedModelType !== selectedTrainingModelType ||
        selectedModel !== selectedTrainingModel
      ) {
        selectModel(selectedTrainingModel, selectedTrainingModelType);
      }
      setSelectedTargetMode("preset");
      setSelectedSnapshotId("");
      clearExperimentTarget();
      onTargetPresetSelected?.();
      setSelectedPreset(preset);
      setSnapshotEditorDraft({});
      return true;
    },
    [
      clearExperimentTarget,
      onTargetPresetSelected,
      selectModel,
      selectedModel,
      selectedModelType,
      selectedTrainingModel,
      selectedTrainingModelType,
      setSelectedPreset,
      trainingPresetNames,
    ],
  );

  const prepareTrainingSelectedSnapshotEdit = useCallback(
    (snapshotId: string) => {
      const snapshot = trainingConfigSnapshots.find(
        (candidate) => candidate.id === snapshotId,
      );
      if (!snapshot || !trainingPresetNames.includes(snapshot.preset)) {
        return false;
      }
      setPendingConfigSnapshot(snapshot);
      if (
        snapshot.modelType !== selectedModelType ||
        snapshot.model !== selectedModel
      ) {
        selectModel(snapshot.model, snapshot.modelType);
      }
      return true;
    },
    [
      selectModel,
      selectedModel,
      selectedModelType,
      trainingConfigSnapshots,
      trainingPresetNames,
    ],
  );

  const selectAllTrainingPresets = useCallback(() => {
    allowEmptyTrainingPresetDraftRef.current = false;
    if (!selectedTrainingPrimaryPreset) {
      setSelectedTrainingPresets([]);
      return;
    }
    setSelectedTrainingPresets([
      selectedTrainingPrimaryPreset,
      ...trainingPresetNames.filter(
        (preset) => preset !== selectedTrainingPrimaryPreset,
      ),
    ]);
  }, [selectedTrainingPrimaryPreset, trainingPresetNames]);

  const selectPrimaryTrainingPreset = useCallback(() => {
    allowEmptyTrainingPresetDraftRef.current = false;
    setSelectedTrainingPresets(
      selectedTrainingPrimaryPreset ? [selectedTrainingPrimaryPreset] : [],
    );
  }, [selectedTrainingPrimaryPreset]);

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

  const selectTrainingExperimentTask = useCallback(
    (experimentTask: string) => {
      const nextTask = normalizeExperimentTask(
        experimentTask,
        trainingDefaultExperimentTask,
        trainingDatasetGroups,
      );
      const nextDatasetNames = datasetsForExperimentTask(
        trainingDatasetGroups,
        nextTask,
      ).map((dataset) => dataset.name);
      setSelectedTrainingExperimentTask(nextTask);
      setSelectedTrainingDatasets((current) =>
        normalizeSelection(current, nextDatasetNames),
      );
    },
    [trainingDatasetGroups, trainingDefaultExperimentTask],
  );

  const setTrainingDatasetSelection = useCallback(
    (datasets: string[]) => {
      setSelectedTrainingDatasets((current) =>
        normalizeSelection(datasets, trainingDatasetNames, current),
      );
    },
    [trainingDatasetNames],
  );

  const toggleTrainingDataset = useCallback(
    (dataset: string) => {
      setSelectedTrainingDatasets((current) => {
        const next = current.includes(dataset)
          ? current.filter((item) => item !== dataset)
          : [...current, dataset];
        return normalizeSelection(next, trainingDatasetNames, current);
      });
    },
    [trainingDatasetNames],
  );

  const selectAllTrainingDatasets = useCallback(() => {
    setSelectedTrainingDatasets(trainingDatasetNames);
  }, [trainingDatasetNames]);

  const selectFirstTrainingDataset = useCallback(() => {
    setSelectedTrainingDatasets(
      trainingDatasetNames[0] ? [trainingDatasetNames[0]] : [],
    );
  }, [trainingDatasetNames]);

  const toggleMonitor = useCallback((monitor: string) => {
    setSelectedTrainingMonitors((current) =>
      current.includes(monitor)
        ? current.filter((item) => item !== monitor)
        : [...current, monitor],
    );
  }, []);

  const setMonitorSelection = useCallback(
    (monitorSelection: string[]) => {
      const monitorNames = trainingMonitors.map((monitor) => monitor.name);
      setSelectedTrainingMonitors(
        uniqueValidValues(monitorSelection, monitorNames),
      );
    },
    [trainingMonitors],
  );

  const selectAllMonitors = useCallback(() => {
    setSelectedTrainingMonitors(trainingMonitors.map((monitor) => monitor.name));
  }, [trainingMonitors]);

  const clearMonitors = useCallback(() => {
    setSelectedTrainingMonitors([]);
  }, []);

  const updatePreview = useCallback(() => {
    const {
      targetMode,
      targetId,
      preset: previewPreset,
      experimentTask: previewExperimentTask,
      dataset: previewDataset,
    } = resolvePreviewTarget({
      selectedTargetMode,
      selectedSnapshotId,
      selectedExperimentTarget,
      selectedPreset,
      selectedExperimentTask: activeExperimentTask,
      selectedDatasets,
    });
    if (!selectedModel || !previewPreset || !previewDataset) {
      return;
    }
    lastRequestedPreviewTargetKeyRef.current = previewTargetKey({
      modelType: selectedModelType,
      model: selectedModel,
      preset: previewPreset,
      experimentTask: previewExperimentTask,
      dataset: previewDataset,
      mode: targetMode,
      target: targetId,
      overrides: activeOverrides,
    });
    resetGraphSelectionAndExpansion();
    if (targetMode === "experiment" && !targetId) {
      clearPreview();
      return;
    }
    requestPreview({
      modelType: selectedModelType,
      model: selectedModel,
      preset: previewPreset,
      experimentTask: previewExperimentTask || undefined,
      dataset: previewDataset,
      overrides: { ...activeOverrides },
      targetMode,
      targetId,
      logRunId: targetMode === "experiment" ? targetId : undefined,
    });
  }, [
    activeOverrides,
    activeExperimentTask,
    clearPreview,
    requestPreview,
    resetGraphSelectionAndExpansion,
    selectedDatasets,
    selectedExperimentTarget,
    selectedModel,
    selectedModelType,
    selectedPreset,
    selectedSnapshotId,
    selectedTargetMode,
  ]);

  const resetTargetOverrides = useCallback(
    (preserveTargetSelection: boolean) => {
      if (!preserveTargetSelection) {
        setSelectedTargetMode("preset");
        setSelectedSnapshotId("");
        clearExperimentTarget();
        onTargetPresetSelected?.();
      }
      const resetSnapshotDraft =
        preserveTargetSelection &&
        selectedTargetMode === "snapshot" &&
        selectedSnapshotId;
      if (resetSnapshotDraft) {
        setSnapshotEditorDraft({});
      } else {
        clearPresetOverrides();
      }
      resetGraphExpansion();
      const resolvedTarget = preserveTargetSelection
          ? resolvePreviewTarget({
              selectedTargetMode,
              selectedSnapshotId,
              selectedExperimentTarget,
              selectedPreset,
              selectedExperimentTask: activeExperimentTask,
              selectedDatasets,
            })
          : {
              targetMode: "preset" as const,
              targetId: selectedPreset,
              preset: selectedPreset,
              experimentTask: activeExperimentTask,
              dataset: selectedDatasets[0] ?? "",
            };
      const {
        targetMode,
        targetId,
        preset: previewPreset,
        experimentTask: previewExperimentTask,
        dataset: previewDataset,
      } = resolvedTarget;
      if (selectedModel && previewPreset && previewDataset) {
        const nextOverrides = {};
        lastRequestedPreviewTargetKeyRef.current = previewTargetKey({
          modelType: selectedModelType,
          model: selectedModel,
          preset: previewPreset,
          experimentTask: previewExperimentTask,
          dataset: previewDataset,
          mode: targetMode,
          target: targetId,
          overrides: nextOverrides,
        });
        if (targetMode === "experiment" && !targetId) {
          clearPreview();
          return;
        }
        const previewRequest: PreviewInspectionRequest = {
          modelType: selectedModelType,
          model: selectedModel,
          preset: previewPreset,
          experimentTask: previewExperimentTask || undefined,
          dataset: previewDataset,
          overrides: nextOverrides,
          targetMode,
          targetId,
        };
        requestPreview(
          targetMode === "experiment"
            ? { ...previewRequest, logRunId: targetId }
            : previewRequest,
        );
      }
    },
    [
      clearPreview,
      clearPresetOverrides,
      activeExperimentTask,
      requestPreview,
      resetGraphExpansion,
      clearExperimentTarget,
      selectedDatasets,
      selectedExperimentTarget,
      selectedModel,
      selectedModelType,
      selectedPreset,
      selectedSnapshotId,
      selectedTargetMode,
      onTargetPresetSelected,
    ],
  );

  const resetOverrides = useCallback(() => {
    resetTargetOverrides(false);
  }, [resetTargetOverrides]);

  const resetOverridesPreservingTargetSelection = useCallback(() => {
    resetTargetOverrides(true);
  }, [resetTargetOverrides]);

  const updateTargetOverride = useCallback(
    (
      key: string,
      value: string,
      options?: { preserveTargetSelection?: boolean },
    ) => {
      if (
        !options?.preserveTargetSelection &&
        (selectedTargetMode === "snapshot" || selectedTargetMode === "experiment")
      ) {
        setSelectedTargetMode("preset");
        setSelectedSnapshotId("");
        clearExperimentTarget();
        onTargetPresetSelected?.();
      }
      setPresetOverrides((current) =>
        normalizeAdaptiveOptionOverrides(
          configFields,
          normalizeConfigOverrides(configFields, {
            ...current,
            [key]: value,
          }),
        ),
      );
    },
    [
      clearExperimentTarget,
      configFields,
      onTargetPresetSelected,
      selectedTargetMode,
      setPresetOverrides,
    ],
  );

  const clearTargetOverride = useCallback(
    (key: string, options?: { preserveTargetSelection?: boolean }) => {
      if (
        !options?.preserveTargetSelection &&
        (selectedTargetMode === "snapshot" || selectedTargetMode === "experiment")
      ) {
        setSelectedTargetMode("preset");
        setSelectedSnapshotId("");
        clearExperimentTarget();
        onTargetPresetSelected?.();
      }
      setPresetOverrides((current) =>
        normalizeAdaptiveOptionOverrides(
          configFields,
          normalizeConfigOverrides(configFields, withoutOverride(current, key)),
        ),
      );
    },
    [
      clearExperimentTarget,
      configFields,
      onTargetPresetSelected,
      selectedTargetMode,
      setPresetOverrides,
    ],
  );

  const updateSnapshotEditorDraftOverride = useCallback(
    (key: string, value: string) => {
      setSnapshotEditorDraft((current) =>
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

  const clearSnapshotEditorDraftOverride = useCallback((key: string) => {
    setSnapshotEditorDraft((current) => {
      const next = withoutOverride(current, key);
      return normalizeAdaptiveOptionOverrides(
        configFields,
        normalizeConfigOverrides(configFields, next),
      );
    });
  }, [configFields]);

  const resetSnapshotEditorDraft = useCallback(() => {
    setSnapshotEditorDraft({});
  }, []);

  const updateTrainingOverride = useCallback(
    (key: string, value: string) => {
      setTrainingBulkOverrides((current) =>
        normalizeAdaptiveOptionOverrides(
          trainingConfigFields,
          normalizeConfigOverrides(trainingConfigFields, {
            ...current,
            [key]: value,
          }),
        ),
      );
    },
    [trainingConfigFields],
  );

  const clearTrainingOverride = useCallback(
    (key: string) => {
      setTrainingBulkOverrides((current) =>
        normalizeAdaptiveOptionOverrides(
          trainingConfigFields,
          normalizeConfigOverrides(
            trainingConfigFields,
            withoutOverride(current, key),
          ),
        ),
      );
    },
    [trainingConfigFields],
  );

  const resetTrainingOverrides = useCallback(() => {
    setTrainingBulkOverrides({});
  }, []);

  const apiOnline = healthQuery.data?.status === "ok";

  const target = useMemo(
    () => ({
      selectedModelType,
      selectModelType,
      selectedModel,
      selectModel,
      selectedTrainingModelType,
      selectTrainingModelType,
      selectedTrainingModel,
      selectTrainingModel,
      selectedTargetMode,
      activateTargetPresetMode,
      activateTargetSnapshotMode,
      activateTargetExperimentMode,
      selectedPreset,
      selectPreset: selectTargetPreset,
      selectTargetPreset,
      selectedTrainingPrimaryPreset,
      selectTrainingPrimaryPreset,
      selectedSnapshotId,
      selectedConfigSnapshot,
      selectedExperimentRunId,
      selectedExperimentTarget,
      selectedExperimentName,
      selectedExperimentPreset,
      selectedExperimentDataset,
      selectTargetSnapshot,
      prepareSelectedSnapshotEdit,
      selectedPresetMeta,
      selectedTrainingPresets,
      setTrainingPresetSelection,
      selectedTrainingSnapshotIds,
      selectedTrainingSnapshots,
      setTrainingSnapshotSelection,
      toggleTrainingPreset,
      toggleDraftTrainingPreset,
      excludeDraftTrainingPreset,
      makeTrainingPresetPrimary,
      selectAllTrainingPresets,
      selectPrimaryTrainingPreset,
      selectedDatasets,
      setDatasetSelection,
      toggleDataset,
      selectAllDatasets,
      selectFirstDataset,
      selectExperimentTask,
      selectedTrainingDatasets,
      setTrainingDatasetSelection,
      toggleTrainingDataset,
      selectAllTrainingDatasets,
      selectFirstTrainingDataset,
      selectTrainingExperimentTask,
      selectedMonitors: selectedTrainingMonitors,
      selectedTrainingMonitors,
      toggleMonitor,
      setMonitorSelection,
      selectAllMonitors,
      clearMonitors,
      presetOverrides,
      effectivePresetOverrides: effectivePresetOverrideValues,
      snapshotEditorDraft,
      activeOverrides,
      activeOverrideScope,
      activeOverrideScopeLabel: activeOverrideScopeLabel(activeOverrideScope),
      inactiveLockedOverrides,
      inactiveLockedOverrideCount,
      snapshotOverrideWarning,
      overrides: activeOverrides,
      trainingOverrides: trainingBulkOverrideValues,
      trainingBulkOverrides: trainingBulkOverrideValues,
      configSections,
      trainingConfigSections,
      overrideCount,
      trainingOverrideCount: Object.keys(trainingBulkOverrideValues).length,
      presetOwnedFieldCount,
      trainingPresetOwnedFieldCount,
      fieldCount,
      trainingFieldCount,
      trainingInactiveLockedOverrideCount,
      capabilities,
      apiOnline,
      trainingSearch,
      setTrainingSearch,
      configSnapshots: visibleConfigSnapshots,
      allConfigSnapshots: modelConfigSnapshots,
      configSnapshotLibrary,
      configSnapshotGroups,
      allConfigSnapshotGroups: modelConfigSnapshotGroups,
      configSnapshotCount: visibleConfigSnapshots.length,
      allConfigSnapshotCount: modelConfigSnapshots.length,
      trainingConfigSnapshots: visibleTrainingConfigSnapshots,
      allTrainingConfigSnapshots: trainingModelConfigSnapshots,
      trainingConfigSnapshotGroups,
      allTrainingConfigSnapshotGroups: trainingModelConfigSnapshotGroups,
      trainingConfigSnapshotCount: visibleTrainingConfigSnapshots.length,
      allTrainingConfigSnapshotCount: trainingModelConfigSnapshots.length,
      configSnapshotLibraryCount: configSnapshotLibrary.length,
      addConfigSnapshot,
      removeConfigSnapshot,
      renameConfigSnapshot,
      updateSelectedConfigSnapshot,
      loadConfigSnapshot,
      includeConfigSnapshot,
      excludeConfigSnapshot,
      toggleConfigSnapshotRunSelection,
      preparePresetSnapshotDraft,
      prepareTrainingPresetSnapshotDraft,
      prepareTrainingSelectedSnapshotEdit,
      updateOverride: updateTargetOverride,
      clearOverride: clearTargetOverride,
      updateTrainingOverride,
      clearTrainingOverride,
      resetTrainingOverrides,
      updateSnapshotEditorDraftOverride,
      clearSnapshotEditorDraftOverride,
      resetSnapshotEditorDraft,
      updatePreview,
      resetOverrides,
      resetOverridesPreservingTargetSelection,
      models,
      modelsLoading,
      isModelsError,
      modelsError,
      presets,
      trainingPresets,
      presetsReady,
      isPresetsError,
      presetsError,
      selectedExperimentTask: activeExperimentTask,
      experimentTaskOptions: experimentTaskOptionsList,
      selectedTrainingExperimentTask: activeTrainingExperimentTask,
      trainingExperimentTaskOptions,
      datasets,
      trainingDatasets,
      isDatasetsError,
      datasetsError,
      targetMonitors,
      targetMonitorsLoading,
      monitors: trainingMonitors,
      trainingMonitors,
      monitorsLoading: trainingMonitorsLoading,
      trainingMonitorsLoading,
      isSchemaReady,
      isTrainingSchemaReady: trainingSchemaQuery.isSuccess,
      schemaLoading,
      trainingSchemaLoading: trainingSchemaQuery.isLoading,
      isSchemaError,
      schemaError,
      searchAxes: trainingSearchAxes,
      trainingSearchAxes,
      searchAxesLoading: trainingSearchAxesLoading,
      trainingSearchAxesLoading,
      libraryLoading,
      isLibraryError,
      libraryError,
    }),
    [
      activateTargetExperimentMode,
      activateTargetPresetMode,
      activateTargetSnapshotMode,
      addConfigSnapshot,
      activeExperimentTask,
      apiOnline,
      activeTrainingExperimentTask,
      capabilities,
      clearSnapshotEditorDraftOverride,
      clearMonitors,
      clearTargetOverride,
      clearTrainingOverride,
      configSections,
      configSnapshotGroups,
      configSnapshotLibrary,
      datasets,
      datasetsError,
      excludeConfigSnapshot,
      excludeDraftTrainingPreset,
      experimentTaskOptionsList,
      fieldCount,
      includeConfigSnapshot,
      isDatasetsError,
      isLibraryError,
      isModelsError,
      isPresetsError,
      isSchemaError,
      isSchemaReady,
      libraryError,
      libraryLoading,
      loadConfigSnapshot,
      makeTrainingPresetPrimary,
      modelConfigSnapshotGroups,
      modelConfigSnapshots,
      models,
      modelsError,
      modelsLoading,
      overrideCount,
      activeOverrideScope,
      activeOverrides,
      effectivePresetOverrideValues,
      inactiveLockedOverrideCount,
      inactiveLockedOverrides,
      preparePresetSnapshotDraft,
      prepareSelectedSnapshotEdit,
      prepareTrainingPresetSnapshotDraft,
      prepareTrainingSelectedSnapshotEdit,
      presetOwnedFieldCount,
      presetOverrides,
      presets,
      presetsError,
      presetsReady,
      removeConfigSnapshot,
      renameConfigSnapshot,
      resetOverrides,
      resetOverridesPreservingTargetSelection,
      resetSnapshotEditorDraft,
      resetTrainingOverrides,
      schemaError,
      schemaLoading,
      selectAllDatasets,
      selectAllMonitors,
      selectAllTrainingDatasets,
      selectAllTrainingPresets,
      selectFirstDataset,
      selectExperimentTask,
      selectFirstTrainingDataset,
      selectTrainingExperimentTask,
      selectModel,
      selectModelType,
      selectPrimaryTrainingPreset,
      selectTargetPreset,
      selectTargetSnapshot,
      selectTrainingModel,
      selectTrainingModelType,
      selectTrainingPrimaryPreset,
      selectedConfigSnapshot,
      selectedDatasets,
      selectedExperimentDataset,
      selectedExperimentName,
      selectedExperimentPreset,
      selectedExperimentRunId,
      selectedExperimentTarget,
      selectedModel,
      selectedModelType,
      selectedPreset,
      selectedPresetMeta,
      selectedSnapshotId,
      selectedTargetMode,
      selectedTrainingDatasets,
      selectedTrainingModel,
      selectedTrainingModelType,
      selectedTrainingMonitors,
      selectedTrainingPrimaryPreset,
      selectedTrainingPresets,
      selectedTrainingSnapshotIds,
      selectedTrainingSnapshots,
      setDatasetSelection,
      setMonitorSelection,
      setTrainingDatasetSelection,
      setTrainingPresetSelection,
      setTrainingSnapshotSelection,
      targetMonitors,
      targetMonitorsLoading,
      toggleConfigSnapshotRunSelection,
      toggleDataset,
      toggleDraftTrainingPreset,
      toggleMonitor,
      toggleTrainingDataset,
      toggleTrainingPreset,
      trainingConfigSections,
      trainingConfigSnapshotGroups,
      trainingDatasets,
      trainingExperimentTaskOptions,
      trainingModelConfigSnapshotGroups,
      trainingModelConfigSnapshots,
      trainingMonitors,
      trainingMonitorsLoading,
      trainingPresets,
      trainingBulkOverrideValues,
      trainingFieldCount,
      trainingInactiveLockedOverrideCount,
      trainingPresetOwnedFieldCount,
      trainingSchemaQuery.isLoading,
      trainingSchemaQuery.isSuccess,
      trainingSearchAxes,
      trainingSearchAxesLoading,
      trainingSearch,
      snapshotEditorDraft,
      snapshotOverrideWarning,
      updateSnapshotEditorDraftOverride,
      updateTrainingOverride,
      updatePreview,
      updateSelectedConfigSnapshot,
      updateTargetOverride,
      visibleTrainingConfigSnapshots,
      visibleConfigSnapshots,
    ],
  );
  const selection = useMemo(
    () => ({
      selectedModel,
      selectedPreset,
      selectedDatasets,
    }),
    [selectedDatasets, selectedModel, selectedPreset],
  );

  return useMemo(
    () => ({
      target,
      selection,
      syncSelectedLogRun,
      clearSelectedExperimentRun,
    }),
    [clearSelectedExperimentRun, selection, syncSelectedLogRun, target],
  );
}
