import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type ConfigSnapshotRecord,
  type Dataset,
  type LogRun,
  type ModelIdentity,
  type MonitorOption,
  type Preset,
  type SearchAxis,
} from "@/lib/api";
import {
  activeOverrideScopeLabel,
  effectivePresetOverrides,
  lockedOverrideKeys,
  overrideDigest,
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
} from "@/features/viewer/state/graph-monitor/use-preview-inspection";
import {
  useConfigSnapshotLibrary,
  useConfigSnapshots,
} from "@/features/viewer/state/target/use-config-snapshots";
import { useTargetOverridesState } from "@/features/viewer/state/target/use-target-overrides";
import {
  readPersistedTargetSelection,
  writePersistedTargetSelection,
} from "@/features/viewer/state/target/target-selection-storage";
import {
  LOCAL_DEFAULT_CAPABILITIES,
  useViewerQueries,
} from "@/features/viewer/state/use-viewer-queries";
import {
  deriveTargetSelectionState,
} from "@/features/viewer/state/target/target-selection";

const EMPTY_MODEL_IDS: ModelIdentity[] = [];
const EMPTY_PRESETS: Preset[] = [];
const EMPTY_DATASETS: Dataset[] = [];
const EMPTY_MONITORS: MonitorOption[] = [];
const EMPTY_SEARCH_AXES: SearchAxis[] = [];

type TargetMode = "preset" | "snapshot" | "experiment";

type TargetConfigStateOptions = {
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

function createSnapshotId() {
  return globalThis.crypto?.randomUUID?.() ?? `snapshot-${Date.now()}`;
}

type TargetSnapshotSelectionOptions = {
  includeTrainingSnapshot?: boolean;
};

type PresetSnapshotDraftOptions = {
  includeTrainingPreset?: boolean;
};

function previewTargetKey({
  modelType,
  model,
  dataset,
  mode,
  target,
  overrides,
}: {
  modelType: string;
  model: string;
  dataset: string;
  mode: TargetMode;
  target: string;
  overrides: OverrideValues;
}) {
  return `${modelType}\u0000${model}\u0000${dataset}\u0000${mode}\u0000${target}\u0000${overrideDigest(overrides)}`;
}

export function useTargetConfigState({
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
    selectPreset,
    updatePresetOverride,
    clearPresetOverride,
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
  const [selectedExperimentRunId, setSelectedExperimentRunId] = useState("");
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [selectedTrainingPresets, setSelectedTrainingPresets] = useState<string[]>(
    initialTargetSelection?.selectedPreset &&
      initialTargetSelection.selectedTargetMode !== "snapshot"
      ? [initialTargetSelection.selectedPreset]
      : [],
  );
  const [selectedMonitors, setSelectedMonitors] = useState<string[]>([]);
  const lastRequestedPreviewTargetKeyRef = useRef("");
  const suppressedAutomaticPreviewTargetKeyRef = useRef("");
  const [isRestoringTargetSelection, setIsRestoringTargetSelection] =
    useState(Boolean(initialTargetSelection));
  const allowEmptyTrainingPresetDraftRef = useRef(false);
  const skipNextTrainingPresetPrimarySyncRef = useRef(false);
  const selectedModelIdentity = useMemo(
    () => ({ modelType: selectedModelType, model: selectedModel }),
    [selectedModel, selectedModelType],
  );
  const {
    query: configSnapshotsQuery,
    snapshots: configSnapshots,
    createMutation: createSnapshotMutation,
    renameMutation: renameSnapshotMutation,
    updateMutation: updateSnapshotMutation,
    deleteMutation: deleteSnapshotMutation,
  } = useConfigSnapshots(selectedModelIdentity);
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
    searchSpaceQuery,
  } = useViewerQueries(
    selectedModelType,
    selectedModel,
    selectedPreset,
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
  const datasets = datasetsQuery.data?.datasets ?? EMPTY_DATASETS;
  const isDatasetsError = datasetsQuery.isError;
  const datasetsError = datasetsQuery.error;
  const monitors = monitorsQuery.data?.monitors ?? EMPTY_MONITORS;
  const monitorsLoading = monitorsQuery.isLoading;
  const isSchemaReady = schemaQuery.isSuccess;
  const schemaLoading = schemaQuery.isLoading;
  const isSchemaError = schemaQuery.isError;
  const schemaError = schemaQuery.error;
  const searchAxes = searchSpaceQuery.data?.axes ?? EMPTY_SEARCH_AXES;
  const searchAxesLoading = searchSpaceQuery.isLoading;
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
        selectedTrainingPresets,
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
      selectedTrainingPresets,
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
    overrideCount: rawPresetOverrideCount,
    presetOwnedFieldCount,
    fieldCount,
  } = targetSelectionState;
  const selectedConfigSnapshot = useMemo(
    () =>
      modelConfigSnapshots.find(
        (snapshot) => snapshot.id === selectedSnapshotId,
      ),
    [modelConfigSnapshots, selectedSnapshotId],
  );
  const selectedTrainingSnapshots = useMemo(() => {
    const selectedSnapshotIds = new Set(selectedTrainingSnapshotIds);
    return modelConfigSnapshots.filter((snapshot) =>
      selectedSnapshotIds.has(snapshot.id),
    );
  }, [modelConfigSnapshots, selectedTrainingSnapshotIds]);
  const effectivePresetOverrideValues = useMemo(
    () => effectivePresetOverrides(configFields, presetOverrides),
    [configFields, presetOverrides],
  );
  const inactiveLockedOverrideKeys = useMemo(
    () => lockedOverrideKeys(configFields, presetOverrides),
    [configFields, presetOverrides],
  );
  const inactiveLockedOverrides = useMemo(
    () =>
      Object.fromEntries(
        inactiveLockedOverrideKeys.map((key) => [key, presetOverrides[key] ?? ""]),
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
  const snapshotOverrideWarning =
    selectedTrainingSnapshotIds.length > 0 && rawPresetOverrideCount > 0
      ? "Preset overrides apply only to preset rows. Config snapshots use their saved editable values."
      : "";

  const selectModel = useCallback(
    (model: string, modelType = selectedModelType) => {
      const nextModel = modelNameForId(model);
      const nextModelType = model.includes("/") ? modelTypeForId(model) : modelType;
      lastRequestedPreviewTargetKeyRef.current = "";
      clearPreview();
      setSelectedModelType(nextModel ? nextModelType : "");
      setSelectedTargetMode("preset");
      setSelectedSnapshotId("");
      setSelectedExperimentRunId("");
      selectTargetModel(nextModel);
      setSnapshotEditorDraft({});
      setSelectedDatasets([]);
      allowEmptyTrainingPresetDraftRef.current = false;
      setSelectedTrainingPresets([]);
      setSelectedTrainingSnapshotIds([]);
      setSelectedMonitors([]);
      onModelSelected?.();
      resetGraphSelectionAndExpansion();
    },
    [
      clearPreview,
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
      if (firstModel && firstModel.model !== selectedModel) {
        selectModel(firstModel.model, firstModel.modelType);
      }
    },
    [catalogModels, selectModel, selectedModel],
  );

  // Selection cascade: model types/models load -> first type/model auto-selected -> presets/datasets
  // load -> first preset + dataset auto-selected and the initial preview is
  // requested -> dataset/monitor lists are pruned to what the model supports.
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
        setSelectedExperimentRunId("");
      }
      setSelectedPreset(nextPreset);
      allowEmptyTrainingPresetDraftRef.current = Boolean(pendingSnapshotForModel);
      setSelectedTrainingPresets(pendingSnapshotForModel ? [] : [nextPreset]);
    }
  }, [
    datasetNames,
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
      allowEmptyTrainingPresetDraftRef.current =
        selectedTrainingPresets.length === 0;
      return;
    }
    if (!schemaQuery.isSuccess) {
      return;
    }
    allowEmptyTrainingPresetDraftRef.current =
      selectedTrainingPresets.length === 0;
    setSelectedTargetMode("snapshot");
    setSelectedSnapshotId(pendingConfigSnapshot.id);
    setSelectedExperimentRunId("");
    setSelectedTrainingSnapshotIds((current) =>
      current.includes(pendingConfigSnapshot.id)
        ? current
        : [...current, pendingConfigSnapshot.id],
    );
    setSnapshotEditorDraft({ ...pendingConfigSnapshot.overrides });
    lastRequestedPreviewTargetKeyRef.current = "";
    setPendingConfigSnapshot(null);
  }, [
    configSnapshotsQuery.isSuccess,
    pendingConfigSnapshot,
    presetNames,
    presetsQuery.isSuccess,
    schemaQuery.isSuccess,
    selectedModel,
    selectedModelType,
    selectedPreset,
    selectedTrainingPresets.length,
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
      setSelectedExperimentRunId("");
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
      setSelectedExperimentRunId("");
      setSnapshotEditorDraft({});
      setIsRestoringTargetSelection(false);
      return;
    }
    if (selectedPreset !== snapshot.preset) {
      setSelectedPreset(snapshot.preset);
      allowEmptyTrainingPresetDraftRef.current =
        selectedTrainingPresets.length === 0;
      return;
    }
    if (!schemaQuery.isSuccess) {
      return;
    }

    allowEmptyTrainingPresetDraftRef.current =
      selectedTrainingPresets.length === 0;
    setSelectedTargetMode("snapshot");
    setSelectedSnapshotId(snapshot.id);
    setSelectedExperimentRunId("");
    setSelectedTrainingSnapshotIds((current) =>
      current.includes(snapshot.id) ? current : [...current, snapshot.id],
    );
    setSnapshotEditorDraft((current) =>
      overrideValuesEqual(current, snapshot.overrides)
        ? current
        : { ...snapshot.overrides },
    );
    lastRequestedPreviewTargetKeyRef.current = "";
    setIsRestoringTargetSelection(false);
  }, [
    configSnapshotsQuery.isSuccess,
    configSnapshotsQuery.isError,
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
    selectedTrainingPresets.length,
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
    const previewDataset = selectedDatasets[0];
    if (!selectedModel || !selectedPreset || !previewDataset) {
      return;
    }
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

    const targetMode =
      selectedTargetMode === "snapshot" && selectedSnapshotId
        ? "snapshot"
        : selectedTargetMode === "experiment" && selectedExperimentRunId
          ? "experiment"
        : "preset";
    const targetKey = previewTargetKey({
      modelType: selectedModelType,
      model: selectedModel,
      dataset: previewDataset,
      mode: targetMode,
      target:
        targetMode === "snapshot"
          ? selectedSnapshotId
          : targetMode === "experiment"
            ? selectedExperimentRunId
            : selectedPreset,
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
    requestPreview({
      modelType: selectedModelType,
      model: selectedModel,
      preset: selectedPreset,
      dataset: previewDataset,
      overrides: { ...activeOverrides },
      targetMode,
      targetId:
        targetMode === "snapshot"
          ? selectedSnapshotId
          : targetMode === "experiment"
            ? selectedExperimentRunId
            : selectedPreset,
    });
  }, [
    activeOverrides,
    isRestoringTargetSelection,
    pendingConfigSnapshot,
    requestPreview,
    resetGraphSelectionAndExpansion,
    selectedConfigSnapshot,
    selectedDatasets,
    selectedModel,
    selectedModelType,
    selectedPreset,
    selectedSnapshotId,
    selectedExperimentRunId,
    selectedTargetMode,
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
    if (
      isRestoringTargetSelection &&
      initialTargetSelection?.selectedTargetMode === "snapshot"
    ) {
      return;
    }
    if (!selectedPreset || presetNames.length === 0) {
      setSelectedTrainingPresets((current) => {
        if (current.length === 0 && allowEmptyTrainingPresetDraftRef.current) {
          return current;
        }
        return current.length === 0 ? current : [];
      });
      return;
    }
    setSelectedTrainingPresets((current) => {
      if (skipNextTrainingPresetPrimarySyncRef.current) {
        skipNextTrainingPresetPrimarySyncRef.current = false;
        return current;
      }
      if (current.length === 0 && allowEmptyTrainingPresetDraftRef.current) {
        return current;
      }
      const next = normalizePrimarySelection(
        current,
        presetNames,
        selectedPreset || undefined,
      );
      return selectionValuesEqual(current, next) ? current : next;
    });
  }, [
    initialTargetSelection?.selectedTargetMode,
    isRestoringTargetSelection,
    presetNames,
    selectedPreset,
  ]);

  useEffect(() => {
    const snapshotIds = modelConfigSnapshots.map((snapshot) => snapshot.id);
    setSelectedTrainingSnapshotIds((current) => {
      const next = uniqueValidValues(current, snapshotIds);
      return selectionValuesEqual(current, next) ? current : next;
    });
  }, [modelConfigSnapshots]);

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
    const monitorNames = monitors.map((monitor) => monitor.name);
    setSelectedMonitors((current) => {
      const next = current.filter((monitor) => monitorNames.includes(monitor));
      return next.length === current.length ? current : next;
    });
  }, [monitors]);

  const syncSelectedLogRun = useCallback(
    (selectedLogRun: LogRun) => {
      if (!selectedModel) {
        return;
      }
      const preset = resolveRunPresetName(
        selectedLogRun,
        presets,
      );
      const dataset = datasetNames.includes(selectedLogRun.dataset)
        ? selectedLogRun.dataset
        : "";
      if (!preset || !dataset) {
        return;
      }
      const desiredTrainingPresets = [preset];
      const desiredDatasets = [dataset];
      const overridesAlreadyEmpty = overridesAreEmpty(presetOverrides);
      const alreadySynced =
        selectedTargetMode === "experiment" &&
        selectedExperimentRunId === selectedLogRun.id &&
        selectedPreset === preset &&
        selectedSnapshotId === "" &&
        selectionValuesEqual(selectedTrainingPresets, desiredTrainingPresets) &&
        selectionValuesEqual(selectedDatasets, desiredDatasets) &&
        overridesAlreadyEmpty;
      if (alreadySynced) {
        return;
      }

      setSelectedTargetMode("experiment");
      setSelectedSnapshotId("");
      setSelectedExperimentRunId(selectedLogRun.id);
      if (selectedPreset !== preset) {
        setSelectedPreset(preset);
      }
      allowEmptyTrainingPresetDraftRef.current = false;
      setSelectedTrainingPresets((current) =>
        selectionValuesEqual(current, desiredTrainingPresets)
          ? current
          : desiredTrainingPresets,
      );
      setSelectedDatasets((current) =>
        selectionValuesEqual(current, desiredDatasets) ? current : desiredDatasets,
      );
      if (!overridesAlreadyEmpty) {
        clearPresetOverrides();
      }
      lastRequestedPreviewTargetKeyRef.current = previewTargetKey({
        modelType: selectedModelType,
        model: selectedModel,
        dataset,
        mode: "experiment",
        target: selectedLogRun.id,
        overrides: {},
      });
      resetGraphSelectionAndExpansion();
      requestPreview({
        modelType: selectedModelType,
        model: selectedModel,
        preset,
        dataset,
        overrides: {},
        targetMode: "experiment",
        targetId: selectedLogRun.id,
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
      selectedExperimentRunId,
      selectedModel,
      selectedModelType,
      selectedPreset,
      selectedSnapshotId,
      selectedTargetMode,
      selectedTrainingPresets,
      setSelectedPreset,
    ],
  );

  const addConfigSnapshot = useCallback(
    (
      name: string,
      draftOverrides: OverrideValues = activeOverrides,
    ): ConfigSnapshotCreateResult => {
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
        overrides: draftOverrides,
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
        overrides: draftOverrides,
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
      if (!modelConfigSnapshots.some((snapshot) => snapshot.id === snapshotId)) {
        return;
      }
      setSelectedTrainingSnapshotIds((current) =>
        current.includes(snapshotId) ? current : [...current, snapshotId],
      );
    },
    [modelConfigSnapshots],
  );

  const excludeConfigSnapshot = useCallback(
    (snapshotId: string) => {
      if (!modelConfigSnapshots.some((snapshot) => snapshot.id === snapshotId)) {
        return;
      }
      setSelectedTrainingSnapshotIds((current) =>
        current.includes(snapshotId)
          ? current.filter((id) => id !== snapshotId)
          : current,
      );
    },
    [modelConfigSnapshots],
  );

  const setTrainingSnapshotSelection = useCallback(
    (snapshotIds: string[]) => {
      const validSnapshotIds = modelConfigSnapshots.map((snapshot) => snapshot.id);
      setSelectedTrainingSnapshotIds(uniqueValidValues(snapshotIds, validSnapshotIds));
    },
    [modelConfigSnapshots],
  );

  const toggleConfigSnapshotRunSelection = useCallback(
    (snapshotId: string) => {
      if (!modelConfigSnapshots.some((snapshot) => snapshot.id === snapshotId)) {
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
      modelConfigSnapshots,
      selectedTrainingSnapshotIds,
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
      const previewDataset = selectedDatasets[0];
      suppressedAutomaticPreviewTargetKeyRef.current =
        selectedModel && preset && previewDataset
          ? previewTargetKey({
              modelType: selectedModelType,
              model: selectedModel,
              dataset: previewDataset,
              mode: "preset",
              target: preset,
              overrides: effectivePresetOverrideValues,
            })
          : "";
    },
    [
      effectivePresetOverrideValues,
      selectedDatasets,
      selectedModel,
      selectedModelType,
    ],
  );

  const selectTrainingPrimaryPreset = useCallback(
    (preset: string) => {
      suppressAutomaticPreviewForPreset(preset);
      setSelectedTargetMode("preset");
      setSelectedSnapshotId("");
      setSelectedExperimentRunId("");
      onTargetPresetSelected?.();
      selectPreset(preset);
      allowEmptyTrainingPresetDraftRef.current = false;
      setSelectedTrainingPresets([preset]);
    },
    [onTargetPresetSelected, selectPreset, suppressAutomaticPreviewForPreset],
  );

  const selectTargetPreset = useCallback(
    (preset: string) => {
      const shouldRefreshPreview =
        selectedTargetMode !== "preset" ||
        selectedSnapshotId !== "" ||
        selectedExperimentRunId !== "" ||
        selectedPreset !== preset ||
        !overridesAreEmpty(effectivePresetOverrideValues);
      selectTrainingPrimaryPreset(preset);
      if (!shouldRefreshPreview) {
        return;
      }

      const previewDataset = selectedDatasets[0];
      if (!selectedModel || !preset || !previewDataset) {
        lastRequestedPreviewTargetKeyRef.current = "";
        return;
      }

      lastRequestedPreviewTargetKeyRef.current = previewTargetKey({
        modelType: selectedModelType,
        model: selectedModel,
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
        dataset: previewDataset,
        overrides: { ...effectivePresetOverrideValues },
        targetMode: "preset",
        targetId: preset,
      });
    },
    [
      effectivePresetOverrideValues,
      requestPreview,
      resetGraphSelectionAndExpansion,
      selectTrainingPrimaryPreset,
      selectedDatasets,
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
      { includeTrainingSnapshot = true }: TargetSnapshotSelectionOptions = {},
    ) => {
      const snapshot = modelConfigSnapshots.find(
        (candidate) => candidate.id === snapshotId,
      );
      if (!snapshot || !presetNames.includes(snapshot.preset)) {
        return false;
      }

      setSelectedTargetMode("snapshot");
      setSelectedSnapshotId(snapshot.id);
      setSelectedExperimentRunId("");
      onTargetSnapshotSelected?.();
      setSelectedPreset(snapshot.preset);
      allowEmptyTrainingPresetDraftRef.current =
        selectedTrainingPresets.length === 0;
      if (includeTrainingSnapshot) {
        setSelectedTrainingSnapshotIds((current) =>
          current.includes(snapshot.id) ? current : [...current, snapshot.id],
        );
      }
      setSnapshotEditorDraft({ ...snapshot.overrides });

      const previewDataset = selectedDatasets[0];
      if (!selectedModel || !snapshot.preset || !previewDataset) {
        lastRequestedPreviewTargetKeyRef.current = "";
        return true;
      }

      lastRequestedPreviewTargetKeyRef.current = previewTargetKey({
        modelType: selectedModelType,
        model: selectedModel,
        dataset: previewDataset,
        mode: "snapshot",
        target: snapshot.id,
        overrides: snapshot.overrides,
      });
      resetGraphSelectionAndExpansion();
      requestPreview({
        modelType: selectedModelType,
        model: selectedModel,
        preset: snapshot.preset,
        dataset: previewDataset,
        overrides: { ...snapshot.overrides },
        targetMode: "snapshot",
        targetId: snapshot.id,
      });
      return true;
    },
    [
      modelConfigSnapshots,
      presetNames,
      requestPreview,
      resetGraphSelectionAndExpansion,
      selectedDatasets,
      selectedModel,
      selectedModelType,
      selectedTrainingPresets.length,
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
      setSelectedExperimentRunId("");
      onTargetPresetSelected?.();
      lastRequestedPreviewTargetKeyRef.current = "";
      return;
    }
    selectTrainingPrimaryPreset(selectedPreset);

    const previewDataset = selectedDatasets[0];
    if (!selectedModel || !previewDataset) {
      lastRequestedPreviewTargetKeyRef.current = "";
      return;
    }

    lastRequestedPreviewTargetKeyRef.current = previewTargetKey({
      modelType: selectedModelType,
      model: selectedModel,
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
      dataset: previewDataset,
      overrides: { ...effectivePresetOverrideValues },
      targetMode: "preset",
      targetId: selectedPreset,
    });
  }, [
    effectivePresetOverrideValues,
    onTargetPresetSelected,
    requestPreview,
    resetGraphSelectionAndExpansion,
    selectTrainingPrimaryPreset,
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

  const setTrainingPresetSelection = useCallback(
    (presets: string[]) => {
      const validPresets = uniqueValidValues(presets, presetNames);
      if (validPresets.length === 0 && selectedTrainingSnapshotIds.length > 0) {
        allowEmptyTrainingPresetDraftRef.current = true;
        setSelectedTrainingPresets([]);
        return;
      }
      const fallbackPreset =
        selectedPreset && presetNames.includes(selectedPreset)
          ? selectedPreset
          : presetNames[0] ?? "";
      const nextPrimary = validPresets.includes(selectedPreset)
        ? selectedPreset
        : validPresets.length > 0
          ? validPresets[0]
          : fallbackPreset;

      allowEmptyTrainingPresetDraftRef.current = false;
      setSelectedTrainingPresets(
        normalizePrimarySelection(
          validPresets,
          presetNames,
          nextPrimary || undefined,
        ),
      );
    },
    [
      presetNames,
      selectedPreset,
      selectedTrainingSnapshotIds.length,
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
      if (!presetNames.includes(preset)) {
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
          presetNames,
          preset === selectedPreset || current.includes(selectedPreset)
            ? selectedPreset
            : undefined,
        );
      });
    },
    [presetNames, selectedPreset],
  );

  const excludeDraftTrainingPreset = useCallback(
    (preset: string) => {
      if (!presetNames.includes(preset)) {
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
    [presetNames],
  );

  const makeTrainingPresetPrimary = useCallback(
    (preset: string) => {
      if (!presetNames.includes(preset)) {
        return;
      }
      suppressAutomaticPreviewForPreset(preset);
      selectPreset(preset);
      setSelectedTargetMode("preset");
      setSelectedSnapshotId("");
      setSelectedExperimentRunId("");
      allowEmptyTrainingPresetDraftRef.current = false;
      setSelectedTrainingPresets((current) =>
        normalizePrimarySelection(
          [...current, preset],
          presetNames,
          preset || undefined,
        ),
      );
    },
    [presetNames, selectPreset, suppressAutomaticPreviewForPreset],
  );

  const preparePresetSnapshotDraft = useCallback(
    (
      preset: string,
      { includeTrainingPreset = true }: PresetSnapshotDraftOptions = {},
    ) => {
      if (!presetNames.includes(preset)) {
        return false;
      }
      suppressAutomaticPreviewForPreset(preset);
      selectPreset(preset);
      setSelectedTargetMode("preset");
      setSelectedSnapshotId("");
      setSelectedExperimentRunId("");
      if (!includeTrainingPreset) {
        allowEmptyTrainingPresetDraftRef.current =
          selectedTrainingPresets.length === 0;
        skipNextTrainingPresetPrimarySyncRef.current = true;
      }
      if (includeTrainingPreset) {
        allowEmptyTrainingPresetDraftRef.current = false;
        setSelectedTrainingPresets((current) =>
          normalizePrimarySelection(
            [...current, preset],
            presetNames,
            preset || undefined,
          ),
        );
      }
      setSnapshotEditorDraft({});
      return true;
    },
    [
      presetNames,
      selectPreset,
      selectedTrainingPresets.length,
      suppressAutomaticPreviewForPreset,
    ],
  );

  const selectAllTrainingPresets = useCallback(() => {
    allowEmptyTrainingPresetDraftRef.current = false;
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
    allowEmptyTrainingPresetDraftRef.current = false;
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

  const setMonitorSelection = useCallback(
    (monitorSelection: string[]) => {
      const monitorNames = monitors.map((monitor) => monitor.name);
      setSelectedMonitors(uniqueValidValues(monitorSelection, monitorNames));
    },
    [monitors],
  );

  const selectAllMonitors = useCallback(() => {
    setSelectedMonitors(monitors.map((monitor) => monitor.name));
  }, [monitors]);

  const clearMonitors = useCallback(() => {
    setSelectedMonitors([]);
  }, []);

  const updatePreview = useCallback(() => {
    const previewDataset = selectedDatasets[0];
    if (!selectedModel || !selectedPreset || !previewDataset) {
      return;
    }
    const targetMode =
      selectedTargetMode === "snapshot" && selectedSnapshotId
        ? "snapshot"
        : selectedTargetMode === "experiment" && selectedExperimentRunId
          ? "experiment"
        : "preset";
    lastRequestedPreviewTargetKeyRef.current = previewTargetKey({
      modelType: selectedModelType,
      model: selectedModel,
      dataset: previewDataset,
      mode: targetMode,
      target:
        targetMode === "snapshot"
          ? selectedSnapshotId
          : targetMode === "experiment"
            ? selectedExperimentRunId
            : selectedPreset,
      overrides: activeOverrides,
    });
    resetGraphSelectionAndExpansion();
    requestPreview({
      modelType: selectedModelType,
      model: selectedModel,
      preset: selectedPreset,
      dataset: previewDataset,
      overrides: { ...activeOverrides },
      targetMode,
      targetId:
        targetMode === "snapshot"
          ? selectedSnapshotId
          : targetMode === "experiment"
            ? selectedExperimentRunId
            : selectedPreset,
    });
  }, [
    activeOverrides,
    requestPreview,
    resetGraphSelectionAndExpansion,
    selectedDatasets,
    selectedModel,
    selectedModelType,
    selectedPreset,
    selectedExperimentRunId,
    selectedSnapshotId,
    selectedTargetMode,
  ]);

  const resetTargetOverrides = useCallback(
    (preserveTargetSelection: boolean) => {
      if (!preserveTargetSelection) {
        setSelectedTargetMode("preset");
        setSelectedSnapshotId("");
        setSelectedExperimentRunId("");
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
      const previewDataset = selectedDatasets[0];
      if (selectedModel && selectedPreset && previewDataset) {
        const targetMode =
          preserveTargetSelection &&
          selectedTargetMode === "snapshot" &&
          selectedSnapshotId
            ? "snapshot"
            : "preset";
        const nextOverrides = {};
        lastRequestedPreviewTargetKeyRef.current = previewTargetKey({
          modelType: selectedModelType,
          model: selectedModel,
          dataset: previewDataset,
          mode: targetMode,
          target: targetMode === "snapshot" ? selectedSnapshotId : selectedPreset,
          overrides: nextOverrides,
        });
        requestPreview({
          modelType: selectedModelType,
          model: selectedModel,
          preset: selectedPreset,
          dataset: previewDataset,
          overrides: nextOverrides,
          targetMode,
          targetId: targetMode === "snapshot" ? selectedSnapshotId : selectedPreset,
        });
      }
    },
    [
      clearPresetOverrides,
      requestPreview,
      resetGraphExpansion,
      selectedDatasets,
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
        setSelectedExperimentRunId("");
        onTargetPresetSelected?.();
      }
      updatePresetOverride(key, value);
    },
    [onTargetPresetSelected, selectedTargetMode, updatePresetOverride],
  );

  const clearTargetOverride = useCallback(
    (key: string, options?: { preserveTargetSelection?: boolean }) => {
      if (
        !options?.preserveTargetSelection &&
        (selectedTargetMode === "snapshot" || selectedTargetMode === "experiment")
      ) {
        setSelectedTargetMode("preset");
        setSelectedSnapshotId("");
        setSelectedExperimentRunId("");
        onTargetPresetSelected?.();
      }
      clearPresetOverride(key);
    },
    [clearPresetOverride, onTargetPresetSelected, selectedTargetMode],
  );

  const updateSnapshotEditorDraftOverride = useCallback(
    (key: string, value: string) => {
      setSnapshotEditorDraft((current) => ({ ...current, [key]: value }));
    },
    [],
  );

  const clearSnapshotEditorDraftOverride = useCallback((key: string) => {
    setSnapshotEditorDraft((current) => {
      const next = { ...current };
      delete next[key];
      return next;
    });
  }, []);

  const resetSnapshotEditorDraft = useCallback(() => {
    setSnapshotEditorDraft({});
  }, []);

  const apiOnline = healthQuery.data?.status === "ok";

  const target = useMemo(
    () => ({
      selectedModelType,
      selectModelType,
      selectedModel,
      selectModel,
      selectedTargetMode,
      activateTargetPresetMode,
      activateTargetSnapshotMode,
      activateTargetExperimentMode,
      selectedPreset,
      selectPreset: selectTrainingPrimaryPreset,
      selectTargetPreset,
      selectedSnapshotId,
      selectedConfigSnapshot,
      selectedExperimentRunId,
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
      selectedMonitors,
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
      configSections,
      overrideCount,
      presetOwnedFieldCount,
      fieldCount,
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
      updateOverride: updateTargetOverride,
      clearOverride: clearTargetOverride,
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
      presetsReady,
      isPresetsError,
      presetsError,
      datasets,
      isDatasetsError,
      datasetsError,
      monitors,
      monitorsLoading,
      isSchemaReady,
      schemaLoading,
      isSchemaError,
      schemaError,
      searchAxes,
      searchAxesLoading,
      libraryLoading,
      isLibraryError,
      libraryError,
    }),
    [
      activateTargetExperimentMode,
      activateTargetPresetMode,
      activateTargetSnapshotMode,
      addConfigSnapshot,
      apiOnline,
      capabilities,
      clearSnapshotEditorDraftOverride,
      clearMonitors,
      clearTargetOverride,
      configSections,
      configSnapshotGroups,
      configSnapshotLibrary,
      datasets,
      datasetsError,
      excludeConfigSnapshot,
      excludeDraftTrainingPreset,
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
      monitors,
      monitorsLoading,
      overrideCount,
      activeOverrideScope,
      activeOverrides,
      effectivePresetOverrideValues,
      inactiveLockedOverrideCount,
      inactiveLockedOverrides,
      preparePresetSnapshotDraft,
      prepareSelectedSnapshotEdit,
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
      schemaError,
      schemaLoading,
      searchAxes,
      searchAxesLoading,
      selectAllDatasets,
      selectAllMonitors,
      selectAllTrainingPresets,
      selectFirstDataset,
      selectModel,
      selectModelType,
      selectPrimaryTrainingPreset,
      selectTargetPreset,
      selectTargetSnapshot,
      selectTrainingPrimaryPreset,
      selectedConfigSnapshot,
      selectedDatasets,
      selectedExperimentRunId,
      selectedModel,
      selectedModelType,
      selectedMonitors,
      selectedPreset,
      selectedPresetMeta,
      selectedSnapshotId,
      selectedTargetMode,
      selectedTrainingPresets,
      selectedTrainingSnapshotIds,
      selectedTrainingSnapshots,
      setDatasetSelection,
      setMonitorSelection,
      setTrainingPresetSelection,
      setTrainingSnapshotSelection,
      toggleConfigSnapshotRunSelection,
      toggleDataset,
      toggleDraftTrainingPreset,
      toggleMonitor,
      toggleTrainingPreset,
      trainingSearch,
      snapshotEditorDraft,
      snapshotOverrideWarning,
      updateSnapshotEditorDraftOverride,
      updatePreview,
      updateSelectedConfigSnapshot,
      updateTargetOverride,
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
    }),
    [selection, syncSelectedLogRun, target],
  );
}
