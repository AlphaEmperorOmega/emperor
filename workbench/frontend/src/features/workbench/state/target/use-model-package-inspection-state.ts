import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type DatasetGroup,
  type LogRun,
  type ModelIdentity,
  type MonitorOption,
  type Preset,
} from "@/lib/api";
import {
  configKeyToken,
  effectivePresetOverrides,
  lockedOverrideKeys,
  normalizeAdaptiveOptionOverrides,
  normalizeConfigOverrides,
  overrideDigest,
  type ActiveOverrideScope,
  type OverrideValues,
} from "@/lib/config";
import {
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
  normalizeSelection,
  selectionValuesEqual,
} from "@/lib/selection";
import {
  useConfigSnapshotRecords,
} from "@/features/workbench/state/config-snapshots/use-config-snapshot-records";
import {
  readPersistedTargetSelection,
  writePersistedTargetSelection,
} from "@/features/workbench/state/target/target-selection-storage";
import { useWorkbenchQueries } from "@/features/workbench/state/use-workbench-queries";
import {
  deriveModelPackageSelection,
  datasetsForExperimentTask,
  experimentTaskOptions,
  normalizeExperimentTask,
} from "@/features/workbench/state/model-package/model-package-selection";
import {
  inspectionTargetKey,
  resolveInspectionTarget,
  type InspectionPreviewRequest,
  type TargetMode,
  useInspectionPreviewState,
} from "@/features/workbench/state/target/_inspection-preview";
import {
  useInspectionTargetState,
} from "@/features/workbench/state/target/_inspection-target-state";

const EMPTY_MODEL_IDS: ModelIdentity[] = [];
const EMPTY_PRESETS: Preset[] = [];
const EMPTY_DATASET_GROUPS: DatasetGroup[] = [];
const EMPTY_MONITORS: MonitorOption[] = [];

type ModelPackageInspectionStateOptions = {
  onModelSelected?: () => void;
  onTargetPresetSelected?: () => void;
  onTargetSnapshotSelected?: () => void;
  protectedReadsEnabled?: boolean;
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

function snapshotTargetSemanticKey({
  id,
  preset,
  overrides,
}: {
  id: string;
  preset: string;
  overrides: OverrideValues;
}) {
  return `${id}\u0000${preset}\u0000${overrideDigest(overrides)}`;
}

function shallowValuesEqual<Target extends Record<PropertyKey, unknown>>(
  left: Target,
  right: Target,
) {
  const keys = Reflect.ownKeys(left);
  return (
    keys.length === Reflect.ownKeys(right).length &&
    keys.every((key) => Object.is(left[key], right[key]))
  );
}

function useStableStateSlice<Target extends Record<PropertyKey, unknown>>(
  value: Target,
) {
  const stableValue = useRef(value);
  if (!shallowValuesEqual(stableValue.current, value)) {
    stableValue.current = value;
  }
  return stableValue.current;
}

export function useModelPackageInspectionState({
  onModelSelected,
  onTargetPresetSelected,
  onTargetSnapshotSelected,
  protectedReadsEnabled = true,
}: ModelPackageInspectionStateOptions) {
  const inspectionPreview = useInspectionPreviewState();
  const requestPreview = inspectionPreview.ensure;
  const clearPreview = inspectionPreview.clear;
  const refreshPreview = inspectionPreview.refresh;
  const clearPreviewForConnectionChange =
    inspectionPreview.clearForConnectionChange;
  const initialTargetSelection = useMemo(readPersistedTargetSelection, []);
  const [selectedModel, setSelectedModel] = useState(
    initialTargetSelection?.selectedModel ?? "",
  );
  const [selectedPreset, setSelectedPreset] = useState(
    initialTargetSelection?.selectedPreset ?? "",
  );
  const [presetOverrides, setPresetOverrides] = useState<OverrideValues>({});
  const clearPresetOverrides = useCallback(() => {
    setPresetOverrides({});
  }, []);
  const [selectedModelType, setSelectedModelType] = useState(
    initialTargetSelection?.selectedModelType ?? "",
  );
  const inspectionTargetState = useInspectionTargetState({
    initialPreset: initialTargetSelection?.selectedPreset ?? "",
    restorePersistedTarget: Boolean(initialTargetSelection),
  });
  const targetSelection = inspectionTargetState.target;
  const targetTransitions = inspectionTargetState.transitions;
  const targetRestoration = inspectionTargetState.restoration;
  const selectTargetModel = useCallback(
    (model: string) => {
      setSelectedModel(model);
      setSelectedPreset("");
      setPresetOverrides({});
      targetTransitions.toPreset("");
    },
    [targetTransitions],
  );
  const selectPreset = useCallback(
    (preset: string) => {
      setSelectedPreset(preset);
      if (targetSelection.kind === "preset") {
        targetTransitions.toPreset(preset);
      }
    },
    [targetSelection.kind, targetTransitions],
  );
  const selectedTargetMode: TargetMode =
    targetSelection.kind === "historical-run"
      ? "experiment"
      : targetSelection.kind;
  const [selectedTargetBrowserMode, setSelectedTargetBrowserMode] =
    useState<TargetMode>(initialTargetSelection?.selectedTargetMode ?? "preset");
  const selectedSnapshotId =
    targetSelection.kind === "snapshot" ? targetSelection.snapshotId : "";
  const selectedExperimentTarget =
    targetSelection.kind === "historical-run" ? targetSelection.run : null;
  const selectedExperimentRunId = selectedExperimentTarget?.runId ?? "";
  const selectedExperimentName = selectedExperimentTarget?.experiment ?? "";
  const selectedExperimentPreset = selectedExperimentTarget?.preset ?? "";
  const selectedExperimentDataset = selectedExperimentTarget?.dataset ?? "";
  const [selectedExperimentTask, setSelectedExperimentTask] = useState("");
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const lastRequestedPreviewTargetKeyRef = useRef("");
  const suppressedAutomaticPreviewTargetKeyRef = useRef("");
  const isRestoringTargetSelection = targetRestoration.isRestoring;
  const cancelTargetRestoration = targetRestoration.cancel;
  const settleTargetRestoration = targetRestoration.settle;
  const [inspectionTransition, setInspectionTransition] = useState<{
    revision: number;
    cause: "target-changed" | "inspection-refreshed";
  }>({ revision: 0, cause: "target-changed" });
  const emitInspectionTransition = useCallback(
    (cause: "target-changed" | "inspection-refreshed") => {
      setInspectionTransition((current) => ({
        revision: current.revision + 1,
        cause,
      }));
    },
    [],
  );
  const issueInspectionPreview = useCallback(
    (
      request: InspectionPreviewRequest,
      mode: "ensure" | "refresh" = "ensure",
    ) => {
      if (!protectedReadsEnabled) {
        lastRequestedPreviewTargetKeyRef.current = "";
        suppressedAutomaticPreviewTargetKeyRef.current = "";
        return false;
      }
      lastRequestedPreviewTargetKeyRef.current = inspectionTargetKey(request);
      if (mode === "refresh") {
        refreshPreview(request);
      } else {
        requestPreview(request);
      }
      return true;
    },
    [protectedReadsEnabled, refreshPreview, requestPreview],
  );
  const selectedModelIdentity = useMemo(
    () => ({ modelType: selectedModelType, model: selectedModel }),
    [selectedModel, selectedModelType],
  );
  const {
    records: configSnapshots,
    status: configSnapshotsStatus,
    actions: configSnapshotActions,
  } = useConfigSnapshotRecords(selectedModelIdentity, {
    enabled: protectedReadsEnabled,
  });
  const renameSnapshotRecord = configSnapshotActions.rename;
  const deleteSnapshotRecord = configSnapshotActions.remove;
  const {
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
    { includeSearchSpace: false, protectedReadsEnabled },
  );
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
  const catalogModels = models;
  const availableModelTypeOptions = useMemo(
    () => modelTypeOptions(catalogModels),
    [catalogModels],
  );

  const targetSelectionState = useMemo(
    () =>
      deriveModelPackageSelection({
        datasets,
        presets,
        schemaFields: schemaQuery.data?.fields,
        configSnapshots,
        selectedModelType,
        selectedModel,
        selectedPreset,
      }),
    [
      configSnapshots,
      datasets,
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
    presetOwnedFieldCount,
    fieldCount,
  } = targetSelectionState;
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
  }, [configFields, setPresetOverrides]);
  const selectedConfigSnapshot = useMemo(
    () =>
      modelConfigSnapshots.find(
        (snapshot) => snapshot.id === selectedSnapshotId,
      ),
    [modelConfigSnapshots, selectedSnapshotId],
  );
  const selectedSnapshotSchemaReady = Boolean(
    selectedConfigSnapshot &&
      schemaQuery.isSuccess &&
      selectedPreset === selectedConfigSnapshot.preset,
  );
  const effectivePresetOverrideValues = useMemo(
    () => effectivePresetOverrides(configFields, presetOverrides),
    [configFields, presetOverrides],
  );
  const selectedSnapshotOverrides = useMemo(
    () =>
      selectedConfigSnapshot && selectedSnapshotSchemaReady
        ? normalizeAdaptiveOptionOverrides(
            configFields,
            normalizeConfigOverrides(
              configFields,
              selectedConfigSnapshot.overrides,
            ),
          )
        : {},
    [configFields, selectedConfigSnapshot, selectedSnapshotSchemaReady],
  );
  const activeSnapshotSemanticKeyRef = useRef("");
  useEffect(() => {
    if (selectedTargetMode !== "snapshot") {
      activeSnapshotSemanticKeyRef.current = "";
      return;
    }
    if (
      !selectedConfigSnapshot ||
      !selectedSnapshotSchemaReady ||
      !presetNames.includes(selectedConfigSnapshot.preset)
    ) {
      return;
    }
    const semanticKey = snapshotTargetSemanticKey({
      id: selectedConfigSnapshot.id,
      preset: selectedConfigSnapshot.preset,
      overrides: selectedSnapshotOverrides,
    });
    if (!activeSnapshotSemanticKeyRef.current) {
      activeSnapshotSemanticKeyRef.current = semanticKey;
      return;
    }
    if (activeSnapshotSemanticKeyRef.current === semanticKey) {
      return;
    }
    activeSnapshotSemanticKeyRef.current = semanticKey;
    lastRequestedPreviewTargetKeyRef.current = "";
    emitInspectionTransition("target-changed");
  }, [
    emitInspectionTransition,
    presetNames,
    selectedConfigSnapshot,
    selectedSnapshotSchemaReady,
    selectedSnapshotOverrides,
    selectedTargetMode,
  ]);
  const inactiveLockedOverrideKeys = useMemo(
    () => lockedOverrideKeys(configFields, presetOverrides),
    [configFields, presetOverrides],
  );
  const activeOverrideScope: ActiveOverrideScope =
    selectedTargetMode === "snapshot" && selectedConfigSnapshot
      ? "snapshot"
      : "preset";
  const activeOverrides =
    activeOverrideScope === "snapshot"
      ? selectedSnapshotOverrides
      : effectivePresetOverrideValues;
  const overrideCount = Object.keys(activeOverrides).length;
  const inactiveLockedOverrideCount = inactiveLockedOverrideKeys.length;
  const currentInspectionRequest = useMemo<InspectionPreviewRequest | null>(() => {
    if (selectedTargetMode === "snapshot" && !selectedSnapshotSchemaReady) {
      return null;
    }
    const resolved = resolveInspectionTarget({
      selectedTargetMode,
      selectedSnapshotId,
      selectedExperimentTarget,
      selectedPreset,
      selectedExperimentTask: activeExperimentTask,
      selectedDatasets,
    });
    if (!selectedModel || !resolved.preset || !resolved.dataset) {
      return null;
    }
    return {
      modelType: selectedModelType,
      model: selectedModel,
      preset: resolved.preset,
      experimentTask: resolved.experimentTask || undefined,
      dataset: resolved.dataset,
      overrides: { ...activeOverrides },
      targetMode: resolved.targetMode,
      targetId: resolved.targetId,
      logRunId:
        resolved.targetMode === "experiment" ? resolved.targetId : undefined,
    };
  }, [
    activeExperimentTask,
    activeOverrides,
    selectedDatasets,
    selectedExperimentTarget,
    selectedModel,
    selectedModelType,
    selectedPreset,
    selectedSnapshotId,
    selectedSnapshotSchemaReady,
    selectedTargetMode,
  ]);
  const inspectionResponse = useMemo(() => {
    if (
      !currentInspectionRequest ||
      !inspectionPreview.request ||
      inspectionTargetKey(currentInspectionRequest) !==
        inspectionTargetKey(inspectionPreview.request)
    ) {
      return undefined;
    }
    return inspectionPreview.response;
  }, [
    currentInspectionRequest,
    inspectionPreview.request,
    inspectionPreview.response,
  ]);
  const clearInspectionForConnectionChange = useCallback(() => {
    lastRequestedPreviewTargetKeyRef.current = "";
    suppressedAutomaticPreviewTargetKeyRef.current = "";
    clearPreviewForConnectionChange();
  }, [clearPreviewForConnectionChange]);
  const selectModel = useCallback(
    (model: string, modelType = selectedModelType) => {
      const nextModel = modelNameForId(model);
      const nextModelType = model.includes("/") ? modelTypeForId(model) : modelType;
      if (
        !nextModel ||
        !catalogModels.some(
          (candidate) =>
            candidate.modelType === nextModelType &&
            candidate.model === nextModel,
        )
      ) {
        return false;
      }
      cancelTargetRestoration();
      lastRequestedPreviewTargetKeyRef.current = "";
      clearPreview();
      setSelectedModelType(nextModel ? nextModelType : "");
      setSelectedTargetBrowserMode("preset");
      selectTargetModel(nextModel);
      setSelectedExperimentTask("");
      setSelectedDatasets([]);
      onModelSelected?.();
      emitInspectionTransition("target-changed");
      return true;
    },
    [
      clearPreview,
      cancelTargetRestoration,
      catalogModels,
      onModelSelected,
      emitInspectionTransition,
      selectTargetModel,
      selectedModelType,
    ],
  );

  const selectModelType = useCallback(
    (modelType: string) => {
      const firstModel = modelsForType(catalogModels, modelType)[0];
      if (!firstModel) {
        return false;
      }
      cancelTargetRestoration();
      if (
        (firstModel.modelType !== selectedModelType ||
          firstModel.model !== selectedModel)
      ) {
        return selectModel(firstModel.model, firstModel.modelType);
      }
      return true;
    },
    [
      cancelTargetRestoration,
      catalogModels,
      selectModel,
      selectedModel,
      selectedModelType,
    ],
  );

  // Selection cascade: model types/models load -> first type/model auto-selected -> presets/datasets
  // load -> first preset + dataset auto-selected and the initial preview is
  // requested -> dataset/monitor lists are pruned to what the model supports.
  useEffect(() => {
    if (!datasetsQuery.isSuccess) {
      return;
    }
    setSelectedExperimentTask((current) => {
      const validTaskNames = datasetGroups.map((group) => group.experimentTask);
      const next = current && !validTaskNames.includes(current) ? "" : current;
      return current === next ? current : next;
    });
  }, [datasetGroups, datasetsQuery.isSuccess]);


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
    const firstPreset = presetNames[0];
    if (!firstPreset) {
      return;
    }
    if (!selectedPreset || !presetNames.includes(selectedPreset)) {
      const shouldKeepSnapshotTarget =
        selectedTargetMode === "snapshot" && selectedSnapshotId.length > 0;
      if (!shouldKeepSnapshotTarget) {
        setSelectedTargetBrowserMode("preset");
        targetTransitions.toPreset(firstPreset);
      }
      setSelectedPreset(firstPreset);
    }
  }, [
    datasetNames,
    presetNames,
    selectedPreset,
    selectedModel,
    selectedModelType,
    selectedSnapshotId,
    selectedTargetMode,
    setSelectedPreset,
    targetTransitions,
  ]);

  useEffect(() => {
    if (
      isRestoringTargetSelection ||
      selectedTargetMode !== "snapshot" ||
      !selectedSnapshotId ||
      !configSnapshotsStatus.isReady ||
      !presetsReady ||
      (selectedConfigSnapshot &&
        presetNames.includes(selectedConfigSnapshot.preset))
    ) {
      return;
    }
    clearPreview();
    setSelectedTargetBrowserMode("preset");
    targetTransitions.toPreset(selectedPreset);
    onTargetPresetSelected?.();
    lastRequestedPreviewTargetKeyRef.current = "";
    emitInspectionTransition("target-changed");
  }, [
    clearPreview,
    configSnapshotsStatus.isReady,
    emitInspectionTransition,
    isRestoringTargetSelection,
    onTargetPresetSelected,
    presetNames,
    presetsReady,
    selectedConfigSnapshot,
    selectedPreset,
    selectedSnapshotId,
    selectedTargetMode,
    targetTransitions,
  ]);

  useEffect(() => {
    if (!isRestoringTargetSelection) {
      return;
    }
    if (!initialTargetSelection) {
      settleTargetRestoration();
      return;
    }
    if (!selectedModel) {
      return;
    }
    if (
      selectedModelType !== initialTargetSelection.selectedModelType ||
      selectedModel !== initialTargetSelection.selectedModel
    ) {
      settleTargetRestoration();
      return;
    }
    if (
      !presetsQuery.isSuccess ||
      presetNames.length === 0 ||
      (!datasetsQuery.isSuccess && !datasetsQuery.isError)
    ) {
      return;
    }
    if (!selectedPreset || !presetNames.includes(selectedPreset)) {
      return;
    }
    if (initialTargetSelection.selectedTargetMode !== "snapshot") {
      targetTransitions.toPreset(selectedPreset);
      settleTargetRestoration();
      return;
    }
    if (!initialTargetSelection.selectedSnapshotId) {
      setSelectedTargetBrowserMode("preset");
      targetTransitions.toPreset(selectedPreset);
      settleTargetRestoration();
      return;
    }
    if (configSnapshotsStatus.isError || schemaQuery.isError) {
      setSelectedTargetBrowserMode("preset");
      targetTransitions.toPreset(selectedPreset);
      settleTargetRestoration();
      return;
    }
    if (!configSnapshotsStatus.isReady) {
      return;
    }

    const snapshot = modelConfigSnapshots.find(
      (candidate) => candidate.id === initialTargetSelection.selectedSnapshotId,
    );
    if (!snapshot || !presetNames.includes(snapshot.preset)) {
      setSelectedTargetBrowserMode("preset");
      targetTransitions.toPreset(selectedPreset);
      settleTargetRestoration();
      return;
    }
    if (selectedPreset !== snapshot.preset) {
      setSelectedPreset(snapshot.preset);
      return;
    }
    if (!schemaQuery.isSuccess) {
      return;
    }

    setSelectedTargetBrowserMode("snapshot");
    targetTransitions.toSnapshot(snapshot.id, snapshot.preset);
    lastRequestedPreviewTargetKeyRef.current = "";
    settleTargetRestoration();
  }, [
    configSnapshotsStatus.isReady,
    configSnapshotsStatus.isError,
    datasetsQuery.isError,
    datasetsQuery.isSuccess,
    initialTargetSelection,
    isRestoringTargetSelection,
    modelConfigSnapshots,
    presetNames,
    presetsQuery.isSuccess,
    schemaQuery.isSuccess,
    schemaQuery.isError,
    selectedModel,
    selectedModelType,
    selectedPreset,
    settleTargetRestoration,
    setSelectedPreset,
    targetTransitions,
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
    if (
      !protectedReadsEnabled ||
      !modelsQuery.isSuccess ||
      !presetsQuery.isSuccess ||
      !datasetsQuery.isSuccess
    ) {
      return;
    }
    if (isRestoringTargetSelection) {
      return;
    }
    if (selectedTargetMode === "snapshot" && selectedSnapshotId) {
      if (!selectedConfigSnapshot || !selectedSnapshotSchemaReady) {
        return;
      }
    }

    const {
      targetMode,
      targetId,
      preset: previewPreset,
      experimentTask: previewExperimentTask,
      dataset: previewDataset,
    } = resolveInspectionTarget({
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
    if (!previewPreset || !previewDataset) {
      return;
    }
    const targetKey = inspectionTargetKey({
      modelType: selectedModelType,
      model: selectedModel,
      preset: previewPreset,
      experimentTask: previewExperimentTask,
      dataset: previewDataset,
      targetMode,
      targetId,
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
    const previewRequest: InspectionPreviewRequest = {
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
    datasetsQuery.isSuccess,
    isRestoringTargetSelection,
    modelsQuery.isSuccess,
    presetsQuery.isSuccess,
    protectedReadsEnabled,
    requestPreview,
    selectedConfigSnapshot,
    selectedDatasets,
    selectedExperimentTarget,
    selectedModel,
    selectedModelType,
    selectedPreset,
    selectedSnapshotId,
    selectedSnapshotSchemaReady,
    selectedTargetMode,
  ]);

  useEffect(() => {
    if (!datasetsQuery.isSuccess) {
      return;
    }
    if (datasetNames.length === 0) {
      setSelectedDatasets((current) => (current.length === 0 ? current : []));
      return;
    }
    setSelectedDatasets((current) => {
      const next = normalizeSelection(current, datasetNames);
      return selectionValuesEqual(current, next) ? current : next;
    });
  }, [datasetNames, datasetsQuery.isSuccess]);

  const syncSelectedLogRun = useCallback(
    (selectedLogRun: LogRun) => {
      if (
        !selectedModel ||
        selectedLogRun.modelType !== selectedModelType ||
        selectedLogRun.model !== selectedModel
      ) {
        return;
      }
      cancelTargetRestoration();
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
        catalogDatasetSynced;
      if (alreadySynced) {
        return;
      }

      setSelectedTargetBrowserMode("experiment");
      targetTransitions.toHistoricalRun({
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
      emitInspectionTransition("target-changed");
      issueInspectionPreview({
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
      cancelTargetRestoration,
      datasetNames,
      presets,
      presetOverrides,
      issueInspectionPreview,
      emitInspectionTransition,
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
      targetTransitions,
    ],
  );

  const removeConfigSnapshot = useCallback(
    (snapshotId: string) => {
      if (
        selectedTargetMode === "snapshot" &&
        selectedSnapshotId === snapshotId
      ) {
        cancelTargetRestoration();
        setSelectedTargetBrowserMode("preset");
        targetTransitions.toPreset(selectedPreset);
        onTargetPresetSelected?.();
        lastRequestedPreviewTargetKeyRef.current = "";
        emitInspectionTransition("target-changed");
      }
      deleteSnapshotRecord(snapshotId);
    },
    [
      cancelTargetRestoration,
      deleteSnapshotRecord,
      emitInspectionTransition,
      onTargetPresetSelected,
      selectedSnapshotId,
      selectedPreset,
      selectedTargetMode,
      targetTransitions,
    ],
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

  const suppressAutomaticPreviewForPreset = useCallback(
    (preset: string) => {
      const previewDataset = selectedDatasets[0] ?? selectedExperimentDataset;
      suppressedAutomaticPreviewTargetKeyRef.current =
        selectedModel && preset && previewDataset
          ? inspectionTargetKey({
              modelType: selectedModelType,
              model: selectedModel,
              preset,
              experimentTask: activeExperimentTask,
              dataset: previewDataset,
              targetMode: "preset",
              targetId: preset,
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
      setSelectedTargetBrowserMode("preset");
      targetTransitions.toPreset(preset);
      onTargetPresetSelected?.();
      selectPreset(preset);
    },
    [
      onTargetPresetSelected,
      selectPreset,
      suppressAutomaticPreviewForPreset,
      targetTransitions,
    ],
  );

  const selectTargetPreset = useCallback(
    (preset: string) => {
      if (!presetNames.includes(preset)) {
        return false;
      }
      cancelTargetRestoration();
      const shouldRefreshPreview =
        selectedTargetMode !== "preset" ||
        selectedSnapshotId !== "" ||
        selectedExperimentRunId !== "" ||
        selectedPreset !== preset ||
        !overridesAreEmpty(effectivePresetOverrideValues);
      selectPreviewPreset(preset);
      if (!shouldRefreshPreview) {
        return true;
      }
      emitInspectionTransition("target-changed");

      const previewDataset = selectedDatasets[0] ?? selectedExperimentDataset;
      if (!selectedModel || !preset || !previewDataset) {
        lastRequestedPreviewTargetKeyRef.current = "";
        return true;
      }

      issueInspectionPreview({
        modelType: selectedModelType,
        model: selectedModel,
        preset,
        experimentTask: activeExperimentTask,
        dataset: previewDataset,
        overrides: { ...effectivePresetOverrideValues },
        targetMode: "preset",
        targetId: preset,
      });
      return true;
    },
    [
      effectivePresetOverrideValues,
      activeExperimentTask,
      cancelTargetRestoration,
      issueInspectionPreview,
      emitInspectionTransition,
      presetNames,
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
    (snapshotId: string) => {
      const snapshot = modelConfigSnapshots.find(
        (candidate) => candidate.id === snapshotId,
      );
      if (!snapshot || !presetNames.includes(snapshot.preset)) {
        return false;
      }

      cancelTargetRestoration();
      setSelectedTargetBrowserMode("snapshot");
      targetTransitions.toSnapshot(snapshot.id, snapshot.preset);
      onTargetSnapshotSelected?.();
      setSelectedPreset(snapshot.preset);
      const normalizedSnapshotOverrides = normalizeAdaptiveOptionOverrides(
        configFields,
        normalizeConfigOverrides(configFields, snapshot.overrides),
      );
      const snapshotSchemaReady =
        schemaQuery.isSuccess && selectedPreset === snapshot.preset;
      activeSnapshotSemanticKeyRef.current = snapshotSchemaReady
        ? snapshotTargetSemanticKey({
            id: snapshot.id,
            preset: snapshot.preset,
            overrides: normalizedSnapshotOverrides,
          })
        : "";
      emitInspectionTransition("target-changed");

      const previewDataset = selectedDatasets[0] ?? selectedExperimentDataset;
      if (
        !snapshotSchemaReady ||
        !selectedModel ||
        !snapshot.preset ||
        !previewDataset
      ) {
        lastRequestedPreviewTargetKeyRef.current = "";
        return true;
      }

      issueInspectionPreview({
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
      cancelTargetRestoration,
      configFields,
      presetNames,
      issueInspectionPreview,
      schemaQuery.isSuccess,
      emitInspectionTransition,
      selectedDatasets,
      selectedExperimentDataset,
      selectedModel,
      selectedModelType,
      selectedPreset,
      onTargetSnapshotSelected,
      setSelectedPreset,
      targetTransitions,
    ],
  );

  const activateTargetPresetMode = useCallback(() => {
    cancelTargetRestoration();
    setSelectedTargetBrowserMode("preset");
  }, [cancelTargetRestoration]);

  const activateTargetSnapshotMode = useCallback(() => {
    if (modelConfigSnapshots.length === 0) {
      return false;
    }
    cancelTargetRestoration();
    setSelectedTargetBrowserMode("snapshot");
    return true;
  }, [cancelTargetRestoration, modelConfigSnapshots.length]);

  const activateTargetExperimentMode = useCallback(() => {
    cancelTargetRestoration();
    setSelectedTargetBrowserMode("experiment");
  }, [cancelTargetRestoration]);

  const selectExperimentTask = useCallback(
    (experimentTask: string) => {
      if (
        !datasetGroups.some(
          (group) => group.experimentTask === experimentTask,
        )
      ) {
        return false;
      }
      cancelTargetRestoration();
      const nextTask = normalizeExperimentTask(
        experimentTask,
        defaultExperimentTask,
        datasetGroups,
      );
      const nextDatasetNames = datasetsForExperimentTask(
        datasetGroups,
        nextTask,
      ).map((dataset) => dataset.name);
      const nextDatasets = normalizeSelection(
        selectedDatasets,
        nextDatasetNames,
      );
      if (
        nextTask === activeExperimentTask &&
        selectionValuesEqual(nextDatasets, selectedDatasets)
      ) {
        return true;
      }
      setSelectedExperimentTask(nextTask);
      setSelectedDatasets(nextDatasets);
      if (selectedTargetMode !== "experiment") {
        emitInspectionTransition("target-changed");
      }
      return true;
    },
    [
      activeExperimentTask,
      cancelTargetRestoration,
      datasetGroups,
      defaultExperimentTask,
      emitInspectionTransition,
      selectedDatasets,
      selectedTargetMode,
    ],
  );

  const updatePreview = useCallback(() => {
    cancelTargetRestoration();
    const {
      targetMode,
      targetId,
      preset: previewPreset,
      experimentTask: previewExperimentTask,
      dataset: previewDataset,
    } = resolveInspectionTarget({
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
    emitInspectionTransition("inspection-refreshed");
    issueInspectionPreview({
      modelType: selectedModelType,
      model: selectedModel,
      preset: previewPreset,
      experimentTask: previewExperimentTask || undefined,
      dataset: previewDataset,
      overrides: { ...activeOverrides },
      targetMode,
      targetId,
      logRunId: targetMode === "experiment" ? targetId : undefined,
    }, "refresh");
  }, [
    activeOverrides,
    activeExperimentTask,
    cancelTargetRestoration,
    emitInspectionTransition,
    issueInspectionPreview,
    selectedDatasets,
    selectedExperimentTarget,
    selectedModel,
    selectedModelType,
    selectedPreset,
    selectedSnapshotId,
    selectedTargetMode,
  ]);

  const resetOverrides = useCallback(() => {
    cancelTargetRestoration();
    setSelectedTargetBrowserMode("preset");
    targetTransitions.toPreset(selectedPreset);
    onTargetPresetSelected?.();
    clearPresetOverrides();
    emitInspectionTransition("target-changed");
    const previewDataset = selectedDatasets[0] ?? "";
    if (!selectedModel || !selectedPreset || !previewDataset) {
      lastRequestedPreviewTargetKeyRef.current = "";
      return;
    }
    issueInspectionPreview({
      modelType: selectedModelType,
      model: selectedModel,
      preset: selectedPreset,
      experimentTask: activeExperimentTask || undefined,
      dataset: previewDataset,
      overrides: {},
      targetMode: "preset",
      targetId: selectedPreset,
    });
  }, [
    activeExperimentTask,
    cancelTargetRestoration,
    clearPresetOverrides,
    onTargetPresetSelected,
    issueInspectionPreview,
    emitInspectionTransition,
    selectedDatasets,
    selectedModel,
    selectedModelType,
    selectedPreset,
    targetTransitions,
  ]);

  const updateTargetOverride = useCallback(
    (
      key: string,
      value: string,
    ) => {
      cancelTargetRestoration();
      const exitsNonPresetTarget =
        selectedTargetMode === "snapshot" || selectedTargetMode === "experiment";
      if (exitsNonPresetTarget) {
        setSelectedTargetBrowserMode("preset");
        targetTransitions.toPreset(selectedPreset);
        onTargetPresetSelected?.();
      }
      const nextOverrides = normalizeAdaptiveOptionOverrides(
        configFields,
        normalizeConfigOverrides(configFields, {
          ...presetOverrides,
          [key]: value,
        }),
      );
      if (overrideValuesEqual(nextOverrides, presetOverrides)) {
        if (exitsNonPresetTarget) {
          emitInspectionTransition("target-changed");
        }
        return;
      }
      setPresetOverrides(nextOverrides);
      emitInspectionTransition("target-changed");
    },
    [
      cancelTargetRestoration,
      configFields,
      emitInspectionTransition,
      onTargetPresetSelected,
      presetOverrides,
      selectedPreset,
      selectedTargetMode,
      setPresetOverrides,
      targetTransitions,
    ],
  );

  const clearTargetOverride = useCallback(
    (key: string) => {
      cancelTargetRestoration();
      const exitsNonPresetTarget =
        selectedTargetMode === "snapshot" || selectedTargetMode === "experiment";
      if (exitsNonPresetTarget) {
        setSelectedTargetBrowserMode("preset");
        targetTransitions.toPreset(selectedPreset);
        onTargetPresetSelected?.();
      }
      const nextOverrides = normalizeAdaptiveOptionOverrides(
        configFields,
        normalizeConfigOverrides(
          configFields,
          withoutOverride(presetOverrides, key),
        ),
      );
      if (overrideValuesEqual(nextOverrides, presetOverrides)) {
        if (exitsNonPresetTarget) {
          emitInspectionTransition("target-changed");
        }
        return;
      }
      setPresetOverrides(nextOverrides);
      emitInspectionTransition("target-changed");
    },
    [
      cancelTargetRestoration,
      configFields,
      emitInspectionTransition,
      onTargetPresetSelected,
      presetOverrides,
      selectedPreset,
      selectedTargetMode,
      setPresetOverrides,
      targetTransitions,
    ],
  );

  const inspectionTarget = useMemo(() => {
    const modelPackage = {
      modelType: selectedModelType,
      model: selectedModel,
    };
    if (selectedTargetMode === "experiment" && selectedExperimentTarget) {
      return {
        kind: "historical-run" as const,
        modelPackage,
        preset: selectedExperimentTarget.preset,
        experimentTask: selectedExperimentTarget.experimentTask ?? "",
        datasets: [selectedExperimentTarget.dataset],
        run: selectedExperimentTarget,
      };
    }
    if (selectedTargetMode === "snapshot" && selectedConfigSnapshot) {
      return {
        kind: "snapshot" as const,
        modelPackage,
        preset: selectedConfigSnapshot.preset,
        experimentTask: activeExperimentTask,
        datasets: selectedDatasets,
        snapshot: selectedConfigSnapshot,
      };
    }
    return {
      kind: "preset" as const,
      modelPackage,
      preset: selectedPreset,
      experimentTask: activeExperimentTask,
      datasets: selectedDatasets,
    };
  }, [
    activeExperimentTask,
    selectedConfigSnapshot,
    selectedDatasets,
    selectedExperimentTarget,
    selectedModel,
    selectedModelType,
    selectedPreset,
    selectedTargetMode,
  ]);
  const browser = useStableStateSlice({
    mode: selectedTargetBrowserMode,
    selectedModelType,
    selectedModel,
    selectedPreset,
    selectedSnapshotId,
    selectedExperimentTask: activeExperimentTask,
    selectedDatasets,
  });
  const options = useStableStateSlice({
    presets,
    selectedPreset: selectedPresetMeta,
    experimentTasks: experimentTaskOptionsList,
    monitorMetadata: targetMonitors,
    configSections,
  });
  const runtimeDefaults = useStableStateSlice({
    effectivePreset: effectivePresetOverrideValues,
    active: activeOverrides,
    inactiveLockedCount: inactiveLockedOverrideCount,
    overrideCount,
    presetOwnedFieldCount,
    fieldCount,
  });
  const presetsStatus = useStableStateSlice({
    isError: isPresetsError,
    error: presetsError,
  });
  const datasetsStatus = useStableStateSlice({
    isError: isDatasetsError,
    error: datasetsError,
  });
  const monitorsStatus = useStableStateSlice({
    isLoading: targetMonitorsLoading,
  });
  const schemaStatus = useStableStateSlice({
    isReady: isSchemaReady,
    isLoading: schemaLoading,
    isError: isSchemaError,
    error: schemaError,
  });
  const status = useStableStateSlice({
    presets: presetsStatus,
    datasets: datasetsStatus,
    monitors: monitorsStatus,
    schema: schemaStatus,
  });
  const actions = useStableStateSlice({
    selectModelType,
    selectModelPackage: selectModel,
    selectPresetTarget: selectTargetPreset,
    selectSnapshotTarget: selectTargetSnapshot,
    showPresetTarget: activateTargetPresetMode,
    showSnapshotTarget: activateTargetSnapshotMode,
    browseHistoricalRuns: activateTargetExperimentMode,
    selectExperimentTask,
    editRuntimeDefault: updateTargetOverride,
    clearRuntimeDefault: clearTargetOverride,
    resetRuntimeDefaults: resetOverrides,
    refreshInspection: updatePreview,
  });
  const modelPackages = useStableStateSlice({
    records: models,
    isLoading: modelsLoading,
    isError: isModelsError,
    error: modelsError,
  });
  const catalog = useStableStateSlice({
    modelPackages,
  });
  const model = useStableStateSlice({
    target: inspectionTarget,
    browser,
    options,
    runtimeDefaults,
    status,
    actions,
  });
  const snapshotRecords = useStableStateSlice({
    all: modelConfigSnapshots,
    allGroups: modelConfigSnapshotGroups,
    allCount: modelConfigSnapshots.length,
  });
  const snapshotActions = useStableStateSlice({
    selectTarget: selectTargetSnapshot,
    remove: removeConfigSnapshot,
    rename: renameConfigSnapshot,
  });
  const snapshots = useStableStateSlice({
    records: snapshotRecords,
    actions: snapshotActions,
  });
  const inspection = useStableStateSlice({
    graph: inspectionResponse,
    status: inspectionPreview.status,
    transition: inspectionTransition,
    clear: inspectionPreview.clear,
    clearForConnectionChange: clearInspectionForConnectionChange,
  });
  const contexts = useMemo(
    () => ({ catalog, model, snapshots }),
    [catalog, model, snapshots],
  );
  return useMemo(
    () => ({
      contexts,
      selectHistoricalRunTarget: syncSelectedLogRun,
      inspection,
    }),
    [
      contexts,
      inspection,
      syncSelectedLogRun,
    ],
  );
}
