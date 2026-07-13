import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type LogRun,
} from "@/lib/api";
import {
  effectivePresetOverrides,
  lockedOverrideKeys,
  runtimeDefaultsEditor,
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
import { useModelPackageMetadata } from "@/features/workbench/state/model-package/use-model-package-metadata";
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
  createInspectionTargetLifecycleState,
  useInspectionTargetLifecycle,
} from "@/features/workbench/state/target/_inspection-target-state";
import { useHistoricalTargetBrowsing } from "@/features/workbench/state/target/use-historical-target-browsing";


type ModelPackageInspectionStateOptions = {
  historicalRunsEnabled?: boolean;
  protectedReadsEnabled?: boolean;
};

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
  historicalRunsEnabled = false,
  protectedReadsEnabled = true,
}: ModelPackageInspectionStateOptions) {
  const inspectionPreview = useInspectionPreviewState();
  const requestPreview = inspectionPreview.ensure;
  const clearPreview = inspectionPreview.clear;
  const refreshPreview = inspectionPreview.refresh;
  const clearPreviewForConnectionChange =
    inspectionPreview.clearForConnectionChange;
  const initialTargetSelection = useMemo(readPersistedTargetSelection, []);
  const initialLifecycleState = useMemo(
    () =>
      createInspectionTargetLifecycleState({
        modelPackage: {
          modelType: initialTargetSelection?.selectedModelType ?? "",
          model: initialTargetSelection?.selectedModel ?? "",
        },
        preset: initialTargetSelection?.selectedPreset ?? "",
        restoreSnapshotId:
          initialTargetSelection?.selectedTargetMode === "snapshot"
            ? initialTargetSelection.selectedSnapshotId
            : "",
        restorePersistedTarget: Boolean(initialTargetSelection),
      }),
    [initialTargetSelection],
  );
  const { state: targetLifecycle, send: sendTargetEvent } =
    useInspectionTargetLifecycle(initialLifecycleState);
  const selectedModelType = targetLifecycle.modelPackage.modelType;
  const selectedModel = targetLifecycle.modelPackage.model;
  const selectedPreset = targetLifecycle.selectedPreset;
  const presetOverrides = targetLifecycle.runtimeDefaults.preset;
  const targetSelection = targetLifecycle.target;
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
  const selectedExperimentTask = targetLifecycle.experimentTask;
  const selectedDatasets = targetLifecycle.datasets;
  const lastRequestedPreviewTargetKeyRef = useRef("");
  const isRestoringTargetSelection =
    targetLifecycle.restoration.phase !== "settled";
  const cancelTargetRestoration = useCallback(() => {
    sendTargetEvent({ type: "restoration-cancelled" });
  }, [sendTargetEvent]);
  const settleTargetRestoration = useCallback(() => {
    sendTargetEvent({ type: "restoration-settled" });
  }, [sendTargetEvent]);
  const inspectionTransition = targetLifecycle.transition;
  const issueInspectionPreview = useCallback(
    (
      request: InspectionPreviewRequest,
      mode: "ensure" | "refresh" = "ensure",
    ) => {
      if (!protectedReadsEnabled) {
        lastRequestedPreviewTargetKeyRef.current = "";
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
  const retrySnapshotRecordMutation = configSnapshotActions.retry;
  const dismissSnapshotRecordMutation = configSnapshotActions.dismissMutation;
  const clearSnapshotRecordsForConnectionChange =
    configSnapshotActions.clearForConnectionChange;
  const metadataSelection = useMemo(
    () => ({
      modelPackage: selectedModelIdentity,
      preset: selectedPreset,
      searchPresets: [] as string[],
    }),
    [selectedModelIdentity, selectedPreset],
  );
  const metadata = useModelPackageMetadata(metadataSelection, {
    includeSearchMetadata: false,
    protectedReadsEnabled,
  });
  const models = metadata.modelPackages.records;
  const modelsLoading = metadata.modelPackages.isLoading;
  const isModelsError = metadata.modelPackages.isError;
  const modelsError = metadata.modelPackages.error;
  const presets = metadata.presets.records;
  const presetsReady = metadata.presets.isReady;
  const isPresetsError = metadata.presets.isError;
  const presetsError = metadata.presets.error;
  const datasetGroups = metadata.datasetMetadata.groups;
  const defaultExperimentTask = metadata.datasetMetadata.defaultExperimentTask;
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
  const isDatasetsError = metadata.datasetMetadata.isError;
  const datasetsError = metadata.datasetMetadata.error;
  const targetMonitors = metadata.monitorMetadata.records;
  const targetMonitorsLoading = metadata.monitorMetadata.isLoading;
  const isSchemaReady = metadata.runtimeDefaults.isReady;
  const schemaLoading = metadata.runtimeDefaults.isLoading;
  const isSchemaError = metadata.runtimeDefaults.isError;
  const schemaError = metadata.runtimeDefaults.error;
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
        schemaFields: metadata.runtimeDefaults.fields,
        configSnapshots,
        selectedModelType,
        selectedModel,
        selectedPreset,
      }),
    [
      configSnapshots,
      datasets,
      presets,
      metadata.runtimeDefaults.fields,
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
    const normalized = runtimeDefaultsEditor.normalize(
      configFields,
      presetOverrides,
    );
    sendTargetEvent({
      type: "runtime-defaults-normalized",
      presetRuntimeDefaults: normalized,
      activeRuntimeDefaults: effectivePresetOverrides(configFields, normalized),
      expectedRevision: targetLifecycle.transition.revision,
    });
  }, [
    configFields,
    presetOverrides,
    sendTargetEvent,
    targetLifecycle.transition.revision,
  ]);
  const selectedConfigSnapshot = useMemo(
    () =>
      modelConfigSnapshots.find(
        (snapshot) => snapshot.id === selectedSnapshotId,
      ),
    [modelConfigSnapshots, selectedSnapshotId],
  );
  const selectedSnapshotSchemaReady = Boolean(
    selectedConfigSnapshot &&
      metadata.runtimeDefaults.isReady &&
      selectedPreset === selectedConfigSnapshot.preset,
  );
  const effectivePresetOverrideValues = useMemo(
    () => effectivePresetOverrides(configFields, presetOverrides),
    [configFields, presetOverrides],
  );
  const selectedSnapshotOverrides = useMemo(
    () =>
      selectedConfigSnapshot && selectedSnapshotSchemaReady
        ? runtimeDefaultsEditor.replace(
            configFields,
            selectedConfigSnapshot.overrides,
          )
        : {},
    [configFields, selectedConfigSnapshot, selectedSnapshotSchemaReady],
  );
  useEffect(() => {
    if (
      selectedTargetMode !== "snapshot" ||
      !selectedConfigSnapshot ||
      !selectedSnapshotSchemaReady ||
      !presetNames.includes(selectedConfigSnapshot.preset)
    ) {
      return;
    }
    sendTargetEvent({
      type: "snapshot-runtime-defaults-refreshed",
      snapshotId: selectedConfigSnapshot.id,
      preset: selectedConfigSnapshot.preset,
      runtimeDefaults: selectedSnapshotOverrides,
      expectedRevision: targetLifecycle.transition.revision,
    });
  }, [
    presetNames,
    sendTargetEvent,
    selectedConfigSnapshot,
    selectedSnapshotSchemaReady,
    selectedSnapshotOverrides,
    selectedTargetMode,
    targetLifecycle.transition.revision,
  ]);
  const inactiveLockedOverrideKeys = useMemo(
    () => lockedOverrideKeys(configFields, presetOverrides),
    [configFields, presetOverrides],
  );
  const activeOverrides = targetLifecycle.runtimeDefaults.active;
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
      ...(resolved.targetMode === "experiment"
        ? { logRunId: resolved.targetId }
        : {}),
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
  const syncSelectedLogRun = useCallback(
    (selectedLogRun: LogRun) => {
      if (
        !selectedModel ||
        selectedLogRun.modelType !== selectedModelType ||
        selectedLogRun.model !== selectedModel
      ) {
        return;
      }
      const rawPreset = selectedLogRun.preset;
      const rawDataset = selectedLogRun.dataset;
      const runExperimentTask = selectedLogRun.experimentTask ?? null;
      const catalogPreset = resolveRunPresetName(selectedLogRun, presets);
      const catalogDataset = datasetNames.includes(rawDataset) ? rawDataset : "";
      const catalogPresetSynced = !catalogPreset || selectedPreset === catalogPreset;
      const catalogDatasetSynced =
        !catalogDataset || selectionValuesEqual(selectedDatasets, [catalogDataset]);
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

      cancelTargetRestoration();
      setSelectedTargetBrowserMode("experiment");
      sendTargetEvent({
        type: "historical-run-selected",
        run: {
          runId: selectedLogRun.id,
          experiment: selectedLogRun.experiment,
          preset: rawPreset,
          dataset: rawDataset,
          experimentTask: runExperimentTask,
        },
        catalogPreset,
        catalogDataset,
      });
    },
    [
      cancelTargetRestoration,
      datasetNames,
      presets,
      sendTargetEvent,
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
    ],
  );
  const historicalTargetActive =
    selectedTargetBrowserMode === "experiment" ||
    selectedTargetMode === "experiment";
  const historicalTarget = useHistoricalTargetBrowsing({
    selectedModelType,
    selectedModel,
    selectedExperimentTask: activeExperimentTask,
    runsEnabled:
      protectedReadsEnabled && (historicalRunsEnabled || historicalTargetActive),
    tagsEnabled: protectedReadsEnabled && historicalTargetActive,
  });
  const selectedHistoricalRun = historicalTarget.coordination.selectedRun;
  const clearHistoricalTarget =
    historicalTarget.coordination.clearForTargetChange;
  useEffect(() => {
    if (selectedHistoricalRun) {
      syncSelectedLogRun(selectedHistoricalRun);
    }
  }, [selectedHistoricalRun, syncSelectedLogRun]);
  const clearInspectionForConnectionChange = useCallback(() => {
    lastRequestedPreviewTargetKeyRef.current = "";
    sendTargetEvent({ type: "connection-reset" });
    clearHistoricalTarget();
    clearSnapshotRecordsForConnectionChange();
    clearPreviewForConnectionChange();
  }, [
    clearHistoricalTarget,
    clearPreviewForConnectionChange,
    clearSnapshotRecordsForConnectionChange,
    sendTargetEvent,
  ]);
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
      setSelectedTargetBrowserMode("preset");
      sendTargetEvent({
        type: "model-package-selected",
        modelPackage: { modelType: nextModelType, model: nextModel },
      });
      clearHistoricalTarget();
      return true;
    },
    [
      clearPreview,
      cancelTargetRestoration,
      catalogModels,
      clearHistoricalTarget,
      sendTargetEvent,
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

  // Query results enter the lifecycle as events. Effects never repair an
  // independently-owned target field.
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
    if (!metadata.presets.isReady || !metadata.datasetMetadata.isReady) {
      return;
    }
    const nextPreset = presetNames.includes(selectedPreset)
      ? selectedPreset
      : presetNames[0] ?? "";
    const nextExperimentTask = normalizeExperimentTask(
      selectedExperimentTask,
      defaultExperimentTask,
      datasetGroups,
    );
    const nextDatasetNames = datasetsForExperimentTask(
      datasetGroups,
      nextExperimentTask,
    ).map((dataset) => dataset.name);
    const nextDatasets =
      nextDatasetNames.length === 0
        ? []
        : normalizeSelection(selectedDatasets, nextDatasetNames);
    if (targetSelection.kind === "preset" && selectedPreset !== nextPreset) {
      setSelectedTargetBrowserMode("preset");
    }
    sendTargetEvent({
      type: "metadata-refreshed",
      preset: nextPreset,
      experimentTask: nextExperimentTask,
      datasets: nextDatasets,
      expectedRevision: targetLifecycle.transition.revision,
    });
  }, [
    datasetGroups,
    defaultExperimentTask,
    metadata.datasetMetadata.isReady,
    metadata.presets.isReady,
    presetNames,
    selectedDatasets,
    selectedExperimentTask,
    selectedPreset,
    sendTargetEvent,
    targetSelection.kind,
    targetLifecycle.transition.revision,
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
    sendTargetEvent({ type: "missing-snapshot-fallback" });
    clearHistoricalTarget();
    lastRequestedPreviewTargetKeyRef.current = "";
  }, [
    clearPreview,
    clearHistoricalTarget,
    configSnapshotsStatus.isReady,
    isRestoringTargetSelection,
    presetNames,
    presetsReady,
    selectedConfigSnapshot,
    selectedPreset,
    selectedSnapshotId,
    selectedTargetMode,
    sendTargetEvent,
  ]);

  useEffect(() => {
    if (!isRestoringTargetSelection) {
      return;
    }
    if (!selectedModel) {
      return;
    }
    if (
      !metadata.presets.isReady ||
      presetNames.length === 0 ||
      (!metadata.datasetMetadata.isReady && !metadata.datasetMetadata.isError)
    ) {
      return;
    }
    if (!selectedPreset || !presetNames.includes(selectedPreset)) {
      return;
    }
    const requestedSnapshotId =
      targetLifecycle.restoration.requestedSnapshotId;
    if (!requestedSnapshotId) {
      settleTargetRestoration();
      return;
    }
    if (configSnapshotsStatus.isError || metadata.runtimeDefaults.isError) {
      setSelectedTargetBrowserMode("preset");
      sendTargetEvent({ type: "missing-snapshot-fallback" });
      return;
    }
    if (!configSnapshotsStatus.isReady) {
      return;
    }

    const snapshot = modelConfigSnapshots.find(
      (candidate) => candidate.id === requestedSnapshotId,
    );
    if (!snapshot || !presetNames.includes(snapshot.preset)) {
      setSelectedTargetBrowserMode("preset");
      sendTargetEvent({ type: "missing-snapshot-fallback" });
      return;
    }
    if (selectedPreset !== snapshot.preset) {
      sendTargetEvent({
        type: "preset-metadata-selected",
        preset: snapshot.preset,
      });
      return;
    }
    if (!metadata.runtimeDefaults.isReady) {
      return;
    }

    setSelectedTargetBrowserMode("snapshot");
    sendTargetEvent({
      type: "restoration-snapshot-resolved",
      snapshotId: snapshot.id,
      preset: snapshot.preset,
      runtimeDefaults: runtimeDefaultsEditor.replace(
        configFields,
        snapshot.overrides,
      ),
    });
    lastRequestedPreviewTargetKeyRef.current = "";
  }, [
    configFields,
    configSnapshotsStatus.isReady,
    configSnapshotsStatus.isError,
    metadata.datasetMetadata.isError,
    metadata.datasetMetadata.isReady,
    isRestoringTargetSelection,
    modelConfigSnapshots,
    presetNames,
    metadata.presets.isReady,
    metadata.runtimeDefaults.isReady,
    metadata.runtimeDefaults.isError,
    selectedModel,
    selectedModelType,
    selectedPreset,
    sendTargetEvent,
    settleTargetRestoration,
    targetLifecycle.restoration.requestedSnapshotId,
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
      !metadata.modelPackages.isReady ||
      !metadata.presets.isReady ||
      !metadata.datasetMetadata.isReady
    ) {
      return;
    }
    if (isRestoringTargetSelection) {
      return;
    }
    if (!currentInspectionRequest) {
      return;
    }
    const targetKey = inspectionTargetKey(currentInspectionRequest);
    if (lastRequestedPreviewTargetKeyRef.current === targetKey) {
      return;
    }

    lastRequestedPreviewTargetKeyRef.current = targetKey;
    requestPreview(currentInspectionRequest);
  }, [
    currentInspectionRequest,
    metadata.datasetMetadata.isReady,
    isRestoringTargetSelection,
    metadata.modelPackages.isReady,
    metadata.presets.isReady,
    protectedReadsEnabled,
    requestPreview,
  ]);

  const renameConfigSnapshot = useCallback(
    async (snapshotId: string, name: string) => {
      const snapshot = configSnapshots.find(
        (candidate) => candidate.id === snapshotId,
      );
      if (!snapshot) {
        return {
          ok: false as const,
          error: "Snapshot unavailable.",
        };
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
        return validation;
      }
      return renameSnapshotRecord({
        id: snapshotId,
        name: validation.name,
      });
    },
    [configSnapshots, renameSnapshotRecord],
  );
  const selectTargetPreset = useCallback(
    (preset: string) => {
      if (!presetNames.includes(preset)) {
        return false;
      }
      cancelTargetRestoration();
      setSelectedTargetBrowserMode("preset");
      sendTargetEvent({ type: "preset-selected", preset });
      clearHistoricalTarget();
      lastRequestedPreviewTargetKeyRef.current = "";
      return true;
    },
    [
      cancelTargetRestoration,
      clearHistoricalTarget,
      presetNames,
      sendTargetEvent,
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
      clearHistoricalTarget();
      const normalizedSnapshotOverrides = runtimeDefaultsEditor.replace(
        configFields,
        snapshot.overrides,
      );
      sendTargetEvent({
        type: "snapshot-selected",
        snapshotId: snapshot.id,
        preset: snapshot.preset,
        runtimeDefaults: normalizedSnapshotOverrides,
      });
      lastRequestedPreviewTargetKeyRef.current = "";
      return true;
    },
    [
      modelConfigSnapshots,
      cancelTargetRestoration,
      clearHistoricalTarget,
      configFields,
      presetNames,
      sendTargetEvent,
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
      sendTargetEvent({
        type: "experiment-task-selected",
        experimentTask: nextTask,
        datasets: nextDatasets,
      });
      return true;
    },
    [
      activeExperimentTask,
      cancelTargetRestoration,
      datasetGroups,
      defaultExperimentTask,
      sendTargetEvent,
      selectedDatasets,
    ],
  );

  const updatePreview = useCallback(() => {
    cancelTargetRestoration();
    if (!currentInspectionRequest) {
      return;
    }
    sendTargetEvent({ type: "inspection-refreshed" });
    issueInspectionPreview(currentInspectionRequest, "refresh");
  }, [
    cancelTargetRestoration,
    currentInspectionRequest,
    issueInspectionPreview,
    sendTargetEvent,
  ]);

  const resetOverrides = useCallback(() => {
    cancelTargetRestoration();
    setSelectedTargetBrowserMode("preset");
    clearHistoricalTarget();
    sendTargetEvent({ type: "runtime-defaults-reset" });
    lastRequestedPreviewTargetKeyRef.current = "";
  }, [
    cancelTargetRestoration,
    clearHistoricalTarget,
    sendTargetEvent,
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
        clearHistoricalTarget();
      }
      const nextOverrides = runtimeDefaultsEditor.edit(
        configFields,
        presetOverrides,
        key,
        value,
      );
      if (nextOverrides === presetOverrides && !exitsNonPresetTarget) {
        return;
      }
      sendTargetEvent({
        type: "runtime-defaults-edited",
        presetRuntimeDefaults: nextOverrides,
        activeRuntimeDefaults: effectivePresetOverrides(
          configFields,
          nextOverrides,
        ),
      });
    },
    [
      cancelTargetRestoration,
      clearHistoricalTarget,
      configFields,
      presetOverrides,
      sendTargetEvent,
      selectedTargetMode,
    ],
  );

  const clearTargetOverride = useCallback(
    (key: string) => {
      cancelTargetRestoration();
      const exitsNonPresetTarget =
        selectedTargetMode === "snapshot" || selectedTargetMode === "experiment";
      if (exitsNonPresetTarget) {
        setSelectedTargetBrowserMode("preset");
        clearHistoricalTarget();
      }
      const nextOverrides = runtimeDefaultsEditor.clear(
        configFields,
        presetOverrides,
        key,
      );
      if (nextOverrides === presetOverrides && !exitsNonPresetTarget) {
        return;
      }
      sendTargetEvent({
        type: "runtime-defaults-edited",
        presetRuntimeDefaults: nextOverrides,
        activeRuntimeDefaults: effectivePresetOverrides(
          configFields,
          nextOverrides,
        ),
      });
    },
    [
      cancelTargetRestoration,
      clearHistoricalTarget,
      configFields,
      presetOverrides,
      sendTargetEvent,
      selectedTargetMode,
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
    remove: deleteSnapshotRecord,
    rename: renameConfigSnapshot,
    retryMutation: retrySnapshotRecordMutation,
    dismissMutation: dismissSnapshotRecordMutation,
  });
  const snapshots = useStableStateSlice({
    records: snapshotRecords,
    mutation: configSnapshotsStatus.mutation,
    actions: snapshotActions,
  });
  const inspection = useStableStateSlice({
    graph: inspectionResponse,
    status: inspectionPreview.status,
    transition: inspectionTransition,
    clear: inspectionPreview.clear,
    clearForConnectionChange: clearInspectionForConnectionChange,
  });
  const historical = useStableStateSlice({
    browsing: historicalTarget.browsing,
    graphFacts: historicalTarget.graphFacts,
  });
  const contexts = useMemo(
    () => ({ catalog, model, snapshots }),
    [catalog, model, snapshots],
  );
  return useMemo(
    () => ({
      contexts,
      historical,
      inspection,
    }),
    [
      contexts,
      historical,
      inspection,
    ],
  );
}
