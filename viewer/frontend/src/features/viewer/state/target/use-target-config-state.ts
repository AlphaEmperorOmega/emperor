import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type ConfigSnapshotRecord, type LogRun } from "@/lib/api";
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
import {
  useConfigSnapshotLibrary,
  useConfigSnapshots,
} from "@/features/viewer/state/target/use-config-snapshots";
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

type TargetMode = "preset" | "snapshot" | "experiment";

type TargetConfigStateOptions = {
  requestPreview: (request: PreviewInspectionRequest) => void;
  resetGraphSelectionAndExpansion: () => void;
  resetGraphExpansion: () => void;
  onModelSelected?: () => void;
  onTargetPresetSelected?: () => void;
  onTargetSnapshotSelected?: () => void;
};

function overridesAreEmpty(overrides: OverrideValues) {
  return Object.keys(overrides).length === 0;
}

function createSnapshotId() {
  return globalThis.crypto?.randomUUID?.() ?? `snapshot-${Date.now()}`;
}

function previewTargetKey({
  model,
  dataset,
  mode,
  target,
}: {
  model: string;
  dataset: string;
  mode: TargetMode;
  target: string;
}) {
  return `${model}\u0000${dataset}\u0000${mode}\u0000${target}`;
}

export function useTargetConfigState({
  requestPreview,
  resetGraphSelectionAndExpansion,
  resetGraphExpansion,
  onModelSelected,
  onTargetPresetSelected,
  onTargetSnapshotSelected,
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
  const [selectedTargetMode, setSelectedTargetMode] =
    useState<TargetMode>("preset");
  const [selectedSnapshotId, setSelectedSnapshotId] = useState("");
  const [selectedExperimentRunId, setSelectedExperimentRunId] = useState("");
  const [selectedDatasets, setSelectedDatasets] = useState<string[]>([]);
  const [selectedTrainingPresets, setSelectedTrainingPresets] = useState<string[]>([]);
  const [selectedMonitors, setSelectedMonitors] = useState<string[]>([]);
  const lastRequestedPreviewTargetKeyRef = useRef("");
  const allowEmptyTrainingPresetDraftRef = useRef(false);
  const {
    query: configSnapshotsQuery,
    snapshots: configSnapshots,
    createMutation: createSnapshotMutation,
    renameMutation: renameSnapshotMutation,
    deleteMutation: deleteSnapshotMutation,
  } = useConfigSnapshots(selectedModel);
  const {
    query: configSnapshotLibraryQuery,
    snapshots: configSnapshotLibrary,
  } = useConfigSnapshotLibrary();
  const [pendingConfigSnapshot, setPendingConfigSnapshot] =
    useState<ConfigSnapshotRecord | null>(null);
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
    modelConfigSnapshots,
    modelConfigSnapshotGroups,
    visibleConfigSnapshots,
    configSnapshotGroups,
    overrideCount,
    presetOwnedFieldCount,
    fieldCount,
  } = targetSelectionState;

  const selectModel = useCallback(
    (model: string) => {
      lastRequestedPreviewTargetKeyRef.current = "";
      setSelectedModelType(model ? modelTypeForId(model) : "");
      setSelectedTargetMode("preset");
      setSelectedSnapshotId("");
      setSelectedExperimentRunId("");
      selectTargetModel(model);
      setSelectedDatasets([]);
      allowEmptyTrainingPresetDraftRef.current = false;
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
      setSelectedTargetMode("preset");
      setSelectedSnapshotId("");
      setSelectedExperimentRunId("");
      setSelectedPreset(firstPreset);
      allowEmptyTrainingPresetDraftRef.current = false;
      setSelectedTrainingPresets([firstPreset]);
      setOverrides({});
    }
  }, [
    datasetNames,
    presetsQuery.data,
    selectedPreset,
    setOverrides,
    setSelectedPreset,
  ]);

  useEffect(() => {
    if (!pendingConfigSnapshot) {
      return;
    }
    if (selectedModel !== pendingConfigSnapshot.model) {
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
      allowEmptyTrainingPresetDraftRef.current = false;
      setSelectedTrainingPresets((current) =>
        normalizePrimarySelection(
          [...current, pendingConfigSnapshot.preset],
          presetNames,
          pendingConfigSnapshot.preset || undefined,
        ),
      );
      setOverrides({});
      return;
    }
    if (!schemaQuery.isSuccess) {
      return;
    }
    allowEmptyTrainingPresetDraftRef.current = false;
    setSelectedTargetMode("snapshot");
    setSelectedSnapshotId(pendingConfigSnapshot.id);
    setSelectedExperimentRunId("");
    setSelectedTrainingPresets((current) =>
      normalizePrimarySelection(
        [...current, pendingConfigSnapshot.preset],
        presetNames,
        pendingConfigSnapshot.preset || undefined,
      ),
    );
    setOverrides({ ...pendingConfigSnapshot.overrides });
    lastRequestedPreviewTargetKeyRef.current = "";
    setPendingConfigSnapshot(null);
  }, [
    configSnapshotsQuery.isSuccess,
    pendingConfigSnapshot,
    presetNames,
    presetsQuery.isSuccess,
    schemaQuery.isSuccess,
    selectedModel,
    selectedPreset,
    setOverrides,
    setSelectedPreset,
  ]);

  useEffect(() => {
    const previewDataset = selectedDatasets[0];
    if (!selectedModel || !selectedPreset || !previewDataset) {
      return;
    }
    if (
      pendingConfigSnapshot &&
      selectedModel === pendingConfigSnapshot.model
    ) {
      return;
    }

    if (lastRequestedPreviewTargetKeyRef.current !== "") {
      return;
    }

    const targetMode =
      selectedTargetMode === "snapshot" && selectedSnapshotId
        ? "snapshot"
        : selectedTargetMode === "experiment" && selectedExperimentRunId
          ? "experiment"
        : "preset";
    const targetKey = previewTargetKey({
      model: selectedModel,
      dataset: previewDataset,
      mode: targetMode,
      target:
        targetMode === "snapshot"
          ? selectedSnapshotId
          : targetMode === "experiment"
            ? selectedExperimentRunId
            : selectedPreset,
    });
    lastRequestedPreviewTargetKeyRef.current = targetKey;
    resetGraphSelectionAndExpansion();
    requestPreview({
      model: selectedModel,
      preset: selectedPreset,
      dataset: previewDataset,
      overrides: { ...overrides },
    });
  }, [
    overrides,
    pendingConfigSnapshot,
    requestPreview,
    resetGraphSelectionAndExpansion,
    selectedDatasets,
    selectedModel,
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
        setOverrides({});
      }
      lastRequestedPreviewTargetKeyRef.current = previewTargetKey({
        model: selectedModel,
        dataset,
        mode: "experiment",
        target: selectedLogRun.id,
      });
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
      selectedExperimentRunId,
      selectedModel,
      selectedPreset,
      selectedSnapshotId,
      selectedTargetMode,
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

  const includeConfigSnapshot = useCallback(
    (snapshotId: string) => {
      const snapshot = modelConfigSnapshots.find(
        (candidate) => candidate.id === snapshotId,
      );
      if (!snapshot || !presetNames.includes(snapshot.preset)) {
        return;
      }
      allowEmptyTrainingPresetDraftRef.current = false;
      setSelectedTrainingPresets((current) =>
        normalizePrimarySelection(
          [...current, snapshot.preset],
          presetNames,
          selectedPreset || undefined,
        ),
      );
      setDeselectedSnapshotIds((current) =>
        current.includes(snapshotId)
          ? current.filter((id) => id !== snapshotId)
          : current,
      );
    },
    [modelConfigSnapshots, presetNames, selectedPreset],
  );

  const excludeConfigSnapshot = useCallback(
    (snapshotId: string) => {
      if (!modelConfigSnapshots.some((snapshot) => snapshot.id === snapshotId)) {
        return;
      }
      setDeselectedSnapshotIds((current) =>
        current.includes(snapshotId) ? current : [...current, snapshotId],
      );
    },
    [modelConfigSnapshots],
  );

  const toggleConfigSnapshotRunSelection = useCallback(
    (snapshotId: string) => {
      const snapshot = modelConfigSnapshots.find(
        (candidate) => candidate.id === snapshotId,
      );
      if (!snapshot) {
        return;
      }
      const isIncluded =
        selectedTrainingPresets.includes(snapshot.preset) &&
        !deselectedSnapshotIds.includes(snapshotId);
      if (isIncluded) {
        excludeConfigSnapshot(snapshotId);
        return;
      }
      includeConfigSnapshot(snapshotId);
    },
    [
      deselectedSnapshotIds,
      excludeConfigSnapshot,
      includeConfigSnapshot,
      modelConfigSnapshots,
      selectedTrainingPresets,
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
      if (snapshot.model !== selectedModel) {
        selectModel(snapshot.model);
      }
      return true;
    },
    [configSnapshotLibrary, configSnapshots, selectModel, selectedModel],
  );

  const selectTrainingPrimaryPreset = useCallback(
    (preset: string) => {
      setSelectedTargetMode("preset");
      setSelectedSnapshotId("");
      setSelectedExperimentRunId("");
      onTargetPresetSelected?.();
      selectPreset(preset);
      allowEmptyTrainingPresetDraftRef.current = false;
      setSelectedTrainingPresets([preset]);
    },
    [onTargetPresetSelected, selectPreset],
  );

  const selectTargetPreset = useCallback(
    (preset: string) => {
      const shouldRefreshPreview =
        selectedTargetMode !== "preset" ||
        selectedSnapshotId !== "" ||
        selectedExperimentRunId !== "" ||
        selectedPreset !== preset ||
        !overridesAreEmpty(overrides);
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
        model: selectedModel,
        dataset: previewDataset,
        mode: "preset",
        target: preset,
      });
      resetGraphSelectionAndExpansion();
      requestPreview({
        model: selectedModel,
        preset,
        dataset: previewDataset,
        overrides: {},
      });
    },
    [
      overrides,
      requestPreview,
      resetGraphSelectionAndExpansion,
      selectTrainingPrimaryPreset,
      selectedDatasets,
      selectedExperimentRunId,
      selectedModel,
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

      setSelectedTargetMode("snapshot");
      setSelectedSnapshotId(snapshot.id);
      setSelectedExperimentRunId("");
      onTargetSnapshotSelected?.();
      setSelectedPreset(snapshot.preset);
      allowEmptyTrainingPresetDraftRef.current = false;
      setSelectedTrainingPresets((current) =>
        normalizePrimarySelection(
          [...current, snapshot.preset],
          presetNames,
          snapshot.preset || undefined,
        ),
      );
      setOverrides({ ...snapshot.overrides });

      const previewDataset = selectedDatasets[0];
      if (!selectedModel || !snapshot.preset || !previewDataset) {
        lastRequestedPreviewTargetKeyRef.current = "";
        return true;
      }

      lastRequestedPreviewTargetKeyRef.current = previewTargetKey({
        model: selectedModel,
        dataset: previewDataset,
        mode: "snapshot",
        target: snapshot.id,
      });
      resetGraphSelectionAndExpansion();
      requestPreview({
        model: selectedModel,
        preset: snapshot.preset,
        dataset: previewDataset,
        overrides: { ...snapshot.overrides },
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
      onTargetSnapshotSelected,
      setOverrides,
      setSelectedPreset,
    ],
  );

  const activateTargetPresetMode = useCallback(() => {
    if (!selectedPreset) {
      setSelectedTargetMode("preset");
      setSelectedSnapshotId("");
      setSelectedExperimentRunId("");
      onTargetPresetSelected?.();
      setOverrides({});
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
      model: selectedModel,
      dataset: previewDataset,
      mode: "preset",
      target: selectedPreset,
    });
    resetGraphSelectionAndExpansion();
    requestPreview({
      model: selectedModel,
      preset: selectedPreset,
      dataset: previewDataset,
      overrides: {},
    });
  }, [
    onTargetPresetSelected,
    requestPreview,
    resetGraphSelectionAndExpansion,
    selectTrainingPrimaryPreset,
    selectedDatasets,
    selectedModel,
    selectedPreset,
    setOverrides,
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
      const fallbackPreset =
        selectedPreset && presetNames.includes(selectedPreset)
          ? selectedPreset
          : presetNames[0] ?? "";
      const nextPrimary = validPresets.includes(selectedPreset)
        ? selectedPreset
        : validPresets[0] ?? fallbackPreset;

      if (nextPrimary && nextPrimary !== selectedPreset) {
        setSelectedTargetMode("preset");
        setSelectedSnapshotId("");
        setSelectedExperimentRunId("");
        selectPreset(nextPrimary);
      }

      allowEmptyTrainingPresetDraftRef.current = false;
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
    [presetNames, selectPreset],
  );

  const preparePresetSnapshotDraft = useCallback(
    (preset: string) => {
      if (!presetNames.includes(preset)) {
        return false;
      }
      const shouldClearOverrides = selectedPreset !== preset;
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
      if (shouldClearOverrides) {
        setOverrides({});
      }
      return true;
    },
    [
      presetNames,
      selectPreset,
      selectedPreset,
      setOverrides,
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
      model: selectedModel,
      dataset: previewDataset,
      mode: targetMode,
      target:
        targetMode === "snapshot"
          ? selectedSnapshotId
          : targetMode === "experiment"
            ? selectedExperimentRunId
            : selectedPreset,
    });
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
    selectedExperimentRunId,
    selectedSnapshotId,
    selectedTargetMode,
  ]);

  const resetOverrides = useCallback(() => {
    setSelectedTargetMode("preset");
    setSelectedSnapshotId("");
    setSelectedExperimentRunId("");
    onTargetPresetSelected?.();
    clearOverrides();
    resetGraphExpansion();
    const previewDataset = selectedDatasets[0];
    if (selectedModel && selectedPreset && previewDataset) {
      lastRequestedPreviewTargetKeyRef.current = previewTargetKey({
        model: selectedModel,
        dataset: previewDataset,
        mode: "preset",
        target: selectedPreset,
      });
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
    onTargetPresetSelected,
  ]);

  const updateTargetOverride = useCallback(
    (key: string, value: string) => {
      if (selectedTargetMode === "snapshot" || selectedTargetMode === "experiment") {
        setSelectedTargetMode("preset");
        setSelectedSnapshotId("");
        setSelectedExperimentRunId("");
        onTargetPresetSelected?.();
      }
      updateOverride(key, value);
    },
    [onTargetPresetSelected, selectedTargetMode, updateOverride],
  );

  const clearTargetOverride = useCallback(
    (key: string) => {
      if (selectedTargetMode === "snapshot" || selectedTargetMode === "experiment") {
        setSelectedTargetMode("preset");
        setSelectedSnapshotId("");
        setSelectedExperimentRunId("");
        onTargetPresetSelected?.();
      }
      clearOverride(key);
    },
    [clearOverride, onTargetPresetSelected, selectedTargetMode],
  );

  const apiOnline = healthQuery.data?.status === "ok";

  return {
    target: {
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
      selectedExperimentRunId,
      selectTargetSnapshot,
      selectedPresetMeta,
      selectedTrainingPresets,
      setTrainingPresetSelection,
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
      allConfigSnapshots: modelConfigSnapshots,
      configSnapshotLibrary,
      configSnapshotGroups,
      allConfigSnapshotGroups: modelConfigSnapshotGroups,
      configSnapshotCount: visibleConfigSnapshots.length,
      allConfigSnapshotCount: modelConfigSnapshots.length,
      configSnapshotLibraryCount: configSnapshotLibrary.length,
      configSnapshotLibraryQuery,
      deselectedSnapshotIds,
      addConfigSnapshot,
      removeConfigSnapshot,
      renameConfigSnapshot,
      loadConfigSnapshot,
      includeConfigSnapshot,
      excludeConfigSnapshot,
      toggleConfigSnapshotRunSelection,
      preparePresetSnapshotDraft,
      updateOverride: updateTargetOverride,
      clearOverride: clearTargetOverride,
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
    syncSelectedLogRun,
  };
}
