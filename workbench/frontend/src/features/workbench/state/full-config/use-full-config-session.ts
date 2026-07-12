import { useCallback, useMemo } from "react";
import { buildTrainingCommand } from "@/lib/training-command";
import {
  useConfigSnapshotEditor,
  useConfigSnapshotRecords,
  useModelPackageInspection,
} from "@/features/workbench/providers/workbench-providers";
import {
  isWorkbenchProtectedAccessReady,
  useWorkbenchCapabilities,
  useWorkbenchConnection,
} from "@/features/workbench/providers/workbench-connection-provider";
import { useTrainingConfiguration } from "@/features/workbench/providers/training-provider";
import { useConfigSnapshotEditorState } from "@/features/workbench/state/config-snapshots/use-config-snapshot-editor";
import {
  type FullConfigDialogMode,
  type FullConfigDialogScope,
} from "@/features/workbench/state/use-workbench-workspace-shell";

export type FullConfigSessionKind =
  | "model"
  | "training"
  | "snapshot-draft"
  | "snapshot-edit";

export function useFullConfigSession({
  mode,
  scope,
  onClose,
}: {
  mode: FullConfigDialogMode;
  scope: FullConfigDialogScope;
  onClose: () => void;
}) {
  const { capabilities } = useWorkbenchCapabilities();
  const modelPackage = useModelPackageInspection();
  const snapshotLibrary = useConfigSnapshotRecords();
  const snapshotEditorSession = useConfigSnapshotEditor();
  const workbenchConnection = useWorkbenchConnection();
  const snapshotEditor = useConfigSnapshotEditorState({
    sessionState: snapshotEditorSession,
    protectedReadsEnabled: isWorkbenchProtectedAccessReady(workbenchConnection),
  });
  const training = useTrainingConfiguration();
  const editorSession = snapshotEditor.session;

  const kind: FullConfigSessionKind =
    mode === "snapshotDraft"
      ? "snapshot-draft"
      : mode === "snapshotEdit"
        ? "snapshot-edit"
        : scope === "training"
          ? "training"
          : "model";
  const isSnapshot = kind === "snapshot-draft" || kind === "snapshot-edit";
  const isTraining = kind === "training";

  const modelType = isSnapshot
    ? editorSession.modelType
    : isTraining
      ? training.selectedModelType
      : modelPackage.browser.selectedModelType;
  const model = isSnapshot
    ? editorSession.model
    : isTraining
      ? training.selectedModel
      : modelPackage.browser.selectedModel;
  const preset = isSnapshot
    ? editorSession.preset
    : isTraining
      ? training.selectedPrimaryPreset
      : modelPackage.browser.selectedPreset;
  const sections = isSnapshot
    ? editorSession.configSections
    : isTraining
      ? training.configSections
      : modelPackage.options.configSections;
  const fieldCount = isSnapshot
    ? editorSession.fieldCount
    : isTraining
      ? training.fieldCount
      : modelPackage.runtimeDefaults.fieldCount;
  const overrides = isSnapshot
    ? editorSession.draft
    : isTraining
      ? training.bulkOverrides
      : modelPackage.runtimeDefaults.active;
  const lockedOverrideCount = isSnapshot
    ? 0
    : isTraining
      ? training.inactiveLockedOverrideCount
      : modelPackage.runtimeDefaults.inactiveLockedCount;
  const isLoading = isSnapshot
    ? editorSession.status.isLoading
    : isTraining
      ? training.schemaLoading
      : modelPackage.status.schema.isLoading;
  const editOverride = isSnapshot
    ? snapshotEditor.actions.updateOverride
    : isTraining
      ? training.updateOverride
      : modelPackage.actions.editRuntimeDefault;
  const clearOverride = isSnapshot
    ? snapshotEditor.actions.clearOverride
    : isTraining
      ? training.clearOverride
      : modelPackage.actions.clearRuntimeDefault;
  const resetOverrides = isSnapshot
    ? snapshotEditor.actions.reset
    : isTraining
      ? training.resetOverrides
      : modelPackage.actions.resetRuntimeDefaults;

  const records = isSnapshot
    ? editorSession.records
    : snapshotLibrary.records.all;
  const recordGroups = isSnapshot
    ? editorSession.recordGroups
    : snapshotLibrary.records.allGroups;
  const mutation = isSnapshot
    ? editorSession.status.mutation
    : snapshotLibrary.mutation;
  const loadSnapshot = isSnapshot
    ? snapshotEditor.actions.load
    : snapshotLibrary.actions.selectTarget;
  const renameSnapshot = isSnapshot
    ? snapshotEditor.actions.rename
    : snapshotLibrary.actions.rename;
  const removeSnapshot = isSnapshot
    ? snapshotEditor.actions.remove
    : snapshotLibrary.actions.remove;
  const retryMutation = isSnapshot
    ? snapshotEditor.actions.retryMutation
    : snapshotLibrary.actions.retryMutation;
  const dismissMutation = isSnapshot
    ? snapshotEditor.actions.dismissMutation
    : snapshotLibrary.actions.dismissMutation;

  const toggleSnapshotRunSelection = useCallback(
    (snapshotId: string) => {
      if (training.selectedSnapshotIds.includes(snapshotId)) {
        training.excludeSnapshot(snapshotId);
        return;
      }
      training.includeSnapshot(snapshotId);
    },
    [training],
  );
  const openSnapshotSave = useCallback(() => {
    if (kind !== "model") {
      return isSnapshot;
    }
    return snapshotEditorSession.actions.beginDraft({
      modelType,
      model,
      preset,
      overrides,
    });
  }, [isSnapshot, kind, model, modelType, overrides, preset, snapshotEditorSession.actions]);
  const finishTransientSnapshotSave = useCallback(() => {
    if (kind === "model") {
      snapshotEditor.actions.close();
    }
  }, [kind, snapshotEditor.actions]);
  const saveSnapshot = useCallback(
    async (name: string) => {
      const result = await snapshotEditor.actions.save(name);
      if (result.ok) {
        finishTransientSnapshotSave();
      }
      return result;
    },
    [finishTransientSnapshotSave, snapshotEditor.actions],
  );
  const retrySnapshotSave = useCallback(async () => {
    const result = await snapshotEditor.actions.retrySave();
    if (result.ok) {
      finishTransientSnapshotSave();
    }
    return result;
  }, [finishTransientSnapshotSave, snapshotEditor.actions]);
  const closeSnapshotSave = useCallback(() => {
    if (kind === "model") {
      snapshotEditor.actions.close();
    }
  }, [kind, snapshotEditor.actions]);
  const close = useCallback(() => {
    if (
      (isSnapshot ||
        (kind === "model" && snapshotEditorSession.session !== null)) &&
      !snapshotEditor.actions.close()
    ) {
      return;
    }
    onClose();
  }, [
    isSnapshot,
    kind,
    onClose,
    snapshotEditor.actions,
    snapshotEditorSession.session,
  ]);

  const trainingCommand = useMemo(
    () =>
      buildTrainingCommand({
        modelType,
        model,
        preset,
        monitors: training.selectedMonitors,
        sections,
        overrides,
      }),
    [
      model,
      modelType,
      overrides,
      preset,
      sections,
      training.selectedMonitors,
    ],
  );
  const overrideCount = Object.keys(overrides).length;
  const canReset = Boolean(
    model && (overrideCount > 0 || (!isSnapshot && lockedOverrideCount > 0)),
  );
  const canAddSnapshot = Boolean(
    kind === "model" &&
      model &&
      preset &&
      fieldCount > 0 &&
      capabilities.configSnapshotsEnabled,
  );
  const canSaveSnapshot = Boolean(
    isSnapshot &&
      model &&
      preset &&
      fieldCount > 0 &&
      capabilities.configSnapshotsEnabled &&
      (kind !== "snapshot-edit" || editorSession.selectedSnapshot),
  );
  const canUpdatePreview = Boolean(
    kind === "model" &&
      model &&
      preset &&
      modelPackage.browser.selectedDatasets.length > 0,
  );

  return {
    kind,
    identity: { modelType, model, preset },
    runtimeDefaults: {
      sections,
      fieldCount,
      overrides,
      overrideCount,
      lockedOverrideCount,
      isLoading,
    },
    snapshots: {
      records,
      recordGroups,
      selected: editorSession.selectedSnapshot,
      selectedTrainingIds: training.selectedSnapshotIds,
      mutation,
      saveMutation: editorSession.status.mutation,
      canManage: capabilities.configSnapshotsEnabled,
    },
    controls: {
      canReset,
      canAddSnapshot,
      canSaveSnapshot,
      canUpdatePreview,
      showTrainingCommand: !isTraining,
      showSnapshotLibrary: !isTraining,
    },
    trainingCommand,
    actions: {
      editOverride,
      clearOverride,
      resetOverrides,
      updatePreview: modelPackage.actions.refreshInspection,
      close,
      openSnapshotSave,
      closeSnapshotSave,
      saveSnapshot,
      retrySnapshotSave,
      loadSnapshot,
      renameSnapshot,
      removeSnapshot,
      retryMutation,
      dismissMutation,
      dismissSaveMutation: snapshotEditor.actions.dismissMutation,
      toggleSnapshotRunSelection,
    },
  };
}

export type FullConfigSession = ReturnType<typeof useFullConfigSession>;
