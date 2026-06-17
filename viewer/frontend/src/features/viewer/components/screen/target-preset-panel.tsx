import { useCallback, useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  useFixedPopupPosition,
  usePopupDismissal,
} from "@/features/viewer/components/screen/fixed-popup";
import { PresetDescriptionSubmenu } from "@/features/viewer/components/screen/preset-description-submenu";
import { TargetSelectorSection } from "@/features/viewer/components/screen/target-selector-section";
import {
  useHistoricalRuns,
  useTargetSelectorState,
} from "@/features/viewer/providers/viewer-providers";
import { type FullConfigDialogControls } from "@/features/viewer/state/use-viewer-workspace-shell";
import { formatRunTimestamp } from "@/lib/format";
import {
  modelNameForId,
  modelsForType,
  modelTypeOptions as createModelTypeOptions,
} from "@/lib/selection";

export function TargetPresetPanel({
  onOpenFullConfig,
}: {
  onOpenFullConfig: FullConfigDialogControls["open"];
}) {
  const {
    selectedModelType,
    selectedModel,
    selectedTargetMode,
    activateTargetPresetMode,
    activateTargetSnapshotMode,
    activateTargetExperimentMode,
    selectedPreset,
    selectedPresetMeta,
    selectedSnapshotId,
    selectedConfigSnapshot,
    selectedExperimentRunId,
    configSnapshotsEnabled,
    isSchemaReady,
    selectModelType: onSelectModelType,
    selectModel: onSelectModel,
    selectPreset: onSelectPreset,
    selectSnapshot: onSelectSnapshot,
    preparePresetSnapshotDraft,
    prepareSelectedSnapshotEdit,
    models,
    presets,
    snapshots,
  } = useTargetSelectorState();
  const {
    visibleHistoricalRuns,
    selectedLogRunId,
    selectLogRun,
  } = useHistoricalRuns();
  const selectedPresetDescription = selectedPresetMeta?.description;
  const presetDescriptionId = useId();
  const presetSelectId = useId();
  const snapshotSelectId = useId();
  const experimentSelectId = useId();
  const presetDescriptionTriggerRef = useRef<HTMLButtonElement>(null);
  const presetDescriptionSubmenuRef = useRef<HTMLDivElement>(null);
  const [isPresetDescriptionOpen, setIsPresetDescriptionOpen] = useState(false);
  const hasPresetDescription = Boolean(selectedPresetDescription?.trim());
  const modelTypeOptions = createModelTypeOptions(models);
  const modelOptions = modelsForType(models, selectedModelType).map((model) => ({
    value: model.model,
    label: modelNameForId(model),
  }));
  const presetOptions = presets.map((preset) => ({
    value: preset.name,
    label: preset.name,
  }));
  const snapshotOptions = snapshots.map((snapshot) => ({
    value: snapshot.id,
    label: snapshot.name,
  }));
  const experimentOptions = visibleHistoricalRuns.map((run) => ({
    value: run.id,
    label: `${run.experiment} · ${run.preset} · ${run.dataset} · ${formatRunTimestamp(
      run.timestamp ?? run.version,
    )}`,
  }));
  const selectedSnapshotName = selectedConfigSnapshot?.name ?? "";
  const {
    position: presetDescriptionPosition,
    updatePosition: updatePresetDescriptionPosition,
  } = useFixedPopupPosition(presetDescriptionTriggerRef, isPresetDescriptionOpen, {
    minimumHeight: 180,
  });

  const closePresetDescription = useCallback((restoreFocus = false) => {
    setIsPresetDescriptionOpen(false);
    if (restoreFocus) {
      presetDescriptionTriggerRef.current?.focus();
    }
  }, []);

  const closePresetDescriptionWithFocus = useCallback(() => {
    closePresetDescription(true);
  }, [closePresetDescription]);

  const togglePresetDescription = () => {
    if (!hasPresetDescription) {
      return;
    }
    setIsPresetDescriptionOpen((isOpen) => {
      const nextOpen = !isOpen;
      if (nextOpen) {
        updatePresetDescriptionPosition();
      }
      return nextOpen;
    });
  };

  const createPresetSnapshot = () => {
    if (preparePresetSnapshotDraft(selectedPreset)) {
      onOpenFullConfig("snapshotDraft");
    }
  };

  const editSelectedSnapshot = () => {
    if (selectedSnapshotId && prepareSelectedSnapshotEdit(selectedSnapshotId)) {
      onOpenFullConfig("snapshotEdit");
    }
  };

  const duplicateSelectedSnapshot = () => {
    if (selectedSnapshotId && onSelectSnapshot(selectedSnapshotId)) {
      onOpenFullConfig("snapshotDraft");
    }
  };

  useEffect(() => {
    closePresetDescription();
  }, [closePresetDescription, selectedModel, selectedPreset, selectedTargetMode]);

  useEffect(() => {
    if (!hasPresetDescription) {
      closePresetDescription();
    }
  }, [closePresetDescription, hasPresetDescription]);

  usePopupDismissal({
    isOpen: isPresetDescriptionOpen,
    triggerRef: presetDescriptionTriggerRef,
    popupRef: presetDescriptionSubmenuRef,
    onDismiss: closePresetDescription,
    onDismissWithFocus: closePresetDescriptionWithFocus,
  });

  return (
    <>
      <TargetSelectorSection
        presetCount={presets.length}
        selectedModelType={selectedModelType}
        selectedModel={selectedModel}
        selectedTargetMode={selectedTargetMode}
        selectedPreset={selectedPreset}
        selectedSnapshotId={selectedSnapshotId}
        selectedSnapshotName={selectedSnapshotName}
        selectedExperimentRunId={selectedLogRunId ?? selectedExperimentRunId}
        configSnapshotsEnabled={configSnapshotsEnabled}
        isSchemaReady={isSchemaReady}
        modelTypeOptions={modelTypeOptions}
        modelOptions={modelOptions}
        presetOptions={presetOptions}
        snapshotOptions={snapshotOptions}
        experimentOptions={experimentOptions}
        presetSelectId={presetSelectId}
        snapshotSelectId={snapshotSelectId}
        experimentSelectId={experimentSelectId}
        presetDescriptionId={presetDescriptionId}
        presetDescriptionTriggerRef={presetDescriptionTriggerRef}
        isPresetDescriptionOpen={isPresetDescriptionOpen}
        hasPresetDescription={hasPresetDescription}
        onSelectModelType={onSelectModelType}
        onSelectModel={onSelectModel}
        onActivatePresetMode={activateTargetPresetMode}
        onActivateSnapshotMode={activateTargetSnapshotMode}
        onActivateExperimentMode={activateTargetExperimentMode}
        onSelectPreset={onSelectPreset}
        onSelectSnapshot={onSelectSnapshot}
        onSelectExperimentRun={selectLogRun}
        onCreateSnapshot={createPresetSnapshot}
        onEditSnapshot={editSelectedSnapshot}
        onDuplicateSnapshot={duplicateSelectedSnapshot}
        onTogglePresetDescription={togglePresetDescription}
      />

      {isPresetDescriptionOpen &&
        presetDescriptionPosition &&
        hasPresetDescription &&
        selectedPresetDescription &&
        createPortal(
          <PresetDescriptionSubmenu
            id={presetDescriptionId}
            submenuRef={presetDescriptionSubmenuRef}
            presetName={selectedPreset}
            description={selectedPresetDescription}
            position={presetDescriptionPosition}
            onClose={closePresetDescriptionWithFocus}
          />,
          document.body,
        )}
    </>
  );
}
