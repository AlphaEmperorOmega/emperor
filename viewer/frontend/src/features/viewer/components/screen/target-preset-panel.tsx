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
import {
  modelNameForId,
  modelsForType,
  modelTypeOptions as createModelTypeOptions,
} from "@/lib/selection";

function historicalFilterLabel(option: { label: string; count: number }) {
  return option.count > 1 ? `${option.label} (${option.count})` : option.label;
}

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
    historicalExperimentOptions,
    historicalDatasetOptions,
    historicalPresetOptions,
    selectedHistoricalExperimentFilter,
    setSelectedHistoricalExperimentFilter,
    selectedHistoricalDatasetFilter,
    setSelectedHistoricalDatasetFilter,
    selectedHistoricalPreset,
    setSelectedHistoricalPreset,
  } = useHistoricalRuns();
  const selectedPresetDescription = selectedPresetMeta?.description;
  const presetDescriptionId = useId();
  const presetSelectId = useId();
  const snapshotSelectId = useId();
  const experimentSelectId = useId();
  const experimentDatasetSelectId = useId();
  const experimentPresetSelectId = useId();
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
  const experimentOptions = historicalExperimentOptions.map((option) => ({
    value: option.value,
    label: historicalFilterLabel(option),
  }));
  const experimentDatasetOptions = historicalDatasetOptions.map((option) => ({
    value: option.value,
    label: historicalFilterLabel(option),
  }));
  const experimentPresetOptions = historicalPresetOptions.map((option) => ({
    value: option.value,
    label: historicalFilterLabel(option),
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
        selectedHistoricalExperimentFilter={selectedHistoricalExperimentFilter}
        selectedHistoricalDatasetFilter={selectedHistoricalDatasetFilter}
        selectedHistoricalPreset={selectedHistoricalPreset}
        configSnapshotsEnabled={configSnapshotsEnabled}
        isSchemaReady={isSchemaReady}
        modelTypeOptions={modelTypeOptions}
        modelOptions={modelOptions}
        presetOptions={presetOptions}
        snapshotOptions={snapshotOptions}
        experimentOptions={experimentOptions}
        experimentDatasetOptions={experimentDatasetOptions}
        experimentPresetOptions={experimentPresetOptions}
        presetSelectId={presetSelectId}
        snapshotSelectId={snapshotSelectId}
        experimentSelectId={experimentSelectId}
        experimentDatasetSelectId={experimentDatasetSelectId}
        experimentPresetSelectId={experimentPresetSelectId}
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
        onSelectHistoricalExperimentFilter={setSelectedHistoricalExperimentFilter}
        onSelectHistoricalDatasetFilter={setSelectedHistoricalDatasetFilter}
        onSelectHistoricalPreset={setSelectedHistoricalPreset}
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
