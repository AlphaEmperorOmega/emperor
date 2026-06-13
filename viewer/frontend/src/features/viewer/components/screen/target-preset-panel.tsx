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
import { formatRunTimestamp } from "@/lib/format";
import {
  modelNameForId,
  modelsForType,
  modelTypeOptions as createModelTypeOptions,
} from "@/lib/selection";

export function TargetPresetPanel() {
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
    selectedExperimentRunId,
    selectModelType: onSelectModelType,
    selectModel: onSelectModel,
    selectPreset: onSelectPreset,
    selectSnapshot: onSelectSnapshot,
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
    value: model,
    label: modelNameForId(model),
  }));
  const presetOptions = presets.map((preset) => ({
    value: preset.name,
    label: preset.name,
  }));
  const snapshotOptions = snapshots.map((snapshot) => {
    const overrideCount = Object.keys(snapshot.overrides).length;
    return {
      value: snapshot.id,
      label: `${snapshot.name} · ${snapshot.preset} · ${overrideCount} ${
        overrideCount === 1 ? "override" : "overrides"
      }`,
    };
  });
  const experimentOptions = visibleHistoricalRuns.map((run) => ({
    value: run.id,
    label: `${run.experiment} · ${run.preset} · ${run.dataset} · ${formatRunTimestamp(
      run.timestamp ?? run.version,
    )}`,
  }));
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
        selectedExperimentRunId={selectedLogRunId ?? selectedExperimentRunId}
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
