import { useCallback, useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { DatasetSelectorTrigger } from "@/components/features/viewer/screen/dataset-selector-trigger";
import { DatasetSubmenu } from "@/components/features/viewer/screen/dataset-submenu";
import {
  useFixedPopupPosition,
  usePopupDismissal,
} from "@/components/features/viewer/screen/fixed-popup";
import { PresetDescriptionSubmenu } from "@/components/features/viewer/screen/preset-description-submenu";
import { TargetSelectorSection } from "@/components/features/viewer/screen/target-selector-section";
import { useTargetConfig } from "@/components/features/viewer/providers/viewer-providers";

export function TargetPresetPanel() {
  const {
    selectedModel,
    selectedPreset,
    selectedPresetMeta,
    selectedDatasets,
    selectModel: onSelectModel,
    selectPreset: onSelectPreset,
    toggleDataset: onToggleDataset,
    selectAllDatasets: onSelectAllDatasets,
    selectFirstDataset: onSelectFirstDataset,
    modelsQuery,
    presetsQuery,
    datasetsQuery,
  } = useTargetConfig();
  const models = modelsQuery.data?.models ?? [];
  const presets = presetsQuery.data?.presets ?? [];
  const datasets = datasetsQuery.data?.datasets ?? [];
  const selectedPresetDescription = selectedPresetMeta?.description;
  const presetDescriptionId = useId();
  const presetSelectId = useId();
  const datasetSelectorId = useId();
  const presetDescriptionTriggerRef = useRef<HTMLButtonElement>(null);
  const presetDescriptionSubmenuRef = useRef<HTMLDivElement>(null);
  const datasetTriggerRef = useRef<HTMLButtonElement>(null);
  const datasetSubmenuRef = useRef<HTMLDivElement>(null);
  const [isPresetDescriptionOpen, setIsPresetDescriptionOpen] = useState(false);
  const [isDatasetSelectorOpen, setIsDatasetSelectorOpen] = useState(false);
  const hasPresetDescription = Boolean(selectedPresetDescription?.trim());
  const datasetCount = `${selectedDatasets.length} / ${datasets.length}`;
  const modelOptions = models.map((model) => ({ value: model, label: model }));
  const presetOptions = presets.map((preset) => ({
    value: preset.name,
    label: preset.name,
  }));
  const {
    position: presetDescriptionPosition,
    updatePosition: updatePresetDescriptionPosition,
  } = useFixedPopupPosition(presetDescriptionTriggerRef, isPresetDescriptionOpen, {
    minimumHeight: 180,
  });
  const { position: datasetSubmenuPosition, updatePosition: updateDatasetSubmenuPosition } =
    useFixedPopupPosition(datasetTriggerRef, isDatasetSelectorOpen);

  const closePresetDescription = useCallback((restoreFocus = false) => {
    setIsPresetDescriptionOpen(false);
    if (restoreFocus) {
      presetDescriptionTriggerRef.current?.focus();
    }
  }, []);

  const closePresetDescriptionWithFocus = useCallback(() => {
    closePresetDescription(true);
  }, [closePresetDescription]);

  const closeDatasetSelector = useCallback((restoreFocus = false) => {
    setIsDatasetSelectorOpen(false);
    if (restoreFocus) {
      datasetTriggerRef.current?.focus();
    }
  }, []);

  const closeDatasetSelectorWithFocus = useCallback(() => {
    closeDatasetSelector(true);
  }, [closeDatasetSelector]);

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

  const toggleDatasetSelector = () => {
    if (datasets.length === 0) {
      return;
    }
    setIsDatasetSelectorOpen((isOpen) => {
      const nextOpen = !isOpen;
      if (nextOpen) {
        updateDatasetSubmenuPosition();
      }
      return nextOpen;
    });
  };

  useEffect(() => {
    closePresetDescription();
  }, [closePresetDescription, selectedModel, selectedPreset]);

  useEffect(() => {
    if (!hasPresetDescription) {
      closePresetDescription();
    }
  }, [closePresetDescription, hasPresetDescription]);

  useEffect(() => {
    closeDatasetSelector();
  }, [closeDatasetSelector, selectedModel, selectedPreset]);

  useEffect(() => {
    if (datasets.length === 0) {
      closeDatasetSelector();
    }
  }, [closeDatasetSelector, datasets.length]);

  usePopupDismissal({
    isOpen: isPresetDescriptionOpen,
    triggerRef: presetDescriptionTriggerRef,
    popupRef: presetDescriptionSubmenuRef,
    onDismiss: closePresetDescription,
    onDismissWithFocus: closePresetDescriptionWithFocus,
  });
  usePopupDismissal({
    isOpen: isDatasetSelectorOpen,
    triggerRef: datasetTriggerRef,
    popupRef: datasetSubmenuRef,
    onDismiss: closeDatasetSelector,
    onDismissWithFocus: closeDatasetSelectorWithFocus,
  });

  return (
    <>
      <TargetSelectorSection
        presetCount={presets.length}
        selectedModel={selectedModel}
        selectedPreset={selectedPreset}
        modelOptions={modelOptions}
        presetOptions={presetOptions}
        presetSelectId={presetSelectId}
        presetDescriptionId={presetDescriptionId}
        presetDescriptionTriggerRef={presetDescriptionTriggerRef}
        isPresetDescriptionOpen={isPresetDescriptionOpen}
        hasPresetDescription={hasPresetDescription}
        onSelectModel={onSelectModel}
        onSelectPreset={onSelectPreset}
        onTogglePresetDescription={togglePresetDescription}
      />

      <DatasetSelectorTrigger
        datasetTriggerRef={datasetTriggerRef}
        datasetSelectorId={datasetSelectorId}
        isOpen={isDatasetSelectorOpen}
        disabled={datasets.length === 0}
        datasetCount={datasetCount}
        onToggle={toggleDatasetSelector}
      />

      {isDatasetSelectorOpen &&
        datasetSubmenuPosition &&
        createPortal(
          <DatasetSubmenu
            id={datasetSelectorId}
            submenuRef={datasetSubmenuRef}
            datasets={datasets}
            selectedDatasets={selectedDatasets}
            position={datasetSubmenuPosition}
            onToggleDataset={onToggleDataset}
            onSelectAllDatasets={onSelectAllDatasets}
            onSelectFirstDataset={onSelectFirstDataset}
            onClose={closeDatasetSelectorWithFocus}
          />,
          document.body,
        )}
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
