import { useId, useState } from "react";
import { Camera, Database, Layers, SlidersHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { MultiSelectDropdown } from "@/features/viewer/components/screen/multi-select-dropdown";
import { SelectOnlyDropdown } from "@/features/viewer/components/screen/select-only-dropdown";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { SectionHeading } from "@/features/viewer/components/shared/section-heading";
import { StatChip } from "@/features/viewer/components/shared/stat-chip";
import { TrainingFooterField } from "@/features/viewer/components/training/training-footer-field";
import { ViewModeButton } from "@/features/viewer/components/view-mode-button";
import { type Dataset } from "@/lib/api";
import { type ConfigSnapshot } from "@/lib/config-snapshots";

type SelectOption = {
  value: string;
  label: string;
};

type TrainingConfigTab = "presets" | "snapshots";

const footerIconClass = "h-[15px] w-[15px] text-violet";
const defaultFieldLabelClass =
  "text-xs font-semibold tracking-[0.02em] text-ink-dim";

function overrideCountLabel(count: number) {
  return `${count} override${count === 1 ? "" : "s"}`;
}

export function TrainingTargetDatasetPanel({
  modelTypeOptions = [],
  modelOptions,
  selectedModelType = "",
  presetOptions,
  selectedModel,
  selectedPreset,
  selectedTrainingPresets = selectedPreset ? [selectedPreset] : [],
  configSnapshots = [],
  selectedTrainingSnapshotIds = [],
  datasetOptions,
  selectedDatasets,
  onSelectModelType,
  onSelectModel,
  onSelectPreset,
  onSetTrainingPresets,
  onSetTrainingSnapshotSelection,
  onToggleTrainingPreset,
  onMakeTrainingPresetPrimary,
  onSelectAllTrainingPresets,
  onSelectPrimaryTrainingPreset,
  onSetDatasets,
  onToggleDataset,
  onSelectAllDatasets,
  onSelectFirstDataset,
  presentation = "default",
}: {
  modelTypeOptions?: SelectOption[];
  modelOptions: SelectOption[];
  selectedModelType?: string;
  presetOptions: SelectOption[];
  selectedModel: string;
  selectedPreset: string;
  selectedTrainingPresets?: string[];
  configSnapshots?: ConfigSnapshot[];
  selectedTrainingSnapshotIds?: string[];
  datasetOptions: Dataset[];
  selectedDatasets: string[];
  onSelectModelType?: (modelType: string) => void;
  onSelectModel: (model: string) => void;
  onSelectPreset: (preset: string) => void;
  onSetTrainingPresets?: (presets: string[]) => void;
  onSetTrainingSnapshotSelection?: (snapshotIds: string[]) => void;
  onToggleTrainingPreset?: (preset: string) => void;
  onMakeTrainingPresetPrimary?: (preset: string) => void;
  onSelectAllTrainingPresets?: () => void;
  onSelectPrimaryTrainingPreset?: () => void;
  onSetDatasets?: (datasets: string[]) => void;
  onToggleDataset?: (dataset: string) => void;
  onSelectAllDatasets: () => void;
  onSelectFirstDataset: () => void;
  presentation?: "default" | "footer";
}) {
  const isFooterPresentation = presentation === "footer";
  const [activeTrainingConfigTab, setActiveTrainingConfigTab] =
    useState<TrainingConfigTab>("presets");
  const trainingConfigTabsId = useId();
  const presetsTabId = `${trainingConfigTabsId}-presets-tab`;
  const snapshotsTabId = `${trainingConfigTabsId}-snapshots-tab`;
  const presetsPanelId = `${trainingConfigTabsId}-presets-panel`;
  const snapshotsPanelId = `${trainingConfigTabsId}-snapshots-panel`;
  const datasetCount = `${selectedDatasets.length} / ${datasetOptions.length}`;
  const trainingPresetCount = `${selectedTrainingPresets.length} / ${presetOptions.length}`;
  const trainingSnapshotCount = `${selectedTrainingSnapshotIds.length} / ${configSnapshots.length}`;
  const trainingPresetDisabledValues =
    selectedTrainingPresets.length === 1 && selectedTrainingSnapshotIds.length === 0
      ? selectedTrainingPresets
      : [];
  const datasetDisabledValues =
    selectedDatasets.length === 1 ? selectedDatasets : [];
  const trainingPresetOptions = presetOptions.map((preset) => ({
    value: preset.value,
    label: preset.label,
    description: preset.value,
  }));
  const trainingSnapshotOptions = configSnapshots.map((snapshot) => {
    const overrideCount = Object.keys(snapshot.overrides).length;
    return {
      value: snapshot.id,
      label: snapshot.name,
      description: `${snapshot.preset} · ${overrideCountLabel(overrideCount)}`,
      meta: <span>{snapshot.preset}</span>,
    };
  });
  const trainingDatasetOptions = datasetOptions.map((dataset) => ({
    value: dataset.name,
    label: dataset.label,
    description: dataset.name,
    meta: (
      <span>
        {dataset.inputDim} {"->"} {dataset.outputDim}
      </span>
    ),
  }));
  function changeTrainingPresets(nextPresets: string[]) {
    if (onSetTrainingPresets) {
      onSetTrainingPresets(nextPresets);
      return;
    }
    const changedPreset = presetOptions.find(
      (preset) =>
        selectedTrainingPresets.includes(preset.value) !==
        nextPresets.includes(preset.value),
    );
    if (changedPreset) {
      onToggleTrainingPreset?.(changedPreset.value);
    }
  }

  function changeTrainingSnapshots(nextSnapshotIds: string[]) {
    onSetTrainingSnapshotSelection?.(nextSnapshotIds);
  }

  function makeTrainingPresetPrimary(preset: string) {
    if (onMakeTrainingPresetPrimary) {
      onMakeTrainingPresetPrimary(preset);
      return;
    }
    onSelectPreset(preset);
  }

  function changeDatasets(nextDatasets: string[]) {
    if (onSetDatasets) {
      onSetDatasets(nextDatasets);
      return;
    }
    const changedDataset = datasetOptions.find(
      (dataset) =>
        selectedDatasets.includes(dataset.name) !==
        nextDatasets.includes(dataset.name),
    );
    if (changedDataset) {
      onToggleDataset?.(changedDataset.name);
    }
  }

  const modelTypeControl =
    modelTypeOptions.length > 0 && onSelectModelType ? (
      <SelectOnlyDropdown
        label="training model type"
        value={selectedModelType}
        options={modelTypeOptions}
        onChange={onSelectModelType}
        placeholder="Select type"
      />
    ) : null;

  const modelControl = (
    <SelectOnlyDropdown
      label="training model"
      value={selectedModel}
      options={modelOptions}
      onChange={onSelectModel}
      placeholder="Select model"
    />
  );
  const modelSelectorGridClass = modelTypeControl
    ? "grid min-w-0 grid-cols-[minmax(0,0.92fr)_minmax(0,1.08fr)] gap-2"
    : "grid min-w-0 gap-2";

  const modelField = isFooterPresentation ? (
    <TrainingFooterField
      className="min-w-0"
      icon={<Layers className={footerIconClass} aria-hidden />}
      label="Model"
    >
      <div className={modelSelectorGridClass}>
        {modelTypeControl && (
          <div className="grid min-w-0 gap-1.5">
            <span className={defaultFieldLabelClass}>Model type</span>
            {modelTypeControl}
          </div>
        )}
        <div className="grid min-w-0 gap-1.5">
          {modelTypeControl && (
            <span className={defaultFieldLabelClass}>Model name</span>
          )}
          {modelControl}
        </div>
      </div>
    </TrainingFooterField>
  ) : (
    <div className={modelSelectorGridClass}>
      {modelTypeControl && (
        <div className="grid min-w-0 gap-1.5">
          <span className={defaultFieldLabelClass}>Model type</span>
          {modelTypeControl}
        </div>
      )}
      <div className="grid min-w-0 gap-1.5">
        <span className={defaultFieldLabelClass}>Model</span>
        {modelControl}
      </div>
    </div>
  );

  const presetsControls = (
    <>
      <MultiSelectDropdown
        label="Presets"
        values={selectedTrainingPresets}
        options={trainingPresetOptions}
        onChange={changeTrainingPresets}
        disabledValues={trainingPresetDisabledValues}
        primaryValue={selectedPreset}
        onPrimaryChange={makeTrainingPresetPrimary}
        placeholder="Select presets"
        emptyMessage="No presets for this model"
      />
      {presetOptions.length === 0 && (
        <InlineStatus compact>
          No presets for this model
        </InlineStatus>
      )}
      <div className="grid grid-cols-2 gap-2">
        <Button
          variant="secondary"
          onClick={onSelectAllTrainingPresets}
          disabled={presetOptions.length === 0 || !onSelectAllTrainingPresets}
          className="h-9 text-[13px]"
        >
          All
        </Button>
        <Button
          variant="ghost"
          onClick={onSelectPrimaryTrainingPreset}
          disabled={!selectedPreset || !onSelectPrimaryTrainingPreset}
          className="h-9 border border-line bg-white/[0.025] text-[13px]"
        >
          Primary only
        </Button>
      </div>
    </>
  );

  const snapshotControls = (
    <>
      <MultiSelectDropdown
        label="Config snapshots"
        values={selectedTrainingSnapshotIds}
        options={trainingSnapshotOptions}
        onChange={changeTrainingSnapshots}
        placeholder="Select snapshots"
        emptyMessage="No config snapshots for this model"
      />
      {configSnapshots.length === 0 && (
        <InlineStatus compact>
          No config snapshots for this model
        </InlineStatus>
      )}
    </>
  );

  const trainingConfigTabs = (
    <SegmentedControl
      aria-label="Training config selector"
      className="grid w-full grid-cols-2"
    >
      <ViewModeButton
        id={presetsTabId}
        controls={presetsPanelId}
        active={activeTrainingConfigTab === "presets"}
        onClick={() => setActiveTrainingConfigTab("presets")}
      >
        <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden />
        Presets
      </ViewModeButton>
      <ViewModeButton
        id={snapshotsTabId}
        controls={snapshotsPanelId}
        active={activeTrainingConfigTab === "snapshots"}
        onClick={() => setActiveTrainingConfigTab("snapshots")}
      >
        <Camera className="h-3.5 w-3.5" aria-hidden />
        Snapshots
      </ViewModeButton>
    </SegmentedControl>
  );

  const presetsField = isFooterPresentation ? (
    <TrainingFooterField
      className="min-w-0"
      icon={<SlidersHorizontal className={footerIconClass} aria-hidden />}
      label="Presets"
      detail={
        <StatChip>
          {activeTrainingConfigTab === "snapshots"
            ? trainingSnapshotCount
            : trainingPresetCount}
        </StatChip>
      }
    >
      {trainingConfigTabs}
      <div
        id={presetsPanelId}
        role="tabpanel"
        aria-labelledby={presetsTabId}
        aria-label="Presets"
        hidden={activeTrainingConfigTab !== "presets"}
        className="grid gap-2"
      >
        {activeTrainingConfigTab === "presets" ? presetsControls : null}
      </div>
      <div
        id={snapshotsPanelId}
        role="tabpanel"
        aria-labelledby={snapshotsTabId}
        aria-label="Snapshots"
        hidden={activeTrainingConfigTab !== "snapshots"}
        className="grid gap-2"
      >
        {activeTrainingConfigTab === "snapshots" ? snapshotControls : null}
      </div>
    </TrainingFooterField>
  ) : (
    <div className="grid min-w-0 gap-1.5">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className={defaultFieldLabelClass}>Presets</div>
        <StatChip>{trainingPresetCount}</StatChip>
      </div>
      {presetsControls}
    </div>
  );

  const datasetsControls = (
    <>
      <MultiSelectDropdown
        label="Training datasets"
        values={selectedDatasets}
        options={trainingDatasetOptions}
        onChange={changeDatasets}
        disabledValues={datasetDisabledValues}
        placeholder="Select datasets"
        emptyMessage="No datasets for this model"
      />
      {datasetOptions.length === 0 && (
        <InlineStatus compact>
          No datasets for this model
        </InlineStatus>
      )}
      <div className="grid grid-cols-2 gap-2">
        <Button
          variant="secondary"
          onClick={onSelectAllDatasets}
          disabled={datasetOptions.length === 0}
          className="h-9 text-[13px]"
        >
          All
        </Button>
        <Button
          variant="ghost"
          onClick={onSelectFirstDataset}
          disabled={datasetOptions.length === 0}
          className="h-9 border border-line bg-white/[0.025] text-[13px]"
        >
          First
        </Button>
      </div>
    </>
  );

  const datasetsField = isFooterPresentation ? (
    <TrainingFooterField
      className="xl:min-h-0"
      icon={<Database className={footerIconClass} aria-hidden />}
      label="Datasets"
      detail={<StatChip>{datasetCount}</StatChip>}
    >
      {datasetsControls}
    </TrainingFooterField>
  ) : (
    <div className="xl:min-h-0 grid gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <SectionHeading
          icon={<Database className="h-[15px] w-[15px] text-violet" aria-hidden />}
          title="Datasets"
        />
        <StatChip>{datasetCount}</StatChip>
      </div>
      {datasetsControls}
    </div>
  );

  if (isFooterPresentation) {
    return (
      <>
        {modelField}
        {presetsField}
        {datasetsField}
      </>
    );
  }

  return (
    <div className="grid content-start gap-3 xl:h-full xl:grid-rows-[auto_minmax(0,1fr)] xl:content-stretch">
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
        {modelField}
        {presetsField}
      </div>

      {datasetsField}
    </div>
  );
}
