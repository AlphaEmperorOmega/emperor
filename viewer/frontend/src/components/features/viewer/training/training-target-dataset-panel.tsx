import { Database, Layers, SlidersHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { MultiSelectDropdown } from "@/components/features/viewer/screen/multi-select-dropdown";
import { SelectOnlyDropdown } from "@/components/features/viewer/screen/select-only-dropdown";
import { InlineStatus } from "@/components/features/viewer/shared/inline-status";
import { SectionHeading } from "@/components/features/viewer/shared/section-heading";
import { StatChip } from "@/components/features/viewer/shared/stat-chip";
import { TrainingFooterField } from "@/components/features/viewer/training/training-footer-field";
import { type Dataset } from "@/lib/api";

type SelectOption = {
  value: string;
  label: string;
};

const footerIconClass = "h-[15px] w-[15px] text-violet";
const defaultFieldLabelClass =
  "text-xs font-semibold tracking-[0.02em] text-ink-dim";

export function TrainingTargetDatasetPanel({
  modelOptions,
  presetOptions,
  selectedModel,
  selectedPreset,
  selectedTrainingPresets = selectedPreset ? [selectedPreset] : [],
  datasetOptions,
  selectedDatasets,
  onSelectModel,
  onSelectPreset,
  onSetTrainingPresets,
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
  modelOptions: SelectOption[];
  presetOptions: SelectOption[];
  selectedModel: string;
  selectedPreset: string;
  selectedTrainingPresets?: string[];
  datasetOptions: Dataset[];
  selectedDatasets: string[];
  onSelectModel: (model: string) => void;
  onSelectPreset: (preset: string) => void;
  onSetTrainingPresets?: (presets: string[]) => void;
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
  const datasetCount = `${selectedDatasets.length} / ${datasetOptions.length}`;
  const trainingPresetCount = `${selectedTrainingPresets.length} / ${presetOptions.length}`;
  const trainingPresetDisabledValues =
    selectedTrainingPresets.length === 1 ? selectedTrainingPresets : [];
  const datasetDisabledValues =
    selectedDatasets.length === 1 ? selectedDatasets : [];
  const trainingPresetOptions = presetOptions.map((preset) => ({
    value: preset.value,
    label: preset.label,
    description: preset.value,
  }));
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

  const modelControl = (
    <SelectOnlyDropdown
      label="training model"
      value={selectedModel}
      options={modelOptions}
      onChange={onSelectModel}
      placeholder="Select model"
    />
  );

  const modelField = isFooterPresentation ? (
    <TrainingFooterField
      className="min-w-0"
      icon={<Layers className={footerIconClass} aria-hidden />}
      label="Model"
    >
      {modelControl}
    </TrainingFooterField>
  ) : (
    <div className="grid min-w-0 gap-1.5">
      <span className={defaultFieldLabelClass}>Model</span>
      {modelControl}
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

  const presetsField = isFooterPresentation ? (
    <TrainingFooterField
      className="min-w-0"
      icon={<SlidersHorizontal className={footerIconClass} aria-hidden />}
      label="Presets"
      detail={<StatChip>{trainingPresetCount}</StatChip>}
    >
      {presetsControls}
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
