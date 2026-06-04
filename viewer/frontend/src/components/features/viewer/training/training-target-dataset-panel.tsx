import { Database, Layers, SlidersHorizontal } from "lucide-react";
import { Button } from "@/components/ui/button";
import { MultiSelectDropdown } from "@/components/features/viewer/screen/multi-select-dropdown";
import { SelectOnlyDropdown } from "@/components/features/viewer/screen/select-only-dropdown";
import { type Dataset } from "@/lib/api";
import { cn } from "@/lib/utils";

type SelectOption = {
  value: string;
  label: string;
};

const footerFieldBoxClass =
  "grid content-start gap-1.5 rounded-[10px] border border-line bg-white/[0.018] px-2.5 py-2";
const footerFieldHeaderClass =
  "flex min-h-[38px] flex-wrap items-center justify-between gap-2";
const footerHeadingClass =
  "flex items-center gap-2 text-xs font-bold uppercase tracking-[0.09em] text-ink-dim";
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

  const modelField = (
    <div
      className={cn(
        "grid min-w-0 gap-1.5",
        isFooterPresentation && footerFieldBoxClass,
      )}
    >
      {isFooterPresentation ? (
        <div className={footerFieldHeaderClass}>
          <span className={footerHeadingClass}>
            <Layers className={footerIconClass} aria-hidden />
            Model
          </span>
        </div>
      ) : (
        <span className={defaultFieldLabelClass}>Model</span>
      )}
      <SelectOnlyDropdown
        label="training model"
        value={selectedModel}
        options={modelOptions}
        onChange={onSelectModel}
        placeholder="Select model"
      />
    </div>
  );

  const presetsField = (
    <div
      className={cn(
        "grid min-w-0 gap-1.5",
        isFooterPresentation && footerFieldBoxClass,
      )}
    >
      <div
        className={
          isFooterPresentation
            ? footerFieldHeaderClass
            : "flex flex-wrap items-center justify-between gap-2"
        }
      >
        <div
          className={
            isFooterPresentation ? footerHeadingClass : defaultFieldLabelClass
          }
        >
          {isFooterPresentation && (
            <SlidersHorizontal className={footerIconClass} aria-hidden />
          )}
          Presets
        </div>
        <span className="rounded-[7px] border border-line bg-white/[0.04] px-2 py-1 font-mono text-xs text-ink-dim">
          {trainingPresetCount}
        </span>
      </div>
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
        <div className="rounded-[10px] border border-dashed border-faint bg-white/[0.018] p-3 text-sm text-ink-faint">
          No presets for this model
        </div>
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
    </div>
  );

  const datasetsField = (
    <div
      className={cn(
        "xl:min-h-0",
        isFooterPresentation ? footerFieldBoxClass : "grid gap-2",
      )}
    >
      <div
        className={
          isFooterPresentation
            ? footerFieldHeaderClass
            : "flex flex-wrap items-center justify-between gap-2"
        }
      >
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.09em] text-ink-dim">
          <Database className="h-[15px] w-[15px] text-violet" aria-hidden />
          Datasets
        </div>
        <span className="rounded-[7px] border border-line bg-white/[0.04] px-2 py-1 font-mono text-xs text-ink-dim">
          {datasetCount}
        </span>
      </div>
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
        <div className="rounded-[10px] border border-dashed border-faint bg-white/[0.018] p-3 text-sm text-ink-faint">
          No datasets for this model
        </div>
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
