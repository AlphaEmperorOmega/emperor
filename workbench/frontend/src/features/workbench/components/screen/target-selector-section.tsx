import {
  Activity,
  Copy,
  Cpu,
  FilePlus2,
  Layers,
  Pencil,
  SlidersHorizontal,
  Terminal,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { ViewModeButton } from "@/features/workbench/components/view-mode-button";
import { SelectOnlyDropdown } from "@/features/workbench/components/screen/select-only-dropdown";
import { SectionHeading } from "@/components/ui/section-heading";

type TargetMode = "preset" | "snapshot" | "experiment";

export type TargetSelectorOption = {
  value: string;
  label: string;
  description?: string;
};

export type TargetSelectorView = {
  modelPackage: {
    modelType: string;
    model: string;
    modelTypes: TargetSelectorOption[];
    models: TargetSelectorOption[];
  };
  experimentTask: {
    visible: boolean;
    value: string;
    options: TargetSelectorOption[];
  };
  source: {
    activeMode: TargetMode;
    snapshotAvailable: boolean;
    historicalAvailable: boolean;
    preset: {
      value: string;
      options: TargetSelectorOption[];
      trainingCommandDisabled: boolean;
      createSnapshotDisabled: boolean;
    };
    snapshot: {
      value: string;
      name: string;
      options: TargetSelectorOption[];
      trainingCommandDisabled: boolean;
      actionsDisabled: boolean;
    };
    historical: {
      experiment: { value: string; options: TargetSelectorOption[] };
      dataset: { value: string; options: TargetSelectorOption[] };
      preset: { value: string; options: TargetSelectorOption[] };
    };
  };
};

export type TargetSelectorCommands = {
  selectModelType: (modelType: string) => void;
  selectModel: (model: string) => void;
  selectExperimentTask: (experimentTask: string) => void;
  showPreset: () => void;
  showSnapshot: () => void;
  showHistorical: () => void;
  selectPreset: (preset: string) => void;
  selectSnapshot: (snapshotId: string) => boolean | void;
  selectHistoricalExperiment: (experiment: string) => void;
  selectHistoricalDataset: (dataset: string) => void;
  selectHistoricalPreset: (preset: string) => void;
  createSnapshot: () => void;
  editSnapshot: () => void;
  duplicateSnapshot: () => void;
  openPresetTrainingCommand: () => void;
  openSnapshotTrainingCommand: () => void;
};

const fieldIconClassName = "h-[15px] w-[15px] text-violet";

export function TargetSelectorSection({
  view,
  commands,
}: {
  view: TargetSelectorView;
  commands: TargetSelectorCommands;
}) {
  const { modelPackage, experimentTask, source } = view;
  const activeTargetMode = source.activeMode;
  const hasSnapshots = source.snapshotAvailable;
  const hasExperimentTasks = experimentTask.visible;
  const canActivateExperimentMode = source.historicalAvailable;
  const selectedModelType = modelPackage.modelType;
  const selectedModel = modelPackage.model;
  const modelTypeOptions = modelPackage.modelTypes;
  const modelOptions = modelPackage.models;
  const selectedExperimentTask = experimentTask.value;
  const experimentTaskOptions = experimentTask.options;
  const presetControlValue = source.preset.value;
  const presetOptions = source.preset.options;
  const presetTrainingCommandDisabled = source.preset.trainingCommandDisabled;
  const snapshotValue = source.snapshot.value;
  const selectedSnapshotName = source.snapshot.name;
  const snapshotOptions = source.snapshot.options;
  const snapshotTrainingCommandDisabled =
    source.snapshot.trainingCommandDisabled;
  const snapshotActionsDisabled = source.snapshot.actionsDisabled;
  const experimentValue = source.historical.experiment.value;
  const experimentOptions = source.historical.experiment.options;
  const experimentDatasetValue = source.historical.dataset.value;
  const experimentDatasetOptions = source.historical.dataset.options;
  const experimentPresetValue = source.historical.preset.value;
  const experimentPresetOptions = source.historical.preset.options;
  const {
    selectModelType: onSelectModelType,
    selectModel: onSelectModel,
    selectExperimentTask: onSelectExperimentTask,
    showPreset: onActivatePresetMode,
    showSnapshot: onActivateSnapshotMode,
    showHistorical: onActivateExperimentMode,
    selectPreset: onSelectPreset,
    selectSnapshot: onSelectSnapshot,
    selectHistoricalExperiment: onSelectHistoricalExperimentFilter,
    selectHistoricalDataset: onSelectHistoricalDatasetFilter,
    selectHistoricalPreset: onSelectHistoricalPreset,
    createSnapshot: onCreateSnapshot,
    editSnapshot: onEditSnapshot,
    duplicateSnapshot: onDuplicateSnapshot,
    openPresetTrainingCommand: onOpenPresetTrainingCommand,
    openSnapshotTrainingCommand: onOpenSnapshotTrainingCommand,
  } = commands;
  const presetSelectId = "target-preset-select";
  const experimentTaskSelectId = "target-experiment-task-select";
  const snapshotSelectId = "target-snapshot-select";
  const experimentSelectId = "target-experiment-select";
  const experimentDatasetSelectId = "target-experiment-dataset-select";
  const experimentPresetSelectId = "target-experiment-preset-select";

  return (
    <section className="grid gap-3">
      <div className="grid min-w-0 gap-2">
        {hasExperimentTasks && (
          <div className="grid min-w-0 gap-1.5">
            <SectionHeading
              icon={<Activity className={fieldIconClassName} aria-hidden />}
              title="Experiment Task"
            />
            <SelectOnlyDropdown
              id={experimentTaskSelectId}
              label="Experiment Task"
              value={selectedExperimentTask}
              options={experimentTaskOptions}
              onChange={onSelectExperimentTask}
              placeholder="Select task"
            />
          </div>
        )}
        <div className="grid min-w-0 gap-1.5">
          <SectionHeading
            icon={<Layers className={fieldIconClassName} aria-hidden />}
            title="Model Type"
          />
          <SelectOnlyDropdown
            label="model type"
            value={selectedModelType}
            options={modelTypeOptions}
            onChange={onSelectModelType}
            placeholder="Select type"
          />
        </div>
        <div className="grid min-w-0 gap-1.5">
          <SectionHeading
            icon={<Cpu className={fieldIconClassName} aria-hidden />}
            title="Model Name"
          />
          <SelectOnlyDropdown
            label="model"
            value={selectedModel}
            options={modelOptions}
            onChange={onSelectModel}
            placeholder="Select model"
          />
        </div>
      </div>
      <div className="grid gap-1.5">
        <SectionHeading
          icon={<SlidersHorizontal className={fieldIconClassName} aria-hidden />}
          title="Configuration Source"
        />
        <SegmentedControl
          aria-label="Configuration Source"
          className="grid w-full grid-cols-3 [&>button]:justify-center [&>button]:text-center"
        >
          <ViewModeButton
            active={activeTargetMode === "preset"}
            onClick={onActivatePresetMode}
          >
            Presets
          </ViewModeButton>
          <ViewModeButton
            active={activeTargetMode === "snapshot"}
            disabled={!hasSnapshots}
            onClick={onActivateSnapshotMode}
          >
            Snapshots
          </ViewModeButton>
          <ViewModeButton
            active={activeTargetMode === "experiment"}
            disabled={!canActivateExperimentMode}
            onClick={onActivateExperimentMode}
          >
            Experiments
          </ViewModeButton>
        </SegmentedControl>
      </div>
      {activeTargetMode === "preset" ? (
        <div className="grid gap-3">
          <div className="grid grid-cols-[minmax(0,1fr)_40px] gap-2">
            <SelectOnlyDropdown
              id={presetSelectId}
              label="preset"
              value={presetControlValue}
              options={presetOptions}
              onChange={onSelectPreset}
              placeholder="Select preset"
              className="min-w-0"
            />
            <IconButton
              label="Training command for preset"
              icon={<Terminal className="h-4 w-4" aria-hidden />}
              size="md"
              variant="edge"
              className="h-10 w-10"
              aria-haspopup="dialog"
              disabled={presetTrainingCommandDisabled}
              onClick={onOpenPresetTrainingCommand}
            />
          </div>
          <Button
            variant="secondary"
            onClick={onCreateSnapshot}
            disabled={source.preset.createSnapshotDisabled}
            className="h-9 justify-center text-xs"
          >
            <FilePlus2 className="h-3.5 w-3.5" aria-hidden />
            Create Snapshot
          </Button>
        </div>
      ) : activeTargetMode === "snapshot" ? (
        <div className="grid gap-3">
          <div className="grid grid-cols-[minmax(0,1fr)_40px] gap-2">
            <SelectOnlyDropdown
              id={snapshotSelectId}
              label="snapshot"
              value={snapshotValue}
              options={snapshotOptions}
              onChange={onSelectSnapshot}
              placeholder="Select snapshot"
              className="min-w-0"
            />
            <IconButton
              label="Training command for snapshot"
              icon={<Terminal className="h-4 w-4" aria-hidden />}
              size="md"
              variant="edge"
              className="h-10 w-10"
              aria-haspopup="dialog"
              disabled={snapshotTrainingCommandDisabled || !snapshotValue}
              onClick={onOpenSnapshotTrainingCommand}
            />
          </div>
          <div className="grid grid-cols-2 gap-2">
            <Button
              variant="secondary"
              onClick={onEditSnapshot}
              disabled={snapshotActionsDisabled || !snapshotValue}
              className="h-9 justify-center text-xs"
              title={selectedSnapshotName ? `Edit ${selectedSnapshotName}` : undefined}
            >
              <Pencil className="h-3.5 w-3.5" aria-hidden />
              Edit
            </Button>
            <Button
              variant="secondary"
              onClick={onDuplicateSnapshot}
              disabled={snapshotActionsDisabled || !snapshotValue}
              className="h-9 justify-center text-xs"
              title={
                selectedSnapshotName
                  ? `Duplicate ${selectedSnapshotName}`
                  : undefined
              }
            >
              <Copy className="h-3.5 w-3.5" aria-hidden />
              Duplicate
            </Button>
          </div>
        </div>
      ) : (
        <div className="grid gap-2">
          <div className="grid gap-1.5">
            <label
              htmlFor={experimentSelectId}
              className="text-xs font-semibold tracking-[0.02em] text-ink-dim"
            >
              Experiment
            </label>
            <SelectOnlyDropdown
              id={experimentSelectId}
              label="Experiment"
              value={experimentValue}
              options={experimentOptions}
              onChange={onSelectHistoricalExperimentFilter}
              placeholder="Select experiment"
              className="min-w-0"
            />
          </div>
          <div className="grid gap-1.5">
            <label
              htmlFor={experimentDatasetSelectId}
              className="text-xs font-semibold tracking-[0.02em] text-ink-dim"
            >
              Dataset
            </label>
            <SelectOnlyDropdown
              id={experimentDatasetSelectId}
              label="Dataset"
              value={experimentDatasetValue}
              options={experimentDatasetOptions}
              onChange={onSelectHistoricalDatasetFilter}
              placeholder="Select dataset"
              disabled={!experimentValue}
              className="min-w-0"
            />
          </div>
          <div className="grid gap-1.5">
            <label
              htmlFor={experimentPresetSelectId}
              className="text-xs font-semibold tracking-[0.02em] text-ink-dim"
            >
              Preset
            </label>
            <SelectOnlyDropdown
              id={experimentPresetSelectId}
              label="Preset"
              value={experimentPresetValue}
              options={experimentPresetOptions}
              onChange={onSelectHistoricalPreset}
              placeholder="Select preset"
              disabled={!experimentValue || !experimentDatasetValue}
              className="min-w-0"
            />
          </div>
        </div>
      )}
    </section>
  );
}
