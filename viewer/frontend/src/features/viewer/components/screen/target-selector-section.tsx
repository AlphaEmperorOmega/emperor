import {
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
import { ViewModeButton } from "@/features/viewer/components/view-mode-button";
import { SelectOnlyDropdown } from "@/features/viewer/components/screen/select-only-dropdown";
import { SectionHeading } from "@/features/viewer/components/shared/section-heading";

type TargetMode = "preset" | "snapshot" | "experiment";

type SelectOption = {
  value: string;
  label: string;
  description?: string;
};

const fieldIconClassName = "h-[15px] w-[15px] text-violet";

export function TargetSelectorSection({
  selectedModelType,
  selectedModel,
  selectedTargetMode,
  selectedPreset,
  selectedSnapshotId,
  selectedSnapshotName,
  selectedHistoricalExperimentFilter,
  selectedHistoricalDatasetFilter,
  selectedHistoricalPreset,
  configSnapshotsEnabled,
  isSchemaReady,
  modelTypeOptions,
  modelOptions,
  presetOptions,
  snapshotOptions,
  experimentOptions,
  experimentDatasetOptions,
  experimentPresetOptions,
  presetSelectId,
  snapshotSelectId,
  experimentSelectId,
  experimentDatasetSelectId,
  experimentPresetSelectId,
  presetTrainingCommandDisabled,
  snapshotTrainingCommandDisabled,
  onSelectModelType,
  onSelectModel,
  onActivatePresetMode,
  onActivateSnapshotMode,
  onActivateExperimentMode,
  onSelectPreset,
  onSelectSnapshot,
  onSelectHistoricalExperimentFilter,
  onSelectHistoricalDatasetFilter,
  onSelectHistoricalPreset,
  onCreateSnapshot,
  onEditSnapshot,
  onDuplicateSnapshot,
  onOpenPresetTrainingCommand,
  onOpenSnapshotTrainingCommand,
}: {
  selectedModelType: string;
  selectedModel: string;
  selectedTargetMode: TargetMode;
  selectedPreset: string;
  selectedSnapshotId: string;
  selectedSnapshotName: string;
  selectedHistoricalExperimentFilter: string;
  selectedHistoricalDatasetFilter: string;
  selectedHistoricalPreset: string;
  configSnapshotsEnabled: boolean;
  isSchemaReady: boolean;
  modelTypeOptions: SelectOption[];
  modelOptions: SelectOption[];
  presetOptions: SelectOption[];
  snapshotOptions: SelectOption[];
  experimentOptions: SelectOption[];
  experimentDatasetOptions: SelectOption[];
  experimentPresetOptions: SelectOption[];
  presetSelectId: string;
  snapshotSelectId: string;
  experimentSelectId: string;
  experimentDatasetSelectId: string;
  experimentPresetSelectId: string;
  presetTrainingCommandDisabled: boolean;
  snapshotTrainingCommandDisabled: boolean;
  onSelectModelType: (modelType: string) => void;
  onSelectModel: (model: string) => void;
  onActivatePresetMode: () => void;
  onActivateSnapshotMode: () => void;
  onActivateExperimentMode: () => void;
  onSelectPreset: (preset: string) => void;
  onSelectSnapshot: (snapshotId: string) => boolean | void;
  onSelectHistoricalExperimentFilter: (experiment: string) => void;
  onSelectHistoricalDatasetFilter: (dataset: string) => void;
  onSelectHistoricalPreset: (preset: string) => void;
  onCreateSnapshot: () => void;
  onEditSnapshot: () => void;
  onDuplicateSnapshot: () => void;
  onOpenPresetTrainingCommand: () => void;
  onOpenSnapshotTrainingCommand: () => void;
}) {
  const hasSnapshots = snapshotOptions.length > 0;
  const hasExperimentRuns = experimentOptions.length > 0;
  const canActivateExperimentMode = Boolean(selectedModel) || hasExperimentRuns;
  const activeTargetMode =
    selectedTargetMode === "snapshot" && hasSnapshots
      ? "snapshot"
      : selectedTargetMode === "experiment" && canActivateExperimentMode
        ? "experiment"
        : "preset";
  const snapshotValue =
    activeTargetMode === "snapshot" &&
    snapshotOptions.some((option) => option.value === selectedSnapshotId)
      ? selectedSnapshotId
      : "";
  const experimentValue =
    activeTargetMode === "experiment" &&
    experimentOptions.some(
      (option) => option.value === selectedHistoricalExperimentFilter,
    )
      ? selectedHistoricalExperimentFilter
      : "";
  const experimentDatasetValue =
    activeTargetMode === "experiment" &&
    experimentDatasetOptions.some(
      (option) => option.value === selectedHistoricalDatasetFilter,
    )
      ? selectedHistoricalDatasetFilter
      : "";
  const experimentPresetValue =
    activeTargetMode === "experiment" &&
    experimentPresetOptions.some(
      (option) => option.value === selectedHistoricalPreset,
    )
      ? selectedHistoricalPreset
      : "";
  const snapshotActionsDisabled =
    !configSnapshotsEnabled ||
    !isSchemaReady ||
    !selectedModel ||
    !selectedPreset;

  return (
    <section className="grid gap-3">
      <div className="grid min-w-0 gap-2">
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
              value={selectedPreset}
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
            disabled={snapshotActionsDisabled}
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
