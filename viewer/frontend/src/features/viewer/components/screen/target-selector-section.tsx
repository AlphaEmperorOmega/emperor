import { type RefObject } from "react";
import {
  Camera,
  FlaskConical,
  Info,
  SlidersHorizontal,
  Target,
} from "lucide-react";
import { IconButton } from "@/components/ui/icon-button";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { ViewModeButton } from "@/features/viewer/components/view-mode-button";
import { SelectOnlyDropdown } from "@/features/viewer/components/screen/select-only-dropdown";
import { SectionHeading } from "@/features/viewer/components/shared/section-heading";
import { cn } from "@/lib/utils";

type TargetMode = "preset" | "snapshot" | "experiment";

type SelectOption = {
  value: string;
  label: string;
};

export function TargetSelectorSection({
  presetCount,
  selectedModelType,
  selectedModel,
  selectedTargetMode,
  selectedPreset,
  selectedSnapshotId,
  selectedExperimentRunId,
  modelTypeOptions,
  modelOptions,
  presetOptions,
  snapshotOptions,
  experimentOptions,
  presetSelectId,
  snapshotSelectId,
  experimentSelectId,
  presetDescriptionId,
  presetDescriptionTriggerRef,
  isPresetDescriptionOpen,
  hasPresetDescription,
  onSelectModelType,
  onSelectModel,
  onActivatePresetMode,
  onActivateSnapshotMode,
  onActivateExperimentMode,
  onSelectPreset,
  onSelectSnapshot,
  onSelectExperimentRun,
  onTogglePresetDescription,
}: {
  presetCount: number;
  selectedModelType: string;
  selectedModel: string;
  selectedTargetMode: TargetMode;
  selectedPreset: string;
  selectedSnapshotId: string;
  selectedExperimentRunId: string;
  modelTypeOptions: SelectOption[];
  modelOptions: SelectOption[];
  presetOptions: SelectOption[];
  snapshotOptions: SelectOption[];
  experimentOptions: SelectOption[];
  presetSelectId: string;
  snapshotSelectId: string;
  experimentSelectId: string;
  presetDescriptionId: string;
  presetDescriptionTriggerRef: RefObject<HTMLButtonElement | null>;
  isPresetDescriptionOpen: boolean;
  hasPresetDescription: boolean;
  onSelectModelType: (modelType: string) => void;
  onSelectModel: (model: string) => void;
  onActivatePresetMode: () => void;
  onActivateSnapshotMode: () => void;
  onActivateExperimentMode: () => void;
  onSelectPreset: (preset: string) => void;
  onSelectSnapshot: (snapshotId: string) => void;
  onSelectExperimentRun: (runId: string) => void;
  onTogglePresetDescription: () => void;
}) {
  const hasSnapshots = snapshotOptions.length > 0;
  const hasExperimentRuns = experimentOptions.length > 0;
  const activeTargetMode =
    selectedTargetMode === "snapshot" && hasSnapshots
      ? "snapshot"
      : selectedTargetMode === "experiment" && hasExperimentRuns
        ? "experiment"
        : "preset";
  const snapshotValue =
    activeTargetMode === "snapshot" &&
    snapshotOptions.some((option) => option.value === selectedSnapshotId)
      ? selectedSnapshotId
      : "";
  const experimentRunValue =
    activeTargetMode === "experiment" &&
    experimentOptions.some((option) => option.value === selectedExperimentRunId)
      ? selectedExperimentRunId
      : "";

  return (
    <section className="grid gap-3">
      <div className="flex items-center justify-between gap-3">
        <SectionHeading
          icon={<Target className="h-[15px] w-[15px] text-violet" aria-hidden />}
          title="Target"
        />
        <span className="text-xs font-medium text-ink-dim">{presetCount} presets</span>
      </div>
      <div className="grid min-w-0 gap-2">
        <div className="grid min-w-0 gap-1.5">
          <span className="text-xs font-semibold tracking-[0.02em] text-ink-dim">
            Model type
          </span>
          <SelectOnlyDropdown
            label="model type"
            value={selectedModelType}
            options={modelTypeOptions}
            onChange={onSelectModelType}
            placeholder="Select type"
          />
        </div>
        <div className="grid min-w-0 gap-1.5">
          <span className="text-xs font-semibold tracking-[0.02em] text-ink-dim">
            Model
          </span>
          <SelectOnlyDropdown
            label="model"
            value={selectedModel}
            options={modelOptions}
            onChange={onSelectModel}
            placeholder="Select model"
          />
        </div>
      </div>
      <SegmentedControl
        aria-label="Target mode"
        className="grid w-full grid-cols-3"
      >
        <ViewModeButton
          active={activeTargetMode === "preset"}
          onClick={onActivatePresetMode}
        >
          <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden />
          Presets
        </ViewModeButton>
        <ViewModeButton
          active={activeTargetMode === "snapshot"}
          disabled={!hasSnapshots}
          onClick={onActivateSnapshotMode}
        >
          <Camera className="h-3.5 w-3.5" aria-hidden />
          Snapshots
        </ViewModeButton>
        <ViewModeButton
          active={activeTargetMode === "experiment"}
          disabled={!hasExperimentRuns}
          onClick={onActivateExperimentMode}
        >
          <FlaskConical className="h-3.5 w-3.5" aria-hidden />
          Experiments
        </ViewModeButton>
      </SegmentedControl>
      {activeTargetMode === "preset" ? (
        <div className="grid gap-1.5">
          <label
            htmlFor={presetSelectId}
            className="text-xs font-semibold tracking-[0.02em] text-ink-dim"
          >
            Preset
          </label>
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
              ref={presetDescriptionTriggerRef}
              label="Show preset description"
              icon={<Info className="h-4 w-4" aria-hidden />}
              size="md"
              variant="edge"
              className={cn(
                "h-10 w-10",
                isPresetDescriptionOpen &&
                  "border-violet/40 bg-control-selected text-ink",
              )}
              aria-haspopup="dialog"
              aria-expanded={isPresetDescriptionOpen}
              aria-controls={presetDescriptionId}
              disabled={!hasPresetDescription}
              onClick={onTogglePresetDescription}
            />
          </div>
        </div>
      ) : activeTargetMode === "snapshot" ? (
        <div className="grid gap-1.5">
          <label
            htmlFor={snapshotSelectId}
            className="text-xs font-semibold tracking-[0.02em] text-ink-dim"
          >
            Snapshot
          </label>
          <SelectOnlyDropdown
            id={snapshotSelectId}
            label="snapshot"
            value={snapshotValue}
            options={snapshotOptions}
            onChange={onSelectSnapshot}
            placeholder="Select snapshot"
            className="min-w-0"
          />
        </div>
      ) : (
        <div className="grid gap-1.5">
          <label
            htmlFor={experimentSelectId}
            className="text-xs font-semibold tracking-[0.02em] text-ink-dim"
          >
            Experiment run
          </label>
          <SelectOnlyDropdown
            id={experimentSelectId}
            label="experiment run"
            value={experimentRunValue}
            options={experimentOptions}
            onChange={onSelectExperimentRun}
            placeholder="Select run"
            className="min-w-0"
          />
        </div>
      )}
    </section>
  );
}
