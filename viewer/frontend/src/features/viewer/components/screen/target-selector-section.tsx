import { type RefObject } from "react";
import { Info, Target } from "lucide-react";
import { IconButton } from "@/components/ui/icon-button";
import { SelectOnlyDropdown } from "@/features/viewer/components/screen/select-only-dropdown";
import { SectionHeading } from "@/features/viewer/components/shared/section-heading";
import { cn } from "@/lib/utils";

type SelectOption = {
  value: string;
  label: string;
};

export function TargetSelectorSection({
  presetCount,
  selectedModelType,
  selectedModel,
  selectedPreset,
  modelTypeOptions,
  modelOptions,
  presetOptions,
  presetSelectId,
  presetDescriptionId,
  presetDescriptionTriggerRef,
  isPresetDescriptionOpen,
  hasPresetDescription,
  onSelectModelType,
  onSelectModel,
  onSelectPreset,
  onTogglePresetDescription,
}: {
  presetCount: number;
  selectedModelType: string;
  selectedModel: string;
  selectedPreset: string;
  modelTypeOptions: SelectOption[];
  modelOptions: SelectOption[];
  presetOptions: SelectOption[];
  presetSelectId: string;
  presetDescriptionId: string;
  presetDescriptionTriggerRef: RefObject<HTMLButtonElement | null>;
  isPresetDescriptionOpen: boolean;
  hasPresetDescription: boolean;
  onSelectModelType: (modelType: string) => void;
  onSelectModel: (model: string) => void;
  onSelectPreset: (preset: string) => void;
  onTogglePresetDescription: () => void;
}) {
  return (
    <section className="grid gap-3">
      <div className="flex items-center justify-between gap-3">
        <SectionHeading
          icon={<Target className="h-[15px] w-[15px] text-violet" aria-hidden />}
          title="Target"
        />
        <span className="text-xs font-medium text-ink-dim">{presetCount} presets</span>
      </div>
      <div className="grid gap-1.5">
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
      <div className="grid gap-1.5">
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
    </section>
  );
}
