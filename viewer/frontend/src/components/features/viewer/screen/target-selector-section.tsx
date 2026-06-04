import { type RefObject } from "react";
import { Info, Target } from "lucide-react";
import { SelectOnlyDropdown } from "@/components/features/viewer/screen/select-only-dropdown";
import { cn } from "@/lib/utils";

type SelectOption = {
  value: string;
  label: string;
};

export function TargetSelectorSection({
  presetCount,
  selectedModel,
  selectedPreset,
  modelOptions,
  presetOptions,
  presetSelectId,
  presetDescriptionId,
  presetDescriptionTriggerRef,
  isPresetDescriptionOpen,
  hasPresetDescription,
  onSelectModel,
  onSelectPreset,
  onTogglePresetDescription,
}: {
  presetCount: number;
  selectedModel: string;
  selectedPreset: string;
  modelOptions: SelectOption[];
  presetOptions: SelectOption[];
  presetSelectId: string;
  presetDescriptionId: string;
  presetDescriptionTriggerRef: RefObject<HTMLButtonElement | null>;
  isPresetDescriptionOpen: boolean;
  hasPresetDescription: boolean;
  onSelectModel: (model: string) => void;
  onSelectPreset: (preset: string) => void;
  onTogglePresetDescription: () => void;
}) {
  return (
    <section className="grid gap-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-[0.09em] text-ink-dim">
          <Target className="h-[15px] w-[15px] text-violet" aria-hidden />
          Target
        </div>
        <span className="text-xs font-medium text-ink-dim">{presetCount} presets</span>
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
          <button
            ref={presetDescriptionTriggerRef}
            type="button"
            className={cn(
              "grid h-10 w-10 shrink-0 place-items-center rounded-[10px] border border-line bg-white/[0.035] text-ink-dim transition hover:border-white/15 hover:bg-white/[0.07] hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:border-line disabled:hover:bg-white/[0.035] disabled:hover:text-ink-dim",
              isPresetDescriptionOpen &&
                "border-violet/40 bg-[linear-gradient(135deg,rgba(146,113,255,0.1),rgba(111,168,255,0.05))] text-ink",
            )}
            aria-label="Show preset description"
            aria-haspopup="dialog"
            aria-expanded={isPresetDescriptionOpen}
            aria-controls={presetDescriptionId}
            disabled={!hasPresetDescription}
            onClick={onTogglePresetDescription}
          >
            <Info className="h-4 w-4" aria-hidden />
          </button>
        </div>
      </div>
    </section>
  );
}
