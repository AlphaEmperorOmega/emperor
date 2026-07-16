import { type RefCallback } from "react";
import { Badge } from "@/components/ui/badge";
import {
  ConfigFieldOverrideIcon,
  ConfigFieldValueEditor,
} from "@/features/workbench/components/config/config-field-control";
import { surfacePanelClassName } from "@/components/ui/surface-panel";
import {
  type RuntimeDefaultsSearchOptionPresentation,
} from "@/features/workbench/state/runtime-defaults/runtime-defaults-presentation";
import { cn } from "@/lib/utils";

function ConfigSearchResultItem({
  option,
  popupId,
  isSelected,
  isActive,
  onSelect,
  onMouseEnter,
  measureOption,
  onFieldChange,
  onFieldReset,
}: {
  option: RuntimeDefaultsSearchOptionPresentation;
  popupId: string;
  isSelected: boolean;
  isActive: boolean;
  onSelect: (option: RuntimeDefaultsSearchOptionPresentation) => void;
  onMouseEnter: () => void;
  measureOption: RefCallback<HTMLElement>;
  onFieldChange: (key: string, value: string) => void;
  onFieldReset: (key: string) => void;
}) {
  const titleId = `${popupId}-option-${encodeURIComponent(option.key)}`;
  const editorId = `${popupId}-field-${option.key}-value`;
  const isOverridden = option.field.isModified;
  const isPresetOwned = option.field.isPresetOwned;
  const showCurrentValue = isOverridden || isPresetOwned;
  const currentValueLabel = option.field.value;
  const disabledReason = option.field.disabledReason;

  return (
    <div
      ref={measureOption}
      role="group"
      aria-label="Config search result"
      data-config-search-result={option.key}
      data-virtual-option-key={option.key}
      onMouseEnter={onMouseEnter}
      className={cn(
        surfacePanelClassName,
        "min-w-0 gap-2 px-3 py-2.5 transition",
        isPresetOwned ? "border-amber/35 bg-amber/[0.055]" : "border-line-soft bg-black/15",
        isSelected && "border-violet/45 bg-violet/10",
        isActive && "ring-1 ring-violet/55",
        "focus-within:ring-1 focus-within:ring-violet/45 hover:border-line hover:bg-white/[0.035]",
      )}
    >
      <span className="flex min-w-0 items-start justify-between gap-2">
        <button
          id={titleId}
          type="button"
          onClick={() => onSelect(option)}
          tabIndex={-1}
          data-config-field-label=""
          className={cn(
            "flex min-h-touch min-w-0 items-center gap-1.5 rounded-chip text-left text-sm font-semibold underline-offset-4 transition hover:text-violet hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:min-h-control",
            isOverridden ? "text-violet" : "text-ink",
          )}
        >
          <span className="min-w-0 truncate">{option.label}</span>
          {isOverridden && <ConfigFieldOverrideIcon />}
        </button>
        <span className="flex shrink-0 flex-wrap justify-end gap-1">
          {isPresetOwned && <Badge variant="preset">preset</Badge>}
        </span>
      </span>
      <span className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
        <span className="min-w-0 truncate font-mono text-xs text-ink-dim">
          {option.key}
        </span>
      </span>
      <span className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 text-xs text-ink-dim">
        <span className="min-w-0 truncate">{option.sectionTitle}</span>
        {showCurrentValue && (
          <span className="min-w-0 truncate">
            current <span className="font-mono text-ink">{currentValueLabel}</span>
          </span>
        )}
      </span>
      <ConfigFieldValueEditor
        presentation={option.field}
        onChange={onFieldChange}
        onReset={onFieldReset}
        controlId={editorId}
        controlLabel={`${option.label} current value`}
        resetLabel={`Reset ${option.label} search result override`}
        resetTitle={`Reset ${option.label} override`}
        density="compact"
        disabled={Boolean(disabledReason)}
        className="mt-0"
      />
      {disabledReason && (
        <span className="text-xs font-medium text-ink-dim">{disabledReason}</span>
      )}
    </div>
  );
}

export function ConfigSearchResults({
  popupId,
  virtualOptions,
  beforeHeight,
  afterHeight,
  activeIndex,
  selectedFieldKey,
  onSelect,
  measureOption,
  onOptionMouseEnter,
  onFieldChange,
  onFieldReset,
}: {
  popupId: string;
  virtualOptions: Array<{
    index: number;
    key: string;
    option: RuntimeDefaultsSearchOptionPresentation;
  }>;
  beforeHeight: number;
  afterHeight: number;
  activeIndex: number;
  selectedFieldKey: string | null;
  onSelect: (option: RuntimeDefaultsSearchOptionPresentation) => void;
  measureOption: RefCallback<HTMLElement>;
  onOptionMouseEnter: (index: number) => void;
  onFieldChange: (key: string, value: string) => void;
  onFieldReset: (key: string) => void;
}) {
  if (virtualOptions.length === 0) {
    return (
      <div
        className={cn(
          surfacePanelClassName,
          "border-dashed border-line-soft bg-black/15 px-3 py-2.5 text-sm text-ink-dim",
        )}
      >
        No matching fields
      </div>
    );
  }

  return (
    <>
      {beforeHeight > 0 && <div aria-hidden style={{ height: beforeHeight }} />}
      {virtualOptions.map(({ option, index, key }) => (
        <ConfigSearchResultItem
          key={key}
          option={option}
          popupId={popupId}
          isSelected={option.key === selectedFieldKey}
          isActive={index === activeIndex}
          onSelect={onSelect}
          onMouseEnter={() => onOptionMouseEnter(index)}
          measureOption={measureOption}
          onFieldChange={onFieldChange}
          onFieldReset={onFieldReset}
        />
      ))}
      {afterHeight > 0 && <div aria-hidden style={{ height: afterHeight }} />}
    </>
  );
}
