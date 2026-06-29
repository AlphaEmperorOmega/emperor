import { Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  ConfigFieldOverrideIcon,
  ConfigFieldValueEditor,
} from "@/features/viewer/components/config/config-field-control";
import { surfacePanelClassName } from "@/features/viewer/components/shared/surface-panel";
import {
  type ConfigSearchOption,
  type OverrideValues,
  fieldValue,
  hasOverride,
} from "@/lib/config";
import { cn } from "@/lib/utils";

function ConfigSearchResultItem({
  option,
  popupId,
  overrides,
  isActive,
  isSelected,
  disabledReason,
  onSelect,
  onFieldChange,
  onFieldReset,
}: {
  option: ConfigSearchOption;
  popupId: string;
  overrides: OverrideValues;
  isActive: boolean;
  isSelected: boolean;
  disabledReason?: string;
  onSelect: (option: ConfigSearchOption) => void;
  onFieldChange: (key: string, value: string) => void;
  onFieldReset: (key: string) => void;
}) {
  const titleId = `${popupId}-field-${option.key}-title`;
  const editorId = `${popupId}-field-${option.key}-value`;
  const isOverridden = hasOverride(overrides, option.field.key);
  const isPresetOwned = Boolean(option.field.locked);
  const showCurrentValue = isOverridden || isPresetOwned;
  const currentValueLabel = fieldValue(option.field, overrides);

  return (
    <div
      role="group"
      aria-label="Config search result"
      data-config-search-result={option.key}
      className={cn(
        surfacePanelClassName,
        "min-w-0 gap-2 px-3 py-2.5 transition",
        isPresetOwned ? "border-amber/35 bg-amber/[0.055]" : "border-line-soft bg-black/15",
        isSelected && "border-violet/45 bg-violet/10",
        isActive ? "ring-1 ring-violet/45" : "hover:border-line hover:bg-white/[0.035]",
      )}
    >
      <span className="flex min-w-0 items-start justify-between gap-2">
        <button
          id={titleId}
          type="button"
          onClick={() => onSelect(option)}
          data-config-field-label=""
          className={cn(
            "flex min-w-0 items-center gap-1.5 rounded-[6px] text-left text-sm font-semibold underline-offset-4 transition hover:text-violet hover:underline focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
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
        field={option.field}
        overrides={overrides}
        onChange={onFieldChange}
        onReset={onFieldReset}
        controlId={editorId}
        controlLabel="Current value"
        resetLabel="Reset search result override"
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
  visibleOptions,
  isLoadingMore,
  activeIndex,
  selectedFieldKey,
  overrides,
  disabledFieldReasons,
  onSelect,
  onFieldChange,
  onFieldReset,
}: {
  popupId: string;
  visibleOptions: ConfigSearchOption[];
  isLoadingMore: boolean;
  activeIndex: number;
  selectedFieldKey: string | null;
  overrides: OverrideValues;
  disabledFieldReasons?: Map<string, string>;
  onSelect: (option: ConfigSearchOption) => void;
  onFieldChange: (key: string, value: string) => void;
  onFieldReset: (key: string) => void;
}) {
  if (visibleOptions.length === 0) {
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
      {visibleOptions.map((option, index) => (
        <ConfigSearchResultItem
          key={option.key}
          option={option}
          popupId={popupId}
          overrides={overrides}
          isActive={index === activeIndex}
          isSelected={option.key === selectedFieldKey}
          disabledReason={disabledFieldReasons?.get(option.key)}
          onSelect={onSelect}
          onFieldChange={onFieldChange}
          onFieldReset={onFieldReset}
        />
      ))}
      {isLoadingMore && (
        <div
          role="status"
          aria-label="Loading more config matches"
          className="flex items-center justify-center gap-2 px-3 py-3 text-xs font-semibold text-ink-dim"
        >
          <Loader2 className="h-3.5 w-3.5 animate-spin text-violet" aria-hidden />
          Loading more matches
        </div>
      )}
    </>
  );
}
