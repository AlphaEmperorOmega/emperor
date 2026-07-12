import {
  type KeyboardEvent,
  type MouseEvent,
  type ReactNode,
  isValidElement,
  useMemo,
} from "react";
import { Check, ChevronDown, Loader2, Search } from "lucide-react";
import { IconButton } from "@/components/ui/icon-button";
import {
  checkboxCheckedClassName,
  checkboxIndicatorClassName,
  checkboxUncheckedClassName,
  dropdownOptionClassName,
  dropdownOptionStateClassName,
  multiSelectTriggerClassName,
  selectTriggerActiveClassName,
} from "@/components/ui/control-styles";
import { DropdownShell } from "@/features/workbench/components/shared/dropdown-shell";
import { HoverTooltip } from "@/features/workbench/components/shared/hover-tooltip";
import { StatChip } from "@/features/workbench/components/shared/stat-chip";
import { useSearchablePopupInteraction } from "@/features/workbench/components/shared/use-searchable-dropdown";
import { cn } from "@/lib/utils";

export type MultiSelectDropdownOptionAction = {
  label: string;
  tooltip: string;
  icon: ReactNode;
  onAction: (value: string) => void;
};

export type MultiSelectDropdownOption = {
  value: string;
  label: string;
  description?: string;
  meta?: ReactNode;
  metaTooltip?: string;
  wrapLabel?: boolean;
  actions?: MultiSelectDropdownOptionAction[];
};

function textFromReactNode(node: ReactNode): string {
  if (node === null || node === undefined || typeof node === "boolean") {
    return "";
  }
  if (typeof node === "string" || typeof node === "number") {
    return String(node);
  }
  if (Array.isArray(node)) {
    return node.map(textFromReactNode).filter(Boolean).join(" ");
  }
  if (isValidElement(node)) {
    return textFromReactNode(
      (node.props as { children?: ReactNode }).children,
    );
  }
  return "";
}

function optionAccessibleName(option: MultiSelectDropdownOption) {
  return [option.label, option.description, textFromReactNode(option.meta)]
    .filter(Boolean)
    .join(" ");
}

function optionSearchText(option: MultiSelectDropdownOption) {
  return [option.label, option.value, option.description ?? ""].join(" ");
}

export function MultiSelectDropdown({
  id,
  label,
  values,
  options,
  onChange,
  disabledValues = [],
  primaryValue,
  onPrimaryChange,
  placeholder = "Select options",
  searchPlaceholder,
  emptyMessage = "No options",
  noResultsMessage = "No matching options",
  disabled: disabledProp = false,
  className,
  triggerClassName,
  initialVisibleCount = 50,
  pageSize = initialVisibleCount,
}: {
  id?: string;
  label: string;
  values: string[];
  options: MultiSelectDropdownOption[];
  onChange: (values: string[]) => void;
  disabledValues?: string[];
  primaryValue?: string;
  onPrimaryChange?: (value: string) => void;
  placeholder?: string;
  searchPlaceholder?: string;
  emptyMessage?: string;
  noResultsMessage?: string;
  disabled?: boolean;
  className?: string;
  triggerClassName?: string;
  initialVisibleCount?: number;
  pageSize?: number;
}) {
  const disabled = disabledProp || options.length === 0;
  const selectedValueSet = useMemo(() => new Set(values), [values]);
  const disabledValueSet = useMemo(
    () => new Set(disabledValues),
    [disabledValues],
  );
  const optionByValue = useMemo(
    () => new Map(options.map((option) => [option.value, option])),
    [options],
  );
  const selectedOptions = values.map(
    (value) => optionByValue.get(value) ?? { value, label: value },
  );
  const interaction = useSearchablePopupInteraction<
    MultiSelectDropdownOption,
    HTMLDivElement
  >({
    mode: "multi-select",
    id,
    idSuffix: "multiselect",
    options,
    optionKey: (option) => option.value,
    optionSearchText,
    disabled,
    isOptionDisabled: (option) => disabledValueSet.has(option.value),
    onActivate: (option) => {
      const nextValues = selectedValueSet.has(option.value)
        ? values.filter((value) => value !== option.value)
        : [...values, option.value];
      onChange(nextValues);
    },
    pagination: { initialVisibleCount, pageSize },
  });
  const {
    ids,
    state: {
      isOpen,
      query,
      options: visibleOptions,
      activeIndex,
      active: activeOption,
      loading: isLoadingMore,
    },
    root,
    trigger,
    search,
    collection,
    close,
  } = interaction;
  const selectedCount = values.length;
  const countSummary = `${selectedCount} / ${options.length} selected`;
  const visibleChips = selectedOptions.slice(0, 2);
  const hiddenChipCount = Math.max(0, selectedOptions.length - visibleChips.length);
  const canMakeActiveOptionPrimary = Boolean(
    activeOption &&
      selectedValueSet.has(activeOption.value) &&
      primaryValue !== activeOption.value &&
      onPrimaryChange,
  );

  function handlePrimaryAction(
    event: MouseEvent<HTMLButtonElement>,
    value: string,
  ) {
    event.preventDefault();
    event.stopPropagation();
    onPrimaryChange?.(value);
  }

  function handleOptionActionMouseDown(event: MouseEvent<HTMLButtonElement>) {
    event.stopPropagation();
  }

  function handleOptionActionClick(
    event: MouseEvent<HTMLButtonElement>,
    option: MultiSelectDropdownOption,
    action: MultiSelectDropdownOptionAction,
  ) {
    event.preventDefault();
    event.stopPropagation();
    action.onAction(option.value);
  }

  function handleOptionActionKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.stopPropagation();
    }
  }

  return (
    <div
      ref={root.ref}
      onBlur={root.onBlur}
      className={cn("relative min-w-0", isOpen ? "z-30" : "z-20", className)}
    >
      <button
        ref={trigger.ref}
        id={ids.control}
        type="button"
        role="combobox"
        aria-label={`${label} ${countSummary}`}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls={isOpen ? ids.popup : undefined}
        aria-activedescendant={isOpen ? ids.active : undefined}
        disabled={disabled}
        onClick={trigger.onClick}
        onKeyDown={trigger.onKeyDown}
        className={cn(
          multiSelectTriggerClassName,
          isOpen && selectTriggerActiveClassName,
          disabledProp && "cursor-not-allowed opacity-60",
          triggerClassName,
        )}
      >
        <span className="grid min-w-0 gap-1">
          <span className="flex min-w-0 items-center gap-2">
            <span className="min-w-0 truncate">
              {selectedCount > 0 ? countSummary : placeholder}
            </span>
          </span>
          {selectedOptions.length > 0 && (
            <span className="flex min-w-0 items-center gap-1.5 overflow-hidden">
              {visibleChips.map((option) => (
                <StatChip
                  key={option.value}
                  tone="violet"
                  size="xs"
                  className="min-w-0 max-w-[9rem] truncate px-2"
                >
                  {option.label}
                </StatChip>
              ))}
              {hiddenChipCount > 0 && (
                <StatChip size="xs" className="shrink-0 px-2">
                  +{hiddenChipCount}
                </StatChip>
              )}
            </span>
          )}
        </span>
        <ChevronDown
          className={cn(
            "h-4 w-4 shrink-0 text-ink-faint transition",
            isOpen && "rotate-180 text-ink",
          )}
          aria-hidden
        />
      </button>

      {isOpen && (
        <DropdownShell
          className="grid max-h-[320px] grid-rows-[auto_minmax(0,1fr)_auto] overflow-hidden"
          searchSlot={
            <label
              htmlFor={ids.search}
              className="grid grid-cols-[auto_minmax(0,1fr)] items-center gap-2 border-b border-line-soft px-3 py-2 transition-colors focus-within:bg-control-active focus-within:ring-2 focus-within:ring-inset focus-within:ring-focus"
            >
              <Search className="h-4 w-4 text-ink-faint" aria-hidden />
              <input
                ref={search.ref}
                id={ids.search}
                name={`${ids.control}-search`}
                type="search"
                value={query}
                onChange={search.onChange}
                onKeyDown={search.onKeyDown}
                placeholder={searchPlaceholder ?? `Search ${label.toLowerCase()}…`}
                aria-label={`Search ${label}`}
                autoComplete="off"
                spellCheck={false}
                className="h-8 min-w-0 bg-transparent font-sans text-sm font-semibold text-ink outline-none placeholder:text-ink-faint"
              />
            </label>
          }
        >
          <div
            ref={collection.ref}
            id={ids.popup}
            role="listbox"
            aria-label={`${label} options`}
            aria-multiselectable="true"
            onScroll={collection.onScroll}
            className="min-h-0 overflow-y-auto"
          >
            {visibleOptions.map((option, index) => {
              const isSelected = selectedValueSet.has(option.value);
              const isDisabled = disabledValueSet.has(option.value);
              const isPrimary = isSelected && primaryValue === option.value;
              const isActive = index === activeIndex;
              return (
                <div
                  {...collection.option(index, option)}
                  key={option.value}
                  id={`${ids.popup}-option-${index}`}
                  role="option"
                  tabIndex={-1}
                  aria-label={optionAccessibleName(option)}
                  aria-describedby={
                    option.metaTooltip
                      ? `${ids.popup}-option-${index}-meta`
                      : undefined
                  }
                  aria-selected={isSelected}
                  aria-disabled={isDisabled || undefined}
                  className={cn(
                    "group relative grid min-w-0 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
                    dropdownOptionClassName,
                    dropdownOptionStateClassName({
                      active: isActive,
                      disabled: isDisabled,
                      selected: isSelected,
                    }),
                  )}
                >
                  <span
                    className={cn(
                      checkboxIndicatorClassName,
                      isSelected
                        ? checkboxCheckedClassName
                        : checkboxUncheckedClassName,
                      isDisabled ? "cursor-default" : "cursor-pointer",
                    )}
                    aria-hidden
                  >
                    <Check
                      className={cn(
                        "h-3 w-3 text-white transition",
                        !isSelected && "opacity-0",
                      )}
                      aria-hidden
                    />
                  </span>
                  <span className="grid min-w-0 gap-0.5">
                    <span
                      className={cn(
                        option.wrapLabel
                          ? "whitespace-normal break-words leading-4 [overflow-wrap:anywhere]"
                          : "truncate",
                      )}
                    >
                      {option.label}
                    </span>
                    {option.description && (
                      <span
                        className={cn(
                          "font-mono text-xs text-ink-dim",
                          option.wrapLabel
                            ? "whitespace-normal break-words [overflow-wrap:anywhere]"
                            : "truncate",
                        )}
                      >
                        {option.description}
                      </span>
                    )}
                  </span>
                  <span className="flex shrink-0 items-center gap-1.5">
                    {isPrimary && (
                      <span className="rounded-control-md border border-violet/35 bg-violet/[0.12] px-2 py-0.5 font-mono type-meta uppercase tracking-label text-violet">
                        primary
                      </span>
                    )}
                    {option.meta && (
                      <span className="font-mono text-xs text-ink-dim">
                        {option.meta}
                      </span>
                    )}
                  </span>
                  {option.metaTooltip && (
                    <span
                      id={`${ids.popup}-option-${index}-meta`}
                      role="tooltip"
                      className="pointer-events-none absolute right-3 top-1/2 z-30 -translate-y-1/2 whitespace-nowrap rounded-control-md border border-line-soft bg-panel px-2 py-1 font-sans type-meta font-bold leading-none text-ink opacity-0 shadow-panel transition-opacity group-focus:opacity-100 group-hover:opacity-100"
                    >
                      {option.metaTooltip}
                    </span>
                  )}
                </div>
              );
            })}
            {visibleOptions.length === 0 && (
              <div
                role="option"
                aria-selected="false"
                aria-disabled="true"
                className="px-3 py-4 text-sm text-ink-faint"
              >
                {options.length === 0 ? emptyMessage : noResultsMessage}
              </div>
            )}
            {isLoadingMore && (
              <div
                role="option"
                aria-selected="false"
                aria-disabled="true"
                aria-label={`Loading more ${label.toLowerCase()}…`}
                className="flex items-center justify-center gap-2 px-3 py-3 text-xs font-semibold text-ink-dim"
              >
                <Loader2 className="h-3.5 w-3.5 animate-spin text-violet" aria-hidden />
                Loading more…
              </div>
            )}
          </div>
          {activeOption &&
            ((activeOption.actions?.length ?? 0) > 0 ||
              canMakeActiveOptionPrimary) && (
            <div
              role="toolbar"
              aria-label={`${activeOption.label} actions`}
              className="flex flex-wrap items-center gap-1.5 border-t border-line-soft bg-black/15 px-3 py-2"
            >
              {(activeOption.actions ?? []).map((action) => {
                const accessibleLabel = action.label
                  .toLocaleLowerCase()
                  .includes(activeOption.label.toLocaleLowerCase())
                  ? action.label
                  : `${action.label} for ${activeOption.label}`;
                return (
                  <HoverTooltip
                    key={action.label}
                    tooltip={action.tooltip}
                    tooltipClassName="right-0"
                  >
                    {(triggerProps) => (
                      <IconButton
                        label={accessibleLabel}
                        icon={action.icon}
                        size="sm"
                        variant="ghost"
                        onMouseDown={handleOptionActionMouseDown}
                        onClick={(event) =>
                          handleOptionActionClick(event, activeOption, action)
                        }
                        onKeyDown={handleOptionActionKeyDown}
                        className="h-touch w-touch rounded-control-sm text-ink-faint hover:text-violet focus-visible:text-violet md:h-control-sm md:w-control-sm"
                        {...triggerProps}
                      />
                    )}
                  </HoverTooltip>
                );
              })}
              {canMakeActiveOptionPrimary && (
                <button
                  type="button"
                  aria-label={`Make ${activeOption.label} Primary`}
                  onClick={(event) =>
                    handlePrimaryAction(event, activeOption.value)
                  }
                  onKeyDown={(event) => {
                    if (event.key === "Escape") {
                      event.preventDefault();
                      close(true);
                    }
                  }}
                  className="inline-flex h-touch max-w-full items-center rounded-control-sm border border-line bg-control px-2.5 font-sans type-meta font-bold text-ink-dim transition hover:border-violet/35 hover:bg-violet/10 hover:text-violet focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:h-control-sm"
                >
                  <span className="truncate">
                    Make {activeOption.label} Primary
                  </span>
                </button>
              )}
            </div>
          )}
        </DropdownShell>
      )}
    </div>
  );
}
