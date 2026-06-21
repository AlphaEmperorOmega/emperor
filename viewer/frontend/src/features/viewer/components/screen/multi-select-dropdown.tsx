import {
  type KeyboardEvent,
  type MouseEvent,
  type ReactNode,
  isValidElement,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { Check, ChevronDown, Search } from "lucide-react";
import { flushSync } from "react-dom";
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
import { DropdownShell } from "@/features/viewer/components/shared/dropdown-shell";
import { HoverTooltip } from "@/features/viewer/components/shared/hover-tooltip";
import { StatChip } from "@/features/viewer/components/shared/stat-chip";
import { usePopupDismissal } from "@/features/viewer/components/shared/use-popup-dismissal";
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
}) {
  const generatedId = useId();
  const triggerId = id ?? `${generatedId}-multiselect`;
  const listboxId = `${triggerId}-options`;
  const searchId = `${triggerId}-search`;
  const rootRef = useRef<HTMLDivElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const searchRef = useRef<HTMLInputElement | null>(null);
  const optionRefs = useRef<Array<HTMLDivElement | null>>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(-1);

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
  const normalizedQuery = query.trim().toLowerCase();
  const filteredOptions = useMemo(() => {
    if (!normalizedQuery) {
      return options;
    }
    return options.filter((option) =>
      [option.label, option.value, option.description ?? ""]
        .join(" ")
        .toLowerCase()
        .includes(normalizedQuery),
    );
  }, [normalizedQuery, options]);
  const selectedCount = values.length;
  const countSummary = `${selectedCount} / ${options.length} selected`;
  const visibleChips = selectedOptions.slice(0, 2);
  const hiddenChipCount = Math.max(0, selectedOptions.length - visibleChips.length);
  const activeOptionId =
    isOpen && activeIndex >= 0 ? `${listboxId}-option-${activeIndex}` : undefined;
  const activeOption = activeIndex >= 0 ? filteredOptions[activeIndex] : undefined;
  const canMakeActiveOptionPrimary = Boolean(
    activeOption &&
      selectedValueSet.has(activeOption.value) &&
      primaryValue !== activeOption.value &&
      onPrimaryChange,
  );

  const focusOption = useCallback((index: number) => {
    window.requestAnimationFrame(() => {
      optionRefs.current[index]?.focus();
    });
  }, []);

  const openDropdown = useCallback(() => {
    if (disabled) {
      return;
    }
    setQuery("");
    setActiveIndex(0);
    setIsOpen(true);
  }, [disabled]);

  const closeDropdown = useCallback((restoreFocus = false) => {
    setIsOpen(false);
    setActiveIndex(-1);
    setQuery("");
    if (restoreFocus) {
      triggerRef.current?.focus();
    }
  }, []);

  const moveActiveOption = useCallback(
    (direction: 1 | -1) => {
      if (filteredOptions.length === 0) {
        return;
      }
      setActiveIndex((current) => {
        const startIndex = current >= 0 ? current : 0;
        const nextIndex =
          (startIndex + direction + filteredOptions.length) % filteredOptions.length;
        focusOption(nextIndex);
        return nextIndex;
      });
    },
    [filteredOptions.length, focusOption],
  );

  const toggleOption = useCallback(
    (option: MultiSelectDropdownOption) => {
      if (disabledValueSet.has(option.value)) {
        return;
      }
      const nextValues = selectedValueSet.has(option.value)
        ? values.filter((value) => value !== option.value)
        : [...values, option.value];
      onChange(nextValues);
    },
    [disabledValueSet, onChange, selectedValueSet, values],
  );

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    searchRef.current?.focus();
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    setActiveIndex(filteredOptions.length > 0 ? 0 : -1);
  }, [filteredOptions.length, isOpen]);

  usePopupDismissal({
    open: isOpen,
    onClose: closeDropdown,
    triggerRef,
    panelRef,
  });

  useEffect(() => {
    if (disabled) {
      closeDropdown();
    }
  }, [closeDropdown, disabled]);

  function handleTriggerKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      openDropdown();
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (isOpen) {
        closeDropdown();
        return;
      }
      openDropdown();
      return;
    }
    if (event.key === "Escape" && isOpen) {
      event.preventDefault();
      closeDropdown(true);
    }
  }

  function handleSearchKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Escape") {
      event.preventDefault();
      closeDropdown(true);
      return;
    }
    if (event.key === "ArrowDown" && filteredOptions.length > 0) {
      event.preventDefault();
      const nextIndex = activeIndex >= 0 ? activeIndex : 0;
      setActiveIndex(nextIndex);
      focusOption(nextIndex);
    }
  }

  function handleOptionKeyDown(
    event: KeyboardEvent<HTMLDivElement>,
    option: MultiSelectDropdownOption,
  ) {
    if (event.key === "Escape") {
      event.preventDefault();
      closeDropdown(true);
      return;
    }
    if (event.key === "ArrowDown") {
      event.preventDefault();
      moveActiveOption(1);
      return;
    }
    if (event.key === "ArrowUp") {
      event.preventDefault();
      moveActiveOption(-1);
      return;
    }
    if (event.key === "Home" && filteredOptions.length > 0) {
      event.preventDefault();
      setActiveIndex(0);
      focusOption(0);
      return;
    }
    if (event.key === "End" && filteredOptions.length > 0) {
      event.preventDefault();
      const lastIndex = filteredOptions.length - 1;
      setActiveIndex(lastIndex);
      focusOption(lastIndex);
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      toggleOption(option);
    }
  }

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
    flushSync(() => closeDropdown());
    action.onAction(option.value);
  }

  function handleOptionActionKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.stopPropagation();
    }
  }

  return (
    <div
      ref={rootRef}
      onBlur={(event) => {
        const nextTarget = event.relatedTarget as Node | null;
        if (nextTarget && rootRef.current?.contains(nextTarget)) {
          return;
        }
        closeDropdown();
      }}
      className={cn("relative min-w-0", isOpen ? "z-30" : "z-20", className)}
    >
      <button
        ref={triggerRef}
        id={triggerId}
        type="button"
        role="combobox"
        aria-label={`${label} ${countSummary}`}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls={listboxId}
        aria-activedescendant={activeOptionId}
        disabled={disabled}
        onClick={() => {
          if (isOpen) {
            closeDropdown();
            return;
          }
          openDropdown();
        }}
        onKeyDown={handleTriggerKeyDown}
        className={cn(
          multiSelectTriggerClassName,
          isOpen && selectTriggerActiveClassName,
          disabledProp && "cursor-not-allowed opacity-60",
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
          ref={panelRef}
          className="grid max-h-[320px] grid-rows-[auto_minmax(0,1fr)_auto] overflow-hidden"
          searchSlot={
            <label
              htmlFor={searchId}
              className="grid grid-cols-[auto_minmax(0,1fr)] items-center gap-2 border-b border-line-soft px-3 py-2"
            >
              <Search className="h-4 w-4 text-ink-faint" aria-hidden />
              <input
                ref={searchRef}
                id={searchId}
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={handleSearchKeyDown}
                placeholder={searchPlaceholder ?? `Search ${label.toLowerCase()}`}
                aria-label={`Search ${label}`}
                autoComplete="off"
                className="h-8 min-w-0 bg-transparent font-sans text-sm font-semibold text-ink outline-none placeholder:text-ink-faint"
              />
            </label>
          }
        >
          <div
            id={listboxId}
            role="listbox"
            aria-label={`${label} options`}
            aria-multiselectable="true"
            className="min-h-0 overflow-y-auto"
          >
            {filteredOptions.map((option, index) => {
              const isSelected = selectedValueSet.has(option.value);
              const isDisabled = disabledValueSet.has(option.value);
              const isPrimary = isSelected && primaryValue === option.value;
              const isActive = index === activeIndex;
              const actions = option.actions ?? [];
              return (
                <div
                  key={option.value}
                  role="presentation"
                  className={cn(
                    "grid grid-cols-[minmax(0,1fr)_auto] items-center gap-2",
                    dropdownOptionClassName,
                    dropdownOptionStateClassName({
                      active: isActive,
                      disabled: isDisabled,
                      selected: isSelected,
                    }),
                  )}
                >
                  <div
                    ref={(node) => {
                      optionRefs.current[index] = node;
                    }}
                    id={`${listboxId}-option-${index}`}
                    role="option"
                    tabIndex={-1}
                    aria-label={optionAccessibleName(option)}
                    aria-selected={isSelected}
                    aria-disabled={isDisabled || undefined}
                    onMouseDown={(event) => event.preventDefault()}
                    onMouseEnter={() => setActiveIndex(index)}
                    onClick={() => toggleOption(option)}
                    onKeyDown={(event) => handleOptionKeyDown(event, option)}
                    className={cn(
                      "grid min-w-0 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 rounded-[7px] focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
                      isDisabled ? "cursor-default" : "cursor-pointer",
                    )}
                  >
                    <span
                      className={cn(
                        checkboxIndicatorClassName,
                        isSelected
                          ? checkboxCheckedClassName
                          : checkboxUncheckedClassName,
                      )}
                      aria-hidden
                    >
                      <Check
                        className={cn(
                          "h-3 w-3 text-white transition",
                          !isSelected && "opacity-0",
                        )}
                      />
                    </span>
                    <span className="grid min-w-0 gap-0.5">
                      <span className="truncate">{option.label}</span>
                      {option.description && (
                        <span className="truncate font-mono text-xs text-ink-dim">
                          {option.description}
                        </span>
                      )}
                    </span>
                    <span className="flex shrink-0 items-center gap-1.5">
                      {isPrimary && (
                        <span className="rounded-[7px] border border-violet/35 bg-violet/12 px-2 py-0.5 font-mono text-[11px] uppercase tracking-[0.06em] text-violet">
                          primary
                        </span>
                      )}
                      {option.meta && (
                        <span className="font-mono text-xs text-ink-dim">
                          {option.meta}
                        </span>
                      )}
                    </span>
                  </div>
                  {actions.length > 0 && (
                    <span className="flex shrink-0 items-center gap-1.5">
                      {actions.map((action) => (
                        <HoverTooltip
                          key={action.label}
                          tooltip={action.tooltip}
                          tooltipClassName="right-0"
                        >
                          {(triggerProps) => (
                            <IconButton
                              label={action.label}
                              icon={action.icon}
                              size="sm"
                              variant="ghost"
                              onMouseDown={handleOptionActionMouseDown}
                              onClick={(event) =>
                                handleOptionActionClick(event, option, action)
                              }
                              onKeyDown={handleOptionActionKeyDown}
                              className="h-7 w-7 rounded-[7px] text-ink-faint hover:text-violet focus-visible:text-violet"
                              {...triggerProps}
                            />
                          )}
                        </HoverTooltip>
                      ))}
                    </span>
                  )}
                </div>
              );
            })}
            {filteredOptions.length === 0 && (
              <div className="px-3 py-4 text-sm text-ink-faint">
                {options.length === 0 ? emptyMessage : noResultsMessage}
              </div>
            )}
          </div>
          {activeOption && canMakeActiveOptionPrimary && (
            <div className="border-t border-line-soft bg-black/15 px-3 py-2">
              <button
                type="button"
                aria-label={`Make ${activeOption.label} primary`}
                onMouseDown={(event) => event.preventDefault()}
                onClick={(event) => handlePrimaryAction(event, activeOption.value)}
                onKeyDown={(event) => {
                  if (event.key === "Escape") {
                    event.preventDefault();
                    closeDropdown(true);
                  }
                }}
                className="inline-flex h-8 max-w-full items-center rounded-control-sm border border-line bg-control px-2.5 font-sans text-[11.5px] font-bold text-ink-dim transition hover:border-violet/35 hover:bg-violet/10 hover:text-violet focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
              >
                <span className="truncate">Make {activeOption.label} primary</span>
              </button>
            </div>
          )}
        </DropdownShell>
      )}
    </div>
  );
}
