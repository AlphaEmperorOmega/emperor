import {
  type KeyboardEvent,
  type MouseEvent,
  type ReactNode,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { Check, ChevronDown, Search } from "lucide-react";
import { cn } from "@/lib/utils";

export type MultiSelectDropdownOption = {
  value: string;
  label: string;
  description?: string;
  meta?: ReactNode;
};

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
  className?: string;
}) {
  const generatedId = useId();
  const triggerId = id ?? `${generatedId}-multiselect`;
  const listboxId = `${triggerId}-options`;
  const searchId = `${triggerId}-search`;
  const rootRef = useRef<HTMLDivElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const searchRef = useRef<HTMLInputElement | null>(null);
  const optionRefs = useRef<Array<HTMLDivElement | null>>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [activeIndex, setActiveIndex] = useState(-1);

  const disabled = options.length === 0;
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

  useEffect(() => {
    if (!isOpen) {
      return;
    }

    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target as Node | null;
      if (target && rootRef.current?.contains(target)) {
        return;
      }
      closeDropdown();
    };

    document.addEventListener("pointerdown", handlePointerDown);
    return () => document.removeEventListener("pointerdown", handlePointerDown);
  }, [closeDropdown, isOpen]);

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
          "grid min-h-[46px] w-full min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-2 rounded-[10px] border border-line bg-[linear-gradient(155deg,#161622,#0e0e18)] px-3 py-2 text-left font-sans text-[13.5px] font-semibold text-ink outline-none transition hover:border-white/15 focus-visible:border-violet/60 focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50",
          isOpen &&
            "border-violet/45 bg-[linear-gradient(135deg,rgba(146,113,255,0.1),rgba(111,168,255,0.05))]",
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
                <span
                  key={option.value}
                  className="min-w-0 max-w-[9rem] truncate rounded-[7px] border border-violet/30 bg-violet/10 px-2 py-0.5 font-mono text-[11px] text-violet"
                >
                  {option.label}
                </span>
              ))}
              {hiddenChipCount > 0 && (
                <span className="shrink-0 rounded-[7px] border border-line bg-white/[0.04] px-2 py-0.5 font-mono text-[11px] text-ink-dim">
                  +{hiddenChipCount}
                </span>
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
        <div className="absolute left-0 right-0 top-full mt-2 grid max-h-[320px] grid-rows-[auto_minmax(0,1fr)] overflow-hidden rounded-[12px] border border-line bg-panel/95 shadow-[0_22px_50px_-30px_rgba(0,0,0,0.98)] backdrop-blur">
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
              const canMakePrimary =
                isSelected && !isPrimary && Boolean(onPrimaryChange);
              const isActive = index === activeIndex;
              return (
                <div
                  key={option.value}
                  ref={(node) => {
                    optionRefs.current[index] = node;
                  }}
                  id={`${listboxId}-option-${index}`}
                  role="option"
                  tabIndex={-1}
                  aria-selected={isSelected}
                  aria-disabled={isDisabled || undefined}
                  onMouseDown={(event) => event.preventDefault()}
                  onMouseEnter={() => setActiveIndex(index)}
                  onClick={() => toggleOption(option)}
                  onKeyDown={(event) => handleOptionKeyDown(event, option)}
                  className={cn(
                    "grid w-full min-w-0 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 border-b border-line-soft px-3 py-2.5 text-left text-sm font-semibold transition last:border-b-0 focus:outline-none",
                    isSelected
                      ? "bg-violet/14 text-violet"
                      : isActive
                        ? "bg-white/[0.045] text-white"
                        : "bg-transparent text-white hover:bg-white/[0.045]",
                    isDisabled && "cursor-default opacity-75",
                    )}
                  >
                  <span
                    className={cn(
                      "grid h-[19px] w-[19px] place-items-center rounded-md border-[1.6px] transition",
                      isSelected
                        ? "border-transparent bg-grad shadow-[0_3px_10px_-3px_rgba(124,109,255,0.8)]"
                        : "border-white/20 bg-transparent",
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
                    {canMakePrimary && (
                      <button
                        type="button"
                        aria-label={`Make ${option.label} primary`}
                        onMouseDown={(event) => {
                          event.preventDefault();
                          event.stopPropagation();
                        }}
                        onClick={(event) =>
                          handlePrimaryAction(event, option.value)
                        }
                        onKeyDown={(event) => {
                          if (event.key === "Enter" || event.key === " ") {
                            event.stopPropagation();
                          }
                        }}
                        className="rounded-[7px] border border-line bg-white/[0.035] px-2 py-1 font-sans text-[11.5px] font-bold text-ink-dim transition hover:border-violet/35 hover:bg-violet/10 hover:text-violet focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                      >
                        Make primary
                      </button>
                    )}
                    {option.meta && (
                      <span className="font-mono text-xs text-ink-dim">
                        {option.meta}
                      </span>
                    )}
                  </span>
                </div>
              );
            })}
            {filteredOptions.length === 0 && (
              <div className="px-3 py-4 text-sm text-ink-faint">
                {options.length === 0 ? emptyMessage : noResultsMessage}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
