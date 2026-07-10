import {
  type KeyboardEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
} from "react";
import { ChevronDown, Search } from "lucide-react";
import {
  dropdownOptionClassName,
  dropdownOptionStateClassName,
  selectTriggerActiveClassName,
  selectTriggerClassName,
} from "@/components/ui/control-styles";
import { DropdownShell } from "@/features/workbench/components/shared/dropdown-shell";
import {
  useDropdownOptionNavigation,
  useSearchableDropdownCore,
} from "@/features/workbench/components/shared/use-searchable-dropdown";
import { cn } from "@/lib/utils";

export type SelectOnlyDropdownOption = {
  value: string;
  label: string;
  description?: string;
  disabled?: boolean;
};

export function SelectOnlyDropdown({
  id,
  label,
  value,
  options,
  onChange,
  placeholder = "Select",
  searchPlaceholder,
  noResultsMessage = "No matching options",
  disabled = false,
  className,
  triggerClassName,
}: {
  id?: string;
  label: string;
  value: string;
  options: SelectOnlyDropdownOption[];
  onChange: (value: string) => void;
  placeholder?: string;
  searchPlaceholder?: string;
  noResultsMessage?: string;
  disabled?: boolean;
  className?: string;
  triggerClassName?: string;
}) {
  const wasOpenRef = useRef(false);
  const selectedIndex = useMemo(
    () => options.findIndex((option) => option.value === value),
    [options, value],
  );
  const optionSignature = useMemo(
    () =>
      options
        .map((option) =>
          [
            option.value,
            option.label,
            option.description ?? "",
            option.disabled ? "disabled" : "",
          ].join("\u0001"),
        )
        .join("\u0002"),
    [options],
  );
  const selectedOption = selectedIndex >= 0 ? options[selectedIndex] : undefined;
  const isDisabled = disabled || options.length === 0;
  const dropdown = useSearchableDropdownCore({
    id,
    idSuffix: "select",
    options,
    disabled: isDisabled,
  });
  const {
    triggerId,
    listboxId,
    searchId,
    rootRef,
    triggerRef,
    panelRef,
    searchRef,
    isOpen,
    query,
    setQuery,
    filteredOptions,
    filteredOptionsKey: filteredOptionSignature,
    openDropdown: openCoreDropdown,
    closeDropdown,
    handleRootBlur,
  } = dropdown;
  const selectedFilteredIndex = useMemo(
    () => filteredOptions.findIndex((option) => option.value === value),
    [filteredOptions, value],
  );
  const navigation = useDropdownOptionNavigation<HTMLButtonElement>({
    optionCount: filteredOptions.length,
    fallbackIndex: selectedFilteredIndex >= 0 ? selectedFilteredIndex : 0,
    onEscape: () => closeDropdown(true),
  });
  const {
    optionRefs,
    activeIndex,
    setActiveIndex,
    focusOption,
    focusFirstOption,
    focusLastOption,
    moveActiveOption: moveOption,
    handleOptionKeyDown: handleSharedOptionKeyDown,
  } = navigation;
  const activeOptionId =
    isOpen && activeIndex >= 0 ? `${listboxId}-option-${activeIndex}` : undefined;

  const moveActiveOption = useCallback(
    (direction: 1 | -1) => {
      if (filteredOptions.length === 0) {
        return;
      }
      openCoreDropdown();
      moveOption(direction);
    },
    [filteredOptions.length, moveOption, openCoreDropdown],
  );

  const openDropdown = useCallback(() => {
    if (isDisabled) {
      return;
    }
    setActiveIndex(selectedIndex >= 0 ? selectedIndex : 0);
    openCoreDropdown();
  }, [isDisabled, openCoreDropdown, selectedIndex, setActiveIndex]);

  const selectOption = useCallback(
    (option: SelectOnlyDropdownOption) => {
      if (option.disabled) {
        return;
      }
      setActiveIndex(options.findIndex((candidate) => candidate.value === option.value));
      closeDropdown(true);
      if (option.value !== value) {
        onChange(option.value);
      }
    },
    [closeDropdown, onChange, options, setActiveIndex, value],
  );

  useEffect(() => {
    closeDropdown();
    setActiveIndex(selectedIndex >= 0 ? selectedIndex : options.length > 0 ? 0 : -1);
  }, [closeDropdown, optionSignature, options.length, selectedIndex, setActiveIndex, value]);

  useEffect(() => {
    if (!isOpen) {
      wasOpenRef.current = false;
      return;
    }
    if (!wasOpenRef.current) {
      wasOpenRef.current = true;
      return;
    }
    setActiveIndex(
      selectedFilteredIndex >= 0
        ? selectedFilteredIndex
        : filteredOptions.length > 0
          ? 0
          : -1,
    );
  }, [
    filteredOptionSignature,
    filteredOptions.length,
    isOpen,
    selectedFilteredIndex,
    setActiveIndex,
  ]);

  function handleKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
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
    if (event.key === "Home" && isOpen && filteredOptions.length > 0) {
      event.preventDefault();
      focusFirstOption();
      return;
    }
    if (event.key === "End" && isOpen && filteredOptions.length > 0) {
      event.preventDefault();
      focusLastOption();
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (!isOpen) {
        openDropdown();
        return;
      }
      const activeOption = filteredOptions[activeIndex];
      if (activeOption) {
        selectOption(activeOption);
      }
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
      if (activeIndex < 0) {
        setActiveIndex(0);
        focusOption(0);
        return;
      }
      moveActiveOption(1);
      return;
    }
    if (event.key === "ArrowUp" && filteredOptions.length > 0) {
      event.preventDefault();
      if (activeIndex < 0) {
        const lastIndex = filteredOptions.length - 1;
        setActiveIndex(lastIndex);
        focusOption(lastIndex);
        return;
      }
      moveActiveOption(-1);
      return;
    }
    if (event.key === "Home" && filteredOptions.length > 0) {
      event.preventDefault();
      focusFirstOption();
      return;
    }
    if (event.key === "End" && filteredOptions.length > 0) {
      event.preventDefault();
      focusLastOption();
      return;
    }
    if (event.key === "Enter") {
      const activeOption = filteredOptions[activeIndex];
      if (activeOption) {
        event.preventDefault();
        selectOption(activeOption);
      }
    }
  }

  function handleOptionKeyDown(
    event: KeyboardEvent<HTMLButtonElement>,
    option: SelectOnlyDropdownOption,
  ) {
    handleSharedOptionKeyDown(event, () => selectOption(option));
  }

  return (
    <div
      ref={rootRef}
      onBlur={(event) => handleRootBlur(event.relatedTarget)}
      className={cn("relative min-w-0", isOpen ? "z-30" : "z-20", className)}
    >
      <button
        ref={triggerRef}
        id={triggerId}
        type="button"
        role="combobox"
        aria-label={label}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls={listboxId}
        aria-activedescendant={activeOptionId}
        disabled={isDisabled}
        onClick={() => {
          if (isOpen) {
            closeDropdown();
            return;
          }
          openDropdown();
        }}
        onKeyDown={handleKeyDown}
        className={cn(
          selectTriggerClassName,
          isOpen && selectTriggerActiveClassName,
          triggerClassName,
        )}
      >
        <span className="min-w-0 truncate">
          {(selectedOption?.label ?? value) || placeholder}
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
          className="grid max-h-[300px] grid-rows-[auto_minmax(0,1fr)] overflow-hidden"
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
            className="min-h-0 overflow-y-auto"
          >
            {filteredOptions.length === 0 ? (
              <div className="px-3 py-3 text-sm font-semibold text-ink-faint">
                {noResultsMessage}
              </div>
            ) : (
              filteredOptions.map((option, index) => {
                const isActive = index === activeIndex;
                const isSelected = option.value === value;
                const isOptionDisabled = Boolean(option.disabled);
                const descriptionId = option.description
                  ? `${listboxId}-option-${index}-description`
                  : undefined;
                return (
                  <button
                    ref={(node) => {
                      optionRefs.current[index] = node;
                    }}
                    key={option.value}
                    id={`${listboxId}-option-${index}`}
                    type="button"
                    role="option"
                    aria-label={option.label}
                    aria-selected={isSelected}
                    aria-disabled={isOptionDisabled || undefined}
                    aria-describedby={descriptionId}
                    tabIndex={-1}
                    onMouseDown={(event) => event.preventDefault()}
                    onMouseEnter={() => setActiveIndex(index)}
                    onClick={() => selectOption(option)}
                    onKeyDown={(event) => handleOptionKeyDown(event, option)}
                    className={cn(
                      "block",
                      dropdownOptionClassName,
                      dropdownOptionStateClassName({
                        active: isActive,
                        disabled: isOptionDisabled,
                        selected: isSelected,
                      }),
                    )}
                  >
                    <span className="grid min-w-0 gap-0.5">
                      <span className="block min-w-0 truncate">{option.label}</span>
                      {option.description && (
                        <span
                          id={descriptionId}
                          className="block min-w-0 truncate font-mono text-xs text-ink-dim"
                        >
                          {option.description}
                        </span>
                      )}
                    </span>
                  </button>
                );
              })
            )}
          </div>
        </DropdownShell>
      )}
    </div>
  );
}
