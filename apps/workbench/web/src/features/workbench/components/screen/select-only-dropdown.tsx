import { ChevronDown, Search } from "lucide-react";
import {
  dropdownOptionClassName,
  dropdownOptionStateClassName,
  selectTriggerActiveClassName,
  selectTriggerClassName,
} from "@/components/ui/control-styles";
import { DropdownShell } from "@/features/workbench/components/shared/dropdown-shell";
import { useSearchablePopupInteraction } from "@/features/workbench/components/shared/use-searchable-dropdown";
import { cn } from "@/lib/utils";

export type SelectOnlyDropdownOption = {
  value: string;
  label: string;
  description?: string;
  disabled?: boolean;
};

function optionSearchText(option: SelectOnlyDropdownOption) {
  return [option.label, option.value, option.description ?? ""].join(" ");
}

function optionRevision(option: SelectOnlyDropdownOption) {
  return [
    option.value,
    option.label,
    option.description ?? "",
    option.disabled ? "disabled" : "",
  ].join("\u0001");
}

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
  const selectedOption = options.find((option) => option.value === value);
  const isDisabled = disabled || options.length === 0;
  const interaction = useSearchablePopupInteraction<SelectOnlyDropdownOption>({
    mode: "single-select",
    id,
    idSuffix: "select",
    options,
    optionKey: (option) => option.value,
    optionSearchText,
    optionRevision,
    selectedKey: value,
    disabled: isDisabled,
    isOptionDisabled: (option) => Boolean(option.disabled),
    onActivate: (option) => {
      if (option.value !== value) {
        onChange(option.value);
      }
    },
  });
  const {
    controlId,
    searchId,
    popupId,
    activeOptionId,
    isOpen,
    query,
    activeIndex,
    virtualOptions,
    beforeHeight,
    afterHeight,
    rootRef,
    triggerRef,
    searchRef,
    collectionRef,
    measureOption,
    handleRootBlur,
    handleTriggerClick,
    handleTriggerKeyDown,
    handleSearchChange,
    handleSearchKeyDown,
    handleCollectionScroll,
    handleOptionMouseDown,
    handleOptionMouseEnter,
    handleOptionClick,
  } = interaction;

  return (
    <div
      ref={rootRef}
      onBlur={handleRootBlur}
      className={cn("relative min-w-0", isOpen ? "z-30" : "z-20", className)}
    >
      <button
        ref={triggerRef}
        id={controlId}
        type="button"
        aria-label={label}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls={isOpen ? popupId : undefined}
        disabled={isDisabled}
        onClick={handleTriggerClick}
        onKeyDown={handleTriggerKeyDown}
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
          className="grid max-h-[300px] grid-rows-[auto_minmax(0,1fr)] overflow-hidden"
          searchSlot={
            <label
              htmlFor={searchId}
              className="grid grid-cols-[auto_minmax(0,1fr)] items-center gap-2 border-b border-line-soft px-3 py-2 transition-colors focus-within:bg-control-active focus-within:ring-2 focus-within:ring-inset focus-within:ring-focus"
            >
              <Search className="h-4 w-4 text-ink-faint" aria-hidden />
              <input
                ref={searchRef}
                id={searchId}
                name={`${controlId}-search`}
                type="search"
                role="combobox"
                value={query}
                onChange={handleSearchChange}
                onKeyDown={handleSearchKeyDown}
                placeholder={searchPlaceholder ?? `Search ${label.toLowerCase()}…`}
                aria-label={`Search ${label}`}
                aria-autocomplete="list"
                aria-haspopup="listbox"
                aria-expanded={isOpen}
                aria-controls={popupId}
                aria-activedescendant={activeOptionId}
                autoComplete="off"
                spellCheck={false}
                className="h-8 min-w-0 bg-transparent font-sans text-sm font-semibold text-ink outline-none placeholder:text-ink-faint"
              />
            </label>
          }
        >
          <div
            ref={collectionRef}
            id={popupId}
            role="listbox"
            aria-label={`${label} options`}
            onScroll={handleCollectionScroll}
            className="min-h-0 overflow-y-auto"
          >
            {virtualOptions.length === 0 ? (
              <div className="px-3 py-3 text-sm font-semibold text-ink-faint">
                {noResultsMessage}
              </div>
            ) : (
              <>
                {beforeHeight > 0 && (
                  <div aria-hidden style={{ height: beforeHeight }} />
                )}
                {virtualOptions.map(({ option, index, key }) => {
                const isActive = index === activeIndex;
                const isSelected = option.value === value;
                const isOptionDisabled = Boolean(option.disabled);
                const descriptionId = option.description
                  ? `${popupId}-option-${encodeURIComponent(key)}-description`
                  : undefined;
                return (
                  <div
                    ref={measureOption}
                    key={key}
                    id={`${popupId}-option-${encodeURIComponent(key)}`}
                    data-virtual-option-key={key}
                    role="option"
                    aria-label={option.label}
                    aria-selected={isSelected}
                    aria-disabled={isOptionDisabled || undefined}
                    aria-describedby={descriptionId}
                    tabIndex={-1}
                    onMouseDown={handleOptionMouseDown}
                    onMouseEnter={() => handleOptionMouseEnter(index)}
                    onClick={() => handleOptionClick(index)}
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
                  </div>
                );
                })}
                {afterHeight > 0 && (
                  <div aria-hidden style={{ height: afterHeight }} />
                )}
              </>
            )}
          </div>
        </DropdownShell>
      )}
    </div>
  );
}
