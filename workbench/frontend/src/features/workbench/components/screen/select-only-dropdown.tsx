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
  const interaction = useSearchablePopupInteraction<
    SelectOnlyDropdownOption,
    HTMLButtonElement
  >({
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
    ids,
    state: { isOpen, query, options: visibleOptions, activeIndex },
    root,
    trigger,
    search,
    collection,
  } = interaction;

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
        aria-label={label}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls={ids.popup}
        aria-activedescendant={ids.active}
        disabled={isDisabled}
        onClick={trigger.onClick}
        onKeyDown={trigger.onKeyDown}
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
            className="min-h-0 overflow-y-auto"
          >
            {visibleOptions.length === 0 ? (
              <div className="px-3 py-3 text-sm font-semibold text-ink-faint">
                {noResultsMessage}
              </div>
            ) : (
              visibleOptions.map((option, index) => {
                const isActive = index === activeIndex;
                const isSelected = option.value === value;
                const isOptionDisabled = Boolean(option.disabled);
                const descriptionId = option.description
                  ? `${ids.popup}-option-${index}-description`
                  : undefined;
                return (
                  <button
                    {...collection.option(index, option)}
                    key={option.value}
                    id={`${ids.popup}-option-${index}`}
                    type="button"
                    role="option"
                    aria-label={option.label}
                    aria-selected={isSelected}
                    aria-disabled={isOptionDisabled || undefined}
                    aria-describedby={descriptionId}
                    tabIndex={-1}
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
