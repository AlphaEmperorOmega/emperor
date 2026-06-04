import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Search, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ConfigSearchResults } from "@/components/features/viewer/config/config-search-results";
import {
  type ConfigSearchOption,
  type OverrideValues,
  configSearchOptionMatchesQuery,
} from "@/lib/config";

const RESULT_LIMIT = 8;

export function ConfigFieldSearch({
  options,
  query,
  selectedFieldKey,
  overrides,
  onQueryChange,
  onClear,
  onSelect,
  onFieldChange,
  onFieldReset,
}: {
  options: ConfigSearchOption[];
  query: string;
  selectedFieldKey: string | null;
  overrides: OverrideValues;
  onQueryChange: (query: string) => void;
  onClear: () => void;
  onSelect: (option: ConfigSearchOption) => void;
  onFieldChange: (key: string, value: string) => void;
  onFieldReset: (key: string) => void;
}) {
  const generatedId = useId();
  const inputId = `${generatedId}-config-field-search`;
  const popupId = `${inputId}-results`;
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [hasFocusWithin, setHasFocusWithin] = useState(false);
  const [isOpen, setIsOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const trimmedQuery = query.trim();
  const matchingOptions = useMemo(
    () =>
      trimmedQuery
        ? options.filter((option) => configSearchOptionMatchesQuery(option, trimmedQuery))
        : [],
    [options, trimmedQuery],
  );
  const visibleOptions = matchingOptions.slice(0, RESULT_LIMIT);
  const hiddenResultCount = Math.max(matchingOptions.length - visibleOptions.length, 0);
  const isPopupOpen = hasFocusWithin && isOpen && trimmedQuery.length > 0;
  const activeOption =
    isPopupOpen && visibleOptions.length > 0 ? visibleOptions[activeIndex] : undefined;

  useEffect(() => {
    setActiveIndex(visibleOptions.length > 0 ? 0 : -1);
  }, [trimmedQuery, visibleOptions.length]);

  function updateQuery(value: string) {
    onQueryChange(value);
    setIsOpen(value.trim().length > 0);
  }

  function selectOption(option: ConfigSearchOption) {
    onSelect(option);
    setIsOpen(false);
  }

  function clearSearch() {
    onClear();
    setIsOpen(false);
    inputRef.current?.focus();
  }

  function moveActiveOption(direction: 1 | -1) {
    if (visibleOptions.length === 0) {
      return;
    }

    setIsOpen(true);
    setActiveIndex((current) => {
      if (current < 0) {
        return 0;
      }
      return (current + direction + visibleOptions.length) % visibleOptions.length;
    });
  }

  return (
    <div
      className="relative z-20"
      onFocus={() => {
        setHasFocusWithin(true);
        setIsOpen(trimmedQuery.length > 0);
      }}
      onBlur={(event) => {
        const nextFocusedElement = event.relatedTarget;
        if (
          nextFocusedElement &&
          event.currentTarget.contains(nextFocusedElement as Node)
        ) {
          return;
        }

        setHasFocusWithin(false);
        setIsOpen(false);
      }}
    >
      <div className="relative">
        <Search
          className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint"
          aria-hidden
        />
        <Input
          ref={inputRef}
          id={inputId}
          role="combobox"
          aria-label="Search config fields"
          aria-haspopup="dialog"
          aria-expanded={isPopupOpen}
          aria-controls={popupId}
          autoComplete="off"
          placeholder="Search fields, keys, flags, or sections"
          value={query}
          onChange={(event) => updateQuery(event.target.value)}
          onKeyDown={(event) => {
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
            if (event.key === "Enter" && isPopupOpen && activeOption) {
              event.preventDefault();
              selectOption(activeOption);
              return;
            }
            if (event.key === "Escape" && isPopupOpen) {
              event.preventDefault();
              setIsOpen(false);
            }
          }}
          className="h-10 rounded-[12px] border-line bg-black/25 pl-9 pr-10 text-sm shadow-[0_14px_34px_-32px_rgba(0,0,0,0.95)] focus-visible:ring-focus"
        />
        {query.length > 0 && (
          <Button
            type="button"
            variant="ghost"
            aria-label="Clear config search"
            onMouseDown={(event) => event.preventDefault()}
            onClick={clearSearch}
            className="absolute right-1.5 top-1/2 h-7 w-7 -translate-y-1/2 rounded-[8px] p-0 text-ink-faint hover:text-ink"
          >
            <X className="h-3.5 w-3.5" aria-hidden />
          </Button>
        )}
      </div>

      {isPopupOpen && (
        <div
          id={popupId}
          role="dialog"
          aria-label="Matching config fields"
          className="absolute left-0 right-0 top-full mt-2 max-h-[min(34rem,calc(100vh-14rem))] overflow-y-auto rounded-[12px] border border-line bg-panel/95 p-2 shadow-[0_22px_50px_-30px_rgba(0,0,0,0.98)] backdrop-blur"
        >
          <ConfigSearchResults
            popupId={popupId}
            visibleOptions={visibleOptions}
            hiddenResultCount={hiddenResultCount}
            activeIndex={activeIndex}
            selectedFieldKey={selectedFieldKey}
            overrides={overrides}
            onSelect={selectOption}
            onFieldChange={onFieldChange}
            onFieldReset={onFieldReset}
          />
        </div>
      )}
    </div>
  );
}
