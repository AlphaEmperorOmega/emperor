import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Search, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ConfigSearchResults } from "@/features/workbench/components/config/config-search-results";
import { DropdownShell } from "@/features/workbench/components/shared/dropdown-shell";
import { useIncrementalVisibleOptions } from "@/features/workbench/components/shared/use-incremental-visible-options";
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
  disabledFieldReasons,
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
  disabledFieldReasons?: Map<string, string>;
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
  const pendingActiveIndexRef = useRef<number | null>(null);
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
  const matchingOptionsKey = useMemo(
    () => matchingOptions.map((option) => option.key).join("\u0000"),
    [matchingOptions],
  );
  const {
    scrollContainerRef,
    visibleOptions,
    hasMore,
    isLoadingMore,
    loadMore,
    handleScroll,
  } = useIncrementalVisibleOptions({
    options: matchingOptions,
    resetKey: matchingOptionsKey,
    initialVisibleCount: RESULT_LIMIT,
    pageSize: RESULT_LIMIT,
  });
  const isPopupOpen = hasFocusWithin && isOpen && trimmedQuery.length > 0;
  const activeOption =
    isPopupOpen && visibleOptions.length > 0 ? visibleOptions[activeIndex] : undefined;

  useEffect(() => {
    setActiveIndex(matchingOptions.length > 0 ? 0 : -1);
  }, [matchingOptions.length, matchingOptionsKey]);

  useEffect(() => {
    const pendingIndex = pendingActiveIndexRef.current;
    if (pendingIndex === null || pendingIndex >= visibleOptions.length) {
      return;
    }
    pendingActiveIndexRef.current = null;
    setActiveIndex(pendingIndex);
  }, [visibleOptions.length]);

  useEffect(() => {
    if (activeIndex < visibleOptions.length) {
      return;
    }
    setActiveIndex(visibleOptions.length > 0 ? visibleOptions.length - 1 : -1);
  }, [activeIndex, visibleOptions.length]);

  useEffect(() => {
    pendingActiveIndexRef.current = null;
  }, [matchingOptionsKey]);

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
        return direction === 1 ? 0 : visibleOptions.length - 1;
      }
      if (direction === 1 && current >= visibleOptions.length - 1 && hasMore) {
        pendingActiveIndexRef.current = visibleOptions.length;
        loadMore();
        return current;
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
          placeholder="Search fields, keys, or sections"
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
        <DropdownShell
          ref={scrollContainerRef}
          id={popupId}
          role="dialog"
          ariaLabel="Matching config fields"
          onScroll={handleScroll}
          className="max-h-[min(34rem,calc(100vh-14rem))] overflow-y-auto p-2"
        >
          <ConfigSearchResults
            popupId={popupId}
            visibleOptions={visibleOptions}
            isLoadingMore={isLoadingMore}
            activeIndex={activeIndex}
            selectedFieldKey={selectedFieldKey}
            overrides={overrides}
            disabledFieldReasons={disabledFieldReasons}
            onSelect={selectOption}
            onFieldChange={onFieldChange}
            onFieldReset={onFieldReset}
          />
        </DropdownShell>
      )}
    </div>
  );
}
