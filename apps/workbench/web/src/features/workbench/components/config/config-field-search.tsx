import { Search, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ConfigSearchResults } from "@/features/workbench/components/config/config-search-results";
import { DropdownShell } from "@/features/workbench/components/shared/dropdown-shell";
import { useSearchableDialogInteraction } from "@/features/workbench/components/shared/use-searchable-dialog";
import { type RuntimeDefaultsSearchOptionPresentation } from "@/features/workbench/state/runtime-defaults/runtime-defaults-presentation";

const RESULT_LIMIT = 8;

export function ConfigFieldSearch({
  options,
  query,
  selectedFieldKey,
  matchesQuery,
  onQueryChange,
  onClear,
  onSelect,
  onFieldChange,
  onFieldReset,
}: {
  options: RuntimeDefaultsSearchOptionPresentation[];
  query: string;
  selectedFieldKey: string | null;
  matchesQuery: (
    option: RuntimeDefaultsSearchOptionPresentation,
    query: string,
  ) => boolean;
  onQueryChange: (query: string) => void;
  onClear: () => void;
  onSelect: (option: RuntimeDefaultsSearchOptionPresentation) => void;
  onFieldChange: (key: string, value: string) => void;
  onFieldReset: (key: string) => void;
}) {
  const interaction = useSearchableDialogInteraction<RuntimeDefaultsSearchOptionPresentation>({
    idSuffix: "config-field-search",
    options,
    optionKey: (option) => option.key,
    matchesQuery,
    query,
    onQueryChange,
    onClear,
    onActivate: onSelect,
    initialVisibleCount: RESULT_LIMIT,
    pageSize: RESULT_LIMIT,
  });
  const {
    controlId,
    popupId,
    activeOptionId,
    isOpen,
    activeIndex,
    matchingCount,
    virtualOptions,
    beforeHeight,
    afterHeight,
    rootRef,
    searchRef,
    collectionRef,
    measureOption,
    handleRootFocus,
    handleRootBlur,
    handleSearchFocus,
    handleSearchChange,
    handleSearchClick,
    handleSearchKeyDown,
    handlePopupKeyDown,
    handleCollectionScroll,
    handleOptionMouseEnter,
    clearSearch,
    activate,
  } = interaction;

  return (
    <div
      ref={rootRef}
      className="relative z-20"
      onFocus={handleRootFocus}
      onBlur={handleRootBlur}
    >
      <div className="relative">
        <Search
          className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint"
          aria-hidden
        />
        <Input
          ref={searchRef}
          id={controlId}
          name="config-field-search"
          role="combobox"
          aria-label="Search config fields"
          aria-haspopup="dialog"
          aria-expanded={isOpen}
          aria-controls={isOpen ? popupId : undefined}
          aria-activedescendant={isOpen ? activeOptionId : undefined}
          aria-describedby={`${controlId}-result-count`}
          autoComplete="off"
          placeholder="Search fields, keys, or sections…"
          spellCheck={false}
          value={query}
          onFocus={handleSearchFocus}
          onChange={handleSearchChange}
          onClick={handleSearchClick}
          onKeyDown={handleSearchKeyDown}
          className="h-touch rounded-panel border-line bg-control-field pl-9 pr-12 type-body shadow-panel focus-visible:ring-focus md:h-control-lg md:pr-10"
        />
        {query.length > 0 && (
          <Button
            type="button"
            variant="ghost"
            aria-label="Clear config search"
            onMouseDown={(event) => event.preventDefault()}
            onClick={clearSearch}
            className="absolute right-0 top-1/2 h-touch w-touch -translate-y-1/2 rounded-control-md p-0 text-ink-faint hover:text-ink md:right-1.5 md:h-control-sm md:w-control-sm"
          >
            <X className="h-3.5 w-3.5" aria-hidden />
          </Button>
        )}
      </div>
      <span
        id={`${controlId}-result-count`}
        role="status"
        aria-live="polite"
        className="sr-only"
      >
        {query.trim()
          ? `${matchingCount} matching config ${matchingCount === 1 ? "field" : "fields"}.`
          : ""}
      </span>

      {isOpen && (
        <DropdownShell
          ref={collectionRef}
          id={popupId}
          role="dialog"
          ariaLabel="Matching config fields"
          onKeyDown={handlePopupKeyDown}
          onScroll={handleCollectionScroll}
          className="max-h-[min(34rem,calc(100dvh-14rem))] overflow-y-auto p-2"
        >
          <ConfigSearchResults
            popupId={popupId}
            virtualOptions={virtualOptions}
            beforeHeight={beforeHeight}
            afterHeight={afterHeight}
            activeIndex={activeIndex}
            selectedFieldKey={selectedFieldKey}
            measureOption={measureOption}
            onOptionMouseEnter={handleOptionMouseEnter}
            onSelect={activate}
            onFieldChange={onFieldChange}
            onFieldReset={onFieldReset}
          />
        </DropdownShell>
      )}
    </div>
  );
}
