import { Search, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ConfigSearchResults } from "@/features/workbench/components/config/config-search-results";
import { DropdownShell } from "@/features/workbench/components/shared/dropdown-shell";
import { useSearchableDialogInteraction } from "@/features/workbench/components/shared/use-searchable-dialog";
import { type RuntimeDefaultsSearchOptionPresentation } from "@/features/workbench/state/full-config/runtime-defaults-schema-presentation";

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
    ids,
    state: {
      isOpen,
      visibleOptions,
      isLoadingMore,
      matchingCount,
    },
    root,
    search,
    popup,
    collection,
    actions,
  } = interaction;

  return (
    <div
      ref={root.ref}
      className="relative z-20"
      onFocus={root.onFocus}
      onBlur={root.onBlur}
    >
      <div className="relative">
        <Search
          className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-faint"
          aria-hidden
        />
        <Input
          ref={search.ref}
          id={ids.control}
          name="config-field-search"
          role="combobox"
          aria-label="Search config fields"
          aria-haspopup="dialog"
          aria-expanded={isOpen}
          aria-controls={isOpen ? ids.popup : undefined}
          aria-describedby={`${ids.control}-result-count`}
          autoComplete="off"
          placeholder="Search fields, keys, or sections…"
          spellCheck={false}
          value={query}
          onChange={search.onChange}
          onClick={search.onClick}
          onKeyDown={search.onKeyDown}
          className="h-touch rounded-panel border-line bg-control-field pl-9 pr-12 type-body shadow-panel focus-visible:ring-focus md:h-control-lg md:pr-10"
        />
        {query.length > 0 && (
          <Button
            type="button"
            variant="ghost"
            aria-label="Clear config search"
            onMouseDown={(event) => event.preventDefault()}
            onClick={search.clear}
            className="absolute right-0 top-1/2 h-touch w-touch -translate-y-1/2 rounded-control-md p-0 text-ink-faint hover:text-ink md:right-1.5 md:h-control-sm md:w-control-sm"
          >
            <X className="h-3.5 w-3.5" aria-hidden />
          </Button>
        )}
      </div>
      <span
        id={`${ids.control}-result-count`}
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
          ref={popup.ref}
          id={ids.popup}
          role="dialog"
          ariaLabel="Matching config fields"
          onKeyDown={popup.onKeyDown}
          onScroll={collection.onScroll}
          className="max-h-[min(34rem,calc(100vh-14rem))] overflow-y-auto p-2"
        >
          <ConfigSearchResults
            popupId={ids.popup}
            visibleOptions={visibleOptions}
            isLoadingMore={isLoadingMore}
            selectedFieldKey={selectedFieldKey}
            onSelect={actions.activate}
            optionTitle={actions.optionTitle}
            onFieldChange={onFieldChange}
            onFieldReset={onFieldReset}
          />
        </DropdownShell>
      )}
    </div>
  );
}
