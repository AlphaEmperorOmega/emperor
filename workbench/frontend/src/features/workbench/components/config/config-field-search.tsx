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
    state: { isOpen, visibleOptions, isLoadingMore, activeIndex },
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
          role="combobox"
          aria-label="Search config fields"
          aria-haspopup="dialog"
          aria-expanded={isOpen}
          aria-controls={ids.popup}
          autoComplete="off"
          placeholder="Search fields, keys, or sections"
          value={query}
          onChange={search.onChange}
          onKeyDown={search.onKeyDown}
          className="h-10 rounded-[12px] border-line bg-black/25 pl-9 pr-10 text-sm shadow-[0_14px_34px_-32px_rgba(0,0,0,0.95)] focus-visible:ring-focus"
        />
        {query.length > 0 && (
          <Button
            type="button"
            variant="ghost"
            aria-label="Clear config search"
            onMouseDown={(event) => event.preventDefault()}
            onClick={search.clear}
            className="absolute right-1.5 top-1/2 h-7 w-7 -translate-y-1/2 rounded-[8px] p-0 text-ink-faint hover:text-ink"
          >
            <X className="h-3.5 w-3.5" aria-hidden />
          </Button>
        )}
      </div>

      {isOpen && (
        <DropdownShell
          ref={popup.ref}
          id={ids.popup}
          role="dialog"
          ariaLabel="Matching config fields"
          onScroll={collection.onScroll}
          className="max-h-[min(34rem,calc(100vh-14rem))] overflow-y-auto p-2"
        >
          <ConfigSearchResults
            popupId={ids.popup}
            visibleOptions={visibleOptions}
            isLoadingMore={isLoadingMore}
            activeIndex={activeIndex}
            selectedFieldKey={selectedFieldKey}
            onSelect={actions.activate}
            onFieldChange={onFieldChange}
            onFieldReset={onFieldReset}
          />
        </DropdownShell>
      )}
    </div>
  );
}
