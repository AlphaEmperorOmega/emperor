import {
  type ChangeEvent,
  type FocusEvent,
  type KeyboardEvent,
  useCallback,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  searchableCollectionKey,
  useSearchableCollectionWindow,
} from "@/features/workbench/components/shared/use-searchable-collection";

function resultId(popupId: string, key: string) {
  return `${popupId}-option-${encodeURIComponent(key)}`;
}

export function useSearchableDialogInteraction<Option>({
  id,
  idSuffix,
  options,
  optionKey,
  matchesQuery,
  query,
  onQueryChange,
  onClear,
  onActivate,
  initialVisibleCount,
}: {
  id?: string;
  idSuffix: string;
  options: Option[];
  optionKey: (option: Option) => string;
  matchesQuery: (option: Option, query: string) => boolean;
  query: string;
  onQueryChange: (query: string) => void;
  onClear: () => void;
  onActivate: (option: Option) => void;
  initialVisibleCount: number;
  pageSize: number;
}) {
  const generatedId = useId();
  const controlId = id ?? `${generatedId}-${idSuffix}`;
  const popupId = `${controlId}-results`;
  const rootRef = useRef<HTMLDivElement | null>(null);
  const searchRef = useRef<HTMLInputElement | null>(null);
  const [hasFocusWithin, setHasFocusWithin] = useState(false);
  const [openRequested, setOpenRequested] = useState(false);
  const [activeState, setActiveState] = useState({
    collectionKey: "",
    index: -1,
  });
  const trimmedQuery = query.trim();
  const matchingOptions = useMemo(
    () =>
      trimmedQuery
        ? options.filter((option) => matchesQuery(option, trimmedQuery))
        : [],
    [matchesQuery, options, trimmedQuery],
  );
  const matchingKey = searchableCollectionKey(matchingOptions, optionKey);
  const activeIndex =
    activeState.collectionKey === matchingKey
      ? Math.min(activeState.index, matchingOptions.length - 1)
      : matchingOptions.length > 0
        ? 0
        : -1;
  const activeOption =
    activeIndex >= 0 ? matchingOptions[activeIndex] : undefined;
  const activeOptionId = activeOption
    ? resultId(popupId, optionKey(activeOption))
    : undefined;
  const isOpen =
    hasFocusWithin && openRequested && trimmedQuery.length > 0;
  const handleScrollIndexChange = useCallback(
    (index: number) => {
      setActiveState({ collectionKey: matchingKey, index });
    },
    [matchingKey],
  );
  const {
    collectionRef,
    measureOption,
    handleCollectionScroll,
    scrollToIndex,
    resetCollectionScroll,
    virtualOptions,
    beforeHeight,
    afterHeight,
  } = useSearchableCollectionWindow({
    options: matchingOptions,
    optionKey,
    estimatedRowHeight: 156,
    fallbackViewportHeight: initialVisibleCount * 156,
    overscanRows: 1,
    onScrollIndexChange: handleScrollIndexChange,
  });

  const dismiss = useCallback(() => {
    setOpenRequested(false);
  }, []);

  const restoreSearchFocus = useCallback(() => {
    dismiss();
    queueMicrotask(() => searchRef.current?.focus());
  }, [dismiss]);

  const setActiveIndex = useCallback(
    (index: number) => {
      const nextIndex =
        matchingOptions.length === 0
          ? -1
          : Math.max(0, Math.min(index, matchingOptions.length - 1));
      setActiveState({ collectionKey: matchingKey, index: nextIndex });
      scrollToIndex(nextIndex);
    },
    [matchingKey, matchingOptions.length, scrollToIndex],
  );

  const activate = useCallback(
    (option: Option) => {
      onActivate(option);
      dismiss();
    },
    [dismiss, onActivate],
  );

  const handleRootFocus = useCallback(() => {
    setHasFocusWithin(true);
  }, []);

  const handleRootBlur = useCallback((event: FocusEvent<HTMLDivElement>) => {
    const nextTarget = event.relatedTarget as Node | null;
    if (nextTarget && rootRef.current?.contains(nextTarget)) {
      return;
    }
    setHasFocusWithin(false);
    setOpenRequested(false);
  }, []);

  const handleSearchFocus = useCallback(() => {
    setHasFocusWithin(true);
  }, []);

  const handleSearchChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const nextQuery = event.target.value;
      onQueryChange(nextQuery);
      setOpenRequested(nextQuery.trim().length > 0);
      setActiveState({ collectionKey: "", index: -1 });
      resetCollectionScroll();
    },
    [onQueryChange, resetCollectionScroll],
  );

  const handleSearchClick = useCallback(() => {
    setOpenRequested(trimmedQuery.length > 0);
  }, [trimmedQuery.length]);

  const handleSearchKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (event.key === "Escape" && isOpen) {
        event.preventDefault();
        restoreSearchFocus();
        return;
      }
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        if (matchingOptions.length === 0) {
          return;
        }
        event.preventDefault();
        setOpenRequested(true);
        const direction = event.key === "ArrowDown" ? 1 : -1;
        const current =
          activeIndex < 0 ? (direction === 1 ? -1 : 0) : activeIndex;
        setActiveIndex(
          (current + direction + matchingOptions.length) %
            matchingOptions.length,
        );
        return;
      }
      if (event.key === "Home" || event.key === "End") {
        event.preventDefault();
        setActiveIndex(event.key === "Home" ? 0 : matchingOptions.length - 1);
        return;
      }
      if (event.key === "Enter" && activeOption) {
        event.preventDefault();
        activate(activeOption);
      }
    },
    [
      activate,
      activeIndex,
      activeOption,
      isOpen,
      matchingOptions.length,
      restoreSearchFocus,
      setActiveIndex,
    ],
  );

  const handlePopupKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.key !== "Escape") {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      restoreSearchFocus();
    },
    [restoreSearchFocus],
  );

  const clearSearch = useCallback(() => {
    onClear();
    dismiss();
    searchRef.current?.focus();
  }, [dismiss, onClear]);

  const handleOptionMouseEnter = useCallback(
    (index: number) => {
      setActiveState({ collectionKey: matchingKey, index });
    },
    [matchingKey],
  );

  return {
    controlId,
    popupId,
    activeOptionId,
    isOpen,
    activeIndex,
    matchingCount: matchingOptions.length,
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
  };
}
