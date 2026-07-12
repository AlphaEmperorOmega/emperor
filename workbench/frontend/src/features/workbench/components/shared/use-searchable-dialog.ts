import {
  type ChangeEvent,
  type FocusEvent,
  type KeyboardEvent,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";

const SCROLL_LOAD_THRESHOLD_PX = 48;
const LOCAL_APPEND_DELAY_MS = 100;

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
  pageSize,
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
  const popupRef = useRef<HTMLDivElement | null>(null);
  const optionTitleRefs = useRef(new Map<string, HTMLButtonElement>());
  const appendTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingActiveIndexRef = useRef<number | null>(null);
  const pendingFocusIndexRef = useRef<number | null>(null);
  const suppressNextFocusOpenRef = useRef(false);
  const onActivateRef = useRef(onActivate);
  const onQueryChangeRef = useRef(onQueryChange);
  const onClearRef = useRef(onClear);
  const [hasFocusWithin, setHasFocusWithin] = useState(false);
  const [openRequested, setOpenRequested] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const [visibleCount, setVisibleCount] = useState(0);
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const trimmedQuery = query.trim();
  const matchingOptions = useMemo(
    () =>
      trimmedQuery
        ? options.filter((option) => matchesQuery(option, trimmedQuery))
        : [],
    [matchesQuery, options, trimmedQuery],
  );
  const matchingKey = matchingOptions.map(optionKey).join("\u0000");
  const visibleOptions = useMemo(
    () => matchingOptions.slice(0, visibleCount),
    [matchingOptions, visibleCount],
  );
  const hasMore = visibleCount < matchingOptions.length;
  const isOpen = hasFocusWithin && openRequested && trimmedQuery.length > 0;
  onActivateRef.current = onActivate;
  onQueryChangeRef.current = onQueryChange;
  onClearRef.current = onClear;

  const clearAppendTimer = useCallback(() => {
    if (appendTimerRef.current === null) {
      return;
    }
    clearTimeout(appendTimerRef.current);
    appendTimerRef.current = null;
  }, []);

  const dismiss = useCallback(() => {
    setOpenRequested(false);
    pendingActiveIndexRef.current = null;
    pendingFocusIndexRef.current = null;
  }, []);

  const restoreSearchFocus = useCallback(() => {
    suppressNextFocusOpenRef.current = true;
    dismiss();
    queueMicrotask(() => searchRef.current?.focus());
  }, [dismiss]);

  const loadMore = useCallback(() => {
    if (
      appendTimerRef.current !== null ||
      visibleCount >= matchingOptions.length
    ) {
      return false;
    }
    setIsLoadingMore(true);
    appendTimerRef.current = setTimeout(() => {
      appendTimerRef.current = null;
      setVisibleCount((current) =>
        Math.min(current + pageSize, matchingOptions.length),
      );
      setIsLoadingMore(false);
    }, LOCAL_APPEND_DELAY_MS);
    return true;
  }, [matchingOptions.length, pageSize, visibleCount]);

  const focusOption = useCallback(
    (index: number) => {
      const option = visibleOptions[index];
      if (!option) return false;
      setActiveIndex(index);
      const key = optionKey(option);
      queueMicrotask(() => optionTitleRefs.current.get(key)?.focus());
      return true;
    },
    [optionKey, visibleOptions],
  );

  const moveActiveOption = useCallback(
    (direction: 1 | -1, currentIndex = activeIndex) => {
      if (visibleOptions.length === 0) {
        return;
      }
      setOpenRequested(true);
      if (
        direction === 1 &&
        currentIndex >= visibleOptions.length - 1 &&
        hasMore
      ) {
        pendingActiveIndexRef.current = visibleOptions.length;
        pendingFocusIndexRef.current = visibleOptions.length;
        loadMore();
        return;
      }
      const nextIndex =
        currentIndex < 0
          ? direction === 1
            ? 0
            : visibleOptions.length - 1
          : (currentIndex + direction + visibleOptions.length) %
            visibleOptions.length;
      focusOption(nextIndex);
    },
    [activeIndex, focusOption, hasMore, loadMore, visibleOptions.length],
  );

  const activate = useCallback(
    (option: Option) => {
      onActivateRef.current(option);
      dismiss();
    },
    [dismiss],
  );

  const handleRootFocus = useCallback(() => {
    setHasFocusWithin(true);
    if (suppressNextFocusOpenRef.current) {
      suppressNextFocusOpenRef.current = false;
      return;
    }
    setOpenRequested(trimmedQuery.length > 0);
  }, [trimmedQuery.length]);

  const handleRootBlur = useCallback((event: FocusEvent<HTMLDivElement>) => {
    const nextTarget = event.relatedTarget as Node | null;
    if (nextTarget && rootRef.current?.contains(nextTarget)) {
      return;
    }
    setHasFocusWithin(false);
    setOpenRequested(false);
  }, []);

  const handleSearchChange = useCallback((event: ChangeEvent<HTMLInputElement>) => {
    const nextQuery = event.target.value;
    onQueryChangeRef.current(nextQuery);
    setOpenRequested(nextQuery.trim().length > 0);
  }, []);

  const handleSearchClick = useCallback(() => {
    setOpenRequested(trimmedQuery.length > 0);
  }, [trimmedQuery.length]);

  const handleSearchKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        setOpenRequested(true);
        focusOption(event.key === "ArrowDown" ? 0 : visibleOptions.length - 1);
        return;
      }
      if (event.key === "Escape" && isOpen) {
        event.preventDefault();
        restoreSearchFocus();
      }
    },
    [focusOption, isOpen, restoreSearchFocus, visibleOptions.length],
  );

  const handleOptionKeyDown = useCallback(
    (index: number, event: KeyboardEvent<HTMLButtonElement>) => {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        moveActiveOption(event.key === "ArrowDown" ? 1 : -1, index);
      } else if (event.key === "Home") {
        event.preventDefault();
        focusOption(0);
      } else if (event.key === "End") {
        event.preventDefault();
        focusOption(visibleOptions.length - 1);
      }
    },
    [focusOption, moveActiveOption, visibleOptions.length],
  );

  const handlePopupKeyDown = useCallback(
    (event: KeyboardEvent<HTMLDivElement>) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      event.stopPropagation();
      restoreSearchFocus();
    },
    [restoreSearchFocus],
  );

  const handleScroll = useCallback(() => {
    const popup = popupRef.current;
    if (!popup || !hasMore || isLoadingMore) {
      return;
    }
    const distanceFromBottom =
      popup.scrollHeight - popup.scrollTop - popup.clientHeight;
    if (distanceFromBottom <= SCROLL_LOAD_THRESHOLD_PX) {
      loadMore();
    }
  }, [hasMore, isLoadingMore, loadMore]);

  const clearSearch = useCallback(() => {
    onClearRef.current();
    setOpenRequested(false);
    searchRef.current?.focus();
  }, []);

  useEffect(() => {
    setActiveIndex(matchingOptions.length > 0 ? 0 : -1);
  }, [matchingKey, matchingOptions.length]);

  useEffect(() => {
    clearAppendTimer();
    pendingActiveIndexRef.current = null;
    setIsLoadingMore(false);
    setVisibleCount(Math.min(initialVisibleCount, matchingOptions.length));
  }, [
    clearAppendTimer,
    initialVisibleCount,
    matchingKey,
    matchingOptions.length,
  ]);

  useEffect(() => {
    const pendingIndex = pendingActiveIndexRef.current;
    if (pendingIndex === null || pendingIndex >= visibleOptions.length) {
      return;
    }
    pendingActiveIndexRef.current = null;
    setActiveIndex(pendingIndex);
    if (pendingFocusIndexRef.current === pendingIndex) {
      pendingFocusIndexRef.current = null;
      const option = visibleOptions[pendingIndex];
      if (option) {
        const key = optionKey(option);
        queueMicrotask(() => optionTitleRefs.current.get(key)?.focus());
      }
    }
  }, [optionKey, visibleOptions, visibleOptions.length]);

  useEffect(() => {
    if (activeIndex < visibleOptions.length) {
      return;
    }
    setActiveIndex(visibleOptions.length > 0 ? visibleOptions.length - 1 : -1);
  }, [activeIndex, visibleOptions.length]);

  useEffect(() => {
    const popup = popupRef.current;
    if (
      !popup ||
      !hasMore ||
      isLoadingMore ||
      popup.clientHeight <= 0
    ) {
      return;
    }
    const autoFillTimer = setTimeout(() => {
      if (
        popup.scrollHeight <= popup.clientHeight &&
        popupRef.current === popup
      ) {
        loadMore();
      }
    }, 0);
    return () => clearTimeout(autoFillTimer);
  }, [hasMore, isLoadingMore, loadMore, visibleOptions.length]);

  useEffect(() => clearAppendTimer, [clearAppendTimer]);

  return {
    ids: { control: controlId, popup: popupId },
    state: {
      isOpen,
      visibleOptions,
      isLoadingMore,
      activeIndex,
      matchingCount: matchingOptions.length,
    },
    root: { ref: rootRef, onFocus: handleRootFocus, onBlur: handleRootBlur },
    search: {
      ref: searchRef,
      onChange: handleSearchChange,
      onClick: handleSearchClick,
      onKeyDown: handleSearchKeyDown,
      clear: clearSearch,
    },
    popup: { ref: popupRef, onKeyDown: handlePopupKeyDown },
    collection: { onScroll: handleScroll },
    actions: {
      activate,
      optionTitle: (option: Option, index: number) => ({
        ref: (element: HTMLButtonElement | null) => {
          const key = optionKey(option);
          if (element) optionTitleRefs.current.set(key, element);
          else optionTitleRefs.current.delete(key);
        },
        tabIndex: index === activeIndex ? 0 : -1,
        onFocus: () => setActiveIndex(index),
        onKeyDown: (event: KeyboardEvent<HTMLButtonElement>) =>
          handleOptionKeyDown(index, event),
      }),
    },
  };
}
