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
  const appendTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingActiveIndexRef = useRef<number | null>(null);
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
  const activeOption =
    isOpen && activeIndex >= 0 ? visibleOptions[activeIndex] : undefined;

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
  }, []);

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

  const moveActiveOption = useCallback(
    (direction: 1 | -1) => {
      if (visibleOptions.length === 0) {
        return;
      }
      setOpenRequested(true);
      setActiveIndex((current) => {
        if (
          direction === 1 &&
          current >= visibleOptions.length - 1 &&
          hasMore
        ) {
          pendingActiveIndexRef.current = visibleOptions.length;
          loadMore();
          return current;
        }
        if (current < 0) {
          return direction === 1 ? 0 : visibleOptions.length - 1;
        }
        return (
          (current + direction + visibleOptions.length) % visibleOptions.length
        );
      });
    },
    [hasMore, loadMore, visibleOptions.length],
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

  const handleSearchKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        moveActiveOption(event.key === "ArrowDown" ? 1 : -1);
        return;
      }
      if (event.key === "Enter" && activeOption) {
        event.preventDefault();
        activate(activeOption);
        return;
      }
      if (event.key === "Escape" && isOpen) {
        event.preventDefault();
        dismiss();
      }
    },
    [activate, activeOption, dismiss, isOpen, moveActiveOption],
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
  }, [visibleOptions.length]);

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
    state: { isOpen, visibleOptions, isLoadingMore, activeIndex },
    root: { ref: rootRef, onFocus: handleRootFocus, onBlur: handleRootBlur },
    search: {
      ref: searchRef,
      onChange: handleSearchChange,
      onKeyDown: handleSearchKeyDown,
      clear: clearSearch,
    },
    popup: { ref: popupRef },
    collection: { onScroll: handleScroll },
    actions: { activate },
  };
}
