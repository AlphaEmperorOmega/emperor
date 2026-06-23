import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const SCROLL_LOAD_THRESHOLD_PX = 48;
const LOCAL_APPEND_DELAY_MS = 100;

export function useIncrementalVisibleOptions<T>({
  options,
  initialVisibleCount,
  pageSize,
}: {
  options: T[];
  initialVisibleCount: number;
  pageSize: number;
}) {
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const appendTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const optionsLengthRef = useRef(options.length);
  const [visibleCount, setVisibleCount] = useState(() =>
    Math.min(initialVisibleCount, options.length),
  );
  const [isLoadingMore, setIsLoadingMore] = useState(false);

  const hasMore = visibleCount < options.length;
  const visibleOptions = useMemo(
    () => options.slice(0, visibleCount),
    [options, visibleCount],
  );

  const clearAppendTimer = useCallback(() => {
    if (appendTimerRef.current === null) {
      return;
    }
    clearTimeout(appendTimerRef.current);
    appendTimerRef.current = null;
  }, []);

  const loadMore = useCallback(() => {
    if (
      appendTimerRef.current !== null ||
      visibleCount >= optionsLengthRef.current
    ) {
      return false;
    }

    setIsLoadingMore(true);
    appendTimerRef.current = setTimeout(() => {
      appendTimerRef.current = null;
      setVisibleCount((current) =>
        Math.min(current + pageSize, optionsLengthRef.current),
      );
      setIsLoadingMore(false);
    }, LOCAL_APPEND_DELAY_MS);

    return true;
  }, [pageSize, visibleCount]);

  const handleScroll = useCallback(() => {
    const scrollContainer = scrollContainerRef.current;
    if (!scrollContainer || !hasMore || isLoadingMore) {
      return;
    }

    const distanceFromBottom =
      scrollContainer.scrollHeight -
      scrollContainer.scrollTop -
      scrollContainer.clientHeight;

    if (distanceFromBottom <= SCROLL_LOAD_THRESHOLD_PX) {
      loadMore();
    }
  }, [hasMore, isLoadingMore, loadMore]);

  useEffect(() => {
    optionsLengthRef.current = options.length;
  }, [options.length]);

  useEffect(() => {
    clearAppendTimer();
    setIsLoadingMore(false);
    setVisibleCount(Math.min(initialVisibleCount, options.length));
  }, [clearAppendTimer, initialVisibleCount, options]);

  useEffect(() => {
    const scrollContainer = scrollContainerRef.current;
    if (
      !scrollContainer ||
      !hasMore ||
      isLoadingMore ||
      scrollContainer.clientHeight <= 0
    ) {
      return;
    }

    const autoFillTimer = setTimeout(() => {
      if (
        scrollContainer.scrollHeight <= scrollContainer.clientHeight &&
        scrollContainerRef.current === scrollContainer
      ) {
        loadMore();
      }
    }, 0);

    return () => clearTimeout(autoFillTimer);
  }, [hasMore, isLoadingMore, loadMore, visibleOptions.length]);

  useEffect(() => clearAppendTimer, [clearAppendTimer]);

  return {
    scrollContainerRef,
    visibleOptions,
    visibleCount,
    hasMore,
    isLoadingMore,
    loadMore,
    handleScroll,
  };
}
