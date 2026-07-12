import {
  type ChangeEvent,
  type FocusEvent,
  type KeyboardEvent,
  type MouseEvent,
  type RefCallback,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";

const SCROLL_LOAD_THRESHOLD_PX = 48;
const LOCAL_APPEND_DELAY_MS = 100;

type SearchablePopupMode = "single-select" | "multi-select";

type SearchablePopupInteractionOptions<Option> = {
  mode: SearchablePopupMode;
  id?: string;
  idSuffix: string;
  options: Option[];
  optionKey: (option: Option) => string;
  optionSearchText: (option: Option) => string;
  optionRevision?: (option: Option) => string;
  selectedKey?: string;
  disabled?: boolean;
  isOptionDisabled?: (option: Option) => boolean;
  onActivate: (option: Option) => void;
  pagination?: { initialVisibleCount: number; pageSize: number };
};

type OptionInteraction<ElementType extends HTMLElement> = {
  ref: RefCallback<ElementType>;
  onMouseDown: (event: MouseEvent<ElementType>) => void;
  onMouseEnter: () => void;
  onClick: () => void;
  onKeyDown: (event: KeyboardEvent<ElementType>) => void;
};

function joinedOptionKey<Option>(
  options: Option[],
  keyForOption: (option: Option) => string,
) {
  return options.map(keyForOption).join("\u0000");
}

function clearTimer(timer: {
  current: ReturnType<typeof setTimeout> | null;
}) {
  if (timer.current !== null) {
    clearTimeout(timer.current);
    timer.current = null;
  }
}

export function useSearchablePopupInteraction<
  Option,
  OptionElement extends HTMLElement = HTMLElement,
>({
  mode,
  id,
  idSuffix,
  options,
  optionKey,
  optionSearchText,
  optionRevision,
  selectedKey,
  disabled = false,
  isOptionDisabled,
  onActivate,
  pagination,
}: SearchablePopupInteractionOptions<Option>) {
  const generatedId = useId();
  const singleSelect = mode === "single-select";
  const controlId = id ?? `${generatedId}-${idSuffix}`;
  const searchId = `${controlId}-search`;
  const popupId = `${controlId}-options`;
  const rootRef = useRef<HTMLDivElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const searchRef = useRef<HTMLInputElement | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);
  const optionRefs = useRef<Array<OptionElement | null>>([]);
  const appendTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingKeyboardIndexRef = useRef<number | null>(null);
  const onActivateRef = useRef(onActivate);
  const [query, setQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const normalizedQuery = query.trim().toLowerCase();
  const matchingOptions = useMemo(
    () =>
      normalizedQuery
        ? options.filter((option) =>
            optionSearchText(option).toLowerCase().includes(normalizedQuery),
          )
        : options,
    [normalizedQuery, optionSearchText, options],
  );
  const matchingKey = joinedOptionKey(matchingOptions, optionKey);
  const sourceRevisionKey = joinedOptionKey(
    options,
    optionRevision ?? optionKey,
  );
  const sourceInitialIndex =
    options.length === 0
      ? -1
      : singleSelect && selectedKey !== undefined
        ? Math.max(
            0,
            options.findIndex((option) => optionKey(option) === selectedKey),
          )
        : 0;
  const matchingInitialIndex =
    matchingOptions.length === 0
      ? -1
      : singleSelect && selectedKey !== undefined
        ? Math.max(
            0,
            matchingOptions.findIndex(
              (option) => optionKey(option) === selectedKey,
            ),
          )
        : 0;
  const initialVisibleCount =
    pagination?.initialVisibleCount ?? matchingOptions.length;
  const pageSize = pagination?.pageSize ?? initialVisibleCount;
  const [visibleCount, setVisibleCount] = useState(() =>
    Math.min(initialVisibleCount, matchingOptions.length),
  );
  const [isLoadingMore, setIsLoadingMore] = useState(false);
  const [activeIndex, setActiveIndex] = useState(matchingInitialIndex);
  const visibleOptions = matchingOptions.slice(0, visibleCount);
  const hasMore = visibleCount < matchingOptions.length;
  const activeOption =
    isOpen && activeIndex >= 0 ? visibleOptions[activeIndex] : undefined;
  const activeOptionId = activeOption
    ? `${popupId}-option-${activeIndex}`
    : undefined;

  onActivateRef.current = onActivate;

  function focusOption(index: number) {
    window.requestAnimationFrame(() => optionRefs.current[index]?.focus());
  }

  const dismiss = useCallback((restoreFocus = false) => {
    setIsOpen(false);
    setQuery("");
    pendingKeyboardIndexRef.current = null;
    if (restoreFocus) {
      triggerRef.current?.focus();
    }
  }, []);

  function loadMore() {
    if (
      appendTimerRef.current !== null ||
      visibleCount >= matchingOptions.length
    ) {
      return;
    }
    setIsLoadingMore(true);
    appendTimerRef.current = setTimeout(() => {
      appendTimerRef.current = null;
      setVisibleCount((current) =>
        Math.min(current + pageSize, matchingOptions.length),
      );
      setIsLoadingMore(false);
    }, LOCAL_APPEND_DELAY_MS);
  }

  function moveActiveOption(direction: 1 | -1, moveFocus: boolean) {
    if (visibleOptions.length === 0) {
      return;
    }
    setActiveIndex((current) => {
      if (
        direction === 1 &&
        current >= visibleOptions.length - 1 &&
        hasMore
      ) {
        pendingKeyboardIndexRef.current = visibleOptions.length;
        loadMore();
        return current;
      }
      const nextIndex =
        current < 0
          ? direction === 1
            ? 0
            : visibleOptions.length - 1
          : (current + direction + visibleOptions.length) % visibleOptions.length;
      if (moveFocus) {
        focusOption(nextIndex);
      }
      return nextIndex;
    });
  }

  function focusEdgeOption(edge: "first" | "last") {
    if (visibleOptions.length === 0) {
      return;
    }
    const index = edge === "first" ? 0 : visibleOptions.length - 1;
    setActiveIndex(index);
    focusOption(index);
  }

  function activate(option: Option) {
    if (isOptionDisabled?.(option)) {
      return;
    }
    if (singleSelect) {
      dismiss(true);
    }
    onActivateRef.current(option);
  }

  function open() {
    if (disabled || options.length === 0) {
      return;
    }
    setQuery("");
    setActiveIndex(sourceInitialIndex);
    setIsOpen(true);
  }

  function handleTriggerKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
    if (isOpen && singleSelect) {
      handleNavigableKey(event, activeOption, false, true);
      return;
    }
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (!singleSelect) {
        open();
        return;
      }
      if (disabled || options.length === 0) {
        return;
      }
      const direction = event.key === "ArrowDown" ? 1 : -1;
      const nextIndex =
        (sourceInitialIndex + direction + options.length) % options.length;
      setIsOpen(true);
      setActiveIndex(nextIndex);
      focusOption(nextIndex);
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (isOpen) {
        dismiss();
      } else {
        open();
      }
      return;
    }
    if (event.key === "Escape" && isOpen) {
      event.preventDefault();
      dismiss(true);
    }
  }

  function handleSearchKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (!singleSelect) {
      if (event.key === "Escape") {
        event.preventDefault();
        dismiss(true);
      } else if (event.key === "ArrowDown" && visibleOptions.length > 0) {
        event.preventDefault();
        const nextIndex = activeIndex >= 0 ? activeIndex : 0;
        setActiveIndex(nextIndex);
        focusOption(nextIndex);
      }
      return;
    }
    handleNavigableKey(event, activeOption, true, false);
  }

  function handleNavigableKey<ElementType extends HTMLElement>(
    event: KeyboardEvent<ElementType>,
    option: Option | undefined,
    resetSearch: boolean,
    allowSpace: boolean,
  ) {
    if (event.key === "Escape") {
      event.preventDefault();
      dismiss(true);
      return;
    }
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      if (visibleOptions.length === 0) {
        return;
      }
      event.preventDefault();
      if (resetSearch) {
        setQuery("");
      }
      moveActiveOption(event.key === "ArrowDown" ? 1 : -1, true);
      return;
    }
    if (
      (event.key === "Home" || event.key === "End") &&
      visibleOptions.length > 0
    ) {
      event.preventDefault();
      focusEdgeOption(event.key === "Home" ? "first" : "last");
      return;
    }
    if (event.key === "Enter" || (allowSpace && event.key === " ")) {
      if (allowSpace || option) {
        event.preventDefault();
      }
      if (option) {
        activate(option);
      }
    }
  }

  function handleOptionKeyDown(
    event: KeyboardEvent<OptionElement>,
    option: Option,
  ) {
    handleNavigableKey(event, option, false, true);
  }

  function optionInteraction(
    index: number,
    option: Option,
  ): OptionInteraction<OptionElement> {
    return {
      ref: (node) => {
        optionRefs.current[index] = node;
      },
      onMouseDown: (event) => event.preventDefault(),
      onMouseEnter: () => setActiveIndex(index),
      onClick: () => {
        setActiveIndex(index);
        activate(option);
      },
      onKeyDown: (event) => handleOptionKeyDown(event, option),
    };
  }

  function handleRootBlur(event: FocusEvent<HTMLDivElement>) {
    const nextTarget = event.relatedTarget as Node | null;
    if (!nextTarget || !rootRef.current?.contains(nextTarget)) {
      dismiss();
    }
  }

  function handleScroll() {
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
  }

  useEffect(() => {
    if (singleSelect) {
      dismiss();
      setActiveIndex(sourceInitialIndex);
    }
  }, [dismiss, selectedKey, singleSelect, sourceInitialIndex, sourceRevisionKey]);

  useEffect(() => {
    if (isOpen) {
      setActiveIndex(matchingInitialIndex);
    }
    // Matching identity is the lifecycle reset signal; opening is handled by open().
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [matchingKey]);

  useEffect(() => {
    clearTimer(appendTimerRef);
    pendingKeyboardIndexRef.current = null;
    setIsLoadingMore(false);
    setVisibleCount(Math.min(initialVisibleCount, matchingOptions.length));
    return () => clearTimer(appendTimerRef);
  }, [
    initialVisibleCount,
    matchingKey,
    matchingOptions.length,
  ]);

  useEffect(() => {
    const pendingIndex = pendingKeyboardIndexRef.current;
    if (pendingIndex !== null && pendingIndex < visibleOptions.length) {
      pendingKeyboardIndexRef.current = null;
      setActiveIndex(pendingIndex);
      window.requestAnimationFrame(() =>
        optionRefs.current[pendingIndex]?.focus(),
      );
      return;
    }
    setActiveIndex((current) =>
      current >= visibleOptions.length
        ? visibleOptions.length > 0
          ? visibleOptions.length - 1
          : -1
        : current,
    );
  }, [visibleOptions.length]);

  useEffect(() => {
    if (disabled) {
      dismiss();
    }
  }, [disabled, dismiss]);

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    searchRef.current?.focus();
    const handlePointerDown = (event: PointerEvent) => {
      if (!rootRef.current?.contains(event.target as Node)) {
        dismiss();
      }
    };
    document.addEventListener("pointerdown", handlePointerDown);
    return () => document.removeEventListener("pointerdown", handlePointerDown);
  }, [dismiss, isOpen]);

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
      if (scrollContainer.scrollHeight <= scrollContainer.clientHeight) {
        loadMore();
      }
    }, 0);
    return () => clearTimeout(autoFillTimer);
    // The listed values fully describe the paging closure used by loadMore().
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [hasMore, isLoadingMore, matchingOptions.length, pageSize, visibleCount]);

  return {
    ids: {
      control: controlId,
      search: searchId,
      popup: popupId,
      active: activeOptionId,
    },
    state: {
      isOpen,
      query,
      options: visibleOptions,
      activeIndex,
      active: activeOption,
      loading: isLoadingMore,
    },
    root: { ref: rootRef, onBlur: handleRootBlur },
    trigger: {
      ref: triggerRef,
      onClick: () => (isOpen ? dismiss() : open()),
      onKeyDown: handleTriggerKeyDown,
    },
    search: {
      ref: searchRef,
      onChange: (event: ChangeEvent<HTMLInputElement>) =>
        setQuery(event.target.value),
      onKeyDown: handleSearchKeyDown,
    },
    collection: {
      ref: scrollContainerRef,
      onScroll: handleScroll,
      option: optionInteraction,
    },
    close: dismiss,
  };
}
