import {
  type ChangeEvent,
  type FocusEvent,
  type KeyboardEvent,
  type MouseEvent,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  searchableCollectionKey,
  useSearchableCollectionWindow,
} from "@/features/workbench/components/shared/use-searchable-collection";

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

function optionId(popupId: string, key: string) {
  return `${popupId}-option-${encodeURIComponent(key)}`;
}

export function useSearchablePopupInteraction<Option>({
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
  const [query, setQuery] = useState("");
  const [openRequested, setOpenRequested] = useState(false);
  const [activeState, setActiveState] = useState({
    collectionKey: "",
    index: -1,
  });

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
  const matchingKey = searchableCollectionKey(
    matchingOptions,
    optionRevision ?? optionKey,
  );
  const sourceKey = searchableCollectionKey(
    options,
    optionRevision ?? optionKey,
  );
  const defaultActiveIndex =
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
  const activeIndex =
    activeState.collectionKey === matchingKey
      ? Math.min(activeState.index, matchingOptions.length - 1)
      : defaultActiveIndex;
  const activeOption =
    activeIndex >= 0 ? matchingOptions[activeIndex] : undefined;
  const activeOptionId = activeOption
    ? optionId(popupId, optionKey(activeOption))
    : undefined;
  const isOpen = openRequested && !disabled && options.length > 0;
  const viewportRows = Math.max(
    3,
    Math.min(10, pagination?.initialVisibleCount ?? 6),
  );
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
    beforeHeight,
    afterHeight,
    virtualOptions,
  } = useSearchableCollectionWindow({
    options: matchingOptions,
    optionKey,
    fallbackViewportHeight: viewportRows * 44,
    onScrollIndexChange: handleScrollIndexChange,
  });

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

  const dismiss = useCallback((restoreFocus = false) => {
    setOpenRequested(false);
    setQuery("");
    if (restoreFocus) {
      queueMicrotask(() => triggerRef.current?.focus());
    }
  }, []);

  const open = useCallback(
    (edge?: "first" | "last") => {
      if (disabled || options.length === 0) {
        return;
      }
      const selectedIndex =
        singleSelect && selectedKey !== undefined
          ? Math.max(
              0,
              options.findIndex((option) => optionKey(option) === selectedKey),
            )
          : 0;
      const index =
        edge === "last"
          ? options.length - 1
          : edge === "first"
            ? 0
            : selectedIndex;
      setQuery("");
      setOpenRequested(true);
      setActiveState({ collectionKey: sourceKey, index });
      resetCollectionScroll();
      scrollToIndex(index, edge === "last" ? "end" : "nearest");
    },
    [
      disabled,
      optionKey,
      options,
      resetCollectionScroll,
      scrollToIndex,
      selectedKey,
      singleSelect,
      sourceKey,
    ],
  );

  const activate = useCallback(
    (option: Option) => {
      if (isOptionDisabled?.(option)) {
        return;
      }
      if (singleSelect) {
        dismiss(true);
      }
      onActivate(option);
    },
    [dismiss, isOptionDisabled, onActivate, singleSelect],
  );

  const moveActive = useCallback(
    (direction: 1 | -1) => {
      if (matchingOptions.length === 0) {
        return;
      }
      const current = activeIndex < 0 ? (direction === 1 ? -1 : 0) : activeIndex;
      const next =
        (current + direction + matchingOptions.length) %
        matchingOptions.length;
      setActiveIndex(next);
    },
    [activeIndex, matchingOptions.length, setActiveIndex],
  );

  const handleTriggerClick = useCallback(() => {
    if (isOpen) {
      dismiss();
    } else {
      open();
    }
  }, [dismiss, isOpen, open]);

  const handleTriggerKeyDown = useCallback(
    (event: KeyboardEvent<HTMLButtonElement>) => {
      if (
        event.key === "ArrowDown" ||
        event.key === "ArrowUp" ||
        event.key === "Enter" ||
        event.key === " "
      ) {
        event.preventDefault();
        if (isOpen) {
          dismiss();
        } else {
          open(event.key === "ArrowUp" ? "last" : undefined);
        }
      } else if (event.key === "Escape" && isOpen) {
        event.preventDefault();
        dismiss(true);
      }
    },
    [dismiss, isOpen, open],
  );

  const handleSearchChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      setQuery(event.target.value);
      setActiveState({ collectionKey: "", index: -1 });
      resetCollectionScroll();
    },
    [resetCollectionScroll],
  );

  const handleSearchKeyDown = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (event.key === "Escape") {
        event.preventDefault();
        dismiss(true);
        return;
      }
      if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        moveActive(event.key === "ArrowDown" ? 1 : -1);
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
      activeOption,
      dismiss,
      matchingOptions.length,
      moveActive,
      setActiveIndex,
    ],
  );

  const handleRootBlur = useCallback(
    (event: FocusEvent<HTMLDivElement>) => {
      const nextTarget = event.relatedTarget as Node | null;
      if (!nextTarget || !rootRef.current?.contains(nextTarget)) {
        dismiss();
      }
    },
    [dismiss],
  );

  const handleOptionMouseDown = useCallback(
    (event: MouseEvent<HTMLElement>) => {
      event.preventDefault();
    },
    [],
  );

  const handleOptionMouseEnter = useCallback(
    (index: number) => {
      setActiveState({ collectionKey: matchingKey, index });
    },
    [matchingKey],
  );

  const handleOptionClick = useCallback(
    (index: number) => {
      const option = matchingOptions[index];
      if (!option) {
        return;
      }
      setActiveState({ collectionKey: matchingKey, index });
      activate(option);
    },
    [activate, matchingKey, matchingOptions],
  );

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

  return {
    controlId,
    searchId,
    popupId,
    activeOptionId,
    isOpen,
    query,
    activeIndex,
    activeOption,
    matchingCount: matchingOptions.length,
    virtualOptions,
    beforeHeight,
    afterHeight,
    rootRef,
    triggerRef,
    searchRef,
    collectionRef,
    measureOption,
    handleRootBlur,
    handleTriggerClick,
    handleTriggerKeyDown,
    handleSearchChange,
    handleSearchKeyDown,
    handleCollectionScroll,
    handleOptionMouseDown,
    handleOptionMouseEnter,
    handleOptionClick,
    close: dismiss,
  };
}
