import {
  type KeyboardEvent,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { usePopupDismissal } from "@/features/workbench/components/shared/use-popup-dismissal";

export type SearchableDropdownOption = {
  value: string;
  label: string;
  description?: string;
};

export function useSearchableDropdownCore<
  Option extends SearchableDropdownOption,
>({
  id,
  idSuffix,
  options,
  disabled,
}: {
  id?: string;
  idSuffix: string;
  options: Option[];
  disabled: boolean;
}) {
  const generatedId = useId();
  const triggerId = id ?? `${generatedId}-${idSuffix}`;
  const listboxId = `${triggerId}-options`;
  const searchId = `${triggerId}-search`;
  const rootRef = useRef<HTMLDivElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const searchRef = useRef<HTMLInputElement | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const normalizedQuery = query.trim().toLowerCase();
  const filteredOptions = useMemo(() => {
    if (!normalizedQuery) {
      return options;
    }
    return options.filter((option) =>
      [option.label, option.value, option.description ?? ""]
        .join(" ")
        .toLowerCase()
        .includes(normalizedQuery),
    );
  }, [normalizedQuery, options]);
  const filteredOptionsKey = useMemo(
    () => filteredOptions.map((option) => option.value).join("\u0000"),
    [filteredOptions],
  );

  const openDropdown = useCallback(() => {
    if (disabled) {
      return;
    }
    setQuery("");
    setIsOpen(true);
  }, [disabled]);

  const closeDropdown = useCallback((restoreFocus = false) => {
    setIsOpen(false);
    setQuery("");
    if (restoreFocus) {
      triggerRef.current?.focus();
    }
  }, []);

  const toggleDropdown = useCallback(() => {
    if (isOpen) {
      closeDropdown();
      return;
    }
    openDropdown();
  }, [closeDropdown, isOpen, openDropdown]);

  const handleRootBlur = useCallback(
    (relatedTarget: EventTarget | null) => {
      const nextTarget = relatedTarget as Node | null;
      if (nextTarget && rootRef.current?.contains(nextTarget)) {
        return;
      }
      closeDropdown();
    },
    [closeDropdown],
  );

  useEffect(() => {
    if (isOpen) {
      searchRef.current?.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    if (disabled) {
      closeDropdown();
    }
  }, [closeDropdown, disabled]);

  usePopupDismissal({
    open: isOpen,
    onClose: closeDropdown,
    triggerRef,
    panelRef,
  });

  return {
    triggerId,
    listboxId,
    searchId,
    rootRef,
    triggerRef,
    panelRef,
    searchRef,
    isOpen,
    query,
    setQuery,
    filteredOptions,
    filteredOptionsKey,
    openDropdown,
    closeDropdown,
    toggleDropdown,
    handleRootBlur,
  };
}

type DropdownNavigationOptions = {
  optionCount: number;
  fallbackIndex?: number;
  onMovePastEnd?: () => boolean;
  onEscape: () => void;
};

export function useDropdownOptionNavigation<
  ElementType extends HTMLElement = HTMLElement,
>({
  optionCount,
  fallbackIndex = 0,
  onMovePastEnd,
  onEscape,
}: DropdownNavigationOptions) {
  const optionRefs = useRef<Array<ElementType | null>>([]);
  const [activeIndex, setActiveIndex] = useState(-1);

  const focusOption = useCallback((index: number) => {
    window.requestAnimationFrame(() => {
      optionRefs.current[index]?.focus();
    });
  }, []);

  const focusFirstOption = useCallback(() => {
    if (optionCount === 0) {
      return;
    }
    setActiveIndex(0);
    focusOption(0);
  }, [focusOption, optionCount]);

  const focusLastOption = useCallback(() => {
    if (optionCount === 0) {
      return;
    }
    const lastIndex = optionCount - 1;
    setActiveIndex(lastIndex);
    focusOption(lastIndex);
  }, [focusOption, optionCount]);

  const moveActiveOption = useCallback(
    (direction: 1 | -1) => {
      if (optionCount === 0) {
        return;
      }
      setActiveIndex((current) => {
        const startIndex = current >= 0 ? current : fallbackIndex;
        if (
          direction === 1 &&
          startIndex >= optionCount - 1 &&
          onMovePastEnd?.()
        ) {
          return current;
        }
        const nextIndex =
          (startIndex + direction + optionCount) % optionCount;
        focusOption(nextIndex);
        return nextIndex;
      });
    },
    [fallbackIndex, focusOption, onMovePastEnd, optionCount],
  );

  const handleOptionKeyDown = useCallback(
    (event: KeyboardEvent<ElementType>, onActivate: () => void) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onEscape();
        return;
      }
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
      if (event.key === "Home") {
        event.preventDefault();
        focusFirstOption();
        return;
      }
      if (event.key === "End") {
        event.preventDefault();
        focusLastOption();
        return;
      }
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        onActivate();
      }
    },
    [focusFirstOption, focusLastOption, moveActiveOption, onEscape],
  );

  return {
    optionRefs,
    activeIndex,
    setActiveIndex,
    focusOption,
    focusFirstOption,
    focusLastOption,
    moveActiveOption,
    handleOptionKeyDown,
  };
}
