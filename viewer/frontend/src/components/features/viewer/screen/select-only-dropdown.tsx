import {
  type KeyboardEvent,
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { ChevronDown } from "lucide-react";
import { DropdownShell } from "@/components/features/viewer/shared/dropdown-shell";
import { usePopupDismissal } from "@/components/features/viewer/shared/use-popup-dismissal";
import { cn } from "@/lib/utils";

export type SelectOnlyDropdownOption = {
  value: string;
  label: string;
};

export function SelectOnlyDropdown({
  id,
  label,
  value,
  options,
  onChange,
  placeholder = "Select",
  className,
}: {
  id?: string;
  label: string;
  value: string;
  options: SelectOnlyDropdownOption[];
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
}) {
  const generatedId = useId();
  const triggerId = id ?? `${generatedId}-select`;
  const listboxId = `${triggerId}-options`;
  const rootRef = useRef<HTMLDivElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const selectedIndex = useMemo(
    () => options.findIndex((option) => option.value === value),
    [options, value],
  );
  const selectedOption = selectedIndex >= 0 ? options[selectedIndex] : undefined;
  const [isOpen, setIsOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(
    selectedIndex >= 0 ? selectedIndex : options.length > 0 ? 0 : -1,
  );
  const disabled = options.length === 0;
  const activeOptionId =
    isOpen && activeIndex >= 0 ? `${listboxId}-option-${activeIndex}` : undefined;

  const moveActiveOption = useCallback(
    (direction: 1 | -1) => {
      if (options.length === 0) {
        return;
      }
      setIsOpen(true);
      setActiveIndex((current) => {
        const startIndex = current >= 0 ? current : selectedIndex >= 0 ? selectedIndex : 0;
        return (startIndex + direction + options.length) % options.length;
      });
    },
    [options.length, selectedIndex],
  );

  const openDropdown = useCallback(() => {
    if (disabled) {
      return;
    }
    setActiveIndex(selectedIndex >= 0 ? selectedIndex : 0);
    setIsOpen(true);
  }, [disabled, selectedIndex]);

  const closeDropdown = useCallback((restoreFocus = false) => {
    setIsOpen(false);
    if (restoreFocus) {
      triggerRef.current?.focus();
    }
  }, []);

  const selectOption = useCallback(
    (option: SelectOnlyDropdownOption) => {
      setActiveIndex(options.findIndex((candidate) => candidate.value === option.value));
      setIsOpen(false);
      if (option.value !== value) {
        onChange(option.value);
      }
      triggerRef.current?.focus();
    },
    [onChange, options, value],
  );

  useEffect(() => {
    setIsOpen(false);
    setActiveIndex(selectedIndex >= 0 ? selectedIndex : options.length > 0 ? 0 : -1);
  }, [options.length, selectedIndex, value]);

  usePopupDismissal({
    open: isOpen,
    onClose: closeDropdown,
    triggerRef,
    panelRef,
  });

  function handleKeyDown(event: KeyboardEvent<HTMLButtonElement>) {
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
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      if (!isOpen) {
        openDropdown();
        return;
      }
      const activeOption = options[activeIndex];
      if (activeOption) {
        selectOption(activeOption);
      }
      return;
    }
    if (event.key === "Escape" && isOpen) {
      event.preventDefault();
      closeDropdown(true);
    }
  }

  return (
    <div ref={rootRef} className={cn("relative min-w-0", isOpen ? "z-30" : "z-20", className)}>
      <button
        ref={triggerRef}
        id={triggerId}
        type="button"
        role="combobox"
        aria-label={label}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls={listboxId}
        aria-activedescendant={activeOptionId}
        disabled={disabled}
        onClick={() => {
          if (isOpen) {
            closeDropdown();
            return;
          }
          openDropdown();
        }}
        onBlur={(event) => {
          const nextTarget = event.relatedTarget as Node | null;
          if (nextTarget && rootRef.current?.contains(nextTarget)) {
            return;
          }
          closeDropdown();
        }}
        onKeyDown={handleKeyDown}
        className={cn(
          "grid h-10 w-full min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-2 rounded-[10px] border border-line bg-[linear-gradient(155deg,#161622,#0e0e18)] px-3 text-left font-sans text-[13.5px] font-semibold text-ink outline-none transition hover:border-white/15 focus-visible:border-violet/60 focus-visible:ring-2 focus-visible:ring-focus disabled:cursor-not-allowed disabled:opacity-50",
          isOpen &&
            "border-violet/45 bg-[linear-gradient(135deg,rgba(146,113,255,0.1),rgba(111,168,255,0.05))]",
        )}
      >
        <span className="min-w-0 truncate">
          {(selectedOption?.label ?? value) || placeholder}
        </span>
        <ChevronDown
          className={cn(
            "h-4 w-4 shrink-0 text-ink-faint transition",
            isOpen && "rotate-180 text-ink",
          )}
          aria-hidden
        />
      </button>

      {isOpen && (
        <DropdownShell
          ref={panelRef}
          id={listboxId}
          role="listbox"
          ariaLabel={`${label} options`}
          className="max-h-[260px] overflow-y-auto"
        >
          {options.map((option, index) => {
            const isActive = index === activeIndex;
            const isSelected = option.value === value;
            return (
              <button
                key={option.value}
                id={`${listboxId}-option-${index}`}
                type="button"
                role="option"
                aria-selected={isSelected || isActive}
                tabIndex={-1}
                onMouseDown={(event) => event.preventDefault()}
                onMouseEnter={() => setActiveIndex(index)}
                onClick={() => selectOption(option)}
                className={cn(
                  "block w-full min-w-0 border-b border-line-soft px-3 py-2.5 text-left text-sm font-semibold transition last:border-b-0 focus:outline-none",
                  isSelected
                    ? "bg-violet/14 text-violet"
                    : isActive
                      ? "bg-white/[0.045] text-white"
                      : "bg-transparent text-white hover:bg-white/[0.045]",
                )}
              >
                <span className="block min-w-0 truncate">{option.label}</span>
              </button>
            );
          })}
        </DropdownShell>
      )}
    </div>
  );
}
