import { type HTMLAttributes, type KeyboardEvent } from "react";
import { cn } from "@/lib/utils";

type SegmentedControlProps = Omit<HTMLAttributes<HTMLDivElement>, "role"> & {
  variant?: "radiogroup" | "tablist";
};

const radioNavigationKeys = new Set([
  "ArrowDown",
  "ArrowLeft",
  "ArrowRight",
  "ArrowUp",
  "End",
  "Home",
]);
const tabNavigationKeys = new Set(["ArrowLeft", "ArrowRight", "End", "Home"]);

export function SegmentedControl({
  className,
  onKeyDown,
  variant = "radiogroup",
  ...props
}: SegmentedControlProps) {
  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    onKeyDown?.(event);
    const navigationKeys =
      variant === "radiogroup" ? radioNavigationKeys : tabNavigationKeys;
    if (
      event.defaultPrevented ||
      !navigationKeys.has(event.key)
    ) {
      return;
    }

    const optionRole = variant === "radiogroup" ? "radio" : "tab";
    const options = Array.from(
      event.currentTarget.querySelectorAll<HTMLButtonElement>(
        `button[role="${optionRole}"]`,
      ),
    ).filter((option) => !option.disabled);
    if (options.length === 0) {
      return;
    }

    event.preventDefault();
    const currentIndex = options.indexOf(document.activeElement as HTMLButtonElement);
    const startIndex = currentIndex >= 0 ? currentIndex : 0;
    const nextIndex =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? options.length - 1
          : event.key === "ArrowLeft" || event.key === "ArrowUp"
            ? (startIndex - 1 + options.length) % options.length
            : (startIndex + 1) % options.length;
    const nextOption = options[nextIndex];
    nextOption?.focus();
    nextOption?.click();
  }

  return (
    <div
      role={variant}
      onKeyDown={handleKeyDown}
      className={cn(
        "inline-flex min-w-0 rounded-control border border-line bg-control-field p-1 shadow-control",
        className,
      )}
      {...props}
    />
  );
}
