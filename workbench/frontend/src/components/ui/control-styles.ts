import { cn } from "@/lib/utils";

export const controlFocusClassName =
  "focus:outline-none focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-2 focus-visible:ring-offset-bg";

export const fieldControlFocusClassName =
  "outline-none focus-visible:border-violet/70 focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-1 focus-visible:ring-offset-bg";

export const controlDisabledClassName =
  "disabled:cursor-not-allowed disabled:opacity-50";

export const controlButtonClassName = [
  "inline-flex h-touch items-center justify-center gap-2 rounded-control px-3 type-body font-semibold md:h-control",
  "transition-[color,background-color,border-color,box-shadow,transform,opacity] duration-150 ease-out",
  controlFocusClassName,
  controlDisabledClassName,
].join(" ");

export const iconButtonClassName = [
  "inline-flex shrink-0 items-center justify-center border transition-[color,background-color,border-color,box-shadow,transform,opacity] duration-150 ease-out",
  controlFocusClassName,
  controlDisabledClassName,
].join(" ");

export const fieldControlClassName = [
  "w-full min-w-0 rounded-control border border-line bg-control-field text-ink shadow-control",
  "transition-[color,background-color,border-color,box-shadow] duration-150 ease-out hover:border-line-hover hover:bg-black/40",
  fieldControlFocusClassName,
  "disabled:cursor-not-allowed disabled:opacity-60",
].join(" ");

export const selectTriggerClassName = [
  "grid h-touch w-full min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-2 md:h-control-lg",
  "rounded-control border border-line bg-panel-2/90 px-3 text-left font-sans shadow-control",
  "type-body font-semibold text-ink transition-[color,background-color,border-color,box-shadow] duration-150 ease-out hover:border-line-hover hover:bg-control-hover",
  fieldControlFocusClassName,
  controlDisabledClassName,
].join(" ");

export const multiSelectTriggerClassName = [
  "grid min-h-touch w-full min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-2",
  "rounded-control border border-line bg-panel-2/90 px-3 py-2 text-left font-sans shadow-control",
  "type-body font-semibold text-ink transition-[color,background-color,border-color,box-shadow] duration-150 ease-out hover:border-line-hover hover:bg-control-hover",
  fieldControlFocusClassName,
  controlDisabledClassName,
].join(" ");

export const selectTriggerActiveClassName =
  "border-violet/55 bg-accent-soft shadow-control-selected";

export const dropdownOptionClassName = [
  "min-h-touch w-full min-w-0 border-b border-line-soft px-3 py-2.5 text-left type-body font-semibold [content-visibility:auto] [contain-intrinsic-size:44px] md:min-h-control-lg",
  "transition-colors duration-150 last:border-b-0 focus:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-focus",
].join(" ");

export function dropdownOptionStateClassName({
  active,
  disabled = false,
  selected,
}: {
  active: boolean;
  disabled?: boolean;
  selected: boolean;
}) {
  return cn(
    selected
      ? "bg-accent-soft text-violet-text"
      : active
        ? "bg-control-active text-ink"
        : "bg-transparent text-ink hover:bg-control-hover",
    disabled && "cursor-default opacity-75",
  );
}

export const checkboxIndicatorClassName =
  "grid h-5 w-5 place-items-center rounded-chip border transition-[color,background-color,border-color,box-shadow] duration-150 ease-out";

export const checkboxCheckedClassName =
  "border-violet/70 bg-violet-deep shadow-control-checked";

export const checkboxUncheckedClassName =
  "border-line-hover bg-control-field hover:border-violet/50";
