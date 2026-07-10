import { cn } from "@/lib/utils";

export const controlFocusClassName =
  "focus:outline-none focus-visible:ring-2 focus-visible:ring-focus";

export const fieldControlFocusClassName =
  "outline-none focus-visible:border-violet/60 focus-visible:ring-2 focus-visible:ring-focus";

export const controlDisabledClassName =
  "disabled:cursor-not-allowed disabled:opacity-50";

export const controlButtonClassName = [
  "inline-flex h-9 items-center justify-center gap-2 rounded-control px-3 text-sm font-semibold",
  "transition",
  controlFocusClassName,
  controlDisabledClassName,
].join(" ");

export const iconButtonClassName = [
  "inline-flex shrink-0 items-center justify-center border transition",
  controlFocusClassName,
  controlDisabledClassName,
].join(" ");

export const fieldControlClassName = [
  "w-full min-w-0 rounded-control border border-line bg-control-field text-ink transition",
  "hover:border-line-hover",
  fieldControlFocusClassName,
  "disabled:cursor-not-allowed disabled:opacity-60",
].join(" ");

export const selectTriggerClassName = [
  "grid h-10 w-full min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-2",
  "rounded-control border border-line bg-control-chrome px-3 text-left font-sans",
  "text-[13.5px] font-semibold text-ink transition hover:border-line-hover",
  fieldControlFocusClassName,
  controlDisabledClassName,
].join(" ");

export const multiSelectTriggerClassName = [
  "grid min-h-[46px] w-full min-w-0 grid-cols-[minmax(0,1fr)_auto] items-center gap-2",
  "rounded-control border border-line bg-control-chrome px-3 py-2 text-left font-sans",
  "text-[13.5px] font-semibold text-ink transition hover:border-line-hover",
  fieldControlFocusClassName,
  controlDisabledClassName,
].join(" ");

export const selectTriggerActiveClassName =
  "border-violet/45 bg-control-selected";

export const dropdownOptionClassName = [
  "w-full min-w-0 border-b border-line-soft px-3 py-2.5 text-left text-sm font-semibold",
  "transition-colors last:border-b-0 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus",
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
      ? "bg-violet/14 text-violet"
      : active
        ? "bg-control-active text-white"
        : "bg-transparent text-white hover:bg-control-active",
    disabled && "cursor-default opacity-75",
  );
}

export const checkboxIndicatorClassName =
  "grid h-[19px] w-[19px] place-items-center rounded-md border-[1.6px] transition";

export const checkboxCheckedClassName =
  "border-transparent bg-grad shadow-control-checked";

export const checkboxUncheckedClassName = "border-white/20 bg-transparent";
