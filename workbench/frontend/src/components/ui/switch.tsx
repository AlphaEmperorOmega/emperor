import { ButtonHTMLAttributes } from "react";
import { controlFocusClassName } from "@/components/ui/control-styles";
import { cn } from "@/lib/utils";

type SwitchAccessibleName =
  | { "aria-label": string; "aria-labelledby"?: never }
  | { "aria-label"?: never; "aria-labelledby": string };

type SwitchProps = Omit<
  ButtonHTMLAttributes<HTMLButtonElement>,
  "value" | "onChange" | "aria-label" | "aria-labelledby"
> & SwitchAccessibleName & {
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
};

export function Switch({
  checked,
  onCheckedChange,
  className,
  ...props
}: SwitchProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      className={cn(
        "relative h-touch w-touch shrink-0 rounded-full bg-transparent transition-[color,background-color,border-color,box-shadow] duration-150 ease-out disabled:cursor-not-allowed disabled:opacity-60 md:h-control-sm",
        controlFocusClassName,
        className,
      )}
      onClick={() => onCheckedChange(!checked)}
      {...props}
    >
      <span
        aria-hidden
        className={cn(
          "absolute left-1/2 top-1/2 h-6 w-control-lg -translate-x-1/2 -translate-y-1/2 rounded-full border transition-[color,background-color,border-color,box-shadow] duration-150 ease-out",
          checked
            ? "border-violet/70 bg-violet-deep shadow-switch-checked"
            : "border-line bg-control-track",
        )}
      />
      <span
        aria-hidden
        className={cn(
          "absolute top-1/2 h-5 w-5 -translate-y-1/2 rounded-full bg-white shadow-control-lift transition-[left,transform] duration-150 ease-out",
          checked ? "left-5" : "left-1",
        )}
      />
    </button>
  );
}
