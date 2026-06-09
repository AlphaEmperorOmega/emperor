import { ButtonHTMLAttributes } from "react";
import { controlFocusClassName } from "@/components/ui/control-styles";
import { cn } from "@/lib/utils";

type SwitchProps = Omit<ButtonHTMLAttributes<HTMLButtonElement>, "value" | "onChange"> & {
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
        "relative h-6 w-11 rounded-full border transition disabled:cursor-not-allowed disabled:opacity-60",
        controlFocusClassName,
        checked
          ? "border-violet/40 bg-grad shadow-switch-checked"
          : "border-line bg-control-track",
        className,
      )}
      onClick={() => onCheckedChange(!checked)}
      {...props}
    >
      <span
        className={cn(
          "absolute top-0.5 h-5 w-5 rounded-full bg-white shadow-sm transition-transform",
          checked ? "left-5" : "left-0.5",
        )}
      />
    </button>
  );
}
