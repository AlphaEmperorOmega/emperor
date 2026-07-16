import { type InputHTMLAttributes, forwardRef } from "react";
import { Check } from "lucide-react";
import {
  checkboxCheckedClassName,
  checkboxIndicatorClassName,
  checkboxUncheckedClassName,
} from "@/components/ui/control-styles";
import { cn } from "@/lib/utils";

type CheckboxProps = Omit<
  InputHTMLAttributes<HTMLInputElement>,
  "type" | "checked" | "onChange"
> & {
  name: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
};

// Visually-hidden native checkbox paired with a decorative gradient box, so the
// control stays keyboard- and screen-reader-accessible while the surrounding
// element supplies its own label/layout. Renders a fragment (input + box) to
// drop into an existing <label> without adding a wrapper.
export const Checkbox = forwardRef<HTMLInputElement, CheckboxProps>(
  ({ checked, onCheckedChange, className, ...props }, ref) => (
    <>
      <input
        ref={ref}
        type="checkbox"
        checked={checked}
        onChange={() => onCheckedChange(!checked)}
        className="peer sr-only"
        {...props}
      />
      <span
        className={cn(
          checkboxIndicatorClassName,
          "peer-focus-visible:border-violet/70 peer-focus-visible:ring-2 peer-focus-visible:ring-focus",
          checked
            ? checkboxCheckedClassName
            : checkboxUncheckedClassName,
          className,
        )}
        aria-hidden
      >
        <Check
          className={cn("h-3 w-3 text-white transition", !checked && "opacity-0")}
          aria-hidden
        />
      </span>
    </>
  ),
);

Checkbox.displayName = "Checkbox";
