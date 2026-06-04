import { type InputHTMLAttributes, forwardRef } from "react";
import { Check } from "lucide-react";
import { cn } from "@/lib/utils";

type CheckboxProps = Omit<
  InputHTMLAttributes<HTMLInputElement>,
  "type" | "checked" | "onChange"
> & {
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
        className="sr-only"
        {...props}
      />
      <span
        className={cn(
          "grid h-[19px] w-[19px] place-items-center rounded-md border-[1.6px] transition",
          checked
            ? "border-transparent bg-grad shadow-[0_3px_10px_-3px_rgba(124,109,255,0.8)]"
            : "border-white/20 bg-transparent",
          className,
        )}
        aria-hidden
      >
        <Check className={cn("h-3 w-3 text-white transition", !checked && "opacity-0")} />
      </span>
    </>
  ),
);

Checkbox.displayName = "Checkbox";
