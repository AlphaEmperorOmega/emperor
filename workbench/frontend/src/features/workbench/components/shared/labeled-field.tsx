import { type ReactNode } from "react";
import { cn } from "@/lib/utils";

export type LabeledFieldProps = {
  label: ReactNode;
  id?: string;
  detail?: ReactNode;
  children: ReactNode;
  className?: string;
  labelClassName?: string;
};

export function LabeledField({
  label,
  id,
  detail,
  children,
  className,
  labelClassName,
}: LabeledFieldProps) {
  const hasDetail = detail !== undefined && detail !== null;
  const labelContent =
    hasDetail || labelClassName ? (
      <span
        className={cn(
          "flex min-w-0 items-center justify-between gap-2",
          labelClassName,
        )}
      >
        <span className="min-w-0 truncate">{label}</span>
        {hasDetail && (
          <span className="shrink-0 normal-case text-ink-faint">{detail}</span>
        )}
      </span>
    ) : (
      label
    );

  return (
    <div
      className={cn(
        "grid gap-1 text-[11px] font-bold uppercase text-ink-dim",
        className,
      )}
    >
      {id ? (
        <label htmlFor={id}>{labelContent}</label>
      ) : (
        <span>{labelContent}</span>
      )}
      {children}
    </div>
  );
}
