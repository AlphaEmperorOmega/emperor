import { type AriaRole, type ReactNode } from "react";
import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

export type InlineStatusProps = {
  tone?: "default" | "danger" | "warning" | "success";
  busy?: boolean;
  compact?: boolean;
  role?: AriaRole;
  children: ReactNode;
  className?: string;
};

const toneClasses: Record<NonNullable<InlineStatusProps["tone"]>, string> = {
  default: "border-faint bg-white/[0.018] text-ink-faint",
  danger: "border-danger-line bg-danger-soft text-danger-text",
  warning: "border-amber/40 bg-amber/[0.12] text-amber",
  success: "border-ok/30 bg-ok/10 text-ok",
};

export function InlineStatus({
  tone = "default",
  busy = false,
  compact = false,
  role,
  children,
  className,
}: InlineStatusProps) {
  return (
    <div
      role={role ?? (busy ? "status" : undefined)}
      className={cn(
        "rounded-[10px] border border-dashed text-sm",
        compact ? "p-3" : "p-4",
        toneClasses[tone],
        className,
      )}
    >
      {busy && (
        <Loader2 className="mr-2 inline h-4 w-4 animate-spin text-violet" aria-hidden />
      )}
      {children}
    </div>
  );
}
