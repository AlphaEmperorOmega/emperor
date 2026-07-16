import { cn } from "@/lib/utils";

// Crisp status indicator with a low-key state ring (online = green, offline = red).
export function StatusDot({
  online,
  className,
}: {
  online: boolean;
  className?: string;
}) {
  return (
    <span
      aria-hidden="true"
      className={cn(
        "block h-2 w-2 rounded-full",
        online
          ? "bg-ok shadow-status-ok"
          : "bg-danger shadow-status-danger",
        className,
      )}
    />
  );
}
