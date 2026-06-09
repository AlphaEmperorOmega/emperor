import { cn } from "@/lib/utils";

// Small glowing status indicator dot (online = green, offline = red).
export function StatusDot({
  online,
  className,
}: {
  online: boolean;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "block h-[7px] w-[7px] rounded-full",
        online
          ? "bg-ok shadow-status-ok"
          : "bg-danger shadow-status-danger",
        className,
      )}
    />
  );
}
