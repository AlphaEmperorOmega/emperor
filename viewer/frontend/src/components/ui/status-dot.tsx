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
          ? "bg-ok shadow-[0_0_10px_1px_rgba(86,214,160,0.8)]"
          : "bg-[#fb7185] shadow-[0_0_10px_1px_rgba(251,113,133,0.55)]",
        className,
      )}
    />
  );
}
