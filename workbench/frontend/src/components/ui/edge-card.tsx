import { type HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

type EdgeCardProps = HTMLAttributes<HTMLDivElement> & {
  selected?: boolean;
};

export function EdgeCard({ className, selected = false, ...props }: EdgeCardProps) {
  return (
    <div
      className={cn("edge", selected && "edge-sel", className)}
      {...props}
    />
  );
}
