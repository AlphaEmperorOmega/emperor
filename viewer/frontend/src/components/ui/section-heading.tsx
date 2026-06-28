import { type ReactNode } from "react";
import { cn } from "@/lib/utils";

type SectionHeadingElement = "div" | "h2" | "h3";

export type SectionHeadingProps = {
  as?: SectionHeadingElement;
  icon?: ReactNode;
  title: ReactNode;
  count?: ReactNode;
  actions?: ReactNode;
  className?: string;
};

export function SectionHeading({
  as: Component = "div",
  icon,
  title,
  count,
  actions,
  className,
}: SectionHeadingProps) {
  return (
    <Component
      className={cn(
        "flex items-center gap-2 text-xs font-bold uppercase tracking-[0.09em] text-ink-dim",
        className,
      )}
    >
      {icon}
      {title}
      {count}
      {actions}
    </Component>
  );
}
