import { type ReactNode } from "react";
import { StatusCard } from "@/components/ui/status-card";

export function EmptyState({
  title,
  detail,
  icon,
}: {
  title: string;
  detail?: string;
  icon: ReactNode;
}) {
  return (
    <StatusCard title={title} detail={detail} icon={icon} layout="overlay" />
  );
}
