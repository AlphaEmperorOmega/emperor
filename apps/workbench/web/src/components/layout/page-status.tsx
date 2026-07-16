import { Loader2 } from "lucide-react";
import { type ReactNode } from "react";
import { StatusCard } from "@/components/ui/status-card";

type FullPageStatusProps = {
  title: string;
  detail?: string;
  icon: ReactNode;
  action?: ReactNode;
  busy?: boolean;
};

export function FullPageStatus({
  title,
  detail,
  icon,
  action,
  busy = false,
}: FullPageStatusProps) {
  return (
    <StatusCard
      title={title}
      detail={detail}
      icon={icon}
      busy={busy}
      layout="page"
      actions={action}
    />
  );
}

export function FullPageLoading() {
  return (
    <FullPageStatus
      title="Loading Workbench…"
      icon={<Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
      busy
    />
  );
}
