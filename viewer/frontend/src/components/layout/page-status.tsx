import { Loader2 } from "lucide-react";
import { type ReactNode } from "react";
import { StatusCard } from "@/components/ui/status-card";

type FullPageStatusProps = {
  title: string;
  detail?: string;
  icon: ReactNode;
  action?: ReactNode;
};

export function FullPageStatus({ title, detail, icon, action }: FullPageStatusProps) {
  return (
    <StatusCard title={title} detail={detail} icon={icon} layout="page" actions={action} />
  );
}

export function FullPageLoading() {
  return (
    <FullPageStatus
      title="Loading viewer"
      icon={<Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
    />
  );
}
