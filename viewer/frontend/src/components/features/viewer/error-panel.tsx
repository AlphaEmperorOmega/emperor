import { AlertCircle } from "lucide-react";
import { StatusCard } from "@/components/features/viewer/shared/status-card";

export function ErrorPanel({ title, message }: { title: string; message: string }) {
  return (
    <StatusCard
      title={title}
      detail={message}
      icon={<AlertCircle className="h-4 w-4" aria-hidden />}
      tone="danger"
      layout="inline"
    />
  );
}
