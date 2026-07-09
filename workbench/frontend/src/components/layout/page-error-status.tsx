import { Button } from "@/components/ui/button";
import { StatusCard } from "@/components/ui/status-card";

export function FullPageError({
  message,
  onRetry,
}: {
  message?: string;
  onRetry: () => void;
}) {
  return (
    <StatusCard
      title="Something went wrong"
      detail={
        message ??
        "The workbench hit an unexpected error. You can retry, or check that the backend API is running."
      }
      icon={<span className="text-base font-bold" aria-hidden>!</span>}
      actions={
        <Button variant="primary" onClick={onRetry} className="px-4 font-bold">
          Try again
        </Button>
      }
      layout="page"
    />
  );
}
