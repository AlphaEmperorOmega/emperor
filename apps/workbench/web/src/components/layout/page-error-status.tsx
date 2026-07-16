import { Button } from "@/components/ui/button";
import { StatusCard } from "@/components/ui/status-card";

export function FullPageError({
  message,
  onRetry,
  reference,
}: {
  message?: string;
  onRetry: () => void;
  reference?: string;
}) {
  const detail =
    message ??
    "The workbench hit an unexpected error. Retry the operation, and check the application logs if it continues.";
  return (
    <StatusCard
      title="Something Went Wrong"
      detail={
        <>
          <span className="block">{detail}</span>
          {reference && (
            <span className="mt-2 block font-mono type-caption text-ink-faint">
              Reference: {reference}
            </span>
          )}
        </>
      }
      icon={<span className="text-base font-bold" aria-hidden>!</span>}
      actions={
        <Button variant="primary" onClick={onRetry} className="px-4 font-bold">
          Try Again
        </Button>
      }
      layout="page"
    />
  );
}
