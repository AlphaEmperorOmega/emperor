import { StatusCard } from "@/components/features/viewer/shared/status-card";

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
        "The viewer hit an unexpected error. You can retry, or check that the backend API is running."
      }
      icon={<span className="text-base font-bold" aria-hidden>!</span>}
      actions={
        <button
          type="button"
          onClick={onRetry}
          className="inline-flex h-9 items-center justify-center rounded-ctl bg-grad px-4 text-sm font-bold text-white shadow-primary"
        >
          Try again
        </button>
      }
      layout="page"
    />
  );
}
