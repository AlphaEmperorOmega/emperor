"use client";
// Client boundary: Next.js error routes receive a browser reset callback.

import { useEffect } from "react";
import { FullPageError } from "@/components/layout/page-error-status";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return <FullPageError message={error.message} onRetry={reset} />;
}
