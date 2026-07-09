import { useCallback, useEffect, useState } from "react";

export type CopyStatus = "idle" | "copied" | "failed";

// Copy `text` to the clipboard, exposing a transient status that resets to
// "idle" whenever `text` changes.
export function useCopyToClipboard(text: string) {
  const [status, setStatus] = useState<CopyStatus>("idle");

  useEffect(() => {
    setStatus("idle");
  }, [text]);

  const copy = useCallback(async () => {
    if (!navigator.clipboard?.writeText) {
      setStatus("failed");
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      setStatus("copied");
    } catch {
      setStatus("failed");
    }
  }, [text]);

  return { status, copy };
}
