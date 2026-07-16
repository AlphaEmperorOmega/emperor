import { useCallback, useRef, useState } from "react";

export type CopyStatus = "idle" | "copied" | "failed";

type CopyCompletion = {
  text: string;
  status: Exclude<CopyStatus, "idle">;
};

// Copy `text` to the clipboard, exposing a status only for the current text.
export function useCopyToClipboard(text: string) {
  const [completion, setCompletion] = useState<CopyCompletion | null>(null);
  const attemptRef = useRef(0);

  const copy = useCallback(async () => {
    const attempt = ++attemptRef.current;
    const complete = (status: CopyCompletion["status"]) => {
      if (attempt === attemptRef.current) {
        setCompletion({ text, status });
      }
    };
    if (!navigator.clipboard?.writeText) {
      complete("failed");
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      complete("copied");
    } catch {
      complete("failed");
    }
  }, [text]);

  const status: CopyStatus =
    completion?.text === text ? completion.status : "idle";
  return { status, copy };
}
