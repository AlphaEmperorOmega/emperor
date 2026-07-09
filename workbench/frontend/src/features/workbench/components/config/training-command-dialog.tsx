import { type ReactNode } from "react";
import { createPortal } from "react-dom";
import { Copy, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import { DialogShell } from "@/features/workbench/components/shared/dialog-shell";
import { type CopyStatus } from "@/hooks/use-copy-to-clipboard";

export function TrainingCommandDialog({
  title = "Training Command",
  model,
  preset,
  trainingCommand,
  copyStatus,
  copyButtonLabel = "Copy Command",
  copiedMessage = "Command copied",
  commandAriaLabel = "Training command",
  closeButtonLabel = "Close training command",
  rows = 3,
  controls,
  footerStart,
  onCopy,
  onClose,
}: {
  title?: string;
  model: string;
  preset: string;
  trainingCommand: string;
  copyStatus: CopyStatus;
  copyButtonLabel?: string;
  copiedMessage?: string;
  commandAriaLabel?: string;
  closeButtonLabel?: string;
  rows?: number;
  controls?: ReactNode;
  footerStart?: ReactNode;
  onCopy: () => void;
  onClose: () => void;
}) {
  const dialog = (
    <DialogShell
      titleId="training-command-title"
      panelVariant="surface"
      onClose={onClose}
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/55 p-3 backdrop-blur-sm sm:p-6"
      panelClassName="grid max-h-none w-full max-w-3xl gap-4 overflow-visible p-4 sm:max-h-none sm:p-5"
      header={
        <header className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 id="training-command-title" className="text-base font-semibold text-ink">
              {title}
            </h2>
            <div className="mt-1 truncate font-mono text-xs text-ink-faint">
              {model || "No model"} {preset ? `/ ${preset}` : ""}
            </div>
          </div>
          <IconButton
            label={closeButtonLabel}
            onClick={onClose}
            variant="edge"
            icon={<X className="h-4 w-4" aria-hidden />}
          />
        </header>
      }
      footer={
        <footer className="flex flex-wrap items-center justify-between gap-2">
          <div className="flex min-w-0 flex-wrap items-center gap-2">
            {footerStart}
            <div
              role={copyStatus === "failed" ? "alert" : "status"}
              className="min-h-5 text-xs font-medium text-ink-faint"
            >
              {copyStatus === "copied" && copiedMessage}
              {copyStatus === "failed" && "Clipboard copy failed"}
            </div>
          </div>
          <Button variant="primary" onClick={onCopy}>
            <Copy className="h-4 w-4" aria-hidden />
            {copyButtonLabel}
          </Button>
        </footer>
      }
    >
      {controls}
      <label className="grid gap-2">
        <span className="text-xs font-bold uppercase tracking-[0.09em] text-ink-faint">
          Command
        </span>
        <textarea
          readOnly
          aria-label={commandAriaLabel}
          value={trainingCommand}
          rows={rows}
          className="min-h-24 w-full resize-none rounded-[10px] border border-line bg-black/25 px-3 py-2.5 font-mono text-sm leading-6 text-ink outline-none focus-visible:border-violet/60 focus-visible:ring-2 focus-visible:ring-focus"
        />
      </label>
    </DialogShell>
  );

  return typeof document === "undefined" ? dialog : createPortal(dialog, document.body);
}
