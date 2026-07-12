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
  closeButtonLabel = "Close Training Command",
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
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/80 p-region backdrop-blur-md sm:p-shell-wide"
      panelClassName="grid max-h-none w-full max-w-3xl gap-region overflow-visible p-region sm:max-h-none sm:p-shell"
      header={
        <header className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 id="training-command-title" className="type-title text-balance font-semibold text-ink">
              {title}
            </h2>
            <div
              className="mt-1 truncate font-mono text-xs text-ink-faint"
              translate="no"
            >
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
        <span className="text-xs font-bold uppercase tracking-label text-ink-faint">
          Command
        </span>
        <textarea
          name="training-command-readout"
          readOnly
          aria-label={commandAriaLabel}
          value={trainingCommand}
          rows={rows}
          spellCheck={false}
          translate="no"
          className="min-h-24 w-full resize-none rounded-control border border-line bg-control-field px-3 py-2.5 font-mono type-body leading-6 text-ink outline-none focus-visible:border-violet/70 focus-visible:ring-2 focus-visible:ring-focus focus-visible:ring-offset-1 focus-visible:ring-offset-bg"
        />
      </label>
    </DialogShell>
  );

  return typeof document === "undefined" ? dialog : createPortal(dialog, document.body);
}
