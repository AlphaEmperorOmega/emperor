import { Copy, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import { DialogShell } from "@/components/features/viewer/shared/dialog-shell";
import { type CopyStatus } from "@/hooks/use-copy-to-clipboard";

export function TrainingCommandDialog({
  model,
  preset,
  trainingCommand,
  copyStatus,
  onCopy,
  onClose,
}: {
  model: string;
  preset: string;
  trainingCommand: string;
  copyStatus: CopyStatus;
  onCopy: () => void;
  onClose: () => void;
}) {
  return (
    <DialogShell
      titleId="training-command-title"
      className="absolute inset-0 z-20 flex items-center justify-center bg-black/55 p-3 backdrop-blur-sm sm:p-6"
      panelClassName="edge grid max-h-none w-full max-w-3xl gap-4 overflow-visible rounded-card p-4 shadow-[0_20px_70px_rgba(0,0,0,0.55)] sm:max-h-none sm:p-5"
      header={
        <header className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 id="training-command-title" className="text-base font-semibold text-ink">
              Training Command
            </h2>
            <div className="mt-1 truncate font-mono text-xs text-ink-faint">
              {model || "No model"} {preset ? `/ ${preset}` : ""}
            </div>
          </div>
          <IconButton
            label="Close training command"
            onClick={onClose}
            variant="edge"
            icon={<X className="h-4 w-4" aria-hidden />}
          />
        </header>
      }
      footer={
        <footer className="flex flex-wrap items-center justify-between gap-2">
          <div
            role={copyStatus === "failed" ? "alert" : "status"}
            className="min-h-5 text-xs font-medium text-ink-faint"
          >
            {copyStatus === "copied" && "Command copied"}
            {copyStatus === "failed" && "Clipboard copy failed"}
          </div>
          <Button variant="primary" onClick={onCopy}>
            <Copy className="h-4 w-4" aria-hidden />
            Copy Command
          </Button>
        </footer>
      }
    >
      <label className="grid gap-2">
        <span className="text-xs font-bold uppercase tracking-[0.09em] text-ink-faint">
          Command
        </span>
        <textarea
          readOnly
          aria-label="Training command"
          value={trainingCommand}
          rows={3}
          className="min-h-24 w-full resize-none rounded-[10px] border border-line bg-black/25 px-3 py-2.5 font-mono text-sm leading-6 text-ink outline-none focus-visible:border-violet/60 focus-visible:ring-2 focus-visible:ring-focus"
        />
      </label>
    </DialogShell>
  );
}
