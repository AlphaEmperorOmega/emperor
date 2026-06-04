import { Copy, X } from "lucide-react";
import { Button } from "@/components/ui/button";
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
    <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/55 p-3 backdrop-blur-sm sm:p-6">
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="training-command-title"
        className="edge grid w-full max-w-3xl gap-4 rounded-card p-4 shadow-[0_20px_70px_rgba(0,0,0,0.55)] sm:p-5"
      >
        <header className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2 id="training-command-title" className="text-base font-semibold text-ink">
              Training Command
            </h2>
            <div className="mt-1 truncate font-mono text-xs text-ink-faint">
              {model || "No model"} {preset ? `/ ${preset}` : ""}
            </div>
          </div>
          <button
            type="button"
            aria-label="Close training command"
            onClick={onClose}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-[10px] border border-line bg-white/[0.035] text-ink-faint transition hover:bg-white/[0.07] hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          >
            <X className="h-4 w-4" aria-hidden />
          </button>
        </header>

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
      </section>
    </div>
  );
}
