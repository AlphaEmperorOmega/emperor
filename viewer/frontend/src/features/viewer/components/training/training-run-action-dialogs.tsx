import { X } from "lucide-react";
import { IconButton } from "@/components/ui/icon-button";
import { TrainingCommandDialog } from "@/features/viewer/components/config/training-command-dialog";
import { DialogShell } from "@/features/viewer/components/shared/dialog-shell";
import { useCopyToClipboard } from "@/hooks/use-copy-to-clipboard";
import { type TrainingRun, type TrainingRunPlan } from "@/lib/api";

export function TrainingRunActionDialogs({
  commandRun,
  errorRun,
  plan,
  onCloseCommand,
  onCloseError,
}: {
  commandRun: TrainingRun | null;
  errorRun: TrainingRun | null;
  plan?: TrainingRunPlan;
  onCloseCommand: () => void;
  onCloseError: () => void;
}) {
  const command = commandRun?.command ?? "";
  const fullErrorText = errorRun?.errorTraceback || errorRun?.error || "";
  const { status: copyStatus, copy } = useCopyToClipboard(command);

  return (
    <>
      {commandRun && (
        <TrainingCommandDialog
          model={plan?.model ?? ""}
          preset={`${commandRun.preset} / ${commandRun.dataset}`}
          trainingCommand={command}
          copyStatus={copyStatus}
          onCopy={copy}
          onClose={onCloseCommand}
        />
      )}
      {errorRun && (
        <DialogShell
          titleId="training-run-error-title"
          size="lg"
          onClose={onCloseError}
          className="z-[60]"
          panelClassName="grid max-w-5xl grid-rows-[auto_minmax(0,1fr)]"
          header={
            <header className="flex items-start justify-between gap-3 border-b border-line-soft px-4 py-3 sm:px-5">
              <div className="min-w-0">
                <h2
                  id="training-run-error-title"
                  className="text-base font-semibold text-ink"
                >
                  Training Error
                </h2>
                <div className="mt-1 flex min-w-0 flex-wrap items-center gap-1.5 text-xs text-ink-faint">
                  <span className="font-mono">run {errorRun.index}</span>
                  <span aria-hidden>/</span>
                  <span className="font-mono">{errorRun.preset}</span>
                  <span aria-hidden>/</span>
                  <span className="font-mono">{errorRun.dataset}</span>
                </div>
              </div>
              <IconButton
                label="Close training error"
                onClick={onCloseError}
                variant="edge"
                className="border-line-soft bg-white/[0.025] hover:bg-white/[0.055]"
                icon={<X className="h-4 w-4" aria-hidden />}
              />
            </header>
          }
        >
          <div className="min-h-0 overflow-auto p-4 sm:p-5">
            <pre className="min-h-[18rem] whitespace-pre-wrap rounded-[10px] border border-danger-line bg-black/35 p-3 font-mono text-xs leading-5 text-danger-text">
              {fullErrorText}
            </pre>
          </div>
        </DialogShell>
      )}
    </>
  );
}
