import { RefreshCw, X } from "lucide-react";
import { useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import { TrainingCommandDialog } from "@/features/viewer/components/config/training-command-dialog";
import { DialogShell } from "@/features/viewer/components/shared/dialog-shell";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { TrainingProgressTable } from "@/features/viewer/components/training/training-progress-table";
import { useCopyToClipboard } from "@/hooks/use-copy-to-clipboard";
import {
  type TrainingRun,
  type TrainingRunPlan,
} from "@/lib/api";

export function TrainingProgressDialog({
  plan,
  isLoading,
  error,
  canResample,
  isResampling,
  onResample,
  canRemoveSnapshots = false,
  onRemoveSnapshot,
  onClose,
}: {
  plan: TrainingRunPlan | undefined;
  isLoading: boolean;
  error: string;
  canResample: boolean;
  isResampling: boolean;
  onResample: () => void;
  canRemoveSnapshots?: boolean;
  onRemoveSnapshot?: (snapshotId: string) => void;
  onClose: () => void;
}) {
  const [commandRun, setCommandRun] = useState<TrainingRun | null>(null);
  const [errorRun, setErrorRun] = useState<TrainingRun | null>(null);
  const command = commandRun?.command ?? "";
  const fullErrorText = errorRun?.errorTraceback || errorRun?.error || "";
  const { status: copyStatus, copy } = useCopyToClipboard(command);
  const summary = plan?.summary;
  const summaryText = useMemo(() => {
    if (isLoading) {
      return "Planning...";
    }
    if (error) {
      return "Plan error";
    }
    if (!summary) {
      return "No run plan";
    }
    return `${summary.completedRuns} / ${summary.totalRuns} runs · ${summary.remainingEpochs} epochs left`;
  }, [error, isLoading, summary]);

  const dialog = (
    <DialogShell
      size="fullscreen"
      titleId="training-progress-title"
      onClose={onClose}
      panelClassName="full-config-dialog-shell relative"
      header={
        <header className="full-config-dialog-chrome full-config-dialog-header sticky top-0 z-10 border-b border-line-soft px-4 py-3 backdrop-blur sm:px-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0">
              <h2 id="training-progress-title" className="text-base font-semibold text-ink">
                Training Progress
              </h2>
              <div className="mt-1 flex min-w-0 flex-wrap items-center gap-1.5 text-xs text-ink-faint">
                <span className="max-w-full truncate font-mono">
                  {plan?.model ?? "No model"}
                </span>
                {plan?.preset && <span aria-hidden>/</span>}
                {plan?.preset && (
                  <span className="max-w-full truncate font-mono">
                    {plan.preset}
                  </span>
                )}
                <span aria-hidden>·</span>
                <span>{summaryText}</span>
              </div>
            </div>
            <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
              {summary && (
                <>
                  <Badge>{summary.totalRuns} runs</Badge>
                  <Badge>{summary.remainingEpochs} epochs left</Badge>
                </>
              )}
              {canResample && (
                <Button
                  variant="secondary"
                  onClick={onResample}
                  disabled={isResampling}
                >
                  <RefreshCw
                    className={isResampling ? "h-4 w-4 animate-spin" : "h-4 w-4"}
                    aria-hidden
                  />
                  Resample
                </Button>
              )}
              <IconButton
                label="Close training progress"
                onClick={onClose}
                variant="edge"
                className="border-line-soft bg-white/[0.025] hover:bg-white/[0.055]"
                icon={<X className="h-4 w-4" aria-hidden />}
              />
            </div>
          </div>
        </header>
      }
      overlayChildren={
        <>
          {commandRun && (
            <TrainingCommandDialog
              model={plan?.model ?? ""}
              preset={`${commandRun.preset} / ${commandRun.dataset}`}
              trainingCommand={command}
              copyStatus={copyStatus}
              onCopy={copy}
              onClose={() => setCommandRun(null)}
            />
          )}
          {errorRun && (
            <DialogShell
              titleId="training-run-error-title"
              size="lg"
              onClose={() => setErrorRun(null)}
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
                    onClick={() => setErrorRun(null)}
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
      }
    >
      <div className="full-config-dialog-body min-h-0 flex-1 overflow-auto px-4 py-4 sm:px-5">
        {error ? (
          <div
            role="alert"
            className="rounded-[10px] border border-danger-line bg-danger-soft p-3 text-sm text-danger-text"
          >
            {error}
          </div>
        ) : isLoading ? (
          <InlineStatus>
            Planning training runs
          </InlineStatus>
        ) : !plan || plan.runs.length === 0 ? (
          <InlineStatus>
            No training runs planned
          </InlineStatus>
        ) : (
          <TrainingProgressTable
            runs={plan.runs}
            canRemoveSnapshots={canRemoveSnapshots}
            onRemoveSnapshot={onRemoveSnapshot}
            onCommand={setCommandRun}
            onFullError={setErrorRun}
          />
        )}
      </div>
    </DialogShell>
  );

  if (typeof document === "undefined") {
    return dialog;
  }

  return createPortal(dialog, document.body);
}
