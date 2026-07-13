import { useMemo, useState } from "react";
import {
  AlertTriangle,
  Copy,
  FolderOpen,
  Terminal,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import { TrainingCommandDialog } from "@/features/workbench/components/config/training-command-dialog";
import { InlineStatus } from "@/features/workbench/components/shared/inline-status";
import { TrainingRunActionDialogs } from "@/features/workbench/components/training/training-run-action-dialogs";
import {
  commandForShell,
  TrainingShellSelector,
  type TrainingShell,
  useTrainingShell,
} from "@/features/workbench/components/training/training-shell-selector";
import {
  formatTrainingMetricsText,
  getTrainingRunDraftRemoval,
  trainingRunStatusMeta,
} from "@/features/workbench/components/training/training-run-display";
import { useCopyToClipboard } from "@/hooks/use-copy-to-clipboard";
import { type TrainingRun, type TrainingRunPlan } from "@/lib/api";
import { cn } from "@/lib/utils";

const COMPACT_RUN_LIMIT = 160;
const COMPACT_CHANGE_LIMIT = 4;

function epochText(run: TrainingRun) {
  return `${run.currentEpoch} / ${run.totalEpochs || "-"} epochs`;
}

function visibleRuns(runs: TrainingRun[]) {
  const keep = new Set(runs.slice(0, COMPACT_RUN_LIMIT).map((run) => run.id));
  for (const run of runs) {
    if (
      run.status === "Running" ||
      run.status === "Failed" ||
      run.status === "Cancelled"
    ) {
      keep.add(run.id);
    }
  }
  return runs.filter((run) => keep.has(run.id));
}

function runnableTrainingCommands(
  plan: TrainingRunPlan | undefined,
  shell: TrainingShell,
) {
  return (plan?.runs ?? []).flatMap((run) => {
    const command = commandForShell(run, shell);
    return command.trim() ? [command] : [];
  });
}

function trainingCommandBlock(commands: string[], shell: TrainingShell) {
  if (shell === "powershell") {
    return [
      "$ErrorActionPreference = 'Stop'",
      "& {",
      ...commands.map((command) => `  ${command}`),
      "}",
    ].join("\n");
  }
  return ["(", "  set -e", ...commands.map((command) => `  ${command}`), ")"].join(
    "\n",
  );
}

export function TrainingAllCommandsButton({
  className,
  plan,
}: {
  className?: string;
  plan?: TrainingRunPlan;
}) {
  const [isAllCommandsOpen, setIsAllCommandsOpen] = useState(false);
  const { shell, setShell } = useTrainingShell();
  const runnableCommands = useMemo(
    () => runnableTrainingCommands(plan, shell),
    [plan, shell],
  );
  const allTrainingCommandsBlock = useMemo(
    () => trainingCommandBlock(runnableCommands, shell),
    [runnableCommands, shell],
  );
  const canCopyAllCommands = runnableCommands.length > 0;

  if (!canCopyAllCommands) {
    return null;
  }

  return (
    <>
      <Button
        variant="secondary"
        onClick={() => setIsAllCommandsOpen(true)}
        className={cn("h-10 px-3 type-compact", className)}
        title="Review training commands"
      >
        <Copy className="h-4 w-4" aria-hidden />
        Commands
      </Button>
      {isAllCommandsOpen && (
        <AllTrainingCommandsDialog
          model={plan?.model ?? ""}
          preset={plan?.preset ?? ""}
          shell={shell}
          trainingCommand={allTrainingCommandsBlock}
          onShellChange={setShell}
          onClose={() => setIsAllCommandsOpen(false)}
        />
      )}
    </>
  );
}

function AllTrainingCommandsDialog({
  model,
  preset,
  shell,
  trainingCommand,
  onShellChange,
  onClose,
}: {
  model: string;
  preset: string;
  shell: TrainingShell;
  trainingCommand: string;
  onShellChange: (shell: TrainingShell) => void;
  onClose: () => void;
}) {
  const { status, copy } = useCopyToClipboard(trainingCommand);

  return (
    <TrainingCommandDialog
      title="Training Commands"
      model={model}
      preset={preset}
      trainingCommand={trainingCommand}
      copyStatus={status}
      copyButtonLabel="Copy Commands"
      copiedMessage="Commands copied"
      commandAriaLabel="Training commands"
      closeButtonLabel="Close Training Commands"
      rows={Math.min(12, Math.max(5, trainingCommand.split("\n").length))}
      controls={<TrainingShellSelector shell={shell} onChange={onShellChange} />}
      onCopy={copy}
      onClose={onClose}
    />
  );
}

function RunChangePills({ run }: { run: TrainingRun }) {
  if (run.changes.length === 0) {
    return <span className="font-mono text-xs text-ink-faint">No changes</span>;
  }

  const visibleChanges = run.changes.slice(0, COMPACT_CHANGE_LIMIT);
  const hiddenCount = run.changes.length - visibleChanges.length;

  return (
    <div className="flex min-w-0 flex-wrap gap-1.5">
      {visibleChanges.map((change) => (
        <Badge
          key={`${change.source}-${change.key}-${String(change.value)}`}
          variant={change.source === "search" ? "violet" : "default"}
          className="max-w-[11rem]"
          title={`${change.label}: ${String(change.value)}`}
        >
          <span className="truncate">
            {change.key}={String(change.value)}
          </span>
        </Badge>
      ))}
      {hiddenCount > 0 && <Badge>+{hiddenCount}</Badge>}
    </div>
  );
}

function RunStatus({ run }: { run: TrainingRun }) {
  const meta = trainingRunStatusMeta[run.status];
  const Icon = meta.icon;

  return (
    <span
      className={cn(
        "inline-flex h-7 items-center gap-1.5 rounded-control-md border px-2 font-mono text-xs font-bold",
        meta.className,
      )}
      title={meta.tooltip}
    >
      <Icon className={cn("h-3.5 w-3.5", meta.iconClassName)} aria-hidden />
      {run.status}
    </span>
  );
}

function RunActions({
  canManageDraftRuns,
  onCommand,
  onExcludePreset,
  onExcludeSnapshot,
  onFullError,
  run,
}: {
  canManageDraftRuns: boolean;
  onCommand: (run: TrainingRun) => void;
  onFullError: (run: TrainingRun) => void;
  onExcludePreset?: (preset: string) => void;
  onExcludeSnapshot?: (snapshotId: string) => void;
  run: TrainingRun;
}) {
  const fullError = run.errorTraceback || run.error;
  const draftRemoval = getTrainingRunDraftRemoval({
    run,
    canManageDraftRuns,
    onExcludePreset,
    onExcludeSnapshot,
  });

  return (
    <div className="flex shrink-0 items-center gap-1.5">
      {run.logDir && (
        <IconButton
          label={`Copy log path for run ${run.index}`}
          icon={<FolderOpen className="h-3.5 w-3.5" aria-hidden />}
          size="sm"
          variant="edge"
          className="h-touch w-touch rounded-control-sm border-ok/30 bg-ok/10 text-ok hover:bg-ok/15 md:h-control-sm md:w-control-sm"
          onClick={() => {
            void navigator.clipboard?.writeText(run.logDir ?? "");
          }}
        />
      )}
      {fullError && (
        <IconButton
          label={`Full error for run ${run.index}`}
          icon={<AlertTriangle className="h-3.5 w-3.5" aria-hidden />}
          size="sm"
          variant="danger"
          className="h-touch w-touch rounded-control-sm md:h-control-sm md:w-control-sm"
          onClick={() => onFullError(run)}
        />
      )}
      <IconButton
        label={`Command for run ${run.index}`}
        icon={<Terminal className="h-3.5 w-3.5" aria-hidden />}
        size="sm"
        variant="edge"
        className="h-touch w-touch rounded-control-sm md:h-control-sm md:w-control-sm"
        onClick={() => onCommand(run)}
      />
      {draftRemoval && (
        <IconButton
          label={draftRemoval.label}
          icon={<X className="h-3.5 w-3.5" aria-hidden />}
          size="sm"
          variant="ghost"
          className="h-touch w-touch rounded-control-sm text-ink-faint hover:border-danger-line hover:bg-danger-soft hover:text-danger-text md:h-control-sm md:w-control-sm"
          onClick={draftRemoval.remove}
        />
      )}
    </div>
  );
}

function TrainingCompactRunRow({
  canManageDraftRuns,
  onCommand,
  onExcludePreset,
  onExcludeSnapshot,
  onFullError,
  run,
}: {
  canManageDraftRuns: boolean;
  onCommand: (run: TrainingRun) => void;
  onFullError: (run: TrainingRun) => void;
  onExcludePreset?: (preset: string) => void;
  onExcludeSnapshot?: (snapshotId: string) => void;
  run: TrainingRun;
}) {
  const metrics = formatTrainingMetricsText(run.metrics, 3);
  const snapshotLabel = run.snapshotName ?? run.snapshotId ?? "";

  return (
    <article className="grid min-w-0 gap-2 border-b border-line-soft px-3 py-2.5 [content-visibility:auto] [contain-intrinsic-size:112px] last:border-b-0">
      <div className="grid min-w-0 gap-2 sm:grid-cols-[auto_minmax(0,1fr)_auto] sm:items-start">
        <div className="flex min-w-0 items-center gap-2">
          <Badge className="shrink-0">#{run.index}</Badge>
          <RunStatus run={run} />
        </div>
        <div className="grid min-w-0 gap-1">
          <div className="grid min-w-0 gap-1 font-mono text-xs text-ink sm:grid-cols-[minmax(0,1.1fr)_minmax(0,0.95fr)]">
            <span className="truncate" title={run.preset}>
              {run.preset}
            </span>
            <span className="truncate text-ink-dim" title={run.dataset}>
              {run.dataset}
            </span>
          </div>
          {snapshotLabel && (
            <span
              className="truncate font-mono type-meta text-violet-muted"
              title={snapshotLabel}
            >
              {snapshotLabel}
            </span>
          )}
        </div>
        <RunActions
          run={run}
          onCommand={onCommand}
          onFullError={onFullError}
          canManageDraftRuns={canManageDraftRuns}
          onExcludePreset={onExcludePreset}
          onExcludeSnapshot={onExcludeSnapshot}
        />
      </div>

      <div className="grid min-w-0 gap-2 text-xs md:grid-cols-[minmax(0,1fr)_minmax(8rem,0.45fr)_minmax(0,0.85fr)] md:items-center">
        <RunChangePills run={run} />
        <span className="whitespace-nowrap font-mono text-violet-muted">
          {epochText(run)}
        </span>
        <span className="truncate font-mono text-ink-dim" title={metrics}>
          {metrics}
        </span>
      </div>

      {run.error && (
        <button
          type="button"
          onClick={() => onFullError(run)}
          className="min-h-touch min-w-0 truncate rounded-control-md border border-danger-line bg-danger-soft px-2 py-1 text-left text-xs font-semibold text-danger-text transition hover:bg-danger-hover/40 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus md:min-h-control-sm"
          title={run.errorTraceback ?? run.error}
        >
          {run.error}
        </button>
      )}
    </article>
  );
}

export function TrainingCompactRunList({
  plan,
  isLoading = false,
  error = "",
  canManageDraftRuns = false,
  onExcludePreset,
  onExcludeSnapshot,
}: {
  plan?: TrainingRunPlan;
  isLoading?: boolean;
  error?: string;
  canManageDraftRuns?: boolean;
  onExcludePreset?: (preset: string) => void;
  onExcludeSnapshot?: (snapshotId: string) => void;
}) {
  const [commandRun, setCommandRun] = useState<TrainingRun | null>(null);
  const [errorRun, setErrorRun] = useState<TrainingRun | null>(null);
  const runs = useMemo(() => visibleRuns(plan?.runs ?? []), [plan?.runs]);
  const hiddenRunCount = Math.max(0, (plan?.runs.length ?? 0) - runs.length);

  return (
    <div className="grid h-full min-h-0 grid-rows-[minmax(0,1fr)] overflow-hidden rounded-b-control border border-line bg-black/10">
      <div className="min-h-0 overflow-y-auto">
        {error ? (
          <InlineStatus tone="danger" compact role="alert" className="m-3">
            {error}
          </InlineStatus>
        ) : isLoading ? (
          <InlineStatus busy compact className="m-3">
            Planning training runs…
          </InlineStatus>
        ) : !plan || plan.runs.length === 0 ? (
          <InlineStatus compact className="m-3">
            No training runs planned
          </InlineStatus>
        ) : (
          <>
            {runs.map((run) => (
              <TrainingCompactRunRow
                key={run.id}
                run={run}
                onCommand={setCommandRun}
                onFullError={setErrorRun}
                canManageDraftRuns={canManageDraftRuns}
                onExcludePreset={onExcludePreset}
                onExcludeSnapshot={onExcludeSnapshot}
              />
            ))}
            {hiddenRunCount > 0 && (
              <div className="border-t border-line-soft px-3 py-3 text-center text-xs text-ink-faint">
                Showing {runs.length} of {plan.runs.length} planned runs.
                Running, failed, and cancelled rows stay visible.
              </div>
            )}
          </>
        )}
      </div>

      <TrainingRunActionDialogs
        plan={plan}
        commandRun={commandRun}
        errorRun={errorRun}
        onCloseCommand={() => setCommandRun(null)}
        onCloseError={() => setErrorRun(null)}
      />
    </div>
  );
}
