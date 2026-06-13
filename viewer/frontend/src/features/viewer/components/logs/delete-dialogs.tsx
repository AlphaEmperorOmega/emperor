import { useState } from "react";
import { AlertTriangle, Loader2, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { IconButton } from "@/components/ui/icon-button";
import { Input } from "@/components/ui/input";
import { ErrorPanel } from "@/features/viewer/components/error-panel";
import { DialogShell } from "@/features/viewer/components/shared/dialog-shell";
import { type LogRunDeleteFilters, type LogRunDeletePlan } from "@/lib/api";
import { type ChecklistOption } from "@/features/viewer/state/logs/logs-selectors";
import { errorMessage } from "@/lib/utils";

export type SubsetDeleteKind = "dataset" | "preset";

export type SubsetDeleteTarget = {
  kind: SubsetDeleteKind;
  value: string;
  experiment: string;
  filters: LogRunDeleteFilters;
  key: string;
};

export function DeleteExperimentDialog({
  option,
  error,
  isDeleting,
  onClose,
  onConfirm,
}: {
  option: ChecklistOption;
  error: unknown;
  isDeleting: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void>;
}) {
  const [confirmation, setConfirmation] = useState("");
  const canDelete = confirmation === option.value && !isDeleting;
  const runCount = option.count ?? 0;

  return (
    <DialogShell
      titleId="delete-experiment-title"
      size="sm"
      onClose={onClose}
      closeOnEscape={!isDeleting}
      panelClassName="grid max-h-none max-w-lg gap-4 overflow-visible p-4 sm:max-h-none sm:p-5"
    >
      <header className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 id="delete-experiment-title" className="text-base font-semibold text-ink">
            Delete Experiment
          </h2>
          <div className="mt-1 truncate font-mono text-xs text-ink-faint">
            logs/{option.value}
          </div>
        </div>
        <IconButton
          label="Close delete experiment"
          onClick={onClose}
          disabled={isDeleting}
          variant="edge"
          icon={<X className="h-4 w-4" aria-hidden />}
        />
      </header>

      <div className="grid gap-3 text-sm leading-6 text-ink-dim">
        <p>
          This permanently deletes {runCount} {runCount === 1 ? "run" : "runs"} from this
          experiment and removes the whole experiment folder from disk.
        </p>
        <label className="grid gap-2">
          <span className="text-xs font-bold uppercase tracking-[0.09em] text-ink-faint">
            Type experiment name
          </span>
          <Input
            value={confirmation}
            onChange={(event) => setConfirmation(event.target.value)}
            aria-label="Type experiment name"
            autoComplete="off"
            disabled={isDeleting}
            placeholder={option.value}
          />
        </label>
      </div>

      {error ? (
        <ErrorPanel title="Delete failed" message={errorMessage(error)} />
      ) : null}

      <footer className="flex flex-wrap items-center justify-end gap-2">
        <Button variant="ghost" onClick={onClose} disabled={isDeleting}>
          Cancel
        </Button>
        <Button
          variant="danger"
          onClick={onConfirm}
          disabled={!canDelete}
        >
          {isDeleting && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
          Delete Experiment
        </Button>
      </footer>
    </DialogShell>
  );
}

function AffectedValueGroup({
  label,
  values,
}: {
  label: string;
  values: string[];
}) {
  return (
    <div className="grid gap-1.5">
      <div className="text-[11px] font-bold uppercase tracking-[0.09em] text-ink-faint">
        {label}
      </div>
      <div className="flex flex-wrap gap-1.5">
        {values.length === 0 ? (
          <span className="font-mono text-xs text-ink-faint">None</span>
        ) : (
          values.slice(0, 8).map((value) => (
            <Badge key={value} className="max-w-full truncate">
              {value}
            </Badge>
          ))
        )}
        {values.length > 8 && <Badge>+{values.length - 8}</Badge>}
      </div>
    </div>
  );
}

function DialogStatus({
  title,
  detail,
  busy,
}: {
  title: string;
  detail: string;
  busy?: boolean;
}) {
  return (
    <div className="rounded-[13px] border border-line-soft bg-white/[0.018] px-3 py-4 text-center">
      <div className="grid justify-items-center gap-2">
        {busy && <Loader2 className="h-5 w-5 animate-spin text-violet" aria-hidden />}
        <div className="text-sm font-semibold text-ink">{title}</div>
        <div className="text-xs leading-5 text-ink-faint">{detail}</div>
      </div>
    </div>
  );
}

export function DeleteSubsetRunsDialog({
  target,
  plan,
  error,
  isPlanning,
  isDeleting,
  onClose,
  onConfirm,
}: {
  target: SubsetDeleteTarget;
  plan: LogRunDeletePlan | undefined;
  error: unknown;
  isPlanning: boolean;
  isDeleting: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void>;
}) {
  const targetLabel = target.kind === "dataset" ? "Dataset" : "Preset";
  const title = `Delete ${targetLabel}`;
  const runCount = plan?.candidateCount ?? 0;
  const previewCandidates = plan?.candidates.slice(0, 8) ?? [];
  const overflowCount = Math.max(0, runCount - previewCandidates.length);
  const blockers = plan?.blockedByActiveJobs ?? [];
  const canDelete =
    Boolean(plan?.canDelete) &&
    blockers.length === 0 &&
    !isPlanning &&
    !isDeleting;
  const dialogTitleId = `delete-${target.kind}-title`;

  return (
    <DialogShell
      titleId={dialogTitleId}
      size="md"
      onClose={onClose}
      closeOnEscape={!isDeleting}
      panelClassName="grid max-h-[min(760px,calc(100vh-32px))] max-w-2xl gap-4 overflow-y-auto p-4 sm:p-5"
    >
      <header className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <h2
              id={dialogTitleId}
              className="text-base font-semibold text-ink"
            >
              {title}
            </h2>
            <div className="mt-1 font-mono text-xs text-ink-faint">
              {isPlanning ? "Planning matched version folders" : `${runCount} matched runs`}
            </div>
          </div>
          <IconButton
            label={`Close delete ${target.kind}`}
            onClick={onClose}
            disabled={isDeleting}
            variant="edge"
            icon={<X className="h-4 w-4" aria-hidden />}
          />
      </header>

        {isPlanning ? (
          <DialogStatus
            title="Planning delete"
            detail="Matching the selected experiment and row."
            busy
          />
        ) : plan ? (
          <div className="grid gap-4">
            <div className="grid gap-3 text-sm leading-6 text-ink-dim">
              <p>
                This permanently removes only the matched <code>version_*</code> run
                folders. Other runs and non-empty parent folders are left on disk.
              </p>
              <div className="grid gap-2 rounded-[12px] border border-danger-line/70 bg-danger-soft p-3 sm:grid-cols-3">
                <div className="min-w-0">
                  <div className="text-[11px] font-bold uppercase tracking-[0.09em] text-[#fda4af]">
                    Experiment
                  </div>
                  <div className="mt-1 truncate font-mono text-xs text-ink">
                    {target.experiment}
                  </div>
                </div>
                <div className="min-w-0">
                  <div className="text-[11px] font-bold uppercase tracking-[0.09em] text-[#fda4af]">
                    {targetLabel}
                  </div>
                  <div className="mt-1 truncate font-mono text-xs text-ink">
                    {target.value}
                  </div>
                </div>
                <div className="min-w-0">
                  <div className="text-[11px] font-bold uppercase tracking-[0.09em] text-[#fda4af]">
                    Matched Runs
                  </div>
                  <div className="mt-1 font-mono text-xs text-ink">{runCount}</div>
                </div>
              </div>
              <div className="grid gap-3 rounded-[12px] border border-line-soft bg-black/20 p-3 sm:grid-cols-2">
                <AffectedValueGroup label="Experiments" values={plan.affected.experiments} />
                <AffectedValueGroup label="Datasets" values={plan.affected.datasets} />
                <AffectedValueGroup label="Models" values={plan.affected.models} />
                <AffectedValueGroup label="Presets" values={plan.affected.presets} />
              </div>
            </div>

            {blockers.length > 0 && (
              <div className="grid gap-2 rounded-[12px] border border-danger-line bg-danger-soft p-3 text-sm text-[#fecdd3]">
                <div className="flex items-start gap-2 font-semibold text-[#fda4af]">
                  <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
                  <span>A training job is still writing to this log folder.</span>
                </div>
                <div className="grid gap-1.5">
                  {blockers.map((blocker) => (
                    <div
                      key={`${blocker.id}-${blocker.logFolder}`}
                      className="grid gap-1 rounded-[9px] border border-danger-line/70 bg-black/20 px-2.5 py-2 font-mono text-xs sm:grid-cols-[minmax(0,1fr)_auto]"
                    >
                      <span className="min-w-0 truncate">
                        {blocker.id} · logs/{blocker.logFolder}
                      </span>
                      <span>{blocker.status}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <div className="grid gap-2">
              <div className="text-xs font-bold uppercase tracking-[0.09em] text-ink-faint">
                Path Preview
              </div>
              <div className="grid max-h-56 gap-1.5 overflow-y-auto rounded-[12px] border border-line-soft bg-black/20 p-2">
                {previewCandidates.length === 0 ? (
                  <div className="px-2 py-4 text-center text-sm text-ink-faint">
                    No matched run folders
                  </div>
                ) : (
                  previewCandidates.map((candidate) => (
                    <div
                      key={candidate.id}
                      className="truncate rounded-[8px] border border-line-soft bg-white/[0.025] px-2.5 py-1.5 font-mono text-xs text-ink-dim"
                      title={candidate.relativePath}
                    >
                      {candidate.relativePath}
                    </div>
                  ))
                )}
                {overflowCount > 0 && (
                  <div className="px-2 py-1 font-mono text-xs text-ink-faint">
                    +{overflowCount} more
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : null}

        {error ? (
          <ErrorPanel title="Delete failed" message={errorMessage(error)} />
        ) : null}

        <footer className="flex flex-wrap items-center justify-end gap-2">
          <Button variant="ghost" onClick={onClose} disabled={isDeleting}>
            Cancel
          </Button>
          <Button variant="danger" onClick={onConfirm} disabled={!canDelete}>
            {isDeleting && <Loader2 className="h-4 w-4 animate-spin" aria-hidden />}
            {title}
          </Button>
        </footer>
    </DialogShell>
  );
}
