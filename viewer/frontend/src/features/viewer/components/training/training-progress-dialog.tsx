import {
  Camera,
  FilePlus2,
  ListChecks,
  Pencil,
  RefreshCw,
  SlidersHorizontal,
  Trash2,
  X,
} from "lucide-react";
import { useEffect, useId, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { IconButton } from "@/components/ui/icon-button";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { TrainingCommandDialog } from "@/features/viewer/components/config/training-command-dialog";
import { DialogShell } from "@/features/viewer/components/shared/dialog-shell";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { TrainingProgressTable } from "@/features/viewer/components/training/training-progress-table";
import { ViewModeButton } from "@/features/viewer/components/view-mode-button";
import { useCopyToClipboard } from "@/hooks/use-copy-to-clipboard";
import {
  type TrainingRun,
  type TrainingRunPlan,
} from "@/lib/api";
import {
  groupConfigSnapshotsByPreset,
  type ConfigSnapshot,
} from "@/lib/config-snapshots";
import { cn } from "@/lib/utils";

type ProgressTab = "runs" | "snapshots" | "presets";

type SelectOption = {
  value: string;
  label: string;
};

export type TrainingProgressDraftManagement = {
  enabled: boolean;
  snapshots: ConfigSnapshot[];
  presetOptions: SelectOption[];
  selectedPreset: string;
  selectedTrainingPresets: string[];
  selectedTrainingSnapshotIds: string[];
  onIncludeSnapshot: (snapshotId: string) => void;
  onExcludeSnapshot: (snapshotId: string) => void;
  onTogglePreset: (preset: string) => void;
  onExcludePreset: (preset: string) => void;
  onEditPresetAsSnapshot: (preset: string) => void;
  onEditSnapshotCopy: (snapshotId: string) => void;
};

function overrideCountLabel(count: number) {
  return `${count} override${count === 1 ? "" : "s"}`;
}

function SnapshotOverridePills({ snapshot }: { snapshot: ConfigSnapshot }) {
  const entries = Object.entries(snapshot.overrides);
  if (entries.length === 0) {
    return <span className="text-xs text-ink-faint">No overrides</span>;
  }

  return (
    <div className="flex min-w-0 flex-wrap gap-1.5">
      {entries.slice(0, 3).map(([key, value]) => (
        <span
          key={key}
          className="max-w-full truncate rounded-[7px] border border-line bg-white/[0.035] px-2 py-0.5 font-mono text-xs text-ink-dim"
          title={`${key}=${value || "None"}`}
        >
          {key}={value || "None"}
        </span>
      ))}
      {entries.length > 3 && <Badge>+{entries.length - 3}</Badge>}
    </div>
  );
}

function snapshotIncluded(
  snapshot: ConfigSnapshot,
  draft: TrainingProgressDraftManagement,
) {
  return draft.selectedTrainingSnapshotIds.includes(snapshot.id);
}

function TrainingSnapshotsDraftPanel({
  draft,
  canRemoveSnapshots,
  onDeleteSnapshot,
}: {
  draft: TrainingProgressDraftManagement;
  canRemoveSnapshots: boolean;
  onDeleteSnapshot: (snapshotId: string, snapshotName: string) => void;
}) {
  const presetOrder = useMemo(
    () => draft.presetOptions.map((option) => option.value),
    [draft.presetOptions],
  );
  const groups = useMemo(
    () => groupConfigSnapshotsByPreset(draft.snapshots, presetOrder),
    [draft.snapshots, presetOrder],
  );

  if (draft.snapshots.length === 0) {
    return (
      <InlineStatus>
        No config snapshots saved for this model
      </InlineStatus>
    );
  }

  return (
    <div className="grid gap-3">
      {groups.map((group) => {
        const selectedSnapshotCount = group.snapshots.filter((snapshot) =>
          draft.selectedTrainingSnapshotIds.includes(snapshot.id),
        ).length;
        return (
        <section
          key={group.preset}
          className={cn(
            "grid gap-2 rounded-[10px] border p-3",
            selectedSnapshotCount > 0
              ? "border-violet/35 bg-violet/[0.055]"
              : "border-line-soft bg-black/15",
          )}
        >
          <div className="flex min-w-0 items-center justify-between gap-2">
            <span className="truncate font-mono text-xs font-semibold text-ink">
              {group.preset}
            </span>
            <Badge
              className={
                selectedSnapshotCount > 0
                  ? "border-violet/30 bg-violet/15 text-violet"
                  : undefined
              }
            >
              {selectedSnapshotCount} / {group.snapshots.length}
            </Badge>
          </div>
          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
            {group.snapshots.map((snapshot) => {
              const included = snapshotIncluded(snapshot, draft);
              const overrideCount = Object.keys(snapshot.overrides).length;
              return (
                <div
                  key={snapshot.id}
                  className={cn(
                    "grid gap-2 rounded-[10px] border p-2.5",
                    included
                      ? "border-ok/30 bg-ok/[0.07]"
                      : "border-line bg-white/[0.025]",
                  )}
                >
                  <div className="flex min-w-0 items-start justify-between gap-2">
                    <label className="flex min-w-0 items-start gap-2">
                      <Checkbox
                        checked={included}
                        onCheckedChange={(checked) => {
                          if (checked) {
                            draft.onIncludeSnapshot(snapshot.id);
                            return;
                          }
                          draft.onExcludeSnapshot(snapshot.id);
                        }}
                        aria-label={`Include snapshot ${snapshot.name} in training`}
                        className="mt-0.5 shrink-0"
                      />
                      <span className="grid min-w-0 gap-1">
                        <span
                          className="truncate text-sm font-semibold text-ink"
                          title={snapshot.name}
                        >
                          {snapshot.name}
                        </span>
                        <span className="font-mono text-xs text-ink-faint">
                          {snapshot.preset} · {overrideCountLabel(overrideCount)}
                        </span>
                      </span>
                    </label>
                    <div className="flex shrink-0 items-center gap-1">
                      {included && (
                        <Badge variant="success">
                          Included
                        </Badge>
                      )}
                      {canRemoveSnapshots && (
                        <IconButton
                          label={`Delete snapshot ${snapshot.name}`}
                          onClick={() => onDeleteSnapshot(snapshot.id, snapshot.name)}
                          size="sm"
                          variant="danger"
                          className="border-danger-line bg-danger-soft text-danger-text hover:bg-danger-hover/40 hover:text-white"
                          icon={<Trash2 className="h-3.5 w-3.5" aria-hidden />}
                        />
                      )}
                    </div>
                  </div>
                  <SnapshotOverridePills snapshot={snapshot} />
                  <Button
                    variant="secondary"
                    onClick={() => draft.onEditSnapshotCopy(snapshot.id)}
                    className="h-8 justify-center px-2.5 text-xs"
                  >
                    <Pencil className="h-3.5 w-3.5" aria-hidden />
                    Edit Copy
                  </Button>
                </div>
              );
            })}
          </div>
        </section>
        );
      })}
    </div>
  );
}

function TrainingPresetsDraftPanel({
  draft,
}: {
  draft: TrainingProgressDraftManagement;
}) {
  if (draft.presetOptions.length === 0) {
    return (
      <InlineStatus>
        No presets for this model
      </InlineStatus>
    );
  }

  return (
    <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
      {draft.presetOptions.map((preset) => {
        const selected = draft.selectedTrainingPresets.includes(preset.value);
        const snapshotCount = draft.snapshots.filter(
          (snapshot) => snapshot.preset === preset.value,
        ).length;
        return (
          <div
            key={preset.value}
            className={cn(
              "grid gap-2 rounded-[10px] border p-2.5",
              selected
                ? "border-violet/35 bg-violet/[0.055]"
                : "border-line bg-white/[0.025]",
            )}
          >
            <div className="flex min-w-0 items-start justify-between gap-2">
              <label className="flex min-w-0 items-start gap-2">
                <Checkbox
                  checked={selected}
                  onCheckedChange={() => draft.onTogglePreset(preset.value)}
                  aria-label={`Include preset ${preset.label} in training`}
                  className="mt-0.5 shrink-0"
                />
                <span className="grid min-w-0 gap-1">
                  <span
                    className="truncate text-sm font-semibold text-ink"
                    title={preset.label}
                  >
                    {preset.label}
                  </span>
                  <span className="font-mono text-xs text-ink-faint">
                    {snapshotCount} snapshot{snapshotCount === 1 ? "" : "s"}
                  </span>
                </span>
              </label>
              {selected && (
                <Badge
                  className="shrink-0"
                  variant={preset.value === draft.selectedPreset ? "violet" : "success"}
                >
                  {preset.value === draft.selectedPreset ? "Primary" : "Selected"}
                </Badge>
              )}
            </div>
            <Button
              variant="secondary"
              onClick={() => draft.onEditPresetAsSnapshot(preset.value)}
              className="h-8 justify-center px-2.5 text-xs"
            >
              <FilePlus2 className="h-3.5 w-3.5" aria-hidden />
              Edit as Snapshot
            </Button>
          </div>
        );
      })}
    </div>
  );
}

export function TrainingProgressDialog({
  plan,
  isLoading,
  error,
  canResample,
  isResampling,
  onResample,
  canRemoveSnapshots = false,
  onRemoveSnapshot,
  draftManagement,
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
  draftManagement?: TrainingProgressDraftManagement;
  onClose: () => void;
}) {
  const [commandRun, setCommandRun] = useState<TrainingRun | null>(null);
  const [errorRun, setErrorRun] = useState<TrainingRun | null>(null);
  const [pendingDeleteSnapshot, setPendingDeleteSnapshot] = useState<{
    id: string;
    name: string;
  } | null>(null);
  const [activeTab, setActiveTab] = useState<ProgressTab>("runs");
  const progressTabsId = useId();
  const runsTabId = `${progressTabsId}-runs-tab`;
  const snapshotsTabId = `${progressTabsId}-snapshots-tab`;
  const presetsTabId = `${progressTabsId}-presets-tab`;
  const runsPanelId = `${progressTabsId}-runs-panel`;
  const snapshotsPanelId = `${progressTabsId}-snapshots-panel`;
  const presetsPanelId = `${progressTabsId}-presets-panel`;
  const command = commandRun?.command ?? "";
  const fullErrorText = errorRun?.errorTraceback || errorRun?.error || "";
  const { status: copyStatus, copy } = useCopyToClipboard(command);
  const summary = plan?.summary;
  const draftTabsEnabled = Boolean(draftManagement?.enabled);
  const visibleTab = draftTabsEnabled ? activeTab : "runs";
  const draftManagementWithSnapshotHandoff = useMemo<
    TrainingProgressDraftManagement | undefined
  >(() => {
    if (!draftManagement) {
      return undefined;
    }
    return {
      ...draftManagement,
      onEditPresetAsSnapshot: (preset) => {
        draftManagement.onEditPresetAsSnapshot(preset);
        onClose();
      },
      onEditSnapshotCopy: (snapshotId) => {
        draftManagement.onEditSnapshotCopy(snapshotId);
        onClose();
      },
    };
  }, [draftManagement, onClose]);
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

  useEffect(() => {
    if (!draftTabsEnabled && activeTab !== "runs") {
      setActiveTab("runs");
    }
  }, [activeTab, draftTabsEnabled]);

  function confirmDeleteSnapshot() {
    if (!pendingDeleteSnapshot) {
      return;
    }
    onRemoveSnapshot?.(pendingDeleteSnapshot.id);
    setPendingDeleteSnapshot(null);
  }

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
          {pendingDeleteSnapshot && (
            <DialogShell
              titleId="delete-config-snapshot-title"
              size="sm"
              onClose={() => setPendingDeleteSnapshot(null)}
              className="z-[60] grid place-items-center bg-black/65 p-4 sm:p-4"
              panelClassName="grid max-h-none max-w-md gap-4 overflow-visible p-4 sm:max-h-none"
              header={
                <div className="grid gap-1">
                  <h2
                    id="delete-config-snapshot-title"
                    className="text-base font-semibold text-ink"
                  >
                    Delete Snapshot
                  </h2>
                  <p className="text-sm leading-6 text-ink-dim">
                    Delete{" "}
                    <span className="font-mono text-ink">
                      {pendingDeleteSnapshot.name}
                    </span>{" "}
                    permanently?
                  </p>
                </div>
              }
            >
              <div className="flex justify-end gap-2">
                <Button
                  variant="secondary"
                  onClick={() => setPendingDeleteSnapshot(null)}
                >
                  Cancel
                </Button>
                <Button variant="danger" onClick={confirmDeleteSnapshot}>
                  <Trash2 className="h-4 w-4" aria-hidden />
                  Delete Snapshot
                </Button>
              </div>
            </DialogShell>
          )}
        </>
      }
    >
      <div className="full-config-dialog-body min-h-0 flex-1 overflow-auto px-4 py-4 sm:px-5">
        {draftTabsEnabled && (
          <SegmentedControl
            aria-label="Training progress sections"
            className="mb-3"
          >
            <ViewModeButton
              id={runsTabId}
              controls={runsPanelId}
              active={visibleTab === "runs"}
              onClick={() => setActiveTab("runs")}
            >
              <ListChecks className="h-3.5 w-3.5" aria-hidden />
              Runs
            </ViewModeButton>
            <ViewModeButton
              id={snapshotsTabId}
              controls={snapshotsPanelId}
              active={visibleTab === "snapshots"}
              onClick={() => setActiveTab("snapshots")}
            >
              <Camera className="h-3.5 w-3.5" aria-hidden />
              Snapshots
            </ViewModeButton>
            <ViewModeButton
              id={presetsTabId}
              controls={presetsPanelId}
              active={visibleTab === "presets"}
              onClick={() => setActiveTab("presets")}
            >
              <SlidersHorizontal className="h-3.5 w-3.5" aria-hidden />
              Presets
            </ViewModeButton>
          </SegmentedControl>
        )}

        {draftTabsEnabled ? (
          <>
            <div
              id={runsPanelId}
              role="tabpanel"
              aria-labelledby={runsTabId}
              aria-label="Runs"
              hidden={visibleTab !== "runs"}
            >
              {visibleTab !== "runs" ? null : error ? (
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
                  onCommand={setCommandRun}
                  onFullError={setErrorRun}
                  canManageDraftRuns={draftTabsEnabled}
                  onExcludePreset={draftManagement?.onExcludePreset}
                  onExcludeSnapshot={draftManagement?.onExcludeSnapshot}
                />
              )}
            </div>
            <div
              id={snapshotsPanelId}
              role="tabpanel"
              aria-labelledby={snapshotsTabId}
              aria-label="Snapshots"
              hidden={visibleTab !== "snapshots"}
            >
              {visibleTab !== "snapshots" ? null : draftManagementWithSnapshotHandoff ? (
                <TrainingSnapshotsDraftPanel
                  draft={draftManagementWithSnapshotHandoff}
                  canRemoveSnapshots={
                    canRemoveSnapshots && Boolean(onRemoveSnapshot)
                  }
                  onDeleteSnapshot={(snapshotId, snapshotName) =>
                    setPendingDeleteSnapshot({
                      id: snapshotId,
                      name: snapshotName,
                    })
                  }
                />
              ) : (
                <InlineStatus>
                  No draft controls available
                </InlineStatus>
              )}
            </div>
            <div
              id={presetsPanelId}
              role="tabpanel"
              aria-labelledby={presetsTabId}
              aria-label="Presets"
              hidden={visibleTab !== "presets"}
            >
              {visibleTab !== "presets" ? null : draftManagementWithSnapshotHandoff ? (
                <TrainingPresetsDraftPanel
                  draft={draftManagementWithSnapshotHandoff}
                />
              ) : (
                <InlineStatus>
                  No draft controls available
                </InlineStatus>
              )}
            </div>
          </>
        ) : error ? (
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
            onCommand={setCommandRun}
            onFullError={setErrorRun}
            canManageDraftRuns={false}
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
