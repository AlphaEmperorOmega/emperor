import { useEffect, useMemo, useState, type KeyboardEvent } from "react";
import {
  Camera,
  Check,
  FilePlus2,
  Loader2,
  Pencil,
  Play,
  SlidersHorizontal,
  Trash2,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { IconButton } from "@/components/ui/icon-button";
import { Input } from "@/components/ui/input";
import { SnapshotRestoreDialog } from "@/features/workbench/components/config/config-snapshot-dialogs";
import { DialogShell } from "@/features/workbench/components/shared/dialog-shell";
import { InlineStatus } from "@/features/workbench/components/shared/inline-status";
import { SectionHeading } from "@/components/ui/section-heading";
import {
  SurfacePanel,
  surfacePanelClassName,
} from "@/components/ui/surface-panel";
import {
  configSnapshotOverrideCount,
  configSnapshotOverrideCountLabel,
  configSnapshotOverrideEntries,
  draftMatchesConfigSnapshot,
  validateConfigSnapshotCandidate,
  validateConfigSnapshotName,
  type ConfigSnapshot,
  type ConfigSnapshotCreateResult,
  type ConfigSnapshotGroup,
} from "@/lib/config-snapshots";
import { type ConfigField } from "@/lib/api";
import { type OverrideValues } from "@/lib/config";
import { cn } from "@/lib/utils";
import {
  type ConfigSnapshotMutationOutcome,
  type ConfigSnapshotMutationStatus,
} from "@/features/workbench/state/config-snapshots/use-config-snapshot-records";

type ConfigSnapshotActionResult =
  | { ok: true }
  | { ok: false; error: string };

type ConfigSnapshotActionHandlerResult =
  | ConfigSnapshotActionResult
  | void
  | Promise<ConfigSnapshotActionResult | void>;

const idleConfigSnapshotMutation: ConfigSnapshotMutationStatus = {
  phase: "idle",
  kind: null,
  snapshotId: null,
  error: "",
  canRetry: false,
};

function SnapshotOverrideSummary({ overrides }: { overrides: OverrideValues }) {
  const entries = Object.entries(overrides);
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

export function AddConfigSnapshotDialog({
  modelType,
  model,
  preset,
  fields,
  overrides,
  snapshots,
  title = "Add Config Snapshot",
  actionLabel = "Add Snapshot",
  initialName,
  excludeSnapshotId,
  mutation = idleConfigSnapshotMutation,
  onAdd,
  onRetry,
  onDismissMutation,
  onClose,
}: {
  modelType: string;
  model: string;
  preset: string;
  fields: ConfigField[];
  overrides: OverrideValues;
  snapshots: ConfigSnapshot[];
  title?: string;
  actionLabel?: string;
  initialName?: string;
  excludeSnapshotId?: string;
  mutation?: ConfigSnapshotMutationStatus;
  onAdd: (name: string) => Promise<ConfigSnapshotCreateResult>;
  onRetry?: () => Promise<ConfigSnapshotCreateResult>;
  onDismissMutation?: () => void;
  onClose: () => void;
}) {
  const defaultName = useMemo(() => initialName ?? "", [initialName]);
  const [name, setName] = useState(defaultName);
  const [submittedError, setSubmittedError] = useState("");
  const nameValidation = useMemo(
    () =>
      validateConfigSnapshotName({
        modelType,
        model,
        preset,
        name,
        snapshots,
        excludeSnapshotId,
      }),
    [excludeSnapshotId, model, modelType, name, preset, snapshots],
  );
  const candidate = useMemo(
    () =>
      validateConfigSnapshotCandidate({
        modelType,
        model,
        preset,
        fields,
        overrides,
        snapshots,
        excludeSnapshotId,
      }),
    [excludeSnapshotId, fields, model, modelType, overrides, preset, snapshots],
  );
  const { entries } = useMemo(
    () => configSnapshotOverrideEntries(fields, overrides),
    [fields, overrides],
  );
  const error =
    submittedError ||
    (mutation.phase === "failed" ? mutation.error : "") ||
    (!nameValidation.ok
      ? nameValidation.error
      : !candidate.ok
        ? candidate.error
        : "");

  useEffect(() => {
    setName(defaultName);
    setSubmittedError("");
  }, [defaultName]);

  const isPending = mutation.phase === "pending";
  const canRetry =
    mutation.phase === "failed" && mutation.canRetry && Boolean(onRetry);

  async function confirm() {
    if (isPending) {
      return;
    }
    setSubmittedError("");
    const result = canRetry && onRetry ? await onRetry() : await onAdd(name);
    if (!result.ok) {
      setSubmittedError(result.error);
      return;
    }
    onDismissMutation?.();
    onClose();
  }

  function close() {
    if (isPending) {
      return;
    }
    onDismissMutation?.();
    onClose();
  }

  return (
    <DialogShell
      titleId="add-config-snapshot-title"
      size="sm"
      panelVariant="surface"
      onClose={close}
      closeOnEscape={!isPending}
      className="z-[60] grid place-items-center bg-black/65 p-4 sm:p-4"
      panelClassName="grid max-h-none max-w-lg gap-4 overflow-visible p-4 sm:max-h-none"
    >
      <div className="flex items-start justify-between gap-3">
        <div className="grid gap-1">
          <h2
            id="add-config-snapshot-title"
            className="text-base font-semibold text-ink"
          >
            {title}
          </h2>
          <div className="flex flex-wrap items-center gap-1.5 text-xs text-ink-faint">
            <span className="font-mono">{model || "No model"}</span>
            {preset && <span aria-hidden>/</span>}
            {preset && <span className="font-mono">{preset}</span>}
          </div>
        </div>
        <IconButton
          label="Close add config snapshot"
          onClick={close}
          disabled={isPending}
          size="sm"
          variant="edge"
          className="rounded-[9px] border-line-soft bg-white/[0.025] hover:bg-white/[0.055]"
          icon={<X className="h-4 w-4" aria-hidden />}
        />
      </div>

      <label className="grid gap-1.5">
        <span className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
          Name
        </span>
        <Input
          value={name}
          placeholder="Unique snapshot name"
          onChange={(event) => {
            setName(event.target.value);
            setSubmittedError("");
            onDismissMutation?.();
          }}
          disabled={isPending}
          autoFocus
        />
      </label>

      <SurfacePanel
        icon={<SlidersHorizontal className="h-[15px] w-[15px] text-violet" aria-hidden />}
        className="gap-2 p-3"
        title="Changed fields"
        detail={<Badge>{configSnapshotOverrideCountLabel(entries.length)}</Badge>}
      >
        {entries.length > 0 ? (
          <div className="grid max-h-40 gap-1.5 overflow-y-auto pr-1">
            {entries.map((entry) => (
              <div
                key={entry.key}
                className="flex min-w-0 items-center justify-between gap-2 rounded-[8px] border border-line-soft bg-black/18 px-2 py-1.5 text-xs"
              >
                <span className="truncate text-ink">{entry.label}</span>
                <span className="max-w-[12rem] truncate font-mono text-violet">
                  {entry.displayValue}
                </span>
              </div>
            ))}
          </div>
        ) : (
          <div
            className={cn(
              surfacePanelClassName,
              "border-dashed border-faint bg-black/18 px-2 py-2 text-xs text-ink-faint",
            )}
          >
            No non-default draft overrides.
          </div>
        )}
      </SurfacePanel>

      {error && (
        <div
          role="alert"
          className="rounded-[9px] border border-danger-line bg-danger-soft px-3 py-2 text-sm text-danger-text"
        >
          {error}
        </div>
      )}

      <div className="flex justify-end gap-2">
        <Button variant="secondary" onClick={close} disabled={isPending}>
          Cancel
        </Button>
        <Button
          variant="primary"
          onClick={() => void confirm()}
          disabled={isPending || !nameValidation.ok || !candidate.ok}
        >
          {isPending ? (
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden />
          ) : (
            <FilePlus2 className="h-4 w-4" aria-hidden />
          )}
          {isPending ? "Saving…" : canRetry ? "Retry Save" : actionLabel}
        </Button>
      </div>
    </DialogShell>
  );
}

function SnapshotNameEditor({
  initialName,
  pending,
  onCancel,
  onSave,
}: {
  initialName: string;
  pending: boolean;
  onCancel: () => void;
  onSave: (name: string) => void;
}) {
  const [value, setValue] = useState(initialName);

  function save() {
    onSave(value);
  }

  function keyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === "Enter") {
      event.preventDefault();
      save();
    }
    if (event.key === "Escape") {
      event.preventDefault();
      onCancel();
    }
  }

  return (
    <div className="flex min-w-0 items-center gap-1.5">
      <Input
        value={value}
        aria-label="Snapshot name"
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={keyDown}
        disabled={pending}
        className="h-8 text-xs"
        autoFocus
      />
      <IconButton
        label="Save snapshot name"
        onClick={save}
        disabled={pending}
        size="sm"
        variant="edge"
        className="border-ok/30 bg-ok/10 text-ok hover:bg-ok/15 hover:text-ok"
        icon={<Check className="h-3.5 w-3.5" aria-hidden />}
      />
      <IconButton
        label="Cancel snapshot rename"
        onClick={onCancel}
        disabled={pending}
        size="sm"
        variant="edge"
        className="bg-white/[0.025] hover:bg-white/[0.055]"
        icon={<X className="h-3.5 w-3.5" aria-hidden />}
      />
    </div>
  );
}

export function ConfigSnapshotsTray({
  groups,
  selectedPreset,
  selectedTrainingSnapshotIds,
  overrides,
  canManage,
  mutation = idleConfigSnapshotMutation,
  onLoad,
  onRename,
  onRemove,
  onRetryMutation = async () => null,
  onDismissMutation = () => undefined,
  onToggleSelection,
}: {
  groups: ConfigSnapshotGroup[];
  selectedPreset: string;
  selectedTrainingSnapshotIds: string[];
  overrides: OverrideValues;
  canManage: boolean;
  mutation?: ConfigSnapshotMutationStatus;
  onLoad: (snapshotId: string) => void;
  onRename: (
    snapshotId: string,
    name: string,
  ) => ConfigSnapshotActionHandlerResult;
  onRemove: (snapshotId: string) => ConfigSnapshotActionHandlerResult;
  onRetryMutation?: () => Promise<ConfigSnapshotMutationOutcome | null>;
  onDismissMutation?: () => void;
  onToggleSelection: (snapshotId: string) => void;
}) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [submittedError, setSubmittedError] = useState("");
  const [pendingLoadSnapshot, setPendingLoadSnapshot] =
    useState<ConfigSnapshot | null>(null);

  function requestLoad(snapshot: ConfigSnapshot) {
    if (
      draftMatchesConfigSnapshot({
        snapshot,
        preset: selectedPreset,
        overrides,
      })
    ) {
      onLoad(snapshot.id);
      return;
    }
    setPendingLoadSnapshot(snapshot);
  }

  function confirmLoad() {
    if (pendingLoadSnapshot) {
      onLoad(pendingLoadSnapshot.id);
    }
    setPendingLoadSnapshot(null);
  }

  function dismissMutation() {
    setSubmittedError("");
    onDismissMutation();
  }

  async function renameSnapshot(snapshotId: string, name: string) {
    setSubmittedError("");
    const result = await onRename(snapshotId, name);
    if (!result || result.ok) {
      setEditingId(null);
      return;
    }
    setSubmittedError(result.error);
  }

  async function removeSnapshot(snapshotId: string) {
    setSubmittedError("");
    const result = await onRemove(snapshotId);
    if (result && !result.ok) {
      setSubmittedError(result.error);
    }
  }

  async function retryMutation() {
    setSubmittedError("");
    const result = await onRetryMutation();
    if (!result) {
      setSubmittedError("There is no failed Config Snapshot change to retry.");
      return;
    }
    if (!result.ok) {
      setSubmittedError(result.error);
      return;
    }
    if (result.kind === "rename" && result.snapshotId === editingId) {
      setEditingId(null);
    }
  }

  const isMutationPending = mutation.phase === "pending";
  const mutationError =
    submittedError || (mutation.phase === "failed" ? mutation.error : "");

  if (groups.length === 0) {
    return null;
  }

  return (
    <>
      <section className={cn(surfacePanelClassName, "gap-3 p-3")}>
        <div className="flex min-w-0 flex-wrap items-center justify-between gap-2">
          <div className="grid gap-0.5">
            <SectionHeading
              as="h3"
              icon={<Camera className="h-[15px] w-[15px] text-violet" aria-hidden />}
              title="Config Snapshots"
            />
          </div>
          <Badge>
            {groups.reduce((count, group) => count + group.snapshots.length, 0)} saved
          </Badge>
        </div>

        {isMutationPending && (
          <InlineStatus busy compact>
            {mutation.kind === "rename"
              ? "Renaming Config Snapshot…"
              : mutation.kind === "remove"
                ? "Removing Config Snapshot…"
                : "Updating Config Snapshot…"}
          </InlineStatus>
        )}
        {mutationError && (
          <InlineStatus tone="danger" role="alert" compact>
            <div className="grid gap-2">
              <span>{mutationError}</span>
              <div className="flex flex-wrap gap-2">
                {mutation.phase === "failed" && mutation.canRetry && (
                  <Button
                    variant="secondary"
                    onClick={() => void retryMutation()}
                    disabled={isMutationPending}
                    className="h-8 text-xs"
                  >
                    Retry change
                  </Button>
                )}
                <Button
                  variant="ghost"
                  onClick={dismissMutation}
                  disabled={isMutationPending}
                  className="h-8 text-xs"
                >
                  Dismiss
                </Button>
              </div>
            </div>
          </InlineStatus>
        )}

        <div className="grid gap-2">
          {groups.map((group) => {
            const selectedSnapshotCount = group.snapshots.filter((snapshot) =>
              selectedTrainingSnapshotIds.includes(snapshot.id),
            ).length;
            return (
              <div
                key={group.preset}
                className={cn(
                  surfacePanelClassName,
                  "gap-2 p-2",
                  selectedSnapshotCount > 0
                    ? "border-violet/35 bg-violet/[0.06]"
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
                    const isEditing = editingId === snapshot.id;
                    const overrideCount = configSnapshotOverrideCount(snapshot);
                    const isIncluded = selectedTrainingSnapshotIds.includes(
                      snapshot.id,
                    );
                    return (
                      <div
                        key={snapshot.id}
                        className={cn(
                          surfacePanelClassName,
                          "gap-2 bg-white/[0.025] p-2.5",
                        )}
                      >
                        {isEditing ? (
                          <SnapshotNameEditor
                            initialName={snapshot.name}
                            pending={
                              isMutationPending &&
                              mutation.kind === "rename" &&
                              mutation.snapshotId === snapshot.id
                            }
                            onCancel={() => {
                              dismissMutation();
                              setEditingId(null);
                            }}
                            onSave={(name) =>
                              void renameSnapshot(snapshot.id, name)
                            }
                          />
                        ) : (
                          <div className="flex min-w-0 items-start justify-between gap-2">
                            <div className="flex min-w-0 items-start gap-2">
                              <Checkbox
                                checked={isIncluded}
                                onCheckedChange={() => onToggleSelection(snapshot.id)}
                                aria-label={`Include snapshot ${snapshot.name} in training`}
                                className="mt-0.5 shrink-0"
                              />
                              <div className="grid min-w-0 gap-1">
                                <span
                                  className="truncate text-sm font-semibold text-ink"
                                  title={snapshot.name}
                                >
                                  {snapshot.name}
                                </span>
                                <span className="font-mono text-xs text-ink-faint">
                                  {snapshot.preset} · {configSnapshotOverrideCountLabel(overrideCount)}
                                </span>
                              </div>
                            </div>
                            {canManage && (
                              <div className="flex shrink-0 items-center gap-1">
                                <IconButton
                                  label={`Rename snapshot ${snapshot.name}`}
                                  onClick={() => {
                                    dismissMutation();
                                    setEditingId(snapshot.id);
                                  }}
                                  disabled={isMutationPending}
                                  size="sm"
                                  variant="edge"
                                  className="bg-white/[0.025] hover:bg-white/[0.055]"
                                  icon={<Pencil className="h-3.5 w-3.5" aria-hidden />}
                                />
                                <IconButton
                                  label={`Remove snapshot ${snapshot.name}`}
                                  onClick={() => void removeSnapshot(snapshot.id)}
                                  disabled={isMutationPending}
                                  size="sm"
                                  variant="danger"
                                  className="border-danger-line bg-danger-soft text-danger-text hover:bg-danger-hover/40 hover:text-white"
                                  icon={<Trash2 className="h-3.5 w-3.5" aria-hidden />}
                                />
                              </div>
                            )}
                          </div>
                        )}
                        <SnapshotOverrideSummary overrides={snapshot.overrides} />
                        <Button
                          variant="secondary"
                          onClick={() => requestLoad(snapshot)}
                          className="h-8 justify-center px-2.5 text-xs"
                        >
                          <Play className="h-3.5 w-3.5" aria-hidden />
                          Load
                        </Button>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      </section>
      {pendingLoadSnapshot && (
        <SnapshotRestoreDialog
          snapshot={pendingLoadSnapshot}
          onCancel={() => setPendingLoadSnapshot(null)}
          onConfirm={confirmLoad}
        />
      )}
    </>
  );
}
