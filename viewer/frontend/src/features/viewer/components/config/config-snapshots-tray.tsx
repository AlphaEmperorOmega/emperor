import { useEffect, useMemo, useState, type KeyboardEvent } from "react";
import {
  Check,
  FilePlus2,
  Pencil,
  Play,
  Trash2,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { IconButton } from "@/components/ui/icon-button";
import { Input } from "@/components/ui/input";
import { SnapshotRestoreDialog } from "@/features/viewer/components/config/config-snapshot-dialogs";
import { DialogShell } from "@/features/viewer/components/shared/dialog-shell";
import {
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

function overrideCountLabel(count: number) {
  return `${count} override${count === 1 ? "" : "s"}`;
}

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
  model,
  preset,
  fields,
  overrides,
  snapshots,
  title = "Add Config Snapshot",
  actionLabel = "Add Snapshot",
  initialName,
  excludeSnapshotId,
  onAdd,
  onClose,
}: {
  model: string;
  preset: string;
  fields: ConfigField[];
  overrides: OverrideValues;
  snapshots: ConfigSnapshot[];
  title?: string;
  actionLabel?: string;
  initialName?: string;
  excludeSnapshotId?: string;
  onAdd: (name: string) => ConfigSnapshotCreateResult;
  onClose: () => void;
}) {
  const defaultName = useMemo(() => initialName ?? "", [initialName]);
  const [name, setName] = useState(defaultName);
  const [submittedError, setSubmittedError] = useState("");
  const nameValidation = useMemo(
    () =>
      validateConfigSnapshotName({
        model,
        preset,
        name,
        snapshots,
        excludeSnapshotId,
      }),
    [excludeSnapshotId, model, name, preset, snapshots],
  );
  const candidate = useMemo(
    () =>
      validateConfigSnapshotCandidate({
        model,
        preset,
        fields,
        overrides,
        snapshots,
        excludeSnapshotId,
      }),
    [excludeSnapshotId, fields, model, overrides, preset, snapshots],
  );
  const { entries } = useMemo(
    () => configSnapshotOverrideEntries(fields, overrides),
    [fields, overrides],
  );
  const error =
    submittedError ||
    (!nameValidation.ok
      ? nameValidation.error
      : !candidate.ok
        ? candidate.error
        : "");

  useEffect(() => {
    setName(defaultName);
    setSubmittedError("");
  }, [defaultName]);

  function confirm() {
    const result = onAdd(name);
    if (!result.ok) {
      setSubmittedError(result.error);
      return;
    }
    onClose();
  }

  return (
    <DialogShell
      titleId="add-config-snapshot-title"
      size="sm"
      onClose={onClose}
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
          onClick={onClose}
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
          }}
          autoFocus
        />
      </label>

      <div className="grid gap-2 rounded-[10px] border border-line bg-white/[0.018] p-3">
        <div className="flex items-center justify-between gap-2">
          <span className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
            Changed fields
          </span>
          <Badge>{overrideCountLabel(entries.length)}</Badge>
        </div>
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
          <div className="rounded-[8px] border border-dashed border-faint bg-black/18 px-2 py-2 text-xs text-ink-faint">
            No non-default draft overrides.
          </div>
        )}
      </div>

      {error && (
        <div
          role="alert"
          className="rounded-[9px] border border-danger-line bg-danger-soft px-3 py-2 text-sm text-danger-text"
        >
          {error}
        </div>
      )}

      <div className="flex justify-end gap-2">
        <Button variant="secondary" onClick={onClose}>
          Cancel
        </Button>
        <Button
          variant="primary"
          onClick={confirm}
          disabled={!nameValidation.ok || !candidate.ok}
        >
          <FilePlus2 className="h-4 w-4" aria-hidden />
          {actionLabel}
        </Button>
      </div>
    </DialogShell>
  );
}

function SnapshotNameEditor({
  initialName,
  onCancel,
  onSave,
}: {
  initialName: string;
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
        onChange={(event) => setValue(event.target.value)}
        onKeyDown={keyDown}
        className="h-8 text-xs"
        autoFocus
      />
      <IconButton
        label="Save snapshot name"
        onClick={save}
        size="sm"
        variant="edge"
        className="border-ok/30 bg-ok/10 text-ok hover:bg-ok/15 hover:text-ok"
        icon={<Check className="h-3.5 w-3.5" aria-hidden />}
      />
      <IconButton
        label="Cancel snapshot rename"
        onClick={onCancel}
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
  onLoad,
  onRename,
  onRemove,
  onToggleSelection,
}: {
  groups: ConfigSnapshotGroup[];
  selectedPreset: string;
  selectedTrainingSnapshotIds: string[];
  overrides: OverrideValues;
  canManage: boolean;
  onLoad: (snapshotId: string) => void;
  onRename: (snapshotId: string, name: string) => void;
  onRemove: (snapshotId: string) => void;
  onToggleSelection: (snapshotId: string) => void;
}) {
  const [editingId, setEditingId] = useState<string | null>(null);
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

  if (groups.length === 0) {
    return null;
  }

  return (
    <>
      <section className="grid gap-3 rounded-[12px] border border-line bg-white/[0.018] p-3">
        <div className="flex min-w-0 flex-wrap items-center justify-between gap-2">
          <div className="grid gap-0.5">
            <h3 className="text-xs font-bold uppercase tracking-[0.09em] text-ink-dim">
              Config Snapshots
            </h3>
          </div>
          <Badge>
            {groups.reduce((count, group) => count + group.snapshots.length, 0)} saved
          </Badge>
        </div>

        <div className="grid gap-2">
          {groups.map((group) => {
            const selectedSnapshotCount = group.snapshots.filter((snapshot) =>
              selectedTrainingSnapshotIds.includes(snapshot.id),
            ).length;
            return (
            <div
              key={group.preset}
              className={cn(
                "grid gap-2 rounded-[10px] border p-2",
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
                  const overrideCount = Object.keys(snapshot.overrides).length;
                  const isIncluded = selectedTrainingSnapshotIds.includes(
                    snapshot.id,
                  );
                  return (
                    <div
                      key={snapshot.id}
                      className="grid gap-2 rounded-[10px] border border-line bg-white/[0.025] p-2.5"
                    >
                      {isEditing ? (
                        <SnapshotNameEditor
                          initialName={snapshot.name}
                          onCancel={() => setEditingId(null)}
                          onSave={(name) => {
                            onRename(snapshot.id, name);
                            setEditingId(null);
                          }}
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
                                {snapshot.preset} · {overrideCountLabel(overrideCount)}
                              </span>
                            </div>
                          </div>
                          {canManage && (
                            <div className="flex shrink-0 items-center gap-1">
                              <IconButton
                                label={`Rename snapshot ${snapshot.name}`}
                                onClick={() => setEditingId(snapshot.id)}
                                size="sm"
                                variant="edge"
                                className="bg-white/[0.025] hover:bg-white/[0.055]"
                                icon={<Pencil className="h-3.5 w-3.5" aria-hidden />}
                              />
                              <IconButton
                                label={`Remove snapshot ${snapshot.name}`}
                                onClick={() => onRemove(snapshot.id)}
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
