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
import { Input } from "@/components/ui/input";
import {
  configSnapshotOverrideEntries,
  draftMatchesConfigSnapshot,
  generateDefaultConfigSnapshotName,
  validateConfigSnapshotCandidate,
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
  onAdd,
  onClose,
}: {
  model: string;
  preset: string;
  fields: ConfigField[];
  overrides: OverrideValues;
  snapshots: ConfigSnapshot[];
  onAdd: (name: string) => ConfigSnapshotCreateResult;
  onClose: () => void;
}) {
  const defaultName = useMemo(
    () => generateDefaultConfigSnapshotName({ preset, fields, overrides }),
    [fields, overrides, preset],
  );
  const [name, setName] = useState(defaultName);
  const [submittedError, setSubmittedError] = useState("");
  const candidate = useMemo(
    () =>
      validateConfigSnapshotCandidate({
        model,
        preset,
        fields,
        overrides,
        snapshots,
      }),
    [fields, model, overrides, preset, snapshots],
  );
  const { entries } = useMemo(
    () => configSnapshotOverrideEntries(fields, overrides),
    [fields, overrides],
  );
  const error = submittedError || (!candidate.ok ? candidate.error : "");

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
    <div className="fixed inset-0 z-[60] grid place-items-center bg-black/65 p-4 backdrop-blur-sm">
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="add-config-snapshot-title"
        className="edge grid w-full max-w-lg gap-4 rounded-card p-4 shadow-[0_24px_80px_rgba(0,0,0,0.58)]"
      >
        <div className="flex items-start justify-between gap-3">
          <div className="grid gap-1">
            <h2
              id="add-config-snapshot-title"
              className="text-base font-semibold text-ink"
            >
              Add Config Snapshot
            </h2>
            <div className="flex flex-wrap items-center gap-1.5 text-xs text-ink-faint">
              <span className="font-mono">{model || "No model"}</span>
              {preset && <span aria-hidden>/</span>}
              {preset && <span className="font-mono">{preset}</span>}
            </div>
          </div>
          <button
            type="button"
            aria-label="Close add config snapshot"
            onClick={onClose}
            className="grid h-8 w-8 place-items-center rounded-[9px] border border-line-soft bg-white/[0.025] text-ink-faint transition hover:bg-white/[0.055] hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
          >
            <X className="h-4 w-4" aria-hidden />
          </button>
        </div>

        <label className="grid gap-1.5">
          <span className="text-xs font-bold uppercase tracking-[0.08em] text-ink-dim">
            Name
          </span>
          <Input
            value={name}
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
            className="rounded-[9px] border border-danger-line bg-danger-soft px-3 py-2 text-sm text-[#fda4af]"
          >
            {error}
          </div>
        )}

        <div className="flex justify-end gap-2">
          <Button variant="secondary" onClick={onClose}>
            Cancel
          </Button>
          <Button variant="primary" onClick={confirm} disabled={!candidate.ok}>
            <FilePlus2 className="h-4 w-4" aria-hidden />
            Add Snapshot
          </Button>
        </div>
      </section>
    </div>
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
      <button
        type="button"
        aria-label="Save snapshot name"
        onClick={save}
        className="grid h-8 w-8 shrink-0 place-items-center rounded-[8px] border border-ok/30 bg-ok/10 text-ok transition hover:bg-ok/15 focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
      >
        <Check className="h-3.5 w-3.5" aria-hidden />
      </button>
      <button
        type="button"
        aria-label="Cancel snapshot rename"
        onClick={onCancel}
        className="grid h-8 w-8 shrink-0 place-items-center rounded-[8px] border border-line bg-white/[0.025] text-ink-faint transition hover:bg-white/[0.055] hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
      >
        <X className="h-3.5 w-3.5" aria-hidden />
      </button>
    </div>
  );
}

function ConfirmLoadSnapshotDialog({
  snapshot,
  onCancel,
  onConfirm,
}: {
  snapshot: ConfigSnapshot;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <div className="fixed inset-0 z-[60] grid place-items-center bg-black/65 p-4 backdrop-blur-sm">
      <section
        role="dialog"
        aria-modal="true"
        aria-labelledby="load-config-snapshot-title"
        className="edge grid w-full max-w-md gap-4 rounded-card p-4 shadow-[0_24px_80px_rgba(0,0,0,0.58)]"
      >
        <div className="grid gap-1">
          <h2
            id="load-config-snapshot-title"
            className="text-base font-semibold text-ink"
          >
            Load Snapshot
          </h2>
          <p className="text-sm leading-6 text-ink-dim">
            Loading &quot;{snapshot.name}&quot; replaces the current draft overrides.
          </p>
        </div>
        <div className="flex justify-end gap-2">
          <Button variant="secondary" onClick={onCancel}>
            Cancel
          </Button>
          <Button variant="primary" onClick={onConfirm}>
            Load
          </Button>
        </div>
      </section>
    </div>
  );
}

export function ConfigSnapshotsTray({
  groups,
  selectedPreset,
  overrides,
  onLoad,
  onRename,
  onRemove,
}: {
  groups: ConfigSnapshotGroup[];
  selectedPreset: string;
  overrides: OverrideValues;
  onLoad: (snapshotId: string) => void;
  onRename: (snapshotId: string, name: string) => void;
  onRemove: (snapshotId: string) => void;
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
          {groups.map((group) => (
            <div
              key={group.preset}
              className={cn(
                "grid gap-2 rounded-[10px] border p-2",
                group.preset === selectedPreset
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
                    group.preset === selectedPreset
                      ? "border-violet/30 bg-violet/15 text-violet"
                      : undefined
                  }
                >
                  {group.snapshots.length}
                </Badge>
              </div>
              <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
                {group.snapshots.map((snapshot) => {
                  const isEditing = editingId === snapshot.id;
                  const overrideCount = Object.keys(snapshot.overrides).length;
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
                          <div className="flex shrink-0 items-center gap-1">
                            <button
                              type="button"
                              aria-label={`Rename snapshot ${snapshot.name}`}
                              onClick={() => setEditingId(snapshot.id)}
                              className="grid h-8 w-8 place-items-center rounded-[8px] border border-line bg-white/[0.025] text-ink-faint transition hover:bg-white/[0.055] hover:text-ink focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                            >
                              <Pencil className="h-3.5 w-3.5" aria-hidden />
                            </button>
                            <button
                              type="button"
                              aria-label={`Remove snapshot ${snapshot.name}`}
                              onClick={() => onRemove(snapshot.id)}
                              className="grid h-8 w-8 place-items-center rounded-[8px] border border-danger-line bg-danger-soft text-[#fda4af] transition hover:bg-[#7f1d2d]/40 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-focus"
                            >
                              <Trash2 className="h-3.5 w-3.5" aria-hidden />
                            </button>
                          </div>
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
          ))}
        </div>
      </section>
      {pendingLoadSnapshot && (
        <ConfirmLoadSnapshotDialog
          snapshot={pendingLoadSnapshot}
          onCancel={() => setPendingLoadSnapshot(null)}
          onConfirm={confirmLoad}
        />
      )}
    </>
  );
}
