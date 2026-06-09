import { Button } from "@/components/ui/button";
import { DialogShell } from "@/features/viewer/components/shared/dialog-shell";
import { type ConfigSnapshot } from "@/lib/config-snapshots";

export function SnapshotRestoreDialog({
  snapshot,
  busy = false,
  onCancel,
  onConfirm,
}: {
  snapshot: ConfigSnapshot;
  busy?: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <DialogShell
      titleId="load-config-snapshot-title"
      size="sm"
      onClose={onCancel}
      closeOnEscape={!busy}
      className="z-[60] grid place-items-center bg-black/65 p-4 sm:p-4"
      panelClassName="grid max-h-none max-w-md gap-4 overflow-visible p-4 sm:max-h-none"
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
        <Button variant="secondary" onClick={onCancel} disabled={busy}>
          Cancel
        </Button>
        <Button variant="primary" onClick={onConfirm} disabled={busy}>
          Load
        </Button>
      </div>
    </DialogShell>
  );
}
