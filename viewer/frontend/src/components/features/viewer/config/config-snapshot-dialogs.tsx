import { Button } from "@/components/ui/button";
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
          <Button variant="secondary" onClick={onCancel} disabled={busy}>
            Cancel
          </Button>
          <Button variant="primary" onClick={onConfirm} disabled={busy}>
            Load
          </Button>
        </div>
      </section>
    </div>
  );
}
