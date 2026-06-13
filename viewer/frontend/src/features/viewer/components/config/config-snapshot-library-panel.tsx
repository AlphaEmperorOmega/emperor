import { Library, Play } from "lucide-react";
import { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ErrorPanel } from "@/features/viewer/components/error-panel";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { SectionHeading } from "@/features/viewer/components/shared/section-heading";
import {
  useConfigSnapshotLibraryState,
} from "@/features/viewer/providers/viewer-providers";
import { type ConfigSnapshotRecord } from "@/lib/api";
import { modelNameForId } from "@/lib/selection";
import { errorMessage } from "@/lib/utils";

type SnapshotLibraryGroup = {
  model: string;
  preset: string;
  snapshots: ConfigSnapshotRecord[];
};

function overrideCountLabel(count: number) {
  return `${count} ${count === 1 ? "override" : "overrides"}`;
}

function groupSnapshots(snapshots: ConfigSnapshotRecord[]) {
  const groups = new Map<string, SnapshotLibraryGroup>();
  for (const snapshot of snapshots) {
    const key = `${snapshot.model}\u0000${snapshot.preset}`;
    const group = groups.get(key) ?? {
      model: snapshot.model,
      preset: snapshot.preset,
      snapshots: [],
    };
    group.snapshots = [...group.snapshots, snapshot];
    groups.set(key, group);
  }
  return Array.from(groups.values());
}

export function ConfigSnapshotLibraryPanel() {
  const {
    snapshots,
    snapshotCount,
    isLoading,
    isError,
    error,
    loadConfigSnapshot,
  } = useConfigSnapshotLibraryState();
  const groups = useMemo(() => groupSnapshots(snapshots), [snapshots]);

  if (isError) {
    return (
      <ErrorPanel
        title="Snapshot library failed"
        message={errorMessage(error)}
      />
    );
  }

  return (
    <section className="grid gap-3" aria-label="Snapshots">
      <div className="flex items-center justify-between gap-3">
        <SectionHeading
          as="h2"
          className="min-w-0"
          icon={<Library className="h-[15px] w-[15px] shrink-0 text-violet" aria-hidden />}
          title="Snapshots"
        />
        {snapshotCount > 0 && <Badge>{snapshotCount} total</Badge>}
      </div>

      {isLoading ? (
        <InlineStatus compact>Loading snapshots</InlineStatus>
      ) : groups.length === 0 ? (
        <InlineStatus compact>No snapshots saved</InlineStatus>
      ) : (
        <div className="grid gap-2">
          {groups.map((group) => (
            <div
              key={`${group.model}\u0000${group.preset}`}
              className="grid gap-2 rounded-[8px] border border-line bg-white/[0.018] p-2.5"
            >
              <div className="flex min-w-0 items-center justify-between gap-2">
                <div className="grid min-w-0 gap-0.5">
                  <span className="truncate text-xs font-semibold text-ink">
                    {modelNameForId(group.model)}
                  </span>
                  <span className="truncate font-mono text-[11px] text-ink-faint">
                    {group.preset}
                  </span>
                </div>
                <Badge>{group.snapshots.length}</Badge>
              </div>
              <div className="grid gap-1.5">
                {group.snapshots.map((snapshot) => {
                  const overrideCount = Object.keys(snapshot.overrides).length;
                  return (
                    <div
                      key={snapshot.id}
                      className="flex min-w-0 items-center justify-between gap-2 rounded-[8px] border border-line-soft bg-black/15 p-2"
                    >
                      <div className="grid min-w-0 gap-0.5">
                        <span
                          className="truncate text-xs font-semibold text-ink"
                          title={snapshot.name}
                        >
                          {snapshot.name}
                        </span>
                        <span className="text-[11px] text-ink-faint">
                          {overrideCountLabel(overrideCount)}
                        </span>
                      </div>
                      <Button
                        variant="secondary"
                        onClick={() => loadConfigSnapshot(snapshot.id)}
                        className="h-8 shrink-0 px-2 text-xs"
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
      )}
    </section>
  );
}
