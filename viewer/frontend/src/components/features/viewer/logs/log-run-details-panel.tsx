import { Badge } from "@/components/ui/badge";
import { EdgeCard } from "@/components/ui/edge-card";
import { InlineStatus } from "@/components/features/viewer/shared/inline-status";
import { KeyValueRow } from "@/components/features/viewer/shared/key-value-row";
import { MetricCard } from "@/components/features/viewer/shared/metric-card";
import { SectionHeading } from "@/components/features/viewer/shared/section-heading";
import { SidePanel } from "@/components/features/viewer/shared/side-panel";
import { type LogsWorkspaceState } from "@/components/features/viewer/state/use-logs-workspace-state";
import { formatMetricValue } from "@/lib/logs/helpers";

function FilePresenceRow({
  label,
  value,
  present,
}: {
  label: string;
  value: string | number;
  present: boolean;
}) {
  return (
    <KeyValueRow
      variant="card"
      label={label}
      value={<Badge className={present ? "text-ink" : "text-ink-faint"}>{value}</Badge>}
    />
  );
}

export function LogRunDetailsPanel({
  selectedRun,
}: {
  selectedRun: LogsWorkspaceState["selectedRun"];
}) {
  const run = selectedRun;
  const metrics = Object.entries(selectedRun?.metrics ?? {});

  return (
    <SidePanel
      title="Run Details"
      actions={
        run ? <Badge>{run.hasResult ? "result.json" : "No result.json"}</Badge> : undefined
      }
    >
      {!run ? (
        <div className="edge rounded-card p-4 text-sm text-ink-faint">
          Select a visible run to inspect its metadata.
        </div>
      ) : (
        <div className="grid gap-4">
          <EdgeCard className="rounded-[12px] px-3 py-3">
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold text-ink" title={run.runName}>
                {run.runName}
              </div>
              <div className="mt-1 break-words font-mono text-xs leading-5 text-ink-faint">
                {run.relativePath}
              </div>
            </div>
          </EdgeCard>

          <div className="grid grid-cols-2 gap-[9px]">
            <MetricCard
              label="Experiment"
              value={run.experiment}
              valueClassName="mt-1.5 truncate text-sm font-bold"
            />
            <MetricCard
              label="Dataset"
              value={run.dataset}
              valueClassName="mt-1.5 truncate text-sm font-bold"
            />
            <MetricCard
              label="Model"
              value={run.model}
              valueClassName="mt-1.5 truncate text-sm font-bold"
            />
            <MetricCard
              label="Preset"
              value={run.preset}
              valueClassName="mt-1.5 truncate text-sm font-bold"
            />
            <MetricCard
              label="Version"
              value={run.version}
              valueClassName="mt-1.5 truncate text-sm font-bold"
            />
          </div>

          <section className="grid gap-2">
            <SectionHeading as="h3" title="Files" />
            <div className="grid gap-2">
              <FilePresenceRow
                label="Event files"
                value={run.eventFileCount}
                present={run.eventFileCount > 0}
              />
              <FilePresenceRow
                label="hparams.yaml"
                value={run.hasHparams ? "present" : "missing"}
                present={run.hasHparams}
              />
              <FilePresenceRow
                label="result.json"
                value={run.hasResult ? "present" : "missing"}
                present={run.hasResult}
              />
              <FilePresenceRow
                label="Checkpoints"
                value={run.checkpointCount}
                present={run.checkpointCount > 0}
              />
            </div>
          </section>

          <section className="grid gap-2">
            <SectionHeading as="h3" title="Metrics" />
            {!run.hasResult ? (
              <InlineStatus className="border-line-soft bg-transparent p-0 px-3 py-4">
                No result.json
              </InlineStatus>
            ) : metrics.length === 0 ? (
              <InlineStatus className="border-line-soft bg-transparent p-0 px-3 py-4">
                No metrics found
              </InlineStatus>
            ) : (
              <div className="grid gap-2">
                {metrics.map(([key, value]) => (
                  <KeyValueRow
                    key={key}
                    variant="card"
                    label={<span className="font-mono">{key}</span>}
                    value={formatMetricValue(value)}
                  />
                ))}
              </div>
            )}
          </section>
        </div>
      )}
    </SidePanel>
  );
}
