import { type ReactNode } from "react";
import { Badge } from "@/components/ui/badge";
import { EdgeCard } from "@/components/ui/edge-card";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { KeyValueRow } from "@/features/viewer/components/shared/key-value-row";
import { MetricCard } from "@/features/viewer/components/shared/metric-card";
import { SectionHeading } from "@/features/viewer/components/shared/section-heading";
import { SidePanel } from "@/features/viewer/components/shared/side-panel";
import { useLogsWorkspace } from "@/features/viewer/providers/logs-workspace-provider";
import { useLogRunArtifactsQuery } from "@/features/viewer/state/logs/use-log-queries";
import { formatMetricValue } from "@/features/viewer/state/logs/logs-selectors";
import { type LogsWorkspaceState } from "@/features/viewer/state/logs/use-logs-workspace-state";
import { type LogRunArtifacts } from "@/lib/api";
import { cn } from "@/lib/utils";

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

function DetailRow({
  label,
  value,
  labelTitle,
  valueTitle,
}: {
  label: ReactNode;
  value: ReactNode;
  labelTitle?: string;
  valueTitle?: string;
}) {
  const textContainmentClasses =
    "min-w-0 whitespace-normal break-words [overflow-wrap:anywhere]";

  return (
    <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] items-start gap-2 rounded-[9px] border border-line-soft bg-black/20 px-3 py-2 text-xs">
      <span
        className={cn("font-mono text-ink-dim", textContainmentClasses)}
        title={labelTitle}
      >
        {label}
      </span>
      <span
        className={cn(
          "text-right font-mono font-semibold text-ink",
          textContainmentClasses,
        )}
        title={valueTitle}
      >
        {value}
      </span>
    </div>
  );
}

function formatFileSize(sizeBytes: number) {
  if (!Number.isFinite(sizeBytes) || sizeBytes < 0) {
    return "0 B";
  }
  if (sizeBytes < 1024) {
    return `${sizeBytes} B`;
  }
  const kib = sizeBytes / 1024;
  if (kib < 1024) {
    return `${kib.toFixed(1).replace(/\.0$/, "")} KB`;
  }
  return `${(kib / 1024).toFixed(1).replace(/\.0$/, "")} MB`;
}

function formatModifiedAt(modifiedAt: string) {
  const date = new Date(modifiedAt);
  if (Number.isNaN(date.getTime())) {
    return modifiedAt;
  }
  return date.toLocaleString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function checkpointDetail({
  epoch,
  step,
  sizeBytes,
  modifiedAt,
}: LogRunArtifacts["checkpoints"][number]) {
  return [
    epoch === null ? "epoch unknown" : `epoch ${epoch}`,
    step === null ? "step unknown" : `step ${step}`,
    formatFileSize(sizeBytes),
    formatModifiedAt(modifiedAt),
  ].join(" · ");
}

export function LogRunDetailsPanel({
  selectedRun,
  artifacts,
  artifactsLoading = false,
  artifactsError = null,
}: {
  selectedRun: LogsWorkspaceState["selectedRun"];
  artifacts?: LogRunArtifacts;
  artifactsLoading?: boolean;
  artifactsError?: unknown;
}) {
  const run = selectedRun;
  const metrics = Object.entries(artifacts?.metrics ?? selectedRun?.metrics ?? {});
  const params = Object.entries(artifacts?.params ?? {});
  const checkpoints = artifacts?.checkpoints ?? [];
  const artifactFiles = artifacts?.artifacts ?? [];

  return (
    <SidePanel
      className="min-w-0 overflow-x-hidden"
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
              <div
                className="min-w-0 truncate text-sm font-semibold text-ink"
                title={run.runName}
              >
                {run.runName}
              </div>
              <div
                className="mt-1 min-w-0 break-words font-mono text-xs leading-5 text-ink-faint [overflow-wrap:anywhere]"
                title={run.relativePath}
              >
                {run.relativePath}
              </div>
            </div>
          </EdgeCard>

          <div className="grid grid-cols-2 gap-[9px]">
            <MetricCard
              label="Experiment"
              value={run.experiment}
              className="min-w-0"
              valueTitle={run.experiment}
              valueClassName="mt-1.5 min-w-0 truncate text-sm font-bold"
            />
            <MetricCard
              label="Dataset"
              value={run.dataset}
              className="min-w-0"
              valueTitle={run.dataset}
              valueClassName="mt-1.5 min-w-0 truncate text-sm font-bold"
            />
            <MetricCard
              label="Model"
              value={run.model}
              className="min-w-0"
              valueTitle={run.model}
              valueClassName="mt-1.5 min-w-0 truncate text-sm font-bold"
            />
            <MetricCard
              label="Preset"
              value={run.preset}
              className="min-w-0"
              valueTitle={run.preset}
              valueClassName="mt-1.5 min-w-0 truncate text-sm font-bold"
            />
            <MetricCard
              label="Version"
              value={run.version}
              className="min-w-0"
              valueTitle={run.version}
              valueClassName="mt-1.5 min-w-0 truncate text-sm font-bold"
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
            <SectionHeading as="h3" title="Checkpoints" />
            {artifactsLoading ? (
              <InlineStatus className="border-line-soft bg-transparent p-0 px-3 py-4">
                Loading checkpoints
              </InlineStatus>
            ) : artifactsError ? (
              <InlineStatus className="border-line-soft bg-transparent p-0 px-3 py-4">
                Checkpoint metadata unavailable
              </InlineStatus>
            ) : checkpoints.length === 0 ? (
              <InlineStatus className="border-line-soft bg-transparent p-0 px-3 py-4">
                No checkpoints found
              </InlineStatus>
            ) : (
              <div className="grid gap-2">
                {checkpoints.map((checkpoint) => (
                  <DetailRow
                    key={checkpoint.id}
                    label={checkpoint.filename}
                    labelTitle={checkpoint.filename}
                    value={checkpointDetail(checkpoint)}
                    valueTitle={checkpointDetail(checkpoint)}
                  />
                ))}
              </div>
            )}
          </section>

          <section className="grid gap-2">
            <SectionHeading as="h3" title="Params" />
            {artifactsLoading ? (
              <InlineStatus className="border-line-soft bg-transparent p-0 px-3 py-4">
                Loading params
              </InlineStatus>
            ) : artifactsError ? (
              <InlineStatus className="border-line-soft bg-transparent p-0 px-3 py-4">
                Params unavailable
              </InlineStatus>
            ) : params.length === 0 ? (
              <InlineStatus className="border-line-soft bg-transparent p-0 px-3 py-4">
                No params found
              </InlineStatus>
            ) : (
              <div className="grid gap-2">
                {params.map(([key, value]) => (
                  <DetailRow
                    key={key}
                    label={key}
                    labelTitle={key}
                    value={formatMetricValue(value)}
                    valueTitle={formatMetricValue(value)}
                  />
                ))}
              </div>
            )}
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
                  <DetailRow
                    key={key}
                    label={key}
                    labelTitle={key}
                    value={formatMetricValue(value)}
                    valueTitle={formatMetricValue(value)}
                  />
                ))}
              </div>
            )}
          </section>

          <section className="grid gap-2">
            <SectionHeading as="h3" title="Artifacts" />
            {artifactsLoading ? (
              <InlineStatus className="border-line-soft bg-transparent p-0 px-3 py-4">
                Loading artifacts
              </InlineStatus>
            ) : artifactsError ? (
              <InlineStatus className="border-line-soft bg-transparent p-0 px-3 py-4">
                Artifact metadata unavailable
              </InlineStatus>
            ) : artifactFiles.length === 0 ? (
              <InlineStatus className="border-line-soft bg-transparent p-0 px-3 py-4">
                No artifacts found
              </InlineStatus>
            ) : (
              <div className="grid gap-2">
                {artifactFiles.map((artifact) => (
                  <DetailRow
                    key={artifact.id}
                    label={artifact.label}
                    labelTitle={artifact.label}
                    value={[
                      artifact.kind,
                      formatFileSize(artifact.sizeBytes),
                      formatModifiedAt(artifact.modifiedAt),
                    ].join(" · ")}
                    valueTitle={[
                      artifact.kind,
                      formatFileSize(artifact.sizeBytes),
                      formatModifiedAt(artifact.modifiedAt),
                    ].join(" · ")}
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

export function ConnectedLogRunDetailsPanel() {
  const state = useLogsWorkspace();
  const artifactsQuery = useLogRunArtifactsQuery({
    runId: state.selectedRun?.id,
    enabled: state.enabled,
  });
  return (
    <LogRunDetailsPanel
      selectedRun={state.selectedRun}
      artifacts={artifactsQuery.data}
      artifactsLoading={artifactsQuery.isLoading}
      artifactsError={artifactsQuery.error}
    />
  );
}
