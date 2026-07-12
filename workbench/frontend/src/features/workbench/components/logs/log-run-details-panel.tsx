import { type ReactNode } from "react";
import { Badge } from "@/components/ui/badge";
import { InlineStatus } from "@/features/workbench/components/shared/inline-status";
import { KeyValueRow } from "@/features/workbench/components/shared/key-value-row";
import { MetricCard } from "@/features/workbench/components/shared/metric-card";
import { SectionHeading } from "@/components/ui/section-heading";
import { SidePanel } from "@/features/workbench/components/shared/side-panel";
import { SurfacePanel } from "@/components/ui/surface-panel";
import { useLogRunDetail } from "@/features/workbench/providers/logs-workspace-provider";
import { formatMetricValue } from "@/features/workbench/state/logs/logs-selectors";
import { type LogRun, type LogRunArtifacts } from "@/lib/api";
import { formatBytes, formatDateTime } from "@/lib/format";
import { cn } from "@/lib/utils";

const metadataCardClassName = "w-full min-w-0";

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
    <div className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] items-start gap-2 rounded-control border border-line-soft bg-black/20 px-3 py-2 text-xs">
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

function checkpointDetail({
  epoch,
  step,
  sizeBytes,
  modifiedAt,
}: LogRunArtifacts["checkpoints"][number]) {
  return [
    epoch === null ? "epoch unknown" : `epoch ${epoch}`,
    step === null ? "step unknown" : `step ${step}`,
    formatBytes(sizeBytes),
    formatDateTime(modifiedAt),
  ].join(" · ");
}

export function LogRunDetailsPanel({
  selectedRun,
  artifacts,
  artifactsLoading = false,
  artifactsError = null,
}: {
  selectedRun: LogRun | undefined;
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
        <SurfacePanel padding="roomy" className="text-sm text-ink-faint">
          Select a visible run to inspect its metadata.
        </SurfacePanel>
      ) : (
        <div className="grid gap-4">
          <SurfacePanel padding="roomy" className="min-w-0">
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
          </SurfacePanel>

          <div className="grid w-full min-w-0 grid-cols-1 gap-[9px]">
            <MetricCard
              label="Experiment"
              value={run.experiment}
              className={metadataCardClassName}
              valueTitle={run.experiment}
              valueClassName="mt-1.5 min-w-0 truncate text-sm font-bold"
            />
            <MetricCard
              label="Dataset"
              value={run.dataset}
              className={metadataCardClassName}
              valueTitle={run.dataset}
              valueClassName="mt-1.5 min-w-0 truncate text-sm font-bold"
            />
            <MetricCard
              label="Model"
              value={run.model}
              className={metadataCardClassName}
              valueTitle={run.model}
              valueClassName="mt-1.5 min-w-0 truncate text-sm font-bold"
            />
            <MetricCard
              label="Preset"
              value={run.preset}
              className={metadataCardClassName}
              valueTitle={run.preset}
              valueClassName="mt-1.5 min-w-0 truncate text-sm font-bold"
            />
            <MetricCard
              label="Version"
              value={run.version}
              className={metadataCardClassName}
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
                Loading checkpoints…
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
                Loading params…
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
                Loading artifacts…
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
                      formatBytes(artifact.sizeBytes),
                      formatDateTime(artifact.modifiedAt),
                    ].join(" · ")}
                    valueTitle={[
                      artifact.kind,
                      formatBytes(artifact.sizeBytes),
                      formatDateTime(artifact.modifiedAt),
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
  const detail = useLogRunDetail();
  return (
    <LogRunDetailsPanel
      selectedRun={detail.run}
      artifacts={detail.artifacts}
      artifactsLoading={detail.status.isLoading}
      artifactsError={detail.status.error}
    />
  );
}
