import {
  AlertTriangle,
  BarChart3,
  GitCompare,
  Plus,
  RotateCcw,
  Settings2,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { CompareTargetCard } from "@/features/viewer/components/compare-workspace/compare-target-card";
import {
  ComparisonTable,
  ConfigDiffTable,
} from "@/features/viewer/components/compare-workspace/compare-summary-tables";
import { CompareDataView } from "@/features/viewer/components/compare-workspace/compare-data-view";
import { CompareGraphsView } from "@/features/viewer/components/compare-workspace/compare-graphs-view";
import { CompareRunSelector } from "@/features/viewer/components/compare-workspace/compare-run-selector";
import { useExperimentCompareWorkspaceState } from "@/features/viewer/components/compare-workspace/use-experiment-compare-workspace-state";
import { useCompareWorkspaceState } from "@/features/viewer/components/compare-workspace/use-compare-workspace-state";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { SectionHeading } from "@/features/viewer/components/shared/section-heading";
import { viewerStatusCopy } from "@/features/viewer/components/shared/status-copy";
import { SurfacePanel } from "@/features/viewer/components/shared/surface-panel";
import { ViewModeButton } from "@/features/viewer/components/view-mode-button";
import { errorMessage } from "@/lib/utils";

export function CompareWorkspace({
  onOpenLogs,
  onUseTarget,
}: {
  onOpenLogs: () => void;
  onUseTarget: () => void;
}) {
  const comparison = useExperimentCompareWorkspaceState();

  return (
    <div className="h-full min-h-0 overflow-y-auto bg-bg-2/60 p-4 lg:p-5">
      <div className="mx-auto grid max-w-[1480px] gap-4">
        <SurfacePanel padding="spacious">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <SectionHeading
              as="h2"
              className="normal-case tracking-normal text-ink"
              icon={
                comparison.mode === "runs" ? (
                  <BarChart3 className="h-4 w-4 text-violet" aria-hidden />
                ) : (
                  <GitCompare className="h-4 w-4 text-violet" aria-hidden />
                )
              }
              title={
                <span className="text-base font-bold">
                  {comparison.mode === "runs"
                    ? "Training Run Comparison"
                    : "Model Config Comparison"}
                </span>
              }
              count={
                <Badge>
                  {comparison.mode === "runs"
                    ? `${comparison.entryData.length} ${
                        comparison.entryData.length === 1 ? "target" : "targets"
                      }`
                    : "configs"}
                </Badge>
              }
            />
            <SegmentedControl variant="tablist">
              <ViewModeButton
                variant="tab"
                active={comparison.mode === "runs"}
                onClick={() => comparison.setMode("runs")}
              >
                <BarChart3 className="h-3.5 w-3.5" aria-hidden />
                Runs
              </ViewModeButton>
              <ViewModeButton
                variant="tab"
                active={comparison.mode === "configs"}
                onClick={() => comparison.setMode("configs")}
              >
                <Settings2 className="h-3.5 w-3.5" aria-hidden />
                Configs
              </ViewModeButton>
            </SegmentedControl>
          </div>
        </SurfacePanel>

        {comparison.mode === "runs" ? (
          <ExperimentCompareContent
            comparison={comparison}
            onOpenLogs={onOpenLogs}
          />
        ) : (
          <ConfigCompareContent onUseTarget={onUseTarget} />
        )}
      </div>
    </div>
  );
}

function ExperimentCompareContent({
  comparison,
  onOpenLogs,
}: {
  comparison: ReturnType<typeof useExperimentCompareWorkspaceState>;
  onOpenLogs: () => void;
}) {
  return (
    <>
      {comparison.runsQuery.isError && (
        <InlineStatus tone="danger" compact role="alert">
          {errorMessage(comparison.runsQuery.error)}
        </InlineStatus>
      )}
      {comparison.tagsQuery.isError && (
        <InlineStatus tone="danger" compact role="alert">
          {errorMessage(comparison.tagsQuery.error)}
        </InlineStatus>
      )}
      {comparison.scalarsQuery.isError && (
        <InlineStatus tone="danger" compact role="alert">
          {errorMessage(comparison.scalarsQuery.error)}
        </InlineStatus>
      )}

      {(comparison.runsQuery.isLoading || comparison.tagsQuery.isLoading) && (
        <InlineStatus busy compact>
          Loading historical Training Runs
        </InlineStatus>
      )}

      <CompareRunSelector comparison={comparison} onOpenLogs={onOpenLogs} />

      {comparison.hasTruncatedSeries && (
        <InlineStatus tone="warning" compact>
          Some scalar series are truncated by the log reader.
        </InlineStatus>
      )}
      {comparison.scalarsQuery.isLoading && (
        <InlineStatus busy compact>
          Loading scalar comparison data
        </InlineStatus>
      )}

      {comparison.view === "graphs" ? (
        <CompareGraphsView comparison={comparison} />
      ) : (
        <CompareDataView comparison={comparison} />
      )}
    </>
  );
}

function ConfigCompareContent({ onUseTarget }: { onUseTarget: () => void }) {
  const comparison = useCompareWorkspaceState({ onUseTarget });

  if (comparison.catalog.isLoading) {
    return (
      <div className="grid place-items-center p-6">
        <InlineStatus busy>{viewerStatusCopy.loading.modelCatalog}</InlineStatus>
      </div>
    );
  }

  if (comparison.catalog.isError) {
    return (
      <div className="grid place-items-center p-6">
        <InlineStatus tone="danger" role="alert">
          {errorMessage(comparison.catalog.error)}
        </InlineStatus>
      </div>
    );
  }

  return (
    <>
      <SurfacePanel padding="spacious">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <SectionHeading
            as="h3"
            className="normal-case tracking-normal text-ink"
            icon={<GitCompare className="h-4 w-4 text-violet" aria-hidden />}
            title={<span className="text-sm font-bold">Model Comparison</span>}
            count={<Badge>{comparison.readyEntryCount} targets</Badge>}
          />
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="secondary"
              onClick={comparison.resetEntries}
              disabled={!comparison.canResetEntries}
            >
              <RotateCcw className="h-4 w-4" aria-hidden />
              Reset
            </Button>
            <Button
              variant="primary"
              onClick={comparison.addEntry}
              disabled={!comparison.canAddEntry}
            >
              <Plus className="h-4 w-4" aria-hidden />
              Add Target
            </Button>
          </div>
        </div>
      </SurfacePanel>

      <div className="grid items-start gap-3 xl:grid-cols-2 min-[1900px]:grid-cols-4">
        {comparison.entries.map((entry, index) => (
          <CompareTargetCard
            key={entry.entry.id}
            entryData={entry}
            index={index}
            modelOptions={comparison.modelOptions}
            baselineParameterCount={
              comparison.entries[0]?.inspection?.parameterCount
            }
            canRemove={comparison.canRemoveEntry}
            onRemove={comparison.removeEntry}
            onUpdate={comparison.updateEntry}
            onApply={comparison.applyAsTarget}
          />
        ))}
      </div>

      {comparison.entries.length < 2 && (
        <InlineStatus tone="warning" compact>
          <AlertTriangle className="mr-2 inline h-4 w-4" aria-hidden />
          Add at least two targets to compare model and preset behavior.
        </InlineStatus>
      )}

      <ComparisonTable
        title="Changed Summary Metrics"
        rows={comparison.changedStats}
        entries={comparison.entries}
        emptyMessage="No changed summary metrics for the selected targets."
      />
      <ConfigDiffTable
        rows={comparison.configRows}
        entries={comparison.entries}
      />
    </>
  );
}
