import { AlertTriangle, GitCompare, Plus, RotateCcw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { EdgeCard } from "@/components/ui/edge-card";
import { CompareTargetCard } from "@/components/features/viewer/compare-workspace/compare-target-card";
import {
  ComparisonTable,
  ConfigDiffTable,
} from "@/components/features/viewer/compare-workspace/compare-summary-tables";
import { useCompareWorkspaceState } from "@/components/features/viewer/compare-workspace/use-compare-workspace-state";
import { InlineStatus } from "@/components/features/viewer/shared/inline-status";
import { SectionHeading } from "@/components/features/viewer/shared/section-heading";
import { errorMessage } from "@/lib/utils";

export function CompareWorkspace({
  onUseTarget,
}: {
  onUseTarget: () => void;
}) {
  const comparison = useCompareWorkspaceState({ onUseTarget });

  if (comparison.catalog.isLoading) {
    return (
      <div className="grid h-full place-items-center p-6">
        <InlineStatus busy>Loading model catalog</InlineStatus>
      </div>
    );
  }

  if (comparison.catalog.isError) {
    return (
      <div className="grid h-full place-items-center p-6">
        <InlineStatus tone="danger" role="alert">
          {errorMessage(comparison.catalog.error)}
        </InlineStatus>
      </div>
    );
  }

  return (
    <div className="min-h-0 overflow-y-auto bg-bg-2/60 p-4 lg:p-5">
      <div className="mx-auto grid max-w-[1480px] gap-4">
        <EdgeCard className="rounded-card p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <SectionHeading
              as="h2"
              className="normal-case tracking-normal text-ink"
              icon={<GitCompare className="h-4 w-4 text-violet" aria-hidden />}
              title={<span className="text-base font-bold">Model Comparison</span>}
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
        </EdgeCard>

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
      </div>
    </div>
  );
}
