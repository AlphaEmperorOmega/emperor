import {
  BarChart3,
  CheckCircle2,
  Plus,
  RotateCcw,
  Table2,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { ViewModeButton } from "@/features/viewer/components/view-mode-button";
import {
  MultiSelectDropdown,
  type MultiSelectDropdownOption,
} from "@/features/viewer/components/screen/multi-select-dropdown";
import { SelectOnlyDropdown } from "@/features/viewer/components/screen/select-only-dropdown";
import { InlineStatus } from "@/features/viewer/components/shared/inline-status";
import { StatChip } from "@/features/viewer/components/shared/stat-chip";
import { SurfacePanel } from "@/features/viewer/components/shared/surface-panel";
import {
  compactRunLabel,
  fullRunLabel,
  runMetadataLabel,
} from "@/features/viewer/components/compare-workspace/compare-run-derive";
import {
  type CompareRunTargetEntry,
  type CompareRunTargetEntryData,
  type ExperimentCompareWorkspaceState,
} from "./use-experiment-compare-workspace-state";

function countLabel(count: number, noun: string) {
  return `${count} ${count === 1 ? noun : `${noun}s`}`;
}

function emptyRunTargetMessage(comparison: ExperimentCompareWorkspaceState) {
  const hasRuns = comparison.runs.length > 0;
  const hasEligibleRuns = comparison.eligibleRuns.length > 0;

  if (comparison.runsQuery.isLoading || comparison.tagsQuery.isLoading) {
    return "Loading Training Runs.";
  }
  if (!hasRuns) {
    return "No historical logs are available.";
  }
  if (!hasEligibleRuns) {
    return "Historical logs exist, but none expose scalar metrics.";
  }
  return "No Training Run targets.";
}

export function CompareRunSelector({
  comparison,
  onOpenLogs,
}: {
  comparison: ExperimentCompareWorkspaceState;
  onOpenLogs: () => void;
}) {
  return (
    <div className="grid gap-3">
      <SurfacePanel as="section" padding="spacious" className="min-w-0">
        <div className="grid gap-4">
          <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="flex min-w-0 flex-wrap items-center gap-2">
                <h3 className="text-sm font-bold text-ink">Training Run Targets</h3>
                <Badge variant="violet">
                  {comparison.entryData.length}/{comparison.maxRunCount}{" "}
                  {comparison.entryData.length === 1 ? "target" : "targets"}
                </Badge>
              </div>
              <div className="mt-0.5 truncate font-mono text-xs text-ink-faint">
                {comparison.readyEntryCount} resolved to scalar-capable runs
              </div>
            </div>
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

          {comparison.hasMoreRuns && (
            <InlineStatus tone="warning" compact>
              Showing the first loaded page of historical runs.
            </InlineStatus>
          )}

          {comparison.entryData.length === 0 ? (
            <div className="grid gap-3 rounded-[10px] border border-line-soft bg-black/16 p-4">
              <InlineStatus compact>{emptyRunTargetMessage(comparison)}</InlineStatus>
              {comparison.runs.length === 0 && !comparison.runsQuery.isLoading && (
                <div className="flex flex-wrap items-center gap-2">
                  <Button variant="secondary" onClick={onOpenLogs}>
                    Open Logs
                  </Button>
                </div>
              )}
            </div>
          ) : (
            <div className="grid items-start gap-3 md:grid-cols-2 xl:grid-cols-4">
              {comparison.entryData.map((entryData, index) => (
                <CompareRunTargetCard
                  key={entryData.entry.id}
                  entryData={entryData}
                  index={index}
                  canRemove={comparison.canRemoveEntry}
                  onRemove={comparison.removeEntry}
                  onUpdate={comparison.updateEntry}
                />
              ))}
            </div>
          )}
        </div>
      </SurfacePanel>

      <CompareMetricSelector comparison={comparison} />
    </div>
  );
}

function CompareRunTargetCard({
  canRemove,
  entryData,
  index,
  onRemove,
  onUpdate,
}: {
  canRemove: boolean;
  entryData: CompareRunTargetEntryData;
  index: number;
  onRemove: (id: string) => void;
  onUpdate: (id: string, patch: Partial<CompareRunTargetEntry>) => void;
}) {
  const { entry, resolvedRun } = entryData;

  return (
    <SurfacePanel as="article" padding="spacious" className="min-w-0">
      <div className="grid gap-3">
        <div className="flex min-w-0 items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="text-xs font-bold uppercase tracking-[0.08em] text-ink-faint">
              Run Target {index + 1}
            </div>
            <div className="mt-1 truncate font-mono text-sm font-semibold text-ink">
              {entry.experiment || "No experiment"}
            </div>
          </div>
          <Button
            variant="ghost"
            className="h-8 w-8 shrink-0 px-0"
            aria-label={`Remove run target ${index + 1}`}
            onClick={() => onRemove(entry.id)}
            disabled={!canRemove}
          >
            <X className="h-4 w-4" aria-hidden />
          </Button>
        </div>

        <RunTargetSelect
          label="Experiment"
          value={entry.experiment}
          options={entryData.experimentOptions}
          onChange={(experiment) => onUpdate(entry.id, { experiment })}
          placeholder="Select experiment"
        />

        <RunTargetSelect
          label="Model Type"
          value={entry.modelType}
          options={entryData.modelTypeOptions}
          onChange={(modelType) => onUpdate(entry.id, { modelType })}
          placeholder="Select model type"
        />

        <RunTargetSelect
          label="Model"
          value={entry.model}
          options={entryData.modelOptions}
          onChange={(model) => onUpdate(entry.id, { model })}
          placeholder="Select model"
        />

        <RunTargetSelect
          label="Preset"
          value={entry.preset}
          options={entryData.presetOptions}
          onChange={(preset) => onUpdate(entry.id, { preset })}
          placeholder="Select preset"
        />

        <RunTargetSelect
          label="Dataset"
          value={entry.dataset}
          options={entryData.datasetOptions}
          onChange={(dataset) => onUpdate(entry.id, { dataset })}
          placeholder="Select dataset"
        />

        {resolvedRun ? (
          <div className="grid min-w-0 gap-2 rounded-[10px] border border-line-soft bg-black/16 p-3">
            <div className="flex min-w-0 items-center justify-between gap-2">
              <Badge variant="violet" className="min-w-0 truncate whitespace-nowrap">
                <CheckCircle2 className="mr-1 h-3.5 w-3.5" aria-hidden />
                Latest run
              </Badge>
              <Badge className="min-w-0 truncate whitespace-nowrap">
                {countLabel(entryData.scalarTagCount, "scalar tag")}
              </Badge>
            </div>
            <div
              className="min-w-0 truncate font-mono text-xs font-semibold text-ink"
              title={fullRunLabel(resolvedRun)}
            >
              {compactRunLabel(resolvedRun)}
            </div>
            <div
              className="min-w-0 truncate font-mono text-[11px] text-ink-faint"
              title={runMetadataLabel(resolvedRun)}
            >
              {runMetadataLabel(resolvedRun)}
            </div>
          </div>
        ) : (
          <InlineStatus compact>{entryData.status}</InlineStatus>
        )}
      </div>
    </SurfacePanel>
  );
}

function RunTargetSelect({
  label,
  onChange,
  options,
  placeholder,
  value,
}: {
  label: string;
  value: string;
  options: CompareRunTargetEntryData["experimentOptions"];
  onChange: (value: string) => void;
  placeholder: string;
}) {
  return (
    <div className="grid min-w-0 gap-1.5">
      <span className="text-xs font-semibold text-ink-dim">{label}</span>
      <SelectOnlyDropdown
        label={label}
        className="min-w-0"
        value={value}
        options={options.map((option) => ({
          value: option.value,
          label: option.label,
          description: option.detail,
          disabled: option.disabled,
        }))}
        onChange={onChange}
        placeholder={placeholder}
        searchPlaceholder={`Search ${label.toLowerCase()}`}
        noResultsMessage={`No ${label.toLowerCase()} options`}
        disabled={options.length === 0}
      />
    </div>
  );
}

function CompareMetricSelector({
  comparison,
}: {
  comparison: ExperimentCompareWorkspaceState;
}) {
  const selectedCount = comparison.selectedMetricTags.length;
  const availableCount = comparison.metricOptions.length;
  const dropdownOptions: MultiSelectDropdownOption[] = comparison.metricOptions.map(
    (option) => ({
      value: option.value,
      label: option.label,
      description: option.detail,
      meta:
        option.count === undefined ? undefined : (
          <StatChip size="xs" className="shrink-0">
            {countLabel(option.count, "run")}
          </StatChip>
        ),
    }),
  );

  return (
    <SurfacePanel as="section" padding="spacious" className="min-w-0">
      <div className="grid gap-3">
        <div className="flex min-w-0 flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex min-w-0 flex-wrap items-center gap-2">
              <h3 className="text-sm font-bold text-ink">Scalar Tags</h3>
              <Badge variant="violet">
                {selectedCount} / {availableCount} selected
              </Badge>
            </div>
          </div>
          <SegmentedControl variant="tablist" className="w-fit">
            <ViewModeButton
              variant="tab"
              active={comparison.view === "graphs"}
              onClick={() => comparison.setView("graphs")}
            >
              <BarChart3 className="h-3.5 w-3.5" aria-hidden />
              Graphs
            </ViewModeButton>
            <ViewModeButton
              variant="tab"
              active={comparison.view === "data"}
              onClick={() => comparison.setView("data")}
            >
              <Table2 className="h-3.5 w-3.5" aria-hidden />
              Data
            </ViewModeButton>
          </SegmentedControl>
        </div>

        <div className="grid gap-2">
          <MultiSelectDropdown
            label="Scalar Tags"
            values={comparison.selectedMetricTags}
            options={dropdownOptions}
            onChange={comparison.setMetricTags}
            placeholder="Select scalar tags"
            searchPlaceholder="Search scalar tags"
            emptyMessage="Resolve at least one scalar-capable target to choose scalar tags."
            noResultsMessage="No scalar tags"
          />
          <div className="flex flex-wrap items-center justify-end gap-2">
            <Button
              variant="secondary"
              disabled={availableCount === 0}
              onClick={comparison.selectDefaultMetrics}
            >
              Defaults
            </Button>
            <Button
              variant="secondary"
              disabled={availableCount === 0}
              onClick={comparison.selectAllMetrics}
            >
              All
            </Button>
            <Button
              variant="ghost"
              className="border border-line bg-white/[0.025]"
              disabled={selectedCount === 0}
              onClick={comparison.selectNoMetrics}
            >
              None
            </Button>
          </div>
        </div>
      </div>
    </SurfacePanel>
  );
}
