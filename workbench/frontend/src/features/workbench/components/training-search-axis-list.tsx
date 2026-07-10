import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import { type ConfigValue, type SearchAxis } from "@/lib/api";
import {
  configValueKey,
  configValueLabel,
  formatTrainingSearchList,
  type TrainingSearchState,
} from "@/lib/training-search";
import { valueIsSelected } from "@/lib/selection";
import { cn } from "@/lib/utils";
import { InlineStatus } from "@/features/workbench/components/shared/inline-status";
import { workbenchStatusCopy } from "@/features/workbench/components/shared/status-copy";

function selectedValueCount(axis: SearchAxis, search: TrainingSearchState) {
  return search.selectedValues[axis.key]?.length ?? 0;
}

export function TrainingSearchAxisList({
  axes,
  search,
  isLoading,
  onToggleAxis,
  onToggleValue,
}: {
  axes: SearchAxis[];
  search: TrainingSearchState;
  isLoading: boolean;
  onToggleAxis: (axis: SearchAxis) => void;
  onToggleValue: (axis: SearchAxis, value: ConfigValue) => void;
}) {
  return (
    <div className="grid min-w-0 grid-cols-1 gap-2">
      {axes.map((axis) => {
        const selectedCount = selectedValueCount(axis, search);
        const axisSelected = selectedCount > 0;
        const lockedByPresets = axis.lockedByPresets ?? [];
        const lockReasons =
          axis.lockReasons && axis.lockReasons.length > 0
            ? axis.lockReasons
            : axis.lockedReason
              ? [axis.lockedReason]
              : [];
        const hasLockedValue =
          axis.locked &&
          Object.prototype.hasOwnProperty.call(axis, "lockedValue") &&
          axis.lockedValue !== undefined;
        return (
          <div
            key={axis.key}
            className={cn(
              "grid min-w-0 content-start gap-2 rounded-[10px] border px-2.5 py-2 transition",
              axis.locked
                ? "border-amber/35 bg-amber/[0.045] opacity-75"
                : axisSelected
                  ? "border-violet/35 bg-violet/10"
                  : "border-line bg-white/[0.018]",
            )}
          >
            <label
              className={cn(
                "grid min-w-0 grid-cols-[auto_minmax(0,1fr)_auto] items-start gap-2",
                axis.locked ? "cursor-not-allowed" : "cursor-pointer",
              )}
            >
              <Checkbox
                checked={axisSelected}
                disabled={axis.locked}
                onCheckedChange={() => onToggleAxis(axis)}
                aria-label={`search axis ${axis.key}`}
              />
              <span className="grid min-w-0 gap-0.5">
                <span className="truncate text-sm font-semibold text-ink">
                  {axis.label}
                </span>
                <span className="truncate text-xs text-ink-dim">{axis.section}</span>
              </span>
              <span className="flex shrink-0 items-center gap-1">
                {axis.locked && <Badge variant="preset">preset</Badge>}
                <Badge>
                  {selectedCount}/{axis.values.length}
                </Badge>
              </span>
            </label>

            {axis.locked && (
              <div className="grid gap-1 text-xs leading-4 text-ink-dim [overflow-wrap:anywhere]">
                <div>
                  {hasLockedValue
                    ? `Locked value: ${configValueLabel(axis.lockedValue ?? null)}`
                    : "Preset-owned axis"}
                  {lockedByPresets.length > 0
                    ? ` by ${formatTrainingSearchList(lockedByPresets)}`
                    : ""}
                  .
                </div>
                {lockReasons[0] && <div>{lockReasons[0]}</div>}
              </div>
            )}

            <div className="flex min-w-0 flex-wrap gap-1.5">
              {axis.values.map((value) => {
                const checked = valueIsSelected(
                  search.selectedValues[axis.key] ?? [],
                  value,
                );
                const label = configValueLabel(value);
                return (
                  <label
                    key={`${axis.key}-${configValueKey(value)}`}
                    title={label}
                    className={cn(
                      "inline-flex min-h-8 min-w-0 max-w-full cursor-pointer items-center gap-1.5 overflow-hidden rounded-[8px] border px-2 py-1 text-xs transition",
                      checked
                        ? "border-violet/35 bg-violet/15 text-violet"
                        : "border-line bg-black/10 text-ink-dim hover:border-violet/25 hover:text-ink",
                      axis.locked && "cursor-not-allowed opacity-65",
                    )}
                  >
                    <Checkbox
                      checked={checked}
                      disabled={axis.locked}
                      onCheckedChange={() => onToggleValue(axis, value)}
                      aria-label={`search value ${axis.key} ${label}`}
                      className="h-4 w-4 rounded-[5px]"
                    />
                    <span className="min-w-0 truncate font-mono">{label}</span>
                  </label>
                );
              })}
            </div>
          </div>
        );
      })}
      {!isLoading && axes.length === 0 && (
        <InlineStatus compact>
          {workbenchStatusCopy.empty.searchAxes}
        </InlineStatus>
      )}
      {isLoading && (
        <InlineStatus compact>
          {workbenchStatusCopy.loading.searchAxes}
        </InlineStatus>
      )}
    </div>
  );
}
