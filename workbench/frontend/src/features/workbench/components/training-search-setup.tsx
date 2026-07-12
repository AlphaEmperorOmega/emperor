import { Grid2X2, Search, Shuffle, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { SegmentedControl } from "@/components/ui/segmented-control";
import { SectionHeading } from "@/components/ui/section-heading";
import { InlineStatus } from "@/features/workbench/components/shared/inline-status";
import { TrainingSearchAxisList } from "@/features/workbench/components/training-search-axis-list";
import { ViewModeButton } from "@/features/workbench/components/view-mode-button";
import { type TrainingRunPlanSearch } from "@/features/workbench/state/training/use-training-plan-state";
import { type ConfigValue, type SearchAxis } from "@/lib/api";
import { configValueEquals, valueIsSelected } from "@/lib/selection";
import {
  DEFAULT_RANDOM_SEARCH_SAMPLES,
  DEFAULT_TRAINING_SEARCH_STATE,
  type TrainingSearchMode,
  type TrainingSearchState,
} from "@/lib/training-search";

function updateSelectedValues(
  current: TrainingSearchState,
  key: string,
  values: ConfigValue[],
) {
  const next = { ...current.selectedValues };
  if (values.length === 0) {
    delete next[key];
  } else {
    next[key] = values;
  }
  return { ...current, selectedValues: next };
}

export function TrainingSearchSetup({
  search,
}: {
  search: TrainingRunPlanSearch;
}) {
  const {
    effective,
    axes,
    isLoading,
    conflictKeys,
    lockWarning,
    activeAxisCount,
    combinationCount,
    estimatedRunCount,
    unlockedAxisCount,
    unlockedAxes,
    disabledReason,
    update,
  } = search;
  const isDisabled = Boolean(disabledReason);

  function setMode(mode: TrainingSearchMode) {
    if (isDisabled) {
      return;
    }
    update(
      mode === "off"
        ? DEFAULT_TRAINING_SEARCH_STATE
        : {
            mode,
            selectedValues:
              effective.mode === "off" ? {} : effective.selectedValues,
            randomSamples:
              effective.randomSamples > 0
                ? effective.randomSamples
                : DEFAULT_RANDOM_SEARCH_SAMPLES,
          },
    );
  }

  function toggleAxis(axis: SearchAxis) {
    if (isDisabled || axis.locked) {
      return;
    }
    const isSelected =
      (effective.selectedValues[axis.key]?.length ?? 0) > 0;
    update(
      updateSelectedValues(
        effective,
        axis.key,
        isSelected ? [] : axis.values,
      ),
    );
  }

  function toggleValue(axis: SearchAxis, value: ConfigValue) {
    if (isDisabled || axis.locked) {
      return;
    }
    const values = effective.selectedValues[axis.key] ?? [];
    const selected = valueIsSelected(values, value);
    const nextValues = selected
      ? values.filter((candidate) => !configValueEquals(candidate, value))
      : [...values, value];
    update(updateSelectedValues(effective, axis.key, nextValues));
  }

  function selectAllAxes() {
    if (isDisabled) {
      return;
    }
    update({
      ...effective,
      selectedValues: Object.fromEntries(
        unlockedAxes.map((axis) => [axis.key, axis.values]),
      ),
    });
  }

  function clearAxes() {
    if (isDisabled) {
      return;
    }
    update({ ...effective, selectedValues: {} });
  }

  function updateRandomSamples(value: string) {
    if (isDisabled) {
      return;
    }
    update({
      ...effective,
      randomSamples: value === "" ? 0 : Number(value),
    });
  }

  return (
    <div className="grid gap-2">
      <div className="flex min-h-[38px] flex-wrap items-center justify-between gap-2">
        <SectionHeading
          icon={<Search className="h-[15px] w-[15px] text-violet" aria-hidden />}
          title="Grid Search"
        />
        <SegmentedControl aria-label="Training search mode">
          <ViewModeButton
            active={effective.mode === "off"}
            disabled={isDisabled}
            onClick={() => setMode("off")}
          >
            Off
          </ViewModeButton>
          <ViewModeButton
            active={effective.mode === "grid"}
            disabled={isDisabled}
            onClick={() => setMode("grid")}
          >
            <Grid2X2 className="h-3.5 w-3.5" aria-hidden />
            Grid
          </ViewModeButton>
          <ViewModeButton
            active={effective.mode === "random"}
            disabled={isDisabled}
            onClick={() => setMode("random")}
          >
            <Shuffle className="h-3.5 w-3.5" aria-hidden />
            Random
          </ViewModeButton>
        </SegmentedControl>
      </div>

      {disabledReason && (
        <div className="rounded-[9px] border border-line bg-white/[0.025] px-2.5 py-2 text-xs text-ink-faint">
          {disabledReason}
        </div>
      )}

      {effective.mode !== "off" && (
        <>
          {lockWarning && (
            <InlineStatus tone="warning" compact className="px-2.5 py-2 text-xs">
              {lockWarning}
            </InlineStatus>
          )}

          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex flex-wrap gap-1.5">
              <Badge>{activeAxisCount} axes</Badge>
              <Badge>{combinationCount} combinations</Badge>
              <Badge>{estimatedRunCount} planned runs</Badge>
              {effective.mode === "random" && (
                <Badge>{effective.randomSamples} samples</Badge>
              )}
            </div>
            <div className="flex shrink-0 items-center gap-1.5">
              <Button
                variant="secondary"
                onClick={selectAllAxes}
                disabled={isDisabled || unlockedAxisCount === 0}
                className="h-8 px-2.5 text-xs"
              >
                All axes
              </Button>
              <Button
                variant="ghost"
                onClick={clearAxes}
                disabled={isDisabled || activeAxisCount === 0}
                className="h-8 border border-line bg-white/[0.025] px-2.5 text-xs"
              >
                <X className="h-3.5 w-3.5" aria-hidden />
                Clear
              </Button>
            </div>
          </div>

          {effective.mode === "random" && (
            <label className="grid max-w-[14rem] gap-1.5">
              <span className="text-xs font-semibold tracking-[0.02em] text-ink-dim">
                Samples
              </span>
              <Input
                type="number"
                min={1}
                step={1}
                value={effective.randomSamples || ""}
                onChange={(event) =>
                  updateRandomSamples(event.target.value)
                }
                aria-label="Random search samples"
                disabled={isDisabled}
              />
            </label>
          )}

          {conflictKeys.length > 0 && (
            <div className="rounded-[9px] border border-amber/30 bg-amber/[0.055] px-2.5 py-2 text-xs text-amber">
              {conflictKeys.length} fixed override
              {conflictKeys.length === 1 ? "" : "s"} replaced by search axes.
            </div>
          )}

          <TrainingSearchAxisList
            axes={axes}
            search={effective}
            isLoading={isLoading}
            onToggleAxis={toggleAxis}
            onToggleValue={toggleValue}
          />
        </>
      )}
    </div>
  );
}
