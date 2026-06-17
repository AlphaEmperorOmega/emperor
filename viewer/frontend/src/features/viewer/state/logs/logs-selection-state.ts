import { type LogRun, type LogRunDeleteFilters } from "@/lib/api";
import { modelIdentityFromLegacyId } from "@/lib/selection";
import {
  type ChecklistOption,
  buildCountOptions,
  buildModelCountOptions,
  logRunModelKey,
} from "@/features/viewer/state/logs/logs-selectors";

type NullableSelectionSet = Set<string> | null;

type LogRunSelections = {
  experiments: Set<string>;
  datasets: Set<string>;
  models: Set<string>;
  presets: Set<string>;
  runIds: Set<string>;
};

export function sortedSelectionValues(values: Set<string>) {
  return Array.from(values).sort((a, b) => a.localeCompare(b));
}

export function selectionSetOrEmpty(selection: NullableSelectionSet) {
  return selection ?? new Set<string>();
}

export function buildInitialExperimentSelection({
  experimentOptions,
  startedExperiments,
}: {
  experimentOptions: ChecklistOption[];
  startedExperiments: Set<string>;
}) {
  return new Set([
    ...experimentOptions.map((option) => option.value),
    ...startedExperiments,
  ]);
}

export function buildInitialRunFacetSelection(
  runs: LogRun[],
  key: keyof Pick<LogRun, "dataset" | "model" | "preset">,
) {
  if (key === "model") {
    return new Set(buildModelCountOptions(runs).map((option) => option.value));
  }
  return new Set(buildCountOptions(runs, key).map((option) => option.value));
}

export function buildInitialRunIdSelection(runs: LogRun[]) {
  return new Set(runs.map((run) => run.id));
}

export function startedRunSelections({
  runs,
  startedExperiments,
}: {
  runs: LogRun[];
  startedExperiments: Set<string>;
}) {
  const startedRuns = runs.filter((run) => startedExperiments.has(run.experiment));
  return {
    hasStartedRuns: startedRuns.length > 0,
    experiments: new Set(startedRuns.map((run) => run.experiment)),
    datasets: new Set(startedRuns.map((run) => run.dataset)),
    models: new Set(startedRuns.map((run) => logRunModelKey(run))),
    presets: new Set(startedRuns.map((run) => run.preset)),
    runIds: new Set(startedRuns.map((run) => run.id)),
  };
}

export function addValuesToInitializedSelection(
  selection: NullableSelectionSet,
  values: Set<string>,
): NullableSelectionSet {
  if (selection === null || values.size === 0) {
    return selection;
  }
  const next = new Set(selection);
  for (const value of values) {
    next.add(value);
  }
  return next;
}

export function addValueToInitializedSelection(
  selection: NullableSelectionSet,
  value: string,
): NullableSelectionSet {
  if (selection === null || selection.has(value)) {
    return selection;
  }
  const next = new Set(selection);
  next.add(value);
  return next;
}

export function removeValueFromSelection({
  selection,
  fallbackValues,
  value,
}: {
  selection: NullableSelectionSet;
  fallbackValues: string[];
  value: string;
}) {
  const next = new Set(selection ?? fallbackValues);
  next.delete(value);
  return next;
}

export function removeValuesFromSelection({
  selection,
  fallbackValues,
  values,
}: {
  selection: NullableSelectionSet;
  fallbackValues: string[];
  values: Set<string>;
}) {
  const next = new Set(selection ?? fallbackValues);
  for (const value of values) {
    next.delete(value);
  }
  return next;
}

export function filterVisibleLogRuns(
  runs: LogRun[],
  selections: LogRunSelections,
) {
  return runs.filter(
    (run) =>
      selections.experiments.has(run.experiment) &&
      selections.datasets.has(run.dataset) &&
      selections.models.has(logRunModelKey(run)) &&
      selections.presets.has(run.preset) &&
      selections.runIds.has(run.id),
  );
}

export function nextSelectedDetailRunId(
  selectedDetailRunId: string | null,
  visibleRuns: LogRun[],
) {
  if (visibleRuns.length === 0) {
    return null;
  }
  if (
    selectedDetailRunId &&
    visibleRuns.some((run) => run.id === selectedDetailRunId)
  ) {
    return selectedDetailRunId;
  }
  return visibleRuns[0].id;
}

export function pruneDeletedDetailRunId({
  selectedDetailRunId,
  deletedRunIds,
}: {
  selectedDetailRunId: string | null;
  deletedRunIds: Set<string>;
}) {
  return selectedDetailRunId && deletedRunIds.has(selectedDetailRunId)
    ? null
    : selectedDetailRunId;
}

export function removeStartedExperiment(
  startedExperiments: Set<string>,
  experiment: string,
) {
  if (!startedExperiments.has(experiment)) {
    return startedExperiments;
  }
  const next = new Set(startedExperiments);
  next.delete(experiment);
  return next;
}

export function buildLogRunDeleteFilters({
  experiments,
  datasets,
  models,
  presets,
  runIds,
}: LogRunSelections): LogRunDeleteFilters {
  return {
    experiments: sortedSelectionValues(experiments),
    datasets: sortedSelectionValues(datasets),
    models: sortedSelectionValues(models).map(modelIdentityFromLegacyId),
    presets: sortedSelectionValues(presets),
    runIds: sortedSelectionValues(runIds),
  };
}
