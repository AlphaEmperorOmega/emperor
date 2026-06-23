import { type LogRun, type LogRunDeleteFilters, type LogRunTags } from "@/lib/api";
import { modelIdentityFromLegacyId } from "@/lib/selection";
import {
  type ChecklistOption,
  buildCountOptions,
  buildModelCountOptions,
  isDefaultScalarTag,
  logRunModelKey,
} from "@/features/viewer/state/logs/logs-selectors";

type NullableSelectionSet = Set<string> | null;

type LogRunSelections = {
  experiments: Set<string>;
  datasets: Set<string>;
  models: Set<string>;
  presets: Set<string>;
};

type ExperimentFacetSeedSelections = {
  loadedExperiments: Set<string>;
  datasets: Set<string>;
  models: Set<string>;
  presets: Set<string>;
};

type ExperimentScalarTagSeedSelection = {
  loadedExperiments: Set<string>;
  selectedTags: NullableSelectionSet;
};

export function sortedSelectionValues(values: Set<string>) {
  return Array.from(values).sort((a, b) => a.localeCompare(b));
}

export function selectionSetOrEmpty(selection: NullableSelectionSet) {
  return selection ?? new Set<string>();
}

export function selectionSetOrDefault(
  selection: NullableSelectionSet,
  defaultSelection: Set<string>,
) {
  return selection ?? defaultSelection;
}

export function pruneSelectionToAvailableValues(
  selection: NullableSelectionSet,
  availableValues: string[],
): NullableSelectionSet {
  if (selection === null) {
    return null;
  }
  const available = new Set(availableValues);
  const next = new Set(
    Array.from(selection).filter((value) => available.has(value)),
  );
  if (next.size === selection.size) {
    return selection;
  }
  return next;
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
  };
}

function nullableSelectionIncludes(
  selection: NullableSelectionSet,
  value: string,
) {
  return selection === null || selection.has(value);
}

export function buildExperimentFacetSeedSelections({
  runs,
  pendingExperiments,
  selectedDatasets,
  selectedModels,
  selectedPresets,
}: {
  runs: LogRun[];
  pendingExperiments: Set<string>;
  selectedDatasets: NullableSelectionSet;
  selectedModels: NullableSelectionSet;
  selectedPresets: NullableSelectionSet;
}): ExperimentFacetSeedSelections {
  const selections: ExperimentFacetSeedSelections = {
    loadedExperiments: new Set(),
    datasets: new Set(),
    models: new Set(),
    presets: new Set(),
  };

  for (const experiment of pendingExperiments) {
    const experimentRuns = runs.filter((run) => run.experiment === experiment);
    if (experimentRuns.length === 0) {
      continue;
    }

    selections.loadedExperiments.add(experiment);
    const hasVisibleRun = experimentRuns.some(
      (run) =>
        nullableSelectionIncludes(selectedDatasets, run.dataset) &&
        nullableSelectionIncludes(selectedModels, logRunModelKey(run)) &&
        nullableSelectionIncludes(selectedPresets, run.preset),
    );
    if (hasVisibleRun) {
      continue;
    }

    for (const run of experimentRuns) {
      selections.datasets.add(run.dataset);
      selections.models.add(logRunModelKey(run));
      selections.presets.add(run.preset);
    }
  }

  return selections;
}

function orderedExperimentScalarTags({
  tagRuns,
  tagOptionValues,
}: {
  tagRuns: LogRunTags[];
  tagOptionValues: string[];
}) {
  const scalarTags = new Set<string>();
  for (const runTags of tagRuns) {
    for (const tag of runTags.scalarTags) {
      scalarTags.add(tag);
    }
  }
  return tagOptionValues.filter((tag) => scalarTags.has(tag));
}

function appendUniqueTags(target: string[], tags: string[]) {
  const existing = new Set(target);
  for (const tag of tags) {
    if (!existing.has(tag)) {
      existing.add(tag);
      target.push(tag);
    }
  }
}

export function buildExperimentScalarTagSeedSelection({
  visibleRuns,
  tagRuns,
  pendingExperiments,
  selectedTags,
  tagOptionValues,
  selectAllLimit,
}: {
  visibleRuns: LogRun[];
  tagRuns: LogRunTags[] | undefined;
  pendingExperiments: Set<string>;
  selectedTags: NullableSelectionSet;
  tagOptionValues: string[];
  selectAllLimit: number;
}): ExperimentScalarTagSeedSelection {
  const selections: ExperimentScalarTagSeedSelection = {
    loadedExperiments: new Set(),
    selectedTags,
  };
  const tagsByRunId = new Map((tagRuns ?? []).map((runTags) => [runTags.runId, runTags]));
  const replacementTags: string[] = [];

  for (const experiment of pendingExperiments) {
    const experimentRuns = visibleRuns.filter((run) => run.experiment === experiment);
    if (experimentRuns.length === 0) {
      continue;
    }

    const experimentTagRuns: LogRunTags[] = [];
    let hasAllVisibleRunTags = true;
    for (const run of experimentRuns) {
      const runTags = tagsByRunId.get(run.id);
      if (!runTags) {
        hasAllVisibleRunTags = false;
        break;
      }
      experimentTagRuns.push(runTags);
    }
    if (!hasAllVisibleRunTags) {
      continue;
    }

    selections.loadedExperiments.add(experiment);
    if (selectedTags === null) {
      continue;
    }

    const experimentTags = orderedExperimentScalarTags({
      tagRuns: experimentTagRuns,
      tagOptionValues,
    });
    if (experimentTags.length === 0) {
      continue;
    }

    if (experimentTags.some((tag) => selectedTags.has(tag))) {
      continue;
    }

    const defaultTags = experimentTags.filter((tag) => isDefaultScalarTag(tag));
    appendUniqueTags(
      replacementTags,
      defaultTags.length > 0
        ? defaultTags
        : experimentTags.slice(0, selectAllLimit),
    );
  }

  if (replacementTags.length > 0) {
    selections.selectedTags = new Set(replacementTags.slice(0, selectAllLimit));
  }

  return selections;
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
      selections.presets.has(run.preset),
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

export function buildLogRunDeleteFilters(runs: LogRun[]): LogRunDeleteFilters {
  const experiments = new Set(runs.map((run) => run.experiment));
  const datasets = new Set(runs.map((run) => run.dataset));
  const models = new Set(runs.map((run) => logRunModelKey(run)));
  const presets = new Set(runs.map((run) => run.preset));
  const runIds = new Set(runs.map((run) => run.id));

  return {
    experiments: sortedSelectionValues(experiments),
    datasets: sortedSelectionValues(datasets),
    models: sortedSelectionValues(models).map(modelIdentityFromLegacyId),
    presets: sortedSelectionValues(presets),
    runIds: sortedSelectionValues(runIds),
  };
}
