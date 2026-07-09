import { type LogRun, type LogRunDeleteFilters, type LogRunTags } from "@/lib/api";
import { modelIdentityFromLegacyId } from "@/lib/selection";
import {
  type ChecklistOption,
  buildCountOptions,
  buildModelCountOptions,
  isDefaultScalarTag,
  logRunModelKey,
} from "@/features/workbench/state/logs/logs-selectors";

type NullableSelectionSet = Set<string> | null;

type LogRunSelections = {
  experiments: Set<string>;
  datasets: Set<string>;
  models: Set<string>;
  presets: Set<string>;
};

type CommonRunFacetOptions = {
  datasets: ChecklistOption[];
  models: ChecklistOption[];
  presets: ChecklistOption[];
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

export function firstAvailableSelection(availableValues: string[]) {
  return new Set(availableValues.slice(0, 1));
}

export function normalizeRunFacetSelection({
  selection,
  availableValues,
  selectFirstAvailable,
}: {
  selection: NullableSelectionSet;
  availableValues: string[];
  selectFirstAvailable: boolean;
}): NullableSelectionSet {
  if (selectFirstAvailable) {
    return firstAvailableSelection(availableValues);
  }

  const nextSelection = pruneSelectionToAvailableValues(selection, availableValues);
  if (
    selection !== null &&
    selection.size > 0 &&
    nextSelection !== null &&
    nextSelection.size === 0 &&
    availableValues.length > 0
  ) {
    return firstAvailableSelection(availableValues);
  }
  return nextSelection;
}

export function effectiveSelectionForAvailableValues(
  selection: NullableSelectionSet,
  availableValues: string[],
) {
  return selectionSetOrDefault(
    pruneSelectionToAvailableValues(selection, availableValues),
    new Set(availableValues),
  );
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

function intersectExperimentFacetValues({
  runs,
  selectedExperiments,
  valueForRun,
}: {
  runs: LogRun[];
  selectedExperiments: Set<string>;
  valueForRun: (run: LogRun) => string;
}) {
  let commonValues: Set<string> | null = null;

  for (const experiment of selectedExperiments) {
    const experimentValues = new Set<string>();
    for (const run of runs) {
      if (run.experiment === experiment) {
        experimentValues.add(valueForRun(run));
      }
    }

    if (commonValues === null) {
      commonValues = experimentValues;
      continue;
    }

    const nextCommonValues = new Set<string>();
    for (const value of commonValues) {
      if (experimentValues.has(value)) {
        nextCommonValues.add(value);
      }
    }
    commonValues = nextCommonValues;
  }

  return commonValues ?? new Set<string>();
}

export function buildCommonRunFacetOptions({
  runs,
  selectedExperiments,
}: {
  runs: LogRun[];
  selectedExperiments: Set<string>;
}): CommonRunFacetOptions {
  if (selectedExperiments.size === 0) {
    return {
      datasets: [],
      models: [],
      presets: [],
    };
  }

  const selectedExperimentRuns = runs.filter((run) =>
    selectedExperiments.has(run.experiment),
  );
  const commonDatasets = intersectExperimentFacetValues({
    runs,
    selectedExperiments,
    valueForRun: (run) => run.dataset,
  });
  const commonModels = intersectExperimentFacetValues({
    runs,
    selectedExperiments,
    valueForRun: logRunModelKey,
  });
  const commonPresets = intersectExperimentFacetValues({
    runs,
    selectedExperiments,
    valueForRun: (run) => run.preset,
  });

  return {
    datasets: buildCountOptions(
      selectedExperimentRuns.filter((run) => commonDatasets.has(run.dataset)),
      "dataset",
    ),
    models: buildModelCountOptions(
      selectedExperimentRuns.filter((run) => commonModels.has(logRunModelKey(run))),
    ),
    presets: buildCountOptions(
      selectedExperimentRuns.filter((run) => commonPresets.has(run.preset)),
      "preset",
    ),
  };
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
