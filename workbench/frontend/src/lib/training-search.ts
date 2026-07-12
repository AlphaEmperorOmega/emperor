import {
  type ConfigValue,
  type SearchAxis,
  type TrainingSearchCreateInput,
} from "@/lib/api";
import { configKeyToken, type OverrideValues } from "@/lib/config";
import {
  DEFAULT_RANDOM_SEARCH_SAMPLES,
  DEFAULT_TRAINING_SEARCH_STATE,
  LARGE_GRID_RUN_THRESHOLD,
  type TrainingSearchLockSummary,
  type TrainingSearchMode,
  type TrainingSearchState,
} from "@/lib/training-search-state";

export {
  DEFAULT_RANDOM_SEARCH_SAMPLES,
  DEFAULT_TRAINING_SEARCH_STATE,
  LARGE_GRID_RUN_THRESHOLD,
};
export type {
  TrainingSearchLockSummary,
  TrainingSearchMode,
  TrainingSearchState,
};

export function configValueKey(value: ConfigValue) {
  return value === null ? "null:null" : `${typeof value}:${String(value)}`;
}

export function configValueLabel(value: ConfigValue) {
  return value === null ? "None" : String(value);
}

function axisMap(axes: SearchAxis[]) {
  return new Map(axes.map((axis) => [configKeyToken(axis.key), axis]));
}

function uniqueStrings(values: string[]) {
  return Array.from(new Set(values.filter((value) => value.length > 0)));
}

export function formatTrainingSearchList(values: string[]) {
  if (values.length === 0) {
    return "";
  }
  if (values.length === 1) {
    return values[0];
  }
  if (values.length === 2) {
    return `${values[0]} and ${values[1]}`;
  }
  return `${values.slice(0, -1).join(", ")}, and ${values.at(-1)}`;
}

function selectedSearchEntries(search: TrainingSearchState) {
  if (search.mode === "off") {
    return [];
  }
  return Object.entries(search.selectedValues).filter(
    ([, values]) => values.length > 0,
  );
}

export function selectedSearchAxisKeys(search: TrainingSearchState) {
  return selectedSearchEntries(search).map(([key]) => key);
}

export function selectedSearchAxisCount(search: TrainingSearchState) {
  return selectedSearchAxisKeys(search).length;
}

export function estimateGridCombinations(selectedValues: Record<string, ConfigValue[]>) {
  const counts = Object.values(selectedValues)
    .map((values) => values.length)
    .filter((count) => count > 0);
  if (counts.length === 0) {
    return 0;
  }
  return counts.reduce((total, count) => total * count, 1);
}

export function estimatePlannedRuns(
  search: TrainingSearchState,
  selectedDatasetCount: number,
  selectedPresetCount = 1,
  options: { emptySearchRunsAsBase?: boolean } = {},
) {
  const presetCount = Math.max(0, selectedPresetCount);
  if (search.mode === "off") {
    return presetCount * selectedDatasetCount;
  }
  const combinations = estimateGridCombinations(search.selectedValues);
  if (combinations === 0 || selectedDatasetCount === 0 || presetCount === 0) {
    if (
      combinations === 0 &&
      options.emptySearchRunsAsBase &&
      selectedDatasetCount > 0 &&
      presetCount > 0
    ) {
      return presetCount * selectedDatasetCount;
    }
    return 0;
  }
  const runsPerDataset =
    search.mode === "random"
      ? Math.min(Math.max(1, search.randomSamples), combinations)
      : combinations;
  return runsPerDataset * selectedDatasetCount * presetCount;
}

export function buildEffectiveOverrides(
  overrides: OverrideValues,
  search: TrainingSearchState,
) {
  if (search.mode === "off") {
    return { ...overrides };
  }
  const searchKeys = new Set(selectedSearchAxisKeys(search).map(configKeyToken));
  return Object.fromEntries(
    Object.entries(overrides).filter(([key]) => !searchKeys.has(configKeyToken(key))),
  );
}

export function searchOverrideConflictKeys(
  overrides: OverrideValues,
  search: TrainingSearchState,
) {
  if (search.mode === "off") {
    return [];
  }
  const searchKeys = new Set(selectedSearchAxisKeys(search).map(configKeyToken));
  return Object.keys(overrides).filter((key) => searchKeys.has(configKeyToken(key)));
}

export function unlockedSearchAxes(axes: SearchAxis[]) {
  return axes.filter((axis) => !axis.locked && axis.values.length > 0);
}

export function effectiveUnlockedTrainingSearch(
  search: TrainingSearchState,
  axes: SearchAxis[],
) {
  if (search.mode === "off") {
    return search;
  }
  const axesByKey = axisMap(axes);
  const selectedValues = Object.fromEntries(
    Object.entries(search.selectedValues).filter(([key, values]) => {
      const axis = axesByKey.get(configKeyToken(key));
      return Boolean(axis && !axis.locked && values.length > 0);
    }).map(([key, values]) => [axesByKey.get(configKeyToken(key))?.key ?? key, values]),
  );
  return { ...search, selectedValues };
}

export function deriveTrainingSearchLockSummary(
  search: TrainingSearchState,
  axes: SearchAxis[],
): TrainingSearchLockSummary {
  const lockedAxes = axes.filter((axis) => axis.locked);
  const lockedAxisLabels = lockedAxes.map((axis) => axis.label);
  const lockedPresetLabels = uniqueStrings(
    lockedAxes.flatMap((axis) => axis.lockedByPresets ?? []),
  );
  const lockedAxisWord = lockedAxes.length === 1 ? "axis" : "axes";
  const lockedPresetText =
    lockedPresetLabels.length > 0
      ? ` for ${formatTrainingSearchList(lockedPresetLabels)}`
      : "";
  const lockedLabelText =
    lockedAxisLabels.length > 0 && lockedAxisLabels.length <= 3
      ? `: ${formatTrainingSearchList(lockedAxisLabels)}`
      : "";
  const lockedAxesMessage =
    lockedAxes.length > 0
      ? `${lockedAxes.length} preset-owned ${lockedAxisWord} will be skipped${lockedPresetText}${lockedLabelText}.`
      : "";

  const axesByKey = axisMap(axes);
  const skippedSelectedAxes = selectedSearchEntries(search)
    .map(([key]) => axesByKey.get(configKeyToken(key)))
    .filter((axis): axis is SearchAxis => Boolean(axis?.locked));
  const skippedSelectedAxisLabels = skippedSelectedAxes.map((axis) => axis.label);
  const skippedSelectedAxisCount = skippedSelectedAxes.length;
  const skippedSelectedAxisMessage =
    skippedSelectedAxisCount > 0
      ? `${skippedSelectedAxisCount} selected ${
          skippedSelectedAxisCount === 1 ? "axis was" : "axes were"
        } skipped because ${
          skippedSelectedAxisCount === 1
            ? "a selected preset owns it"
            : "selected presets own them"
        }.`
      : "";

  return {
    lockedAxisCount: lockedAxes.length,
    lockedAxisLabels,
    lockedPresetLabels,
    lockedAxesMessage,
    skippedSelectedAxisCount,
    skippedSelectedAxisLabels,
    skippedSelectedAxisMessage,
  };
}

export function validateTrainingSearch(
  search: TrainingSearchState,
  axes: SearchAxis[],
  options: { allowEmptySelected?: boolean } = {},
) {
  if (search.mode === "off") {
    return { ready: true, message: "" };
  }

  const axesByKey = axisMap(axes);
  const entries = selectedSearchEntries(search);
  if (entries.length === 0) {
    if (options.allowEmptySelected) {
      return { ready: true, message: "" };
    }
    return { ready: false, message: "Select at least one search axis." };
  }
  for (const [key, values] of entries) {
    const axis = axesByKey.get(configKeyToken(key));
    if (!axis) {
      return { ready: false, message: `Unknown search axis: ${key}.` };
    }
    if (axis.locked) {
      return { ready: false, message: `${axis.label} is locked by this preset.` };
    }
    if (values.length === 0) {
      return { ready: false, message: `${axis.label} needs at least one value.` };
    }
  }
  if (search.mode === "random" && !Number.isInteger(search.randomSamples)) {
    return { ready: false, message: "Random samples must be a whole number." };
  }
  if (search.mode === "random" && search.randomSamples < 1) {
    return { ready: false, message: "Random samples must be at least 1." };
  }
  return { ready: true, message: "" };
}

export function buildTrainingSearchPayload(
  search: TrainingSearchState,
): TrainingSearchCreateInput | undefined {
  if (search.mode === "off") {
    return undefined;
  }
  const values = Object.fromEntries(
    Object.entries(search.selectedValues).filter(([, selected]) => selected.length > 0),
  );
  if (Object.keys(values).length === 0) {
    return undefined;
  }
  return search.mode === "random"
    ? { mode: "random", values, randomSamples: search.randomSamples }
    : { mode: "grid", values };
}

export function trainingSearchModeLabel(mode: TrainingSearchMode) {
  if (mode === "grid") {
    return "Grid";
  }
  if (mode === "random") {
    return "Random";
  }
  return "Off";
}
