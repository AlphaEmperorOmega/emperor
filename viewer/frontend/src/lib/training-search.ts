import {
  type ConfigValue,
  type SearchAxis,
  type TrainingSearchCreateInput,
} from "@/lib/api";
import { type OverrideValues } from "@/lib/config";

export type TrainingSearchMode = "off" | "grid" | "random";

export type TrainingSearchState = {
  mode: TrainingSearchMode;
  selectedValues: Record<string, ConfigValue[]>;
  randomSamples: number;
};

export const DEFAULT_RANDOM_SEARCH_SAMPLES = 10;
export const LARGE_GRID_RUN_THRESHOLD = 100;

export const DEFAULT_TRAINING_SEARCH_STATE: TrainingSearchState = {
  mode: "off",
  selectedValues: {},
  randomSamples: DEFAULT_RANDOM_SEARCH_SAMPLES,
};

export function configValueKey(value: ConfigValue) {
  return value === null ? "null:null" : `${typeof value}:${String(value)}`;
}

export function configValueLabel(value: ConfigValue) {
  return value === null ? "None" : String(value);
}

function axisMap(axes: SearchAxis[]) {
  return new Map(axes.map((axis) => [axis.key, axis]));
}

export function selectedSearchAxisKeys(search: TrainingSearchState) {
  if (search.mode === "off") {
    return [];
  }
  return Object.entries(search.selectedValues)
    .filter(([, values]) => values.length > 0)
    .map(([key]) => key);
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
) {
  const presetCount = Math.max(0, selectedPresetCount);
  if (search.mode === "off") {
    return presetCount * selectedDatasetCount;
  }
  const combinations = estimateGridCombinations(search.selectedValues);
  if (combinations === 0 || selectedDatasetCount === 0 || presetCount === 0) {
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
  const searchKeys = new Set(selectedSearchAxisKeys(search));
  return Object.fromEntries(
    Object.entries(overrides).filter(([key]) => !searchKeys.has(key)),
  );
}

export function searchOverrideConflictKeys(
  overrides: OverrideValues,
  search: TrainingSearchState,
) {
  if (search.mode === "off") {
    return [];
  }
  const searchKeys = new Set(selectedSearchAxisKeys(search));
  return Object.keys(overrides).filter((key) => searchKeys.has(key));
}

export function validateTrainingSearch(
  search: TrainingSearchState,
  axes: SearchAxis[],
) {
  if (search.mode === "off") {
    return { ready: true, message: "" };
  }

  const axesByKey = axisMap(axes);
  const entries = Object.entries(search.selectedValues).filter(
    ([, values]) => values.length > 0,
  );
  if (entries.length === 0) {
    return { ready: false, message: "Select at least one search axis." };
  }
  for (const [key, values] of entries) {
    const axis = axesByKey.get(key);
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
