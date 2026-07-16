import type { ConfigValue } from "@/lib/api/schemas";

export type TrainingSearchMode = "off" | "grid" | "random";

export type TrainingSearchState = {
  mode: TrainingSearchMode;
  selectedValues: Record<string, ConfigValue[]>;
  randomSamples: number;
};

export type TrainingSearchLockSummary = {
  lockedAxisCount: number;
  lockedAxisLabels: string[];
  lockedPresetLabels: string[];
  lockedAxesMessage: string;
  skippedSelectedAxisCount: number;
  skippedSelectedAxisLabels: string[];
  skippedSelectedAxisMessage: string;
};

export const DEFAULT_RANDOM_SEARCH_SAMPLES = 10;
export const LARGE_GRID_RUN_THRESHOLD = 100;

export const DEFAULT_TRAINING_SEARCH_STATE: TrainingSearchState = {
  mode: "off",
  selectedValues: {},
  randomSamples: DEFAULT_RANDOM_SEARCH_SAMPLES,
};
