import {
  type TrainingJobCreateInput,
  type TrainingRunPlan,
  type TrainingRunPlanCreateInput,
  type TrainingSearchCreateInput,
} from "@/lib/api";
import { type OverrideValues } from "@/lib/config";

export type TrainingRunPlanRequestInput = {
  canPlan: boolean;
  selectedModelType: string;
  selectedModel: string;
  selectedPreset: string;
  selectedTrainingPresets: string[];
  selectedDatasets: string[];
  effectiveOverrides: OverrideValues;
  logFolder: string;
  searchPayload?: TrainingSearchCreateInput;
  submittedRunPlan?: TrainingRunPlan;
};

export function buildTrainingRunPlanRequest({
  canPlan,
  selectedModelType,
  selectedModel,
  selectedPreset,
  selectedTrainingPresets,
  selectedDatasets,
  effectiveOverrides,
  logFolder,
  searchPayload,
  submittedRunPlan,
}: TrainingRunPlanRequestInput): TrainingRunPlanCreateInput | null {
  if (!canPlan || submittedRunPlan) {
    return null;
  }
  return {
    modelType: selectedModelType,
    model: selectedModel,
    preset: selectedPreset,
    presets: selectedTrainingPresets,
    datasets: selectedDatasets,
    overrides: effectiveOverrides,
    logFolder,
    ...(searchPayload ? { search: searchPayload } : {}),
  };
}

export type TrainingJobRequestInput = {
  selectedModelType: string;
  selectedModel: string;
  selectedPreset: string;
  selectedTrainingPresets: string[];
  selectedDatasets: string[];
  effectiveOverrides: OverrideValues;
  logFolder: string;
  selectedMonitors: string[];
  searchPayload?: TrainingSearchCreateInput;
  runPlan?: TrainingRunPlan;
};

export function buildTrainingJobRequest({
  selectedModel,
  selectedModelType,
  selectedPreset,
  selectedTrainingPresets,
  selectedDatasets,
  effectiveOverrides,
  logFolder,
  selectedMonitors,
  searchPayload,
  runPlan,
}: TrainingJobRequestInput): TrainingJobCreateInput | null {
  if (!runPlan) {
    return null;
  }
  const presets = runPlan.presets.length > 0
    ? runPlan.presets
    : selectedTrainingPresets;
  return {
    modelType: selectedModelType,
    model: selectedModel,
    preset: runPlan.preset || selectedPreset,
    presets,
    datasets: selectedDatasets,
    overrides: effectiveOverrides,
    logFolder,
    monitors: selectedMonitors,
    ...(searchPayload ? { search: searchPayload } : {}),
    runPlan,
  };
}
