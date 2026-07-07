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
  selectedExperimentTask?: string;
  selectedDatasets: string[];
  effectiveOverrides: OverrideValues;
  logFolder: string;
  selectedMonitors: string[];
  searchPayload?: TrainingSearchCreateInput;
  submittedRunPlan?: TrainingRunPlan;
};

export function buildTrainingRunPlanRequest({
  canPlan,
  selectedModelType,
  selectedModel,
  selectedPreset,
  selectedTrainingPresets,
  selectedExperimentTask = "",
  selectedDatasets,
  effectiveOverrides,
  logFolder,
  selectedMonitors,
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
    ...(selectedExperimentTask ? { experimentTask: selectedExperimentTask } : {}),
    datasets: selectedDatasets,
    overrides: effectiveOverrides,
    logFolder,
    monitors: selectedMonitors,
    ...(searchPayload ? { search: searchPayload } : {}),
  };
}

export type TrainingJobRequestInput = {
  selectedModelType: string;
  selectedModel: string;
  selectedPreset: string;
  selectedTrainingPresets: string[];
  selectedExperimentTask?: string;
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
  selectedExperimentTask = "",
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
  const experimentTask = runPlan.experimentTask || selectedExperimentTask;
  return {
    modelType: selectedModelType,
    model: selectedModel,
    preset: runPlan.preset || selectedPreset,
    presets,
    ...(experimentTask ? { experimentTask } : {}),
    datasets: selectedDatasets,
    overrides: effectiveOverrides,
    logFolder,
    monitors: selectedMonitors,
    ...(searchPayload ? { search: searchPayload } : {}),
    runPlan,
  };
}
