import { overrideDigest, type OverrideValues } from "@/lib/config";

export type TargetMode = "preset" | "snapshot" | "experiment";

export type HistoricalExperimentTarget = {
  runId: string;
  experiment: string;
  preset: string;
  dataset: string;
  experimentTask?: string | null;
};

export function previewTargetKey({
  modelType,
  model,
  preset,
  experimentTask,
  dataset,
  mode,
  target,
  overrides,
}: {
  modelType: string;
  model: string;
  preset: string;
  experimentTask?: string | null;
  dataset: string;
  mode: TargetMode;
  target: string;
  overrides: OverrideValues;
}) {
  return `${modelType}\u0000${model}\u0000${preset}\u0000${experimentTask ?? ""}\u0000${dataset}\u0000${mode}\u0000${target}\u0000${overrideDigest(overrides)}`;
}

export function resolvePreviewTarget({
  selectedTargetMode,
  selectedSnapshotId,
  selectedExperimentTarget,
  selectedPreset,
  selectedExperimentTask,
  selectedDatasets,
}: {
  selectedTargetMode: TargetMode;
  selectedSnapshotId: string;
  selectedExperimentTarget: HistoricalExperimentTarget | null;
  selectedPreset: string;
  selectedExperimentTask: string;
  selectedDatasets: string[];
}) {
  const catalogDataset = selectedDatasets[0] ?? "";
  if (selectedTargetMode === "snapshot" && selectedSnapshotId) {
    return {
      targetMode: "snapshot" as const,
      targetId: selectedSnapshotId,
      preset: selectedPreset,
      experimentTask: selectedExperimentTask,
      dataset: catalogDataset,
    };
  }
  if (selectedTargetMode === "experiment") {
    return {
      targetMode: "experiment" as const,
      targetId: selectedExperimentTarget?.runId ?? "",
      preset: selectedExperimentTarget?.preset ?? selectedPreset,
      experimentTask: selectedExperimentTarget?.experimentTask ?? "",
      dataset: selectedExperimentTarget?.dataset ?? catalogDataset,
    };
  }
  return {
    targetMode: "preset" as const,
    targetId: selectedPreset,
    preset: selectedPreset,
    experimentTask: selectedExperimentTask,
    dataset: catalogDataset,
  };
}
