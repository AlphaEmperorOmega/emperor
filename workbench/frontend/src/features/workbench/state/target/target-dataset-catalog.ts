import { type Dataset, type DatasetGroup } from "@/lib/api";

const EMPTY_DATASETS: Dataset[] = [];

export function experimentTaskOptions(groups: DatasetGroup[]) {
  return groups.map((group) => ({
    value: group.experimentTask,
    label: group.label || group.experimentTask,
  }));
}

export function normalizeExperimentTask(
  current: string,
  defaultExperimentTask: string,
  groups: DatasetGroup[],
) {
  const taskNames = groups.map((group) => group.experimentTask);
  if (current && taskNames.includes(current)) {
    return current;
  }
  if (defaultExperimentTask && taskNames.includes(defaultExperimentTask)) {
    return defaultExperimentTask;
  }
  return taskNames[0] ?? "";
}

export function datasetsForExperimentTask(
  groups: DatasetGroup[],
  experimentTask: string,
) {
  return (
    groups.find((group) => group.experimentTask === experimentTask)?.datasets ??
    EMPTY_DATASETS
  );
}
