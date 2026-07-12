import {
  type ConfigField,
  type TrainingRun,
  type TrainingRunChange,
  type TrainingRunPlan,
} from "@/lib/api";
import {
  configKeyToken,
  configSectionsFields,
  overrideValue,
  runtimeDefaultsEditor,
  type ConfigSection,
  type OverrideValues,
} from "@/lib/config";
import { type ConfigSnapshot } from "@/lib/config-snapshots";

function shellQuote(value: string) {
  if (value === "") {
    return "''";
  }
  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(value)) {
    return value;
  }
  return `'${value.replace(/'/g, `'"'"'`)}'`;
}

function snapshotCommandFlag(key: string) {
  return `--${key.toLowerCase().replace(/_/g, "-")}`;
}

function trainingCommand({
  modelType,
  model,
  preset,
  experimentTask,
  dataset,
  overrides,
  logFolder,
}: {
  modelType: string;
  model: string;
  preset: string;
  experimentTask: string;
  dataset: string;
  overrides: OverrideValues;
  logFolder: string;
}) {
  const parts = [
    "source",
    "experiment.sh",
    "--model-type",
    shellQuote(modelType),
    "--model",
    shellQuote(model),
    "--preset",
    shellQuote(preset),
  ];
  if (experimentTask) {
    parts.push("--experiment-task", shellQuote(experimentTask));
  }
  parts.push("--datasets", shellQuote(dataset));
  if (logFolder) {
    parts.push("--logdir", shellQuote(logFolder));
  }
  const overrideEntries = Object.entries(overrides);
  if (overrideEntries.length > 0) {
    parts.push("--config");
    for (const [key, value] of overrideEntries) {
      parts.push(snapshotCommandFlag(key), shellQuote(value || "None"));
    }
  }
  return parts.join(" ");
}

function snapshotRunChanges(
  overrides: OverrideValues,
  fieldsByKey: Map<string, ConfigField>,
) {
  return Object.entries(overrides).map<TrainingRunChange>(([key, value]) => ({
    key,
    label: fieldsByKey.get(configKeyToken(key))?.label ?? key,
    value,
    source: "override",
  }));
}

function epochDefault(fieldsByKey: Map<string, ConfigField>) {
  const field =
    fieldsByKey.get(configKeyToken("num_epochs")) ??
    Array.from(fieldsByKey.values()).find(
      (candidate) => candidate.configKey.toLowerCase() === "num_epochs",
    );
  const defaultEpochs = Number(field?.default ?? 10);
  return Number.isFinite(defaultEpochs)
    ? Math.max(0, Math.trunc(defaultEpochs))
    : 10;
}

function epochsFromOverrides(
  overrides: OverrideValues,
  fieldsByKey: Map<string, ConfigField>,
) {
  const rawEpochs = overrideValue(overrides, "NUM_EPOCHS");
  const epochValue =
    rawEpochs === undefined ? epochDefault(fieldsByKey) : Number(rawEpochs);
  return Number.isFinite(epochValue)
    ? Math.max(0, Math.trunc(epochValue))
    : 0;
}

function presetRun({
  modelType,
  model,
  preset,
  experimentTask,
  dataset,
  index,
  fieldsByKey,
  overrides,
  logFolder,
}: {
  modelType: string;
  model: string;
  preset: string;
  experimentTask: string;
  dataset: string;
  index: number;
  fieldsByKey: Map<string, ConfigField>;
  overrides: OverrideValues;
  logFolder: string;
}): TrainingRun {
  const runOverrides = { ...overrides };
  return {
    id: `preset-${preset}-${dataset}-${index}`,
    index,
    status: "Pending",
    preset,
    experimentTask,
    dataset,
    changes: snapshotRunChanges(runOverrides, fieldsByKey),
    overrides: runOverrides,
    command: trainingCommand({
      model,
      modelType,
      preset,
      experimentTask,
      dataset,
      overrides: runOverrides,
      logFolder,
    }),
    totalEpochs: epochsFromOverrides(runOverrides, fieldsByKey),
    currentEpoch: 0,
    metrics: {},
    logDir: null,
    error: null,
    errorTraceback: null,
  };
}

function summarizeSnapshotRuns(runs: TrainingRun[]) {
  const totalEpochs = runs.reduce((total, run) => total + run.totalEpochs, 0);
  return {
    totalRuns: runs.length,
    completedRuns: 0,
    runningRuns: 0,
    pendingRuns: runs.length,
    failedRuns: 0,
    cancelledRuns: 0,
    skippedRuns: 0,
    totalEpochs,
    completedEpochs: 0,
    remainingEpochs: totalEpochs,
  };
}

function normalizedRunOverrides(
  fields: ConfigField[],
  overrides: OverrideValues | undefined,
) {
  return runtimeDefaultsEditor.replace(fields, overrides);
}

export function materializeConfigSnapshotRunPlan({
  modelType,
  model,
  selectedPreset,
  selectedTrainingPresets,
  selectedExperimentTask = "",
  selectedDatasets,
  snapshots,
  sections,
  fields: suppliedFields,
  bulkOverrides,
  presetOverrides,
  logFolder,
}: {
  modelType: string;
  model: string;
  selectedPreset: string;
  selectedTrainingPresets: string[];
  selectedExperimentTask?: string;
  selectedDatasets: string[];
  snapshots: ConfigSnapshot[];
  sections?: ConfigSection[];
  fields?: ConfigField[];
  bulkOverrides?: OverrideValues;
  presetOverrides?: OverrideValues;
  logFolder: string;
}): TrainingRunPlan | undefined {
  const selectedSnapshots = snapshots.filter(
    (snapshot) => snapshot.modelType === modelType && snapshot.model === model,
  );
  if (
    selectedDatasets.length === 0 ||
    (selectedTrainingPresets.length === 0 && selectedSnapshots.length === 0)
  ) {
    return undefined;
  }

  const fields = sections ? configSectionsFields(sections) : suppliedFields ?? [];
  const fieldsByKey = new Map(
    fields.flatMap((field) => [
      [configKeyToken(field.key), field] as const,
      [configKeyToken(field.configKey), field] as const,
    ]),
  );
  const defaultRunOverrides = normalizedRunOverrides(
    fields,
    bulkOverrides ?? presetOverrides,
  );
  const snapshotBulkOverrides = bulkOverrides
    ? normalizedRunOverrides(fields, bulkOverrides)
    : undefined;
  const runs: TrainingRun[] = [];
  for (const preset of selectedTrainingPresets) {
    for (const dataset of selectedDatasets) {
      const index = runs.length + 1;
      runs.push(
        presetRun({
          model,
          modelType,
          preset,
          experimentTask: selectedExperimentTask,
          dataset,
          index,
          fieldsByKey,
          overrides: defaultRunOverrides,
          logFolder,
        }),
      );
    }
  }
  for (const snapshot of selectedSnapshots) {
    const snapshotOverrides = normalizedRunOverrides(fields, snapshot.overrides);
    const mergedSnapshotOverrides = snapshotBulkOverrides
      ? normalizedRunOverrides(fields, {
          ...snapshotOverrides,
          ...snapshotBulkOverrides,
        })
      : snapshotOverrides;
    for (const dataset of selectedDatasets) {
      const index = runs.length + 1;
      runs.push({
        id: `snapshot-${snapshot.id}-${dataset}-${index}`,
        index,
        status: "Pending",
        preset: snapshot.preset,
        snapshotId: snapshot.id,
        snapshotName: snapshot.name,
        dataset,
        experimentTask: selectedExperimentTask,
        changes: snapshotRunChanges(mergedSnapshotOverrides, fieldsByKey),
        overrides: mergedSnapshotOverrides,
        command: trainingCommand({
          model,
          modelType,
          preset: snapshot.preset,
          experimentTask: selectedExperimentTask,
          dataset,
          overrides: mergedSnapshotOverrides,
          logFolder,
        }),
        totalEpochs: epochsFromOverrides(mergedSnapshotOverrides, fieldsByKey),
        currentEpoch: 0,
        metrics: {},
        logDir: null,
        error: null,
        errorTraceback: null,
      });
    }
  }

  const presets = Array.from(
    new Set([
      ...selectedTrainingPresets,
      ...selectedSnapshots.map((snapshot) => snapshot.preset),
    ]),
  );
  const primaryPreset = presets.includes(selectedPreset)
    ? selectedPreset
    : presets[0] || selectedPreset;
  return {
    modelType,
    model,
    preset: primaryPreset,
    presets,
    experimentTask: selectedExperimentTask,
    datasets: selectedDatasets,
    overrides: {},
    search: null,
    logFolder,
    isRandomSearch: false,
    runs,
    summary: summarizeSnapshotRuns(runs),
  };
}
