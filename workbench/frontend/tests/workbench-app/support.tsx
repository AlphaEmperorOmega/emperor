import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, vi } from "vitest";
import { WorkbenchApp } from "@/features/workbench/components/workbench-app";
import { type Capabilities } from "@/lib/api";
import { loadWorkbenchConnectionRuntime } from "@/lib/api/_connection-runtime";
import { IMPLEMENTED_FEATURES } from "@/lib/feature-catalog";

type MockModelIdentity = { modelType: string; model: string };

type MockConfigSnapshot = {
  id: string;
  modelType: string;
  model: string;
  preset: string;
  name: string;
  overrides: Record<string, string>;
  createdAt: string;
  updatedAt: string;
};

vi.mock("@xyflow/react", async () => import("./react-flow-test-adapter"));

const modelsResponse = {
  models: [
    { modelType: "linears", model: "linear" },
    { modelType: "bert", model: "linear" },
    { modelType: "bert", model: "linear_adaptive" },
    { modelType: "bert", model: "expert_linear" },
    { modelType: "bert", model: "expert_linear_adaptive" },
    { modelType: "vit", model: "linear" },
    { modelType: "vit", model: "linear_adaptive" },
    { modelType: "vit", model: "expert_linear" },
    { modelType: "vit", model: "expert_linear_adaptive" },
  ],
};
const neuronModelsResponse = {
  models: [
    { modelType: "neuron", model: "linear" },
    { modelType: "neuron", model: "linear_adaptive" },
    { modelType: "neuron", model: "expert_linear" },
    { modelType: "neuron", model: "expert_linear_adaptive" },
  ],
};
const presetsResponse = {
  modelType: "linears",
  model: "linear",
  presets: [
    { name: "baseline", label: "BASELINE", description: "Baseline" },
    {
      name: "recurrent-gating-halting",
      label: "RECURRENT_GATING_HALTING",
      description: "Recurrent",
    },
  ],
};
const bertPresetsResponse = {
  modelType: "bert",
  model: "linear",
  presets: [{ name: "bert-baseline", label: "BERT_BASELINE", description: "Bert baseline" }],
};
const vitPresetsResponse = {
  modelType: "vit",
  model: "linear",
  presets: [{ name: "baseline", label: "BASELINE", description: "ViT baseline" }],
};
const neuronPresetsResponse = {
  modelType: "neuron",
  model: "linear",
  presets: [{ name: "baseline", label: "BASELINE", description: "Baseline" }],
};
const datasetsResponse = {
  modelType: "linears",
  model: "linear",
  defaultExperimentTask: "image-classification",
  datasetGroups: [
    {
      experimentTask: "image-classification",
      label: "Image Classification",
      datasets: [
        { name: "Mnist", label: "Mnist", inputDim: 784, outputDim: 10 },
        { name: "Cifar10", label: "Cifar 10", inputDim: 3072, outputDim: 10 },
      ],
    },
  ],
};
const bertDatasetsResponse = {
  modelType: "bert",
  model: "linear",
  defaultExperimentTask: "bert-pretraining",
  datasetGroups: [
    {
      experimentTask: "bert-pretraining",
      label: "Bert Pretraining",
      datasets: [
        { name: "ToyText", label: "Toy Text", inputDim: 128, outputDim: 2 },
      ],
    },
  ],
};
const vitDatasetsResponse = {
  modelType: "vit",
  model: "linear",
  defaultExperimentTask: "image-classification",
  datasetGroups: [
    {
      experimentTask: "image-classification",
      label: "Image Classification",
      datasets: [
        { name: "Mnist", label: "MNIST", inputDim: 784, outputDim: 10 },
      ],
    },
  ],
};
const neuronDatasetsResponse = {
  modelType: "neuron",
  model: "linear",
  defaultExperimentTask: "image-classification",
  datasetGroups: [
    {
      experimentTask: "image-classification",
      label: "Image Classification",
      datasets: [
        { name: "Mnist", label: "Mnist", inputDim: 784, outputDim: 10 },
      ],
    },
  ],
};
const monitorsResponse = {
  modelType: "linears",
  model: "linear",
  monitors: [
    {
      name: "linear",
      label: "Linear layers",
      description: "Logs activation, parameter, and gradient stats.",
      kinds: ["scalar"],
      defaultEnabled: false,
    },
    {
      name: "sampler",
      label: "Sampler usage",
      description: "Logs routing histograms and heatmaps.",
      kinds: ["scalar", "histogram", "image"],
      defaultEnabled: false,
    },
  ],
};
const neuronMonitorsResponse = {
  modelType: "neuron",
  model: "linear",
  monitors: [] as Array<{
    name: string;
    label: string;
    description: string;
    kinds: string[];
    defaultEnabled: boolean;
  }>,
};
const capabilitiesResponse: Capabilities = {
  authMode: "none",
  trainingEnabled: true,
  trainingCancellationCapability: "strict-cgroup",
  logDeletionEnabled: true,
  configSnapshotsEnabled: true,
  historicalLogsEnabled: true,
  liveMonitorDataEnabled: true,
  historicalMonitorDataEnabled: true,
  uploadsEnabled: true,
  maxUploadSize: null,
  maxActiveTrainingJobs: 2,
  trainingJobMemoryLimitBytes: 16 * 1024 ** 3,
  trainingJobCpuLimit: 8,
  trainingJobProcessLimit: 512,
};
const configSnapshotsResponse = {
  modelType: "linears",
  model: "linear",
  snapshots: [] as MockConfigSnapshot[],
};
const configSnapshotLibraryResponse = {
  snapshots: [] as MockConfigSnapshot[],
};
const logRunsResponse = {
  runs: [
    {
      id: "log-mnist",
      group: "test_model",
      experiment: "test_model",
      modelType: "linears",
      model: "linear",
      experimentTask: "image-classification",
      preset: "BASELINE",
      dataset: "Mnist",
      runName: "aaa_20260601_010203",
      timestamp: "2026-06-01 01:02:03",
      version: "version_0",
      relativePath: "test_model/linear/BASELINE/Mnist/aaa_20260601_010203/version_0",
      hasResult: true,
      eventFileCount: 1,
      checkpointCount: 1,
      hasHparams: true,
      metrics: { "test/accuracy": 0.9 },
    },
    {
      id: "log-cifar",
      group: "test_model_2",
      experiment: "test_model_2",
      modelType: "linears",
      model: "linear",
      experimentTask: "image-classification",
      preset: "BASELINE",
      dataset: "Cifar10",
      runName: "bbb_20260601_020304",
      timestamp: "2026-06-01 02:03:04",
      version: "version_0",
      relativePath:
        "test_model_2/linear/BASELINE/Cifar10/bbb_20260601_020304/version_0",
      hasResult: false,
      eventFileCount: 1,
      checkpointCount: 0,
      hasHparams: true,
      metrics: {},
    },
  ],
};
const logExperimentsResponse = {
  experiments: [
    { experiment: "test_model", runCount: 1, relativePath: "test_model" },
    { experiment: "test_model_2", runCount: 1, relativePath: "test_model_2" },
  ],
};
type MockLogTags =
  | string[]
  | {
      scalarTags?: string[];
      histogramTags?: string[];
      imageTags?: string[];
      textTags?: string[];
    };

function logTagsPayload(tags: MockLogTags | undefined) {
  if (Array.isArray(tags)) {
    return {
      scalarTags: tags,
      histogramTags: [],
      imageTags: [],
      textTags: [],
    };
  }
  return {
    scalarTags: tags?.scalarTags ?? [],
    histogramTags: tags?.histogramTags ?? [],
    imageTags: tags?.imageTags ?? [],
    textTags: tags?.textTags ?? [],
  };
}

const logTagsByRun: Record<string, MockLogTags> = {
  "log-mnist": [
    "validation/accuracy_epoch",
    "validation/loss_epoch",
    "train/loss_epoch",
    "train/accuracy_epoch",
    "train/loss",
    "train/accuracy",
    "validation/accuracy",
    "test/accuracy",
    "main_model.0.model/weights/mean",
  ],
  "log-cifar": [
    "validation/accuracy_epoch",
    "validation/loss_epoch",
    "train/loss_epoch",
    "train/accuracy_epoch",
    "train/loss",
    "train/accuracy",
    "validation/accuracy",
    "test/accuracy",
    "main_model.0.model/weights/mean",
  ],
};
const logScalarSeries = [
  {
    runId: "log-mnist",
    tag: "train/loss_epoch",
    points: [
      { step: 1, wallTime: 1780000000, value: 0.72 },
      { step: 2, wallTime: 1780000001, value: 0.31 },
    ],
  },
  {
    runId: "log-cifar",
    tag: "train/loss_epoch",
    points: [
      { step: 1, wallTime: 1780000000, value: 0.91 },
      { step: 2, wallTime: 1780000001, value: 0.54 },
    ],
  },
  {
    runId: "log-mnist",
    tag: "train/accuracy_epoch",
    points: [
      { step: 1, wallTime: 1780000000, value: 0.52 },
      { step: 2, wallTime: 1780000001, value: 0.78 },
    ],
  },
  {
    runId: "log-cifar",
    tag: "train/accuracy_epoch",
    points: [
      { step: 1, wallTime: 1780000000, value: 0.42 },
      { step: 2, wallTime: 1780000001, value: 0.69 },
    ],
  },
  {
    runId: "log-mnist",
    tag: "validation/loss_epoch",
    points: [
      { step: 1, wallTime: 1780000000, value: 0.64 },
      { step: 2, wallTime: 1780000001, value: 0.28 },
    ],
  },
  {
    runId: "log-cifar",
    tag: "validation/loss_epoch",
    points: [
      { step: 1, wallTime: 1780000000, value: 0.86 },
      { step: 2, wallTime: 1780000001, value: 0.47 },
    ],
  },
  {
    runId: "log-mnist",
    tag: "validation/accuracy_epoch",
    points: [
      { step: 1, wallTime: 1780000000, value: 0.6 },
      { step: 2, wallTime: 1780000001, value: 0.8 },
    ],
  },
  {
    runId: "log-cifar",
    tag: "validation/accuracy_epoch",
    points: [
      { step: 1, wallTime: 1780000000, value: 0.4 },
      { step: 2, wallTime: 1780000001, value: 0.55 },
    ],
  },
  {
    runId: "log-mnist",
    tag: "train/loss",
    points: [
      { step: 1, wallTime: 1780000000, value: 0.7 },
      { step: 2, wallTime: 1780000001, value: 0.3 },
    ],
  },
  {
    runId: "log-cifar",
    tag: "train/loss",
    points: [
      { step: 1, wallTime: 1780000000, value: 0.9 },
      { step: 2, wallTime: 1780000001, value: 0.5 },
    ],
  },
  {
    runId: "log-mnist",
    tag: "train/accuracy",
    points: [
      { step: 1, wallTime: 1780000000, value: 0.5 },
      { step: 2, wallTime: 1780000001, value: 0.76 },
    ],
  },
  {
    runId: "log-cifar",
    tag: "train/accuracy",
    points: [
      { step: 1, wallTime: 1780000000, value: 0.45 },
      { step: 2, wallTime: 1780000001, value: 0.7 },
    ],
  },
  {
    runId: "log-mnist",
    tag: "validation/accuracy",
    points: [
      { step: 1, wallTime: 1780000000, value: 0.6 },
      { step: 2, wallTime: 1780000001, value: 0.8 },
    ],
  },
  {
    runId: "log-cifar",
    tag: "validation/accuracy",
    points: [
      { step: 1, wallTime: 1780000000, value: 0.4 },
      { step: 2, wallTime: 1780000001, value: 0.55 },
    ],
  },
  {
    runId: "log-mnist",
    tag: "test/accuracy",
    points: [{ step: 2, wallTime: 1780000001, value: 0.9 }],
  },
  {
    runId: "log-cifar",
    tag: "test/accuracy",
    points: [{ step: 2, wallTime: 1780000001, value: 0.62 }],
  },
  {
    runId: "log-mnist",
    tag: "main_model.0.model/weights/mean",
    points: [
      { step: 1, wallTime: 1780000000, value: 0.12 },
      { step: 2, wallTime: 1780000001, value: 0.18 },
    ],
  },
  {
    runId: "log-cifar",
    tag: "main_model.0.model/weights/mean",
    points: [
      { step: 1, wallTime: 1780000000, value: 0.16 },
      { step: 2, wallTime: 1780000001, value: 0.2 },
    ],
  },
];

type MockLogCheckpoint = {
  id: string;
  runId: string;
  filename: string;
  relativePath: string;
  epoch: number | null;
  step: number | null;
  sizeBytes: number;
  modifiedAt: string;
};

type MockLogArtifact = {
  id: string;
  kind: string;
  label: string;
  relativePath: string;
  sizeBytes: number;
  modifiedAt: string;
};

type MockLogRunArtifacts = {
  runId: string;
  params: Record<string, unknown>;
  metrics: Record<string, unknown>;
  artifacts: MockLogArtifact[];
  checkpoints: MockLogCheckpoint[];
};

const logCheckpointsByRun: Record<string, MockLogCheckpoint[]> = {
  "log-mnist": [
    {
      id: "ckpt-log-mnist-2",
      runId: "log-mnist",
      filename: "epoch=0-step=2.ckpt",
      relativePath:
        "test_model/linear/BASELINE/Mnist/aaa_20260601_010203/version_0/checkpoints/epoch=0-step=2.ckpt",
      epoch: 0,
      step: 2,
      sizeBytes: 2048,
      modifiedAt: "2026-06-01T01:03:00Z",
    },
  ],
  "log-cifar": [],
};

function fallbackCheckpointsForRun(
  run: (typeof logRunsResponse.runs)[number],
  overrides: Record<string, MockLogCheckpoint[]> | undefined,
) {
  if (overrides && run.id in overrides) {
    return overrides[run.id] ?? [];
  }
  if (run.id in logCheckpointsByRun) {
    return logCheckpointsByRun[run.id] ?? [];
  }
  return Array.from({ length: run.checkpointCount }, (_, index) => {
    const step = index + 1;
    const filename = `epoch=0-step=${step}.ckpt`;
    return {
      id: `ckpt-${run.id}-${step}`,
      runId: run.id,
      filename,
      relativePath: `${run.relativePath}/checkpoints/${filename}`,
      epoch: 0,
      step,
      sizeBytes: 2048,
      modifiedAt: "2026-06-01T01:03:00Z",
    };
  });
}

function defaultArtifactsForRun(
  run: (typeof logRunsResponse.runs)[number],
  checkpoints: MockLogCheckpoint[],
): MockLogRunArtifacts {
  const artifacts: MockLogArtifact[] = [
    {
      id: `event-${run.id}`,
      kind: "event_file",
      label: "events.out.tfevents.1780000000",
      relativePath: `${run.relativePath}/events.out.tfevents.1780000000`,
      sizeBytes: 4096,
      modifiedAt: "2026-06-01T01:02:30Z",
    },
  ];
  if (run.hasHparams) {
    artifacts.push({
      id: `hparams-${run.id}`,
      kind: "hparams",
      label: "hparams.yaml",
      relativePath: `${run.relativePath}/hparams.yaml`,
      sizeBytes: 42,
      modifiedAt: "2026-06-01T01:02:40Z",
    });
  }
  if (run.hasResult) {
    artifacts.push({
      id: `result-${run.id}`,
      kind: "result",
      label: "result.json",
      relativePath: `${run.relativePath}/result.json`,
      sizeBytes: 120,
      modifiedAt: "2026-06-01T01:02:50Z",
    });
  }
  artifacts.push(
    ...checkpoints.map((checkpoint) => ({
      id: `artifact-${checkpoint.id}`,
      kind: "checkpoint",
      label: `checkpoints/${checkpoint.filename}`,
      relativePath: checkpoint.relativePath,
      sizeBytes: checkpoint.sizeBytes,
      modifiedAt: checkpoint.modifiedAt,
    })),
  );
  return {
    runId: run.id,
    params:
      run.id === "log-mnist"
        ? { batch_size: 4, learning_rate: 0.01 }
        : { batch_size: 8 },
    metrics: run.metrics,
    artifacts,
    checkpoints,
  };
}

function buildLargeLogFixture(count = 64) {
  const runs = Array.from({ length: count }, (_, index) => {
    const number = String(index + 1).padStart(2, "0");
    const experiment = `experiment_${number}`;
    const dataset = index % 2 === 0 ? "Mnist" : "Cifar10";
    const runName = `run_${number}_20260601_010203`;

    return {
      ...logRunsResponse.runs[0],
      id: `log-${number}`,
      group: experiment,
      experiment,
      dataset,
      runName,
      timestamp: `2026-06-01 01:${number}:03`,
      relativePath: `${experiment}/linear/BASELINE/${dataset}/${runName}/version_0`,
      metrics: { "test/accuracy": 0.8 + index / 100 },
    };
  });

  return {
    logRunsResponse: { runs },
    logExperimentsResponse: {
      experiments: runs.map((run) => ({
        experiment: run.experiment,
        runCount: 1,
        relativePath: run.experiment,
      })),
    },
    logTagsByRun: Object.fromEntries(
      runs.map((run, index) => {
        const number = String(index + 1).padStart(2, "0");
        return [run.id, [`custom/tag-${number}`]];
      }),
    ) as Record<string, MockLogTags>,
    logScalarSeries: runs.map((run, index) => {
      const number = String(index + 1).padStart(2, "0");
      return {
        runId: run.id,
        tag: `custom/tag-${number}`,
        points: [
          { step: 1, wallTime: 1780000000 + index, value: index / 100 },
        ],
      };
    }),
  };
}

function buildKaggleLinearLogFixture() {
  const normalRun = {
    ...logRunsResponse.runs[0],
    id: "normal-linear",
    group: "normal_linear",
    experiment: "normal_linear",
    dataset: "Mnist",
    preset: "BASELINE",
    runName: "normal_linear_20260601_010203",
    timestamp: "2026-06-01 01:02:03",
    relativePath:
      "normal_linear/linear/BASELINE/Mnist/normal_linear_20260601_010203/version_0",
    metrics: { "test/accuracy": 0.82 },
  };
  const kaggleRuns = [
    {
      ...logRunsResponse.runs[0],
      id: "kaggle-linear-fold-0",
      group: "kaggle_linear",
      experiment: "kaggle_linear",
      dataset: "KaggleDigits",
      preset: "KAGGLE_LINEAR",
      runName: "kaggle_linear_fold_0_20260601_020304",
      timestamp: "2026-06-01 02:03:04",
      relativePath:
        "kaggle_linear/linear/KAGGLE_LINEAR/KaggleDigits/kaggle_linear_fold_0_20260601_020304/version_0",
      metrics: { "test/accuracy": 0.88 },
    },
    {
      ...logRunsResponse.runs[0],
      id: "kaggle-linear-fold-1",
      group: "kaggle_linear",
      experiment: "kaggle_linear",
      dataset: "KaggleDigits",
      preset: "KAGGLE_LINEAR",
      runName: "kaggle_linear_fold_1_20260601_030405",
      timestamp: "2026-06-01 03:04:05",
      relativePath:
        "kaggle_linear/linear/KAGGLE_LINEAR/KaggleDigits/kaggle_linear_fold_1_20260601_030405/version_0",
      metrics: { "test/accuracy": 0.9 },
    },
  ];
  const runs = [normalRun, ...kaggleRuns];
  const kaggleTags = ["train/kaggle_logloss", "validation/kaggle_auc"];
  const normalTags = [
    "validation/accuracy_epoch",
    "validation/loss_epoch",
    "train/loss_epoch",
    "train/accuracy_epoch",
    "train/loss",
    "validation/accuracy",
  ];

  return {
    kaggleRunIds: kaggleRuns.map((run) => run.id),
    kaggleTags,
    normalRunId: normalRun.id,
    normalTags,
    logRunsResponse: { runs },
    logExperimentsResponse: {
      experiments: [
        { experiment: "normal_linear", runCount: 1, relativePath: "normal_linear" },
        { experiment: "kaggle_linear", runCount: 2, relativePath: "kaggle_linear" },
      ],
    },
    logTagsByRun: {
      [normalRun.id]: normalTags,
      [kaggleRuns[0].id]: kaggleTags,
      [kaggleRuns[1].id]: kaggleTags,
    } as Record<string, MockLogTags>,
    logScalarSeries: [
      {
        runId: normalRun.id,
        tag: "validation/accuracy_epoch",
        points: [
          { step: 1, wallTime: 1780000000, value: 0.5 },
          { step: 2, wallTime: 1780000100, value: 0.72 },
        ],
      },
      {
        runId: normalRun.id,
        tag: "validation/loss_epoch",
        points: [
          { step: 1, wallTime: 1780000000, value: 0.66 },
          { step: 2, wallTime: 1780000100, value: 0.35 },
        ],
      },
      {
        runId: normalRun.id,
        tag: "train/loss_epoch",
        points: [
          { step: 1, wallTime: 1780000000, value: 0.8 },
          { step: 2, wallTime: 1780000100, value: 0.4 },
        ],
      },
      {
        runId: normalRun.id,
        tag: "train/accuracy_epoch",
        points: [
          { step: 1, wallTime: 1780000000, value: 0.48 },
          { step: 2, wallTime: 1780000100, value: 0.74 },
        ],
      },
      {
        runId: normalRun.id,
        tag: "train/loss",
        points: [
          { step: 1, wallTime: 1780000000, value: 0.8 },
          { step: 2, wallTime: 1780000100, value: 0.4 },
        ],
      },
      {
        runId: normalRun.id,
        tag: "validation/accuracy",
        points: [
          { step: 1, wallTime: 1780000000, value: 0.5 },
          { step: 2, wallTime: 1780000100, value: 0.72 },
        ],
      },
      ...kaggleRuns.flatMap((run, runIndex) =>
        kaggleTags.map((tag, tagIndex) => ({
          runId: run.id,
          tag,
          points: [
            {
              step: 1,
              wallTime: 1780000200 + runIndex * 10 + tagIndex,
              value: tag === "train/kaggle_logloss"
                ? 0.6 - runIndex / 20
                : 0.7 + runIndex / 10,
            },
            {
              step: 2,
              wallTime: 1780000300 + runIndex * 10 + tagIndex,
              value: tag === "train/kaggle_logloss"
                ? 0.3 - runIndex / 20
                : 0.84 + runIndex / 10,
            },
          ],
        })),
      ),
    ],
  };
}

function buildSubsetDeleteFixture() {
  const runs = [
    {
      ...logRunsResponse.runs[0],
      id: "log-mnist-baseline",
      group: "test_model",
      experiment: "test_model",
      preset: "BASELINE",
      dataset: "Mnist",
      runName: "mnist_baseline_20260601_010203",
      timestamp: "2026-06-01 01:02:03",
      relativePath:
        "test_model/linear/BASELINE/Mnist/mnist_baseline_20260601_010203/version_0",
    },
    {
      ...logRunsResponse.runs[0],
      id: "log-mnist-alt",
      group: "test_model",
      experiment: "test_model",
      preset: "ALT",
      dataset: "Mnist",
      runName: "mnist_alt_20260601_011203",
      timestamp: "2026-06-01 01:12:03",
      relativePath: "test_model/linear/ALT/Mnist/mnist_alt_20260601_011203/version_0",
    },
    {
      ...logRunsResponse.runs[0],
      id: "log-cifar-baseline",
      group: "test_model",
      experiment: "test_model",
      preset: "BASELINE",
      dataset: "Cifar10",
      runName: "cifar_baseline_20260601_020304",
      timestamp: "2026-06-01 02:03:04",
      relativePath:
        "test_model/linear/BASELINE/Cifar10/cifar_baseline_20260601_020304/version_0",
    },
    {
      ...logRunsResponse.runs[1],
      id: "log-other-mnist",
      group: "test_model_2",
      experiment: "test_model_2",
      preset: "BASELINE",
      dataset: "Mnist",
      runName: "other_mnist_20260601_030405",
      timestamp: "2026-06-01 03:04:05",
      relativePath:
        "test_model_2/linear/BASELINE/Mnist/other_mnist_20260601_030405/version_0",
    },
  ];

  return {
    logRunsResponse: { runs },
    logExperimentsResponse: {
      experiments: [
        { experiment: "test_model", runCount: 3, relativePath: "test_model" },
        { experiment: "test_model_2", runCount: 1, relativePath: "test_model_2" },
      ],
    },
    logTagsByRun: Object.fromEntries(runs.map((run) => [run.id, []])) as Record<
      string,
      MockLogTags
    >,
  };
}

function buildHistoricalMonitorFixture(count = 6) {
  const runs = Array.from({ length: count }, (_, index) => {
    const number = String(index + 1).padStart(2, "0");
    const runName = `monitor_run_${number}_20260601_${number}0000`;

    return {
      ...logRunsResponse.runs[0],
      id: `historical-${number}`,
      group: "monitor_exp",
      experiment: "monitor_exp",
      dataset: "Mnist",
      runName,
      timestamp: `2026-06-01 ${number}:00:00`,
      relativePath: `monitor_exp/linear/BASELINE/Mnist/${runName}/version_0`,
      metrics: { "test/accuracy": 0.8 + index / 100 },
    };
  });

  return {
    logRunsResponse: { runs },
    logTagsByRun: Object.fromEntries(
      runs.map((run) => [
        run.id,
        {
          scalarTags: [
            "main_model.0.model/output/mean",
            "main_model.0.model/weights/mean",
            "main_model.1.model/output/mean",
            "main_model.1.model/weights/mean",
          ],
          histogramTags: [],
          imageTags: [],
          textTags: [],
        },
      ]),
    ) as Record<string, MockLogTags>,
  };
}

const schemaResponse = {
  modelType: "linears",
  model: "linear",
  fields: [
    {
      key: "hidden_dim",
      configKey: "HIDDEN_DIM",
      flag: "--hidden-dim",
      label: "stack hidden dim",
      section: "Layer Stack Options",
      sectionPath: ["Layer Stack Options"],
      type: "int",
      default: 256,
      nullable: false,
      choices: [],
    },
    {
      key: "stack_num_layers",
      configKey: "STACK_NUM_LAYERS",
      flag: "--stack-num-layers",
      label: "stack num layers",
      section: "Layer Stack Options",
      sectionPath: ["Layer Stack Options"],
      type: "int",
      default: 5,
      nullable: false,
      choices: [],
    },
    {
      key: "gate_flag",
      configKey: "GATE_FLAG",
      flag: "--gate-flag",
      label: "gate flag",
      section: "Gate Options",
      sectionPath: ["Gate Options"],
      type: "bool",
      default: false,
      nullable: false,
      choices: [true, false],
    },
    {
      key: "stack_activation",
      configKey: "STACK_ACTIVATION",
      flag: "--stack-activation",
      label: "stack activation",
      section: "Layer Stack Options",
      sectionPath: ["Layer Stack Options"],
      type: "enum",
      default: "GELU",
      nullable: false,
      choices: ["RELU", "GELU"],
    },
  ],
};
function schemaResponseWithDescriptions(descriptions: Record<string, string>) {
  return {
    ...schemaResponse,
    fields: schemaResponse.fields.map((field) => ({
      ...field,
      description: descriptions[field.key] ?? "",
    })),
  };
}
const searchSpaceResponse = {
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  axes: [
    {
      key: "hidden_dim",
      configKey: "HIDDEN_DIM",
      searchKey: "SEARCH_SPACE_HIDDEN_DIM",
      label: "stack hidden dim",
      section: "Layer Stack Options",
      type: "int",
      values: [64, 128],
      locked: false,
      lockedValue: null,
      lockedReason: "",
    },
    {
      key: "stack_num_layers",
      configKey: "STACK_NUM_LAYERS",
      searchKey: "SEARCH_SPACE_STACK_NUM_LAYERS",
      label: "stack num layers",
      section: "Layer Stack Options",
      type: "int",
      values: [2, 4],
      locked: false,
      lockedValue: null,
      lockedReason: "",
    },
    {
      key: "stack_activation",
      configKey: "STACK_ACTIVATION",
      searchKey: "SEARCH_SPACE_STACK_ACTIVATION",
      label: "stack activation",
      section: "Layer Stack Options",
      type: "enum",
      values: ["RELU", "GELU"],
      locked: false,
      lockedValue: null,
      lockedReason: "",
    },
  ],
};
const inspectResponse = {
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  parameterCount: 65792,
  nodes: [
    {
      id: "model",
      label: "Model",
      typeName: "Model",
      path: "model",
      graphRole: "architecture",
      parameterCount: 65792,
      details: {},
    },
    {
      id: "loss_fn",
      label: "CrossEntropyLoss",
      typeName: "CrossEntropyLoss",
      path: "loss_fn",
      graphRole: "runtime",
      parameterCount: 0,
      details: {},
    },
    {
      id: "metrics",
      label: "ClassifierMetricsLogger",
      typeName: "ClassifierMetricsLogger",
      path: "metrics",
      graphRole: "runtime",
      parameterCount: 0,
      details: {},
    },
    {
      id: "main_model.0",
      label: "Layer",
      typeName: "Layer",
      path: "main_model.0",
      graphRole: "architecture",
      parameterCount: 33024,
      details: {
        dims: "128 -> 128",
        activation: "GELU",
        layerNorm: "BEFORE",
        dropout: 0.2,
      },
    },
    {
      id: "main_model.0.model",
      label: "LinearLayer",
      typeName: "LinearLayer",
      path: "main_model.0.model",
      graphRole: "architecture",
      parameterCount: 16512,
      details: {
        dims: "128 -> 128",
      },
    },
    {
      id: "main_model.0.gate_model",
      label: "Sequential",
      typeName: "Sequential",
      path: "main_model.0.gate_model",
      graphRole: "architecture",
      parameterCount: 0,
      details: {},
    },
    {
      id: "main_model.0.layer_norm_module",
      label: "LayerNorm",
      typeName: "LayerNorm",
      path: "main_model.0.layer_norm_module",
      graphRole: "internal",
      parameterCount: 256,
      details: {},
    },
    {
      id: "main_model.0.dropout_module",
      label: "Dropout",
      typeName: "Dropout",
      path: "main_model.0.dropout_module",
      graphRole: "internal",
      parameterCount: 0,
      details: {},
    },
    {
      id: "main_model.0.processor",
      label: "SelfAttentionProcessor",
      typeName: "SelfAttentionProcessor",
      path: "main_model.0.processor",
      graphRole: "internal",
      parameterCount: 16512,
      details: {},
    },
    {
      id: "main_model.0.processor.projection",
      label: "LinearLayer",
      typeName: "LinearLayer",
      path: "main_model.0.processor.projection",
      graphRole: "architecture",
      parameterCount: 16512,
      details: {
        dims: "128 -> 128",
      },
    },
  ],
  edges: [
    {
      id: "model-loss_fn",
      source: "model",
      target: "loss_fn",
    },
    {
      id: "model-metrics",
      source: "model",
      target: "metrics",
    },
    {
      id: "model-main_model.0",
      source: "model",
      target: "main_model.0",
    },
    {
      id: "main_model.0-main_model.0.model",
      source: "main_model.0",
      target: "main_model.0.model",
    },
    {
      id: "main_model.0-main_model.0.gate_model",
      source: "main_model.0",
      target: "main_model.0.gate_model",
    },
    {
      id: "main_model.0-main_model.0.layer_norm_module",
      source: "main_model.0",
      target: "main_model.0.layer_norm_module",
    },
    {
      id: "main_model.0-main_model.0.dropout_module",
      source: "main_model.0",
      target: "main_model.0.dropout_module",
    },
    {
      id: "main_model.0-main_model.0.processor",
      source: "main_model.0",
      target: "main_model.0.processor",
    },
    {
      id: "main_model.0.processor-main_model.0.processor.projection",
      source: "main_model.0.processor",
      target: "main_model.0.processor.projection",
    },
  ],
};

const parameterShapeInspectResponse = {
  ...inspectResponse,
  nodes: inspectResponse.nodes.map((node) =>
    node.id === "main_model.0.model"
      ? {
          ...node,
          details: {
            ...node.details,
            weightShape: "128 x 128",
            biasShape: "128",
          },
        }
      : node,
  ),
};

const repeatedLayersInspectResponse = {
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  nodes: [
    {
      id: "model",
      label: "Model",
      typeName: "Model",
      path: "model",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "main_model.0",
      label: "Layer",
      typeName: "Layer",
      path: "main_model.0",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "main_model.0.model",
      label: "LinearLayer",
      typeName: "LinearLayer",
      path: "main_model.0.model",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "main_model.1",
      label: "Layer",
      typeName: "Layer",
      path: "main_model.1",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "main_model.1.model",
      label: "LinearLayer",
      typeName: "LinearLayer",
      path: "main_model.1.model",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "main_model.2",
      label: "Layer",
      typeName: "Layer",
      path: "main_model.2",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "main_model.2.model",
      label: "LinearLayer",
      typeName: "LinearLayer",
      path: "main_model.2.model",
      graphRole: "architecture",
      details: {},
    },
  ],
  edges: [
    {
      id: "model-main_model.0",
      source: "model",
      target: "main_model.0",
    },
    {
      id: "main_model.0-main_model.0.model",
      source: "main_model.0",
      target: "main_model.0.model",
    },
    {
      id: "model-main_model.1",
      source: "model",
      target: "main_model.1",
    },
    {
      id: "main_model.1-main_model.1.model",
      source: "main_model.1",
      target: "main_model.1.model",
    },
    {
      id: "model-main_model.2",
      source: "model",
      target: "main_model.2",
    },
    {
      id: "main_model.2-main_model.2.model",
      source: "main_model.2",
      target: "main_model.2.model",
    },
  ],
};

const monitorScopeInspectResponse = {
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  nodes: [
    {
      id: "model",
      label: "Model",
      typeName: "Model",
      path: "model",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "input_model",
      label: "Layer",
      typeName: "Layer",
      path: "input_model",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "input_model.model",
      label: "LinearLayer",
      typeName: "LinearLayer",
      path: "input_model.model",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "main_model.0",
      label: "Layer",
      typeName: "Layer",
      path: "main_model.0",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "main_model.0.model",
      label: "LinearLayer",
      typeName: "LinearLayer",
      path: "main_model.0.model",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "main_model.1",
      label: "Layer",
      typeName: "Layer",
      path: "main_model.1",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "main_model.1.model",
      label: "LinearLayer",
      typeName: "LinearLayer",
      path: "main_model.1.model",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "output_model",
      label: "Layer",
      typeName: "Layer",
      path: "output_model",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "output_model.model",
      label: "LinearLayer",
      typeName: "LinearLayer",
      path: "output_model.model",
      graphRole: "architecture",
      details: {},
    },
  ],
  edges: [
    {
      id: "model-input_model",
      source: "model",
      target: "input_model",
    },
    {
      id: "input_model-input_model.model",
      source: "input_model",
      target: "input_model.model",
    },
    {
      id: "model-main_model.0",
      source: "model",
      target: "main_model.0",
    },
    {
      id: "main_model.0-main_model.0.model",
      source: "main_model.0",
      target: "main_model.0.model",
    },
    {
      id: "model-main_model.1",
      source: "model",
      target: "main_model.1",
    },
    {
      id: "main_model.1-main_model.1.model",
      source: "main_model.1",
      target: "main_model.1.model",
    },
    {
      id: "model-output_model",
      source: "model",
      target: "output_model",
    },
    {
      id: "output_model-output_model.model",
      source: "output_model",
      target: "output_model.model",
    },
  ],
};

const stackContainerInspectResponse = {
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  parameterCount: 65792,
  nodes: [
    {
      id: "model",
      label: "Model",
      typeName: "Model",
      path: "model",
      graphRole: "architecture",
      parameterCount: 65792,
      details: {},
    },
    {
      id: "main_model",
      label: "LayerStack",
      typeName: "LayerStack",
      path: "main_model",
      graphRole: "architecture",
      parameterCount: 65792,
      details: {},
    },
    {
      id: "main_model.layers.0",
      label: "Layer",
      typeName: "Layer",
      path: "main_model.layers.0",
      graphRole: "architecture",
      parameterCount: 33024,
      details: { inputDim: 256, outputDim: 256 },
    },
    {
      id: "main_model.layers.1",
      label: "Layer",
      typeName: "Layer",
      path: "main_model.layers.1",
      graphRole: "architecture",
      parameterCount: 32768,
      details: { inputDim: 256, outputDim: 10 },
    },
  ],
  edges: [
    {
      id: "model-main_model",
      source: "model",
      target: "main_model",
    },
    {
      id: "main_model-main_model.layers.0",
      source: "main_model",
      target: "main_model.layers.0",
    },
    {
      id: "main_model-main_model.layers.1",
      source: "main_model",
      target: "main_model.layers.1",
    },
  ],
};

const manyRepeatedLayersInspectResponse = {
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  nodes: [
    {
      id: "model",
      label: "Model",
      typeName: "Model",
      path: "model",
      graphRole: "architecture",
      details: {},
    },
    ...Array.from({ length: 9 }, (_, index) => [
      {
        id: `main_model.${index}`,
        label: "Layer",
        typeName: "Layer",
        path: `main_model.${index}`,
        graphRole: "architecture",
        details: {},
      },
      {
        id: `main_model.${index}.model`,
        label: "LinearLayer",
        typeName: "LinearLayer",
        path: `main_model.${index}.model`,
        graphRole: "architecture",
        details: {},
      },
    ]).flat(),
  ],
  edges: Array.from({ length: 9 }, (_, index) => [
    {
      id: `model-main_model.${index}`,
      source: "model",
      target: `main_model.${index}`,
    },
    {
      id: `main_model.${index}-main_model.${index}.model`,
      source: `main_model.${index}`,
      target: `main_model.${index}.model`,
    },
  ]).flat(),
};

const mechanismMetadataInspectResponse = {
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  nodes: [
    {
      id: "model",
      label: "Model",
      typeName: "Model",
      path: "model",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "controller",
      label: "Controller",
      typeName: "Controller",
      path: "controller",
      graphRole: "architecture",
      details: {
        gate: true,
        recurrent: {
          maxSteps: 4,
          halting: true,
        },
      },
    },
  ],
  edges: [
    {
      id: "model-controller",
      source: "model",
      target: "controller",
    },
  ],
};

const mechanismChildrenInspectResponse = {
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  nodes: [
    {
      id: "model",
      label: "Model",
      typeName: "Model",
      path: "model",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "controller",
      label: "Controller",
      typeName: "Controller",
      path: "controller",
      graphRole: "architecture",
      details: {
        gate: true,
        halting: true,
      },
    },
    {
      id: "controller.gate_model",
      label: "Sequential",
      typeName: "Sequential",
      path: "controller.gate_model",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "controller.halting_model",
      label: "StickBreaking",
      typeName: "StickBreaking",
      path: "controller.halting_model",
      graphRole: "architecture",
      details: {},
    },
  ],
  edges: [
    {
      id: "model-controller",
      source: "model",
      target: "controller",
    },
    {
      id: "controller-controller.gate_model",
      source: "controller",
      target: "controller.gate_model",
    },
    {
      id: "controller-controller.halting_model",
      source: "controller",
      target: "controller.halting_model",
    },
  ],
};

const tallSummaryInspectResponse = {
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  nodes: [
    {
      id: "model",
      label: "Model",
      typeName: "Model",
      path: "model",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "block",
      label: "Block",
      typeName: "Block",
      path: "block",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "block.linear",
      label: "LinearLayer",
      typeName: "LinearLayer",
      path: "block.linear",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "block.attention",
      label: "AttentionLayer",
      typeName: "AttentionLayer",
      path: "block.attention",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "block.embedding",
      label: "Embedding",
      typeName: "Embedding",
      path: "block.embedding",
      graphRole: "architecture",
      details: {},
    },
    {
      id: "block.output",
      label: "OutputHead",
      typeName: "OutputHead",
      path: "block.output",
      graphRole: "architecture",
      details: {},
    },
  ],
  edges: [
    {
      id: "model-block",
      source: "model",
      target: "block",
    },
    {
      id: "block-block.linear",
      source: "block",
      target: "block.linear",
    },
    {
      id: "block-block.attention",
      source: "block",
      target: "block.attention",
    },
    {
      id: "block-block.embedding",
      source: "block",
      target: "block.embedding",
    },
    {
      id: "block-block.output",
      source: "block",
      target: "block.output",
    },
  ],
};

const longSelectedNodeId = "model.0.model.adaptive_behaviour.mask_model.model.1";
const longSelectedNodeInspectResponse = {
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  nodes: [
    {
      id: longSelectedNodeId,
      label: "LinearLayer",
      typeName: "LinearLayer",
      path: longSelectedNodeId,
      graphRole: "architecture",
      details: {
        dims: "256 -> 256",
      },
    },
  ],
  edges: [],
};

const locationInspectResponse = {
  modelType: "linears",
  model: "linear",
  preset: "baseline",
  parameterCount: 0,
  nodes: [
    {
      id: "model",
      label: "Model",
      typeName: "Model",
      path: "model",
      graphRole: "architecture",
      parameterCount: 0,
      details: {},
    },
    {
      id: "model.cluster",
      label: "Cluster",
      typeName: "NeuronCluster",
      path: "model.cluster",
      graphRole: "architecture",
      parameterCount: 0,
      details: {
        cluster: {
          capacity: [2, 2, 1],
          initial: [1, 1, 1],
          initialStart: [1, 1, 1],
          instantiated: 1,
          coordinates: [[1, 1, 1]],
          maxSteps: 2,
          growthThreshold: 4,
        },
      },
    },
    {
      id: "model.cluster.neuron_1_1_1",
      label: "Neuron",
      typeName: "Neuron",
      path: "model.cluster.neuron_1_1_1",
      graphRole: "internal",
      parameterCount: 0,
      details: {
        terminalReach: {
          position: [1, 1, 1],
          connections: [
            [1, 1, 1],
            [2, 1, 1],
          ],
          total: 2,
        },
      },
    },
    {
      id: "model.cluster.terminal",
      label: "Terminal",
      typeName: "Terminal",
      path: "model.cluster.terminal",
      graphRole: "internal",
      parameterCount: 0,
      details: {
        terminalReach: {
          position: [4, 4, 2],
          connections: [
            [5, 4, 2],
            [4, 5, 2],
          ],
          total: 2,
        },
      },
    },
  ],
  edges: [
    {
      id: "model-model.cluster",
      source: "model",
      target: "model.cluster",
    },
    {
      id: "model.cluster-model.cluster.terminal",
      source: "model.cluster",
      target: "model.cluster.terminal",
    },
    {
      id: "model.cluster-model.cluster.neuron_1_1_1",
      source: "model.cluster",
      target: "model.cluster.neuron_1_1_1",
    },
  ],
};

function jsonResponse(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { "content-type": "application/json" },
    }),
  );
}

type MockTrainingPlanRequest = {
  modelType?: string;
  model?: string;
  preset?: string;
  presets?: string[];
  experimentTask?: string;
  datasets?: string[];
  overrides?: Record<string, unknown>;
  monitors?: string[];
  logFolder?: string;
  snapshotIds?: string[];
  snapshotRevisions?: Array<{
    id: string;
    semanticRevision: string;
  }>;
  search?: {
    mode: "grid" | "random";
    values: Record<string, unknown[]>;
    randomSamples?: number;
  };
};

function mockTrainingSearchCombinations(request: MockTrainingPlanRequest) {
  const search = request.search;
  if (!search) {
    return [{ changes: [] as unknown[], overrides: {} as Record<string, unknown> }];
  }
  const entries = Object.entries(search.values).filter(
    ([, values]) => values.length > 0,
  );
  if (entries.length === 0) {
    return [{ changes: [] as unknown[], overrides: {} as Record<string, unknown> }];
  }
  let combinations = [{ changes: [] as unknown[], overrides: {} as Record<string, unknown> }];
  for (const [key, values] of entries) {
    combinations = combinations.flatMap((combination) =>
      values.map((value) => ({
        changes: [
          ...combination.changes,
          { key, label: key.replaceAll("_", " "), value, source: "search" },
        ],
        overrides: { ...combination.overrides, [key]: value },
      })),
    );
  }
  if (search.mode === "random") {
    return combinations.slice(0, Math.min(search.randomSamples ?? 10, combinations.length));
  }
  return combinations;
}

function mockTrainingCommand(input: {
  modelType: string;
  model: string;
  preset: string;
  experimentTask?: string;
  dataset: string;
  logFolder: string;
  monitors?: string[];
  overrides: Record<string, unknown>;
}) {
  const parts = [
    "source",
    "experiment.sh",
    "--model-type",
    input.modelType,
    "--model",
    input.model,
    "--preset",
    input.preset,
  ];
  if (input.experimentTask) {
    parts.push("--experiment-task", input.experimentTask);
  }
  parts.push("--datasets", input.dataset);
  if (input.logFolder) {
    parts.push("--logdir", input.logFolder);
  }
  if (input.monitors && input.monitors.length > 0) {
    parts.push("--monitors", ...input.monitors);
  }
  const entries = Object.entries(input.overrides);
  if (entries.length > 0) {
    parts.push("--config");
    for (const [key, value] of entries) {
      parts.push(`--${key.replaceAll("_", "-")}`, String(value ?? "None"));
    }
  }
  return parts.join(" ");
}

function mockTrainingRunPlan(
  request: MockTrainingPlanRequest,
  snapshotRecords: MockConfigSnapshot[] = [],
) {
  const modelType = request.modelType ?? "linears";
  const model = request.model ?? "linear";
  const preset = request.preset ?? "baseline";
  const selectedPresets = request.presets ?? [preset];
  const datasets = request.datasets?.length ? request.datasets : ["Cifar10"];
  const overrides = request.overrides ?? {};
  const fixedChanges = Object.entries(overrides).map(([key, value]) => ({
    key,
    label: key.replaceAll("_", " "),
    value,
    source: "override",
  }));
  const combinations = mockTrainingSearchCombinations(request);
  const presetRuns = selectedPresets.flatMap((runPreset) =>
    datasets.flatMap((dataset) =>
      combinations.map((combination) => {
        const rowOverrides = { ...overrides, ...combination.overrides };
        const index = 0;
        return {
          id: "",
          index,
          status: "Pending",
          preset: runPreset,
          dataset,
          changes: [...fixedChanges, ...combination.changes],
          overrides: rowOverrides,
          command: mockTrainingCommand({
            modelType,
            model,
            preset: runPreset,
            experimentTask: request.experimentTask,
            dataset,
            logFolder: request.logFolder ?? "",
            monitors: request.monitors ?? [],
            overrides: rowOverrides,
          }),
          totalEpochs: Number(
            rowOverrides.num_epochs ?? rowOverrides.NUM_EPOCHS ?? 30,
          ),
          currentEpoch: 0,
          metrics: {},
          logDir: null,
          error: null,
        };
      }),
    ),
  );
  const selectedSnapshots = (request.snapshotIds ?? []).flatMap((snapshotId) => {
    const snapshot = snapshotRecords.find((candidate) => candidate.id === snapshotId);
    return snapshot ? [snapshot] : [];
  });
  const snapshotRuns = selectedSnapshots.flatMap((snapshot) =>
    datasets.map((dataset) => {
      const rowOverrides = { ...snapshot.overrides, ...overrides };
      return {
        id: "",
        index: 0,
        status: "Pending",
        preset: snapshot.preset,
        snapshotId: snapshot.id,
        snapshotName: snapshot.name,
        dataset,
        changes: Object.entries(rowOverrides).map(([key, value]) => ({
          key,
          label: key.replaceAll("_", " "),
          value,
          source: "override",
        })),
        overrides: rowOverrides,
        command: mockTrainingCommand({
          modelType,
          model,
          preset: snapshot.preset,
          experimentTask: request.experimentTask,
          dataset,
          logFolder: request.logFolder ?? "",
          monitors: request.monitors ?? [],
          overrides: rowOverrides,
        }),
        totalEpochs: Number(
          rowOverrides.num_epochs ?? rowOverrides.NUM_EPOCHS ?? 30,
        ),
        currentEpoch: 0,
        metrics: {},
        logDir: null,
        error: null,
      };
    }),
  );
  const runs = [...presetRuns, ...snapshotRuns].map((run, index) => ({
    ...run,
    id: `run-${String(index + 1).padStart(4, "0")}`,
    index: index + 1,
  }));
  const planPresets = Array.from(
    new Set([
      ...selectedPresets,
      ...selectedSnapshots.map((snapshot) => snapshot.preset),
    ]),
  );
  const totalEpochs = runs.reduce((total, run) => total + run.totalEpochs, 0);
  return {
    modelType,
    model,
    preset: planPresets.includes(preset) ? preset : planPresets[0] ?? preset,
    presets: planPresets,
    experimentTask: request.experimentTask ?? "",
    datasets,
    overrides,
    search: request.search ?? null,
    logFolder: request.logFolder ?? "",
    isRandomSearch: request.search?.mode === "random",
    runs,
    snapshotRevisions: selectedSnapshots.map((snapshot, index) => ({
      id: snapshot.id,
      semanticRevision: String(index + 1).padStart(64, "0"),
    })),
    summary: {
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
    },
  };
}

type MockSubmittedTrainingRunPlan = {
  runs?: Array<{
    id?: unknown;
    preset?: unknown;
    snapshotId?: unknown;
    snapshotName?: unknown;
    dataset?: unknown;
    overrides?: unknown;
  }>;
};

function mockAcceptedTrainingRunPlan(
  request: MockTrainingPlanRequest,
  submitted: MockSubmittedTrainingRunPlan | undefined,
  preferred?: ReturnType<typeof mockTrainingRunPlan>,
) {
  const submittedRuns = submitted?.runs ?? [];
  if (
    preferred &&
    preferred.runs.length === submittedRuns.length &&
    preferred.runs.every((run, index) => {
      const submittedRun = submittedRuns[index];
      return submittedRun?.id === run.id &&
        submittedRun.preset === run.preset &&
        submittedRun.dataset === run.dataset &&
        JSON.stringify(submittedRun.overrides ?? {}) === JSON.stringify(run.overrides);
    })
  ) {
    return preferred;
  }

  const basePlan = mockTrainingRunPlan(request);
  if (submittedRuns.length === 0) {
    if ((request.snapshotIds?.length ?? 0) > 0 && preferred) {
      return preferred;
    }
    return basePlan;
  }
  const runs = submittedRuns.map((submittedRun, index) => {
    const preset = String(submittedRun.preset ?? request.preset ?? "baseline");
    const dataset = String(submittedRun.dataset ?? request.datasets?.[0] ?? "Mnist");
    const overrides = (
      submittedRun.overrides && typeof submittedRun.overrides === "object"
        ? submittedRun.overrides
        : {}
    ) as Record<string, unknown>;
    const template = basePlan.runs.find(
      (run) => run.preset === preset && run.dataset === dataset,
    ) ?? basePlan.runs[0];
    const totalEpochs = Number(
      overrides.NUM_EPOCHS ?? overrides.num_epochs ?? template?.totalEpochs ?? 30,
    );
    return {
      ...template,
      id: String(submittedRun.id ?? `run-${String(index + 1).padStart(4, "0")}`),
      index: index + 1,
      status: "Pending",
      preset,
      ...(submittedRun.snapshotId !== undefined
        ? { snapshotId: submittedRun.snapshotId }
        : {}),
      ...(submittedRun.snapshotName !== undefined
        ? { snapshotName: submittedRun.snapshotName }
        : {}),
      dataset,
      changes: Object.entries(overrides).map(([key, value]) => ({
        key,
        label: key.replaceAll("_", " "),
        value,
        source: "override",
      })),
      overrides,
      command: mockTrainingCommand({
        modelType: request.modelType ?? "linears",
        model: request.model ?? "linear",
        preset,
        dataset,
        logFolder: request.logFolder ?? "",
        monitors: request.monitors ?? [],
        overrides,
      }),
      totalEpochs,
      currentEpoch: 0,
      metrics: {},
      logDir: null,
      error: null,
    };
  });
  return {
    ...basePlan,
    runs,
    summary: summarizeMockTrainingRuns(runs),
  };
}

type MockTrainingRunSummaryInput = Array<{
  status: string;
  totalEpochs: number;
  currentEpoch: number;
}>;

function summarizeMockTrainingRuns(runs: MockTrainingRunSummaryInput) {
  const totalEpochs = runs.reduce((total, run) => total + run.totalEpochs, 0);
  const completedEpochs = runs.reduce((total, run) => {
    if (run.status === "Completed") {
      return total + run.totalEpochs;
    }
    if (["Running", "Failed", "Cancelled"].includes(run.status)) {
      return total + Math.min(run.currentEpoch, run.totalEpochs);
    }
    return total;
  }, 0);
  const remainingEpochs = runs.reduce((total, run) => {
    if (run.status === "Pending" || run.status === "Running") {
      return total + Math.max(0, run.totalEpochs - run.currentEpoch);
    }
    return total;
  }, 0);
  return {
    totalRuns: runs.length,
    completedRuns: runs.filter((run) => run.status === "Completed").length,
    runningRuns: runs.filter((run) => run.status === "Running").length,
    pendingRuns: runs.filter((run) => run.status === "Pending").length,
    failedRuns: runs.filter((run) => run.status === "Failed").length,
    cancelledRuns: runs.filter((run) => run.status === "Cancelled").length,
    skippedRuns: runs.filter((run) => run.status === "Skipped").length,
    totalEpochs,
    completedEpochs,
    remainingEpochs,
  };
}

function completedMockTrainingRunPlan(
  request: MockTrainingPlanRequest,
  preferred?: ReturnType<typeof mockTrainingRunPlan>,
) {
  const plan = preferred ?? mockTrainingRunPlan(request);
  const runs = plan.runs.map((run) => ({
    ...run,
    status: "Completed",
    currentEpoch: run.totalEpochs,
    metrics: { validation_accuracy: 0.9 },
    logDir: `logs/${request.logFolder ?? "test_model"}`,
  }));
  return {
    ...plan,
    runs,
    summary: {
      ...plan.summary,
      ...summarizeMockTrainingRuns(runs),
    },
  };
}

function cancelledMockTrainingRunPlan(
  request: MockTrainingPlanRequest,
  preferred?: ReturnType<typeof mockTrainingRunPlan>,
) {
  const plan = preferred ?? mockTrainingRunPlan(request);
  const runs = plan.runs.map((run) => ({
    ...run,
    status: run.status === "Running" ? "Cancelled" : "Skipped",
  }));
  return {
    ...plan,
    runs,
    summary: {
      ...plan.summary,
      ...summarizeMockTrainingRuns(runs),
    },
  };
}

function failedMockTrainingRunPlan(
  request: MockTrainingPlanRequest,
  preferred?: ReturnType<typeof mockTrainingRunPlan>,
) {
  const plan = preferred ?? mockTrainingRunPlan(request);
  let failedSeen = false;
  const runs = plan.runs.map((run) => {
    if (!failedSeen) {
      failedSeen = true;
      return {
        ...run,
        status: "Failed",
        error: "Training failed",
      };
    }
    return { ...run, status: "Skipped" };
  });
  return {
    ...plan,
    runs,
    summary: {
      ...plan.summary,
      ...summarizeMockTrainingRuns(runs),
    },
  };
}

type MockTrainingJobStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

function mockTrainingJobPayload(
  request: MockTrainingPlanRequest,
  { status }: { status: MockTrainingJobStatus },
  preferredPlan?: ReturnType<typeof mockTrainingRunPlan>,
) {
  const logFolder = request.logFolder ?? "test_model";
  const runPlan =
    status === "completed"
      ? completedMockTrainingRunPlan(request, preferredPlan)
      : status === "cancelled"
        ? cancelledMockTrainingRunPlan(request, preferredPlan)
        : status === "failed"
          ? failedMockTrainingRunPlan(request, preferredPlan)
      : preferredPlan ?? mockTrainingRunPlan(request);
  const isCompleted = status === "completed";
  const isFailed = status === "failed";
  const isCancelled = status === "cancelled";
  return {
    id: "job-1",
    status,
    modelType: request.modelType ?? "linears",
    model: request.model ?? "linear",
    preset: request.preset ?? "baseline",
    presets: request.presets ?? ["baseline"],
    datasets: request.datasets ?? ["Mnist"],
    overrides: request.overrides ?? {},
    runPlan,
    monitors: request.monitors ?? [],
    logFolder,
    createdAt: "2026-06-01T00:00:00Z",
    updatedAt: isCompleted
      ? "2026-06-01T00:00:01Z"
      : "2026-06-01T00:00:00Z",
    exitCode: isCompleted ? 0 : isFailed ? 1 : isCancelled ? -15 : null,
    pid: 123,
    cancellationMode: "strict-cgroup",
    currentPreset: request.preset ?? "baseline",
    currentDataset: request.datasets?.[0] ?? "Mnist",
    epoch: isCompleted ? 1 : 0,
    step: isCompleted ? 4 : 0,
    metrics: isCompleted ? { validation_accuracy: 0.9 } : {},
    logDir: isCompleted ? `logs/${logFolder}` : null,
    events: isCancelled ? [{ type: "cancelled", status: "cancelled" }] : [],
    eventCount: isCancelled ? 1 : 0,
    eventCounts: isCancelled ? { cancelled: 1 } : {},
    eventsTruncated: false,
    clusterGrowth: [],
    logTail: isCompleted ? ["done"] : [],
    resultLinks: isCompleted
      ? [
          {
            preset: request.preset ?? "baseline",
            dataset: "Mnist",
            logDir: `logs/${logFolder}`,
          },
        ]
      : [],
  };
}

type MockMonitorScalarSeries = {
  tag: string;
  label: string;
  points: Array<{ step: number; wallTime: number; value: number }>;
};

type MockMonitorHistogram = {
  tag: string;
  step: number;
  wallTime: number;
  buckets: Array<{ left: number; right: number; count: number }>;
};

type MockMonitorImage = {
  tag: string;
  step: number;
  wallTime: number;
  mimeType: string;
  dataUrl: string;
};

type MockMonitorPayload = {
  scalarSeries: MockMonitorScalarSeries[];
  histograms: MockMonitorHistogram[];
  images: MockMonitorImage[];
};

type MockMonitorRequestContext = {
  jobId: string;
  nodePath: string;
  preset: string | null;
  dataset: string | null;
  logDir: string | null;
};

const tinyPngDataUrl =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

function mockScalarSeries(
  nodePath: string,
  suffix: string,
  values: [number, number] = [0.1, 0.2],
): MockMonitorScalarSeries {
  return {
    tag: `${nodePath}/${suffix}`,
    label: suffix,
    points: [
      { step: 1, wallTime: 1780000000, value: values[0] },
      { step: 2, wallTime: 1780000001, value: values[1] },
    ],
  };
}

function mockHistogram(nodePath: string, suffix = "histogram/usage_fraction") {
  return {
    tag: `${nodePath}/${suffix}`,
    step: 2,
    wallTime: 1780000001,
    buckets: [
      { left: 0, right: 0.5, count: 2 },
      { left: 0.5, right: 1, count: 1 },
    ],
  };
}

function mockMonitorImage(nodePath: string, suffix = "heatmap/usage_fraction") {
  return {
    tag: `${nodePath}/${suffix}`,
    step: 2,
    wallTime: 1780000001,
    mimeType: "image/png",
    dataUrl: tinyPngDataUrl,
  };
}

function defaultMonitorPayload(nodePath: string): MockMonitorPayload {
  return {
    scalarSeries: [mockScalarSeries(nodePath, "output/mean")],
    histograms: [mockHistogram(nodePath)],
    images: [mockMonitorImage(nodePath)],
  };
}

function defaultLogRunMonitorPayload(nodePath: string): MockMonitorPayload {
  return {
    scalarSeries: [mockScalarSeries(nodePath, "output/mean", [0.11, 0.22])],
    histograms: [],
    images: [],
  };
}

function semanticMonitorPayload(nodePath: string): MockMonitorPayload {
  return {
    scalarSeries: [
      mockScalarSeries(nodePath, "input/mean", [0.1, 0.15]),
      mockScalarSeries(nodePath, "bias/grad_mean", [0.01, 0.02]),
      mockScalarSeries(nodePath, "bias/mean", [0.03, 0.04]),
      mockScalarSeries(nodePath, "weights/grad_norm", [0.09, 0.12]),
      mockScalarSeries(nodePath, "weights/l2_norm", [1.2, 1.4]),
      mockScalarSeries(nodePath, "attention/entropy_mean", [0.8, 0.7]),
      mockScalarSeries(nodePath, "recurrent/actual_steps", [2, 3]),
      mockScalarSeries(nodePath, "gate/output_mean", [0.2, 0.25]),
      mockScalarSeries(nodePath, "parametric/generated_weight_norm", [1.1, 1.3]),
      mockScalarSeries(nodePath, "router/weight_entropy", [0.5, 0.6]),
    ],
    histograms: [mockHistogram(nodePath)],
    images: [mockMonitorImage(nodePath)],
  };
}

function withParameterCounts(body: unknown) {
  if (typeof body !== "object" || body === null) {
    return body;
  }

  const payload = body as {
    parameterCount?: unknown;
    parameterSizeBytes?: unknown;
    nodes?: unknown[];
    [key: string]: unknown;
  };
  if (!Array.isArray(payload.nodes)) {
    return body;
  }

  const nodes = payload.nodes.map((node) => {
    if (typeof node !== "object" || node === null) {
      return node;
    }
    const graphNode = node as {
      parameterCount?: unknown;
      parameterSizeBytes?: unknown;
      [key: string]: unknown;
    };
    const parameterCount =
      typeof graphNode.parameterCount === "number" ? graphNode.parameterCount : 0;
    return {
      ...graphNode,
      config:
        graphNode.config && typeof graphNode.config === "object"
          ? graphNode.config
          : null,
      parameterCount,
      parameterSizeBytes:
        typeof graphNode.parameterSizeBytes === "number"
          ? graphNode.parameterSizeBytes
          : parameterCount * 4,
    };
  });
  const firstNode = nodes[0] as
    | { parameterCount?: unknown; parameterSizeBytes?: unknown }
    | undefined;
  const parameterCount =
    typeof payload.parameterCount === "number"
      ? payload.parameterCount
      : typeof firstNode?.parameterCount === "number"
        ? firstNode.parameterCount
        : 0;

  return {
    ...payload,
    parameterCount,
    parameterSizeBytes:
      typeof payload.parameterSizeBytes === "number"
        ? payload.parameterSizeBytes
        : typeof firstNode?.parameterSizeBytes === "number"
          ? firstNode.parameterSizeBytes
          : parameterCount * 4,
    nodes,
  };
}

function withPreviewIdentity(
  body: unknown,
  request: { modelType?: unknown; model?: unknown; preset?: unknown },
) {
  if (typeof body !== "object" || body === null) {
    return body;
  }
	  return {
	    ...body,
	    ...(typeof request.modelType === "string" ? { modelType: request.modelType } : {}),
	    ...(typeof request.model === "string" ? { model: request.model } : {}),
    ...(typeof request.preset === "string" ? { preset: request.preset } : {}),
  };
}

type FetchScenarioOptions = {
    inspectError?: boolean;
    inspectResponse?: unknown;
    inspectResponseFactory?: (requestIndex: number) => unknown | Promise<unknown>;
    modelsResponse?: typeof modelsResponse;
    logRunsResponse?: typeof logRunsResponse;
    logExperimentsResponse?: typeof logExperimentsResponse;
    logScalarSeries?: typeof logScalarSeries;
    logScalarResponseFactory?: (
      body: {
        runIds: string[];
        tags: string[];
        maxPoints?: number;
        sampling?: string;
      },
      requestIndex: number,
    ) => unknown | Promise<unknown>;
    logMediaResponse?: { images: unknown[]; texts: unknown[] };
    logMediaResponseFactory?: (
      body: {
        runIds: string[];
        imageTags: string[];
        textTags: string[];
      },
      requestIndex: number,
    ) => unknown | Promise<unknown>;
    logTagsResponseFactory?: (
      body: { runIds: string[] },
      requestIndex: number,
    ) => unknown | Promise<unknown>;
    logTagsByRun?: Record<string, MockLogTags>;
    logCheckpointsByRun?: Record<string, MockLogCheckpoint[]>;
    logRunArtifactsByRun?: Record<
      string,
      Partial<Omit<MockLogRunArtifacts, "runId">>
    >;
    deleteLogExperimentError?: string;
    deleteLogRunPlanError?: string;
    deleteLogRunPlanErrorFactory?: (requestIndex: number) => string | undefined;
    deleteLogRunsError?: string;
    deleteLogRunsErrorFactory?: (requestIndex: number) => string | undefined;
    deleteLogRunsBlockers?: Array<{ id: string; logFolder: string; status: string }>;
    logImportError?: string;
    logImportResponse?: {
      extractedFileCount: number;
      skippedFileCount: number;
      destinationRoot: string;
    };
    logImportResponseFactory?: (
      requestIndex: number,
    ) => unknown | Promise<unknown>;
    capabilitiesResponse?: typeof capabilitiesResponse;
    presetsResponse?: typeof presetsResponse;
    schemaResponse?: unknown;
    searchSpaceResponse?: typeof searchSpaceResponse;
    searchSpaceResponseFactory?: (url: string) => unknown;
    datasetsResponse?: typeof datasetsResponse;
    monitorDataResponse?: (
      context: MockMonitorRequestContext,
    ) => MockMonitorPayload | Promise<MockMonitorPayload>;
    logRunMonitorDataResponse?: (
      context: MockMonitorRequestContext,
    ) => MockMonitorPayload | Promise<MockMonitorPayload>;
    parameterStatusResponse?: (context: {
      jobId: string;
      preset: string | null;
      dataset: string | null;
      logDir: string | null;
    }) => unknown;
    logParameterStatusResponse?: (
      context: { runIds: string[] },
    ) => unknown | Promise<unknown>;
    trainingJobStatus?: MockTrainingJobStatus;
    trainingRunPlanResponseFactory?: (
      request: MockTrainingPlanRequest,
      requestIndex: number,
      defaultPlan: ReturnType<typeof mockTrainingRunPlan>,
    ) => unknown | Promise<unknown>;
    trainingJobResponseFactory?: (
      requestIndex: number,
    ) => unknown | Promise<unknown>;
    createTrainingJobResponseFactory?: (
      request: Record<string, unknown>,
    ) => unknown | Promise<unknown>;
    cancelTrainingJobResponseFactory?: (
      request: Record<string, unknown> | undefined,
    ) => unknown | Promise<unknown>;
    cancelTrainingJobError?: string;
	    configSnapshotsResponse?: {
	      modelType: string;
	      model: string;
	      snapshots: MockConfigSnapshot[];
    };
  configSnapshotLibraryResponse?: {
    snapshots: MockConfigSnapshot[];
  };
};

function installFetchMock(options: FetchScenarioOptions = {}) {
  const inspectBodies: unknown[] = [];
  const trainingBodies: unknown[] = [];
  let trainingRunPlanRequestCount = 0;
  let latestTrainingRunPlan: ReturnType<typeof mockTrainingRunPlan> | undefined;
  let trainingJobPollRequestCount = 0;
  const logScalarRequests: Array<{
    runIds: string[];
    tags: string[];
    maxPoints?: number;
    sampling?: string;
  }> = [];
  const logMediaRequests: Array<{
    runIds: string[];
    imageTags: string[];
    textTags: string[];
  }> = [];
  const logTagRequests: Array<{ runIds: string[] }> = [];
  const configSnapshotCreateRequests: Array<{
    model: string;
    preset: string;
    name: string;
    overrides: Record<string, string>;
  }> = [];
  const configSnapshotUpdateRequests: Array<{
    id: string;
    body: {
      name?: string;
      overrides?: Record<string, string>;
    };
  }> = [];
  const logCheckpointRequests: Array<{ runIds: string[] }> = [];
  const logArtifactRequests: string[] = [];
  const logRunRequests: Array<{
    experiments: string[];
    modelTypes: string[];
    models: string[];
    presets: string[];
    datasets: string[];
    experimentTask: string | null;
    hasEventFiles: string | null;
    projection: string | null;
    limit: number;
    offset: number;
  }> = [];
  const deleteExperimentRequests: string[] = [];
	const deletePresetPlanRequests: Array<{ experiment: string; preset: string }> = [];
	const deletePresetRequests: Array<{ experiment: string; preset: string }> = [];
	  const deleteRunPlanRequests: Array<{
	    experiments: string[];
	    datasets: string[];
	    models: MockModelIdentity[];
	    presets: string[];
	    runIds: string[];
	  }> = [];
	  const deleteRunRequests: Array<{
	    experiments: string[];
	    datasets: string[];
	    models: MockModelIdentity[];
	    presets: string[];
	    runIds: string[];
	  }> = [];
  const logImportRequests: RequestInit[] = [];
  const monitorDataRequests: Array<{
    jobId: string;
    nodePath: string | null;
    preset: string | null;
    dataset: string | null;
  }> = [];
  const logRunMonitorDataRequests: Array<{ runId: string; nodePath: string | null }> = [];
  let latestTrainingRequest: Record<string, unknown> | undefined;
  let logResponse = { runs: [...(options.logRunsResponse ?? logRunsResponse).runs] };
  let experimentResponse = {
    experiments: [
      ...(options.logExperimentsResponse ?? logExperimentsResponse).experiments,
    ],
  };
  const tagsByRun = options.logTagsByRun ?? logTagsByRun;
  const scalarSeries = options.logScalarSeries ?? logScalarSeries;
  let configSnapshotResponse = {
    ...(options.configSnapshotsResponse ?? configSnapshotsResponse),
    snapshots: [
      ...(
        options.configSnapshotsResponse ?? configSnapshotsResponse
      ).snapshots,
    ],
  };

  function checkpointsForRun(runId: string) {
    const run = logResponse.runs.find((candidate) => candidate.id === runId);
    return run ? fallbackCheckpointsForRun(run, options.logCheckpointsByRun) : [];
  }

  function uniqueSorted(values: string[]) {
    return Array.from(new Set(values)).sort((a, b) => a.localeCompare(b));
  }

	  function matchingDeleteRuns(filters: {
	    experiments: string[];
	    datasets: string[];
	    models: MockModelIdentity[];
	    presets: string[];
	    runIds: string[];
	  }) {
    if (
      filters.experiments.length === 0 ||
      filters.datasets.length === 0 ||
      filters.models.length === 0 ||
      filters.presets.length === 0 ||
      filters.runIds.length === 0
    ) {
      return [];
    }
    const experiments = new Set(filters.experiments);
    const datasets = new Set(filters.datasets);
	    const models = new Set(
	      filters.models.map((model) => `${model.modelType}/${model.model}`),
	    );
    const presets = new Set(filters.presets);
    const runIds = new Set(filters.runIds);
    return logResponse.runs.filter(
      (run) =>
        experiments.has(run.experiment) &&
        datasets.has(run.dataset) &&
	        models.has(`${run.modelType}/${run.model}`) &&
        presets.has(run.preset) &&
        runIds.has(run.id),
    );
  }

	function deletePlanForCandidates(candidates: typeof logResponse.runs) {
	  const blockers = options.deleteLogRunsBlockers ?? [];
	  const returnedCandidates = candidates.slice(0, 500);
	  const truncated = returnedCandidates.length < candidates.length;
	  return {
	    candidateCount: candidates.length,
	    sourceItemCount: candidates.length,
	    returnedItemCount: returnedCandidates.length,
	    truncated,
	    truncationReason: truncated
	      ? "delete candidates capped at 500 rows"
	      : null,
	    counts: {
	      runs: candidates.length,
	      experiments: uniqueSorted(candidates.map((run) => run.experiment)).length,
	      datasets: uniqueSorted(candidates.map((run) => run.dataset)).length,
	      models: uniqueSorted(candidates.map((run) => `${run.modelType}/${run.model}`)).length,
	      presets: uniqueSorted(candidates.map((run) => run.preset)).length,
	    },
	    affected: {
	      experiments: uniqueSorted(candidates.map((run) => run.experiment)),
	      datasets: uniqueSorted(candidates.map((run) => run.dataset)),
	      models: uniqueSorted(candidates.map((run) => `${run.modelType}/${run.model}`))
	        .map((value) => {
	          const [modelType, model] = value.split("/", 2);
	          return { modelType, model };
	        }),
	      presets: uniqueSorted(candidates.map((run) => run.preset)),
	      runIds: uniqueSorted(candidates.map((run) => run.id)),
	    },
	    candidates: returnedCandidates.map((run) => ({
	      id: run.id,
	      experiment: run.experiment,
	      modelType: run.modelType,
	      model: run.model,
	      preset: run.preset,
	      dataset: run.dataset,
	      runName: run.runName,
	      version: run.version,
	      relativePath: run.relativePath,
	    })),
	    blockedByActiveJobs: blockers,
	    canDelete: candidates.length > 0 && blockers.length === 0,
	  };
	}

	function deletePlanPayload(filters: {
	  experiments: string[];
	  datasets: string[];
	  models: MockModelIdentity[];
	  presets: string[];
	  runIds: string[];
	}) {
	  return deletePlanForCandidates(matchingDeleteRuns(filters));
	}

	function presetDeletePlanPayload(target: { experiment: string; preset: string }) {
	  return deletePlanForCandidates(
	    logResponse.runs.filter(
	      (run) =>
	        run.experiment === target.experiment && run.preset === target.preset,
	    ),
	  );
	}

  const dispatchFetch = (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const endsWithAny = (suffixes: string[]) =>
      suffixes.some((suffix) => url.endsWith(suffix));
    const includesAny = (fragments: string[]) =>
      fragments.some((fragment) => url.includes(fragment));
    if (url.endsWith("/health")) {
      return jsonResponse({ status: "ok" });
    }
    if (url.endsWith("/capabilities")) {
      return jsonResponse(options.capabilitiesResponse ?? capabilitiesResponse);
    }
    if (url.endsWith("/models")) {
      return jsonResponse(options.modelsResponse ?? modelsResponse);
    }
    if (url.endsWith("/config-snapshots/library")) {
      return jsonResponse(
        options.configSnapshotLibraryResponse ?? configSnapshotLibraryResponse,
      );
    }
    if (url.endsWith("/config-snapshots") && init?.method === "POST") {
	      const body = JSON.parse(String(init.body)) as {
	        modelType: string;
	        model: string;
        preset: string;
        name: string;
        overrides: Record<string, string>;
      };
      configSnapshotCreateRequests.push(body);
	      const snapshot = {
	        id: `snapshot-created-${configSnapshotCreateRequests.length}`,
	        modelType: body.modelType,
	        model: body.model,
        preset: body.preset,
        name: body.name,
        overrides: body.overrides,
        createdAt: "2026-06-01T00:00:00.000Z",
        updatedAt: "2026-06-01T00:00:00.000Z",
      };
	      configSnapshotResponse = {
	        modelType: body.modelType,
	        model: body.model,
        snapshots: [...configSnapshotResponse.snapshots, snapshot],
      };
      return jsonResponse(snapshot);
    }
    if (url.includes("/config-snapshots/") && init?.method === "PATCH") {
      const snapshotId = decodeURIComponent(
        url.split("/config-snapshots/")[1] ?? "",
      );
      const body = JSON.parse(String(init.body)) as {
        name?: string;
        overrides?: Record<string, string>;
      };
      configSnapshotUpdateRequests.push({ id: snapshotId, body });
      const existing = configSnapshotResponse.snapshots.find(
        (snapshot) => snapshot.id === snapshotId,
      );
      const snapshot = {
	        ...(existing ?? {
	          id: snapshotId,
	          modelType: configSnapshotResponse.modelType,
	          model: configSnapshotResponse.model,
          preset: "baseline",
          name: "Snapshot",
          overrides: {},
          createdAt: "2026-06-01T00:00:00.000Z",
          updatedAt: "2026-06-01T00:00:00.000Z",
        }),
        ...("name" in body ? { name: body.name ?? "" } : {}),
        ...("overrides" in body ? { overrides: body.overrides ?? {} } : {}),
        updatedAt: "2026-06-01T00:00:01.000Z",
      };
	      configSnapshotResponse = {
	        modelType: snapshot.modelType,
	        model: snapshot.model,
	        snapshots: configSnapshotResponse.snapshots.map((candidate) =>
	          candidate.id === snapshotId ? snapshot : candidate,
	        ),
	      };
	      return jsonResponse(snapshot);
	    }
	    if (url.includes("/config-snapshots/") && init?.method === "DELETE") {
	      const snapshotId = decodeURIComponent(
	        url.split("/config-snapshots/")[1] ?? "",
	      );
	      configSnapshotResponse = {
	        ...configSnapshotResponse,
	        snapshots: configSnapshotResponse.snapshots.filter(
	          (snapshot) => snapshot.id !== snapshotId,
	        ),
	      };
	      return jsonResponse(configSnapshotResponse);
	    }
	    if (url.includes("/config-snapshots?")) {
	      const parsedUrl = new URL(url);
	      const modelType = parsedUrl.searchParams.get("modelType") ?? "linears";
	      const model = parsedUrl.searchParams.get("model") ?? "linear";
	      return jsonResponse({ ...configSnapshotResponse, modelType, model });
	    }
    const neuronModel = [
      "linear",
      "linear_adaptive",
      "expert_linear",
      "expert_linear_adaptive",
    ].find((model) => url.includes(`/models/neuron/${model}/`));
    if (neuronModel && url.endsWith(`/models/neuron/${neuronModel}/presets`)) {
      return jsonResponse({ ...neuronPresetsResponse, model: neuronModel });
    }
    if (neuronModel && url.endsWith(`/models/neuron/${neuronModel}/datasets`)) {
      return jsonResponse({ ...neuronDatasetsResponse, model: neuronModel });
    }
    if (neuronModel && url.endsWith(`/models/neuron/${neuronModel}/monitors`)) {
      return jsonResponse({ ...neuronMonitorsResponse, model: neuronModel });
    }
    if (neuronModel && url.includes(`/models/neuron/${neuronModel}/config-schema`)) {
      const schemaPayload = options.schemaResponse ?? schemaResponse;
      return jsonResponse(
	        typeof schemaPayload === "object" && schemaPayload !== null
	          ? { ...schemaPayload, modelType: "neuron", model: neuronModel }
	          : schemaPayload,
      );
    }
    if (neuronModel && url.includes(`/models/neuron/${neuronModel}/search-space`)) {
      const searchPayload = options.searchSpaceResponse ?? searchSpaceResponse;
      return jsonResponse({
        ...searchPayload,
        modelType: "neuron",
        model: neuronModel,
        preset: "baseline",
      });
    }
    if (endsWithAny(["/models/linear/presets", "/models/linears/linear/presets"])) {
      return jsonResponse(options.presetsResponse ?? presetsResponse);
    }
    if (
      endsWithAny(["/models/linear/datasets", "/models/linears/linear/datasets"])
    ) {
      return jsonResponse(options.datasetsResponse ?? datasetsResponse);
    }
    if (
      endsWithAny(["/models/linear/monitors", "/models/linears/linear/monitors"])
    ) {
      return jsonResponse(monitorsResponse);
    }
    if (
      includesAny([
        "/models/linear/config-schema",
        "/models/linears/linear/config-schema",
      ])
    ) {
      return jsonResponse(options.schemaResponse ?? schemaResponse);
    }
    if (
      includesAny([
        "/models/linear/search-space",
        "/models/linears/linear/search-space",
      ])
    ) {
      const searchPayload = options.searchSpaceResponseFactory
        ? options.searchSpaceResponseFactory(url)
        : options.searchSpaceResponse;
      return jsonResponse(searchPayload ?? searchSpaceResponse);
    }
    if (
      endsWithAny([
        "/models/bert/linear/presets",
        "/models/bert/linear_adaptive/presets",
        "/models/bert/expert_linear/presets",
        "/models/bert/expert_linear_adaptive/presets",
      ])
    ) {
      return jsonResponse(bertPresetsResponse);
    }
    if (
      endsWithAny([
        "/models/vit/linear/presets",
        "/models/vit/linear_adaptive/presets",
        "/models/vit/expert_linear/presets",
        "/models/vit/expert_linear_adaptive/presets",
      ])
    ) {
      return jsonResponse(vitPresetsResponse);
    }
    if (
      endsWithAny([
        "/models/bert/linear/datasets",
        "/models/bert/linear_adaptive/datasets",
        "/models/bert/expert_linear/datasets",
        "/models/bert/expert_linear_adaptive/datasets",
      ])
    ) {
      return jsonResponse(bertDatasetsResponse);
    }
    if (
      endsWithAny([
        "/models/vit/linear/datasets",
        "/models/vit/linear_adaptive/datasets",
        "/models/vit/expert_linear/datasets",
        "/models/vit/expert_linear_adaptive/datasets",
      ])
    ) {
      return jsonResponse(vitDatasetsResponse);
    }
    if (
      endsWithAny([
        "/models/bert/linear/monitors",
        "/models/bert/linear_adaptive/monitors",
        "/models/bert/expert_linear/monitors",
        "/models/bert/expert_linear_adaptive/monitors",
      ])
    ) {
      return jsonResponse({
        modelType: "bert",
        model: "linear",
        monitors: [],
      });
    }
    if (
      endsWithAny([
        "/models/vit/linear/monitors",
        "/models/vit/linear_adaptive/monitors",
        "/models/vit/expert_linear/monitors",
        "/models/vit/expert_linear_adaptive/monitors",
      ])
    ) {
      return jsonResponse({
        modelType: "vit",
        model: "linear",
        monitors: [],
      });
    }
    if (
      includesAny([
        "/models/bert/linear/config-schema",
        "/models/bert/linear_adaptive/config-schema",
        "/models/bert/expert_linear/config-schema",
        "/models/bert/expert_linear_adaptive/config-schema",
      ])
    ) {
      const schemaPayload = options.schemaResponse ?? schemaResponse;
      return jsonResponse(
        typeof schemaPayload === "object" && schemaPayload !== null
          ? {
              ...schemaPayload,
              modelType: "bert",
              model: "linear",
            }
          : schemaPayload,
      );
    }
    if (
      includesAny([
        "/models/vit/linear/config-schema",
        "/models/vit/linear_adaptive/config-schema",
        "/models/vit/expert_linear/config-schema",
        "/models/vit/expert_linear_adaptive/config-schema",
      ])
    ) {
      const schemaPayload = options.schemaResponse ?? schemaResponse;
      return jsonResponse(
        typeof schemaPayload === "object" && schemaPayload !== null
          ? {
              ...schemaPayload,
              modelType: "vit",
              model: "linear",
            }
          : schemaPayload,
      );
    }
    if (
      includesAny([
        "/models/bert/linear/search-space",
        "/models/bert/linear_adaptive/search-space",
        "/models/bert/expert_linear/search-space",
        "/models/bert/expert_linear_adaptive/search-space",
      ])
    ) {
      return jsonResponse({
        modelType: "bert",
        model: "linear",
        preset: "bert-baseline",
        axes: [],
      });
    }
    if (
      includesAny([
        "/models/vit/linear/search-space",
        "/models/vit/linear_adaptive/search-space",
        "/models/vit/expert_linear/search-space",
        "/models/vit/expert_linear_adaptive/search-space",
      ])
    ) {
      return jsonResponse({
        modelType: "vit",
        model: "linear",
        preset: "baseline",
        axes: [],
      });
    }
    if (url.endsWith("/training/run-plan")) {
      const request = JSON.parse(String(init?.body)) as MockTrainingPlanRequest;
      const defaultPlan = mockTrainingRunPlan(
        request,
        configSnapshotResponse.snapshots,
      );
      const requestIndex = trainingRunPlanRequestCount;
      trainingRunPlanRequestCount += 1;
      const responseBody = options.trainingRunPlanResponseFactory
        ? options.trainingRunPlanResponseFactory(
            request,
            requestIndex,
            defaultPlan,
          )
        : defaultPlan;
      return Promise.resolve(responseBody).then((body) => {
        latestTrainingRunPlan = body as ReturnType<typeof mockTrainingRunPlan>;
        return jsonResponse(body);
      });
    }
    if (url.endsWith("/inspect")) {
      const inspectRequest = JSON.parse(String(init?.body)) as {
        model?: unknown;
        preset?: unknown;
      };
      inspectBodies.push(inspectRequest);
      if (options.inspectError) {
        return jsonResponse({ detail: "invalid override value" }, 400);
      }
      const inspectRequestIndex = inspectBodies.length - 1;
      const responseBody = options.inspectResponseFactory
        ? options.inspectResponseFactory(inspectRequestIndex)
        : options.inspectResponse;
      return Promise.resolve(responseBody ?? inspectResponse).then((body) =>
        jsonResponse(withParameterCounts(withPreviewIdentity(body, inspectRequest))),
      );
    }
    if (url.endsWith("/training/jobs")) {
      latestTrainingRequest = JSON.parse(String(init?.body));
      const trainingRequest = latestTrainingRequest as Record<string, unknown>;
      trainingBodies.push(latestTrainingRequest);
      const runPlan = mockAcceptedTrainingRunPlan(
        latestTrainingRequest as MockTrainingPlanRequest,
        latestTrainingRequest?.runPlan as MockSubmittedTrainingRunPlan | undefined,
        latestTrainingRunPlan,
      );
      const logFolder = String(latestTrainingRequest?.logFolder ?? "test_model");
      if (!experimentResponse.experiments.some((entry) => entry.experiment === logFolder)) {
        experimentResponse = {
          experiments: [
            ...experimentResponse.experiments,
            { experiment: logFolder, runCount: 0, relativePath: logFolder },
          ],
        };
      }
      const responseBody = options.createTrainingJobResponseFactory
        ? options.createTrainingJobResponseFactory(trainingRequest)
        : {
            ...mockTrainingJobPayload(
              trainingRequest as MockTrainingPlanRequest,
              {
                status: "running",
              },
              runPlan,
            ),
            runPlan,
            monitors: latestTrainingRequest?.monitors ?? [],
          };
      return Promise.resolve(responseBody).then((body) => jsonResponse(body));
    }
    if (url.endsWith("/training/jobs/job-1/cancel")) {
      if (options.cancelTrainingJobError) {
        return jsonResponse({ detail: options.cancelTrainingJobError }, 400);
      }
      const logFolder = String(latestTrainingRequest?.logFolder ?? "test_model");
      const responseBody = options.cancelTrainingJobResponseFactory
        ? options.cancelTrainingJobResponseFactory(latestTrainingRequest)
        : mockTrainingJobPayload(
            (latestTrainingRequest ?? { logFolder }) as MockTrainingPlanRequest,
            { status: "cancelled" },
            latestTrainingRunPlan,
          );
      return Promise.resolve(responseBody).then((body) => jsonResponse(body));
    }
    if (url.endsWith("/training/jobs/job-1")) {
      const requestIndex = trainingJobPollRequestCount;
      trainingJobPollRequestCount += 1;
      if (options.trainingJobResponseFactory) {
        return Promise.resolve(
          options.trainingJobResponseFactory(requestIndex),
        ).then((body) => jsonResponse(body));
      }
      const logFolder = String(latestTrainingRequest?.logFolder ?? "test_model");
      return jsonResponse(
        mockTrainingJobPayload(
          (latestTrainingRequest ?? { logFolder }) as MockTrainingPlanRequest,
          { status: options.trainingJobStatus ?? "completed" },
          latestTrainingRunPlan,
        ),
      );
    }
    if (url.includes("/training/jobs/job-1/monitor-data")) {
      const parsedUrl = new URL(url);
      const logFolder = String(latestTrainingRequest?.logFolder ?? "test_model");
      const nodePath = parsedUrl.searchParams.get("nodePath") ?? "";
      const preset = parsedUrl.searchParams.get("preset");
      const dataset = parsedUrl.searchParams.get("dataset");
      const logDir = `logs/${logFolder}`;
      monitorDataRequests.push({
        jobId: "job-1",
        nodePath,
        preset,
        dataset,
      });
      const monitorPayload =
        options.monitorDataResponse?.({
          jobId: "job-1",
          nodePath,
          preset,
          dataset,
          logDir,
        }) ?? defaultMonitorPayload(nodePath);
      return Promise.resolve(monitorPayload).then((payload) =>
        jsonResponse({
          jobId: "job-1",
          nodePath,
          preset,
          dataset,
          logDir,
          ...payload,
        }),
      );
    }
    if (url.includes("/training/jobs/job-1/monitor-parameter-status")) {
      const parsedUrl = new URL(url);
      const logFolder = String(latestTrainingRequest?.logFolder ?? "test_model");
      const preset = parsedUrl.searchParams.get("preset");
      const dataset = parsedUrl.searchParams.get("dataset");
      const logDir = `logs/${logFolder}`;
      return jsonResponse(
        options.parameterStatusResponse?.({
          jobId: "job-1",
          preset,
          dataset,
          logDir,
        }) ?? {
          sourceId: "job-1",
          preset,
          dataset,
          logDir,
          nodes: [],
        },
      );
    }
    if (url.endsWith("/logs/runs/preset-delete-plan")) {
      const body = JSON.parse(String(init?.body)) as {
        experiment: string;
        preset: string;
      };
      deletePresetPlanRequests.push(body);
      const planError =
        options.deleteLogRunPlanErrorFactory?.(
          deletePresetPlanRequests.length - 1,
        ) ?? options.deleteLogRunPlanError;
      if (planError) {
        return jsonResponse({ detail: planError }, 400);
      }
      return jsonResponse(presetDeletePlanPayload(body));
    }
    if (url.endsWith("/logs/runs/preset-delete")) {
      const body = JSON.parse(String(init?.body)) as {
        experiment: string;
        preset: string;
      };
      deletePresetRequests.push(body);
      const deleteError =
        options.deleteLogRunsErrorFactory?.(deletePresetRequests.length - 1) ??
        options.deleteLogRunsError;
      if (deleteError) {
        return jsonResponse({ detail: deleteError }, 400);
      }
      const plan = presetDeletePlanPayload(body);
      if (!plan.canDelete) {
        return jsonResponse(
          { detail: "A training job is still writing to this log folder." },
          400,
        );
      }
      const deletedRuns = logResponse.runs.filter(
        (run) =>
          run.experiment === body.experiment && run.preset === body.preset,
      );
      const deletedRunIds = new Set(deletedRuns.map((run) => run.id));
      logResponse = {
        runs: logResponse.runs.filter((run) => !deletedRunIds.has(run.id)),
      };
      const runCounts = new Map<string, number>();
      for (const run of logResponse.runs) {
        runCounts.set(run.experiment, (runCounts.get(run.experiment) ?? 0) + 1);
      }
      experimentResponse = {
        experiments: experimentResponse.experiments
          .map((entry) => ({
            ...entry,
            runCount: runCounts.get(entry.experiment) ?? 0,
          }))
          .filter((entry) => entry.runCount > 0),
      };
      return jsonResponse({
        ...plan,
        deletedRunIds: deletedRuns.map((run) => run.id),
        deletedRunCount: deletedRuns.length,
        deletedRelativePaths: deletedRuns.map((run) => run.relativePath),
      });
    }
    if (url.endsWith("/logs/runs/delete-plan")) {
	      const body = JSON.parse(String(init?.body)) as {
	        experiments: string[];
	        datasets: string[];
	        models: MockModelIdentity[];
	        presets: string[];
	        runIds: string[];
      };
      deleteRunPlanRequests.push(body);
      const planError =
        options.deleteLogRunPlanErrorFactory?.(
          deleteRunPlanRequests.length - 1,
        ) ?? options.deleteLogRunPlanError;
      if (planError) {
        return jsonResponse({ detail: planError }, 400);
      }
      return jsonResponse(deletePlanPayload(body));
    }
    if (url.endsWith("/logs/runs/delete")) {
	      const body = JSON.parse(String(init?.body)) as {
	        experiments: string[];
	        datasets: string[];
	        models: MockModelIdentity[];
	        presets: string[];
	        runIds: string[];
      };
      deleteRunRequests.push(body);
      const deleteError =
        options.deleteLogRunsErrorFactory?.(deleteRunRequests.length - 1) ??
        options.deleteLogRunsError;
      if (deleteError) {
        return jsonResponse({ detail: deleteError }, 400);
      }
      const plan = deletePlanPayload(body);
      if (!plan.canDelete) {
        return jsonResponse(
          { detail: "A training job is still writing to this log folder." },
          400,
        );
      }
      const deletedRunIds = new Set(plan.candidates.map((run) => run.id));
      logResponse = {
        runs: logResponse.runs.filter((run) => !deletedRunIds.has(run.id)),
      };
      const runCounts = new Map<string, number>();
      for (const run of logResponse.runs) {
        runCounts.set(run.experiment, (runCounts.get(run.experiment) ?? 0) + 1);
      }
      experimentResponse = {
        experiments: experimentResponse.experiments
          .map((entry) => ({
            ...entry,
            runCount: runCounts.get(entry.experiment) ?? 0,
          }))
          .filter((entry) => entry.runCount > 0),
      };
      return jsonResponse({
        ...plan,
        deletedRunIds: plan.candidates.map((run) => run.id),
        deletedRunCount: plan.candidates.length,
        deletedRelativePaths: plan.candidates.map((run) => run.relativePath),
      });
    }
    if (url.endsWith("/logs/import")) {
      logImportRequests.push(init ?? {});
      if (options.logImportError) {
        return jsonResponse({ detail: options.logImportError }, 400);
      }
      const requestIndex = logImportRequests.length - 1;
      const responseBody = options.logImportResponseFactory
        ? options.logImportResponseFactory(requestIndex)
        : options.logImportResponse;
      return Promise.resolve(
        responseBody ?? {
          extractedFileCount: 2,
          skippedFileCount: 0,
          destinationRoot: "/workspace/logs",
        },
      ).then((body) => jsonResponse(body));
    }
    if (url.endsWith("/logs/runs") || url.includes("/logs/runs?")) {
      const parsedUrl = new URL(url, "http://testserver");
      const selectedExperiments = parsedUrl.searchParams.getAll("experiment");
      const selectedModelTypes = parsedUrl.searchParams.getAll("modelType");
      const selectedModels = parsedUrl.searchParams.getAll("model");
      const selectedPresets = parsedUrl.searchParams.getAll("preset");
      const selectedDatasets = parsedUrl.searchParams.getAll("dataset");
      const experimentTask = parsedUrl.searchParams.get("experimentTask");
      const hasEventFiles = parsedUrl.searchParams.get("hasEventFiles");
      const projection = parsedUrl.searchParams.get("projection");
      const offset = Number(parsedUrl.searchParams.get("offset") ?? 0);
      const limit = Number(
        parsedUrl.searchParams.get("limit") ?? logResponse.runs.length,
      );
      logRunRequests.push({
        experiments: selectedExperiments,
        modelTypes: selectedModelTypes,
        models: selectedModels,
        presets: selectedPresets,
        datasets: selectedDatasets,
        experimentTask,
        hasEventFiles,
        projection,
        limit,
        offset,
      });
      const filteredRuns = logResponse.runs.filter((run) => {
        if (
          selectedExperiments.length > 0 &&
          !selectedExperiments.includes(run.experiment)
        ) {
          return false;
        }
        if (
          selectedModelTypes.length > 0 &&
          !selectedModelTypes.includes(run.modelType)
        ) {
          return false;
        }
        if (selectedModels.length > 0 && !selectedModels.includes(run.model)) {
          return false;
        }
        if (selectedPresets.length > 0 && !selectedPresets.includes(run.preset)) {
          return false;
        }
        if (selectedDatasets.length > 0 && !selectedDatasets.includes(run.dataset)) {
          return false;
        }
        if (
          experimentTask &&
          !("experimentTask" in run && run.experimentTask === experimentTask)
        ) {
          return false;
        }
        if (hasEventFiles === "true" && run.eventFileCount <= 0) {
          return false;
        }
        if (hasEventFiles === "false" && run.eventFileCount > 0) {
          return false;
        }
        return true;
      });
      const facetCounts = (values: string[]) =>
        Array.from(
          values.reduce((counts, value) => {
            counts.set(value, (counts.get(value) ?? 0) + 1);
            return counts;
          }, new Map<string, number>()),
          ([value, count]) => ({ value, count }),
        ).sort((left, right) => left.value.localeCompare(right.value));
      const facetExperiments = Array.from(
        filteredRuns.reduce((groups, run) => {
          const group = groups.get(run.experiment) ?? [];
          group.push(run);
          groups.set(run.experiment, group);
          return groups;
        }, new Map<string, typeof filteredRuns>()),
      )
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([experiment, runs]) => ({
          experiment,
          runCount: runs.length,
          datasets: facetCounts(runs.map((run) => run.dataset)),
          models: Array.from(
            runs.reduce((counts, run) => {
              const key = `${run.modelType}\u0000${run.model}`;
              const current = counts.get(key);
              counts.set(key, {
                modelType: run.modelType,
                model: run.model,
                count: (current?.count ?? 0) + 1,
              });
              return counts;
            }, new Map<string, { modelType: string; model: string; count: number }>()),
            ([, value]) => value,
          ).sort((left, right) =>
            `${left.modelType}/${left.model}`.localeCompare(
              `${right.modelType}/${right.model}`,
            ),
          ),
          presets: facetCounts(runs.map((run) => run.preset)),
        }));
      return jsonResponse({
        ...logResponse,
        runs: filteredRuns.slice(offset, offset + limit).map((run) =>
          projection === "summary" ? { ...run, metrics: {} } : run,
        ),
        facets: { experiments: facetExperiments },
        total: filteredRuns.length,
        limit,
        offset,
        hasMore: offset + limit < filteredRuns.length,
      });
    }
    if (url.endsWith("/logs/checkpoints")) {
      const body = JSON.parse(String(init?.body)) as { runIds: string[] };
      logCheckpointRequests.push(body);
      return jsonResponse({
        checkpoints: body.runIds.flatMap((runId) => checkpointsForRun(runId)),
      });
    }
    if (url.includes("/logs/runs/") && url.endsWith("/artifacts")) {
      const runId = decodeURIComponent(
        url.split("/logs/runs/")[1]?.split("/artifacts")[0] ?? "",
      );
      logArtifactRequests.push(runId);
      const run = logResponse.runs.find((candidate) => candidate.id === runId);
      if (!run) {
        return jsonResponse({ detail: `Unknown log run id: ${runId}` }, 400);
      }
      const checkpoints = checkpointsForRun(runId);
      return jsonResponse({
        ...defaultArtifactsForRun(run, checkpoints),
        ...(options.logRunArtifactsByRun?.[runId] ?? {}),
        runId,
      });
    }
    if (url.includes("/logs/runs/") && url.includes("/monitor-data")) {
      const parsedUrl = new URL(url);
      const runId = decodeURIComponent(
        url.split("/logs/runs/")[1]?.split("/monitor-data")[0] ?? "",
      );
      const run = logResponse.runs.find((candidate) => candidate.id === runId);
      const nodePath = parsedUrl.searchParams.get("nodePath") ?? "";
      logRunMonitorDataRequests.push({
        runId,
        nodePath,
      });
      const logDir = run?.relativePath ?? null;
      const monitorPayload =
        options.logRunMonitorDataResponse?.({
          jobId: runId,
          nodePath,
          preset: null,
          dataset: run?.dataset ?? null,
          logDir,
        }) ?? defaultLogRunMonitorPayload(nodePath);
      return Promise.resolve(monitorPayload).then((payload) =>
        jsonResponse({
          jobId: runId,
          nodePath,
          dataset: run?.dataset ?? null,
          logDir,
          ...payload,
        }),
      );
    }
    if (url.endsWith("/logs/experiments")) {
      return jsonResponse(experimentResponse);
    }
    if (url.includes("/logs/experiments/")) {
      const experiment = decodeURIComponent(url.split("/logs/experiments/")[1] ?? "");
      deleteExperimentRequests.push(experiment);
      if (options.deleteLogExperimentError) {
        return jsonResponse({ detail: options.deleteLogExperimentError }, 400);
      }
      const deletedRuns = logResponse.runs.filter(
        (run) => run.experiment === experiment,
      );
      logResponse = {
        runs: logResponse.runs.filter((run) => run.experiment !== experiment),
      };
      experimentResponse = {
        experiments: experimentResponse.experiments.filter(
          (entry) => entry.experiment !== experiment,
        ),
      };
      return jsonResponse({
        experiment,
        deletedRunIds: deletedRuns.map((run) => run.id),
        deletedRunCount: deletedRuns.length,
        deletedRelativePath: experiment,
      });
    }
    if (url.endsWith("/logs/tags")) {
      const body = JSON.parse(String(init?.body)) as { runIds: string[] };
      logTagRequests.push(body);
      const responseBody = {
        runs: body.runIds.map((runId) => ({
          runId,
          ...logTagsPayload(tagsByRun[runId]),
        })),
      };
      if (options.logTagsResponseFactory) {
        return Promise.resolve(
          options.logTagsResponseFactory(body, logTagRequests.length - 1),
        ).then((payload) => jsonResponse(payload ?? responseBody));
      }
      return jsonResponse(responseBody);
    }
    if (url.endsWith("/logs/scalars")) {
      const body = JSON.parse(String(init?.body)) as {
        runIds: string[];
        tags: string[];
        maxPoints?: number;
        sampling?: string;
      };
      logScalarRequests.push(body);
      const responseBody = {
        series: scalarSeries.filter(
          (series) => body.runIds.includes(series.runId) && body.tags.includes(series.tag),
        ),
      };
      if (options.logScalarResponseFactory) {
        return Promise.resolve(
          options.logScalarResponseFactory(body, logScalarRequests.length - 1),
        ).then((payload) => jsonResponse(payload ?? responseBody));
      }
      return jsonResponse(responseBody);
    }
    if (url.endsWith("/logs/media")) {
      const body = JSON.parse(String(init?.body)) as {
        runIds: string[];
        imageTags: string[];
        textTags: string[];
      };
      logMediaRequests.push(body);
      const responseBody = options.logMediaResponse ?? { images: [], texts: [] };
      if (options.logMediaResponseFactory) {
        return Promise.resolve(
          options.logMediaResponseFactory(body, logMediaRequests.length - 1),
        ).then((payload) => jsonResponse(payload ?? responseBody));
      }
      return jsonResponse(responseBody);
    }
    if (url.endsWith("/logs/parameter-status")) {
      const body = JSON.parse(String(init?.body)) as { runIds: string[] };
      const responseBody = options.logParameterStatusResponse?.({
        runIds: body.runIds,
      }) ?? {
        runs: body.runIds.map((runId) => {
          const run = logResponse.runs.find((candidate) => candidate.id === runId);
          return {
            sourceId: runId,
            preset: run?.preset ?? null,
            dataset: run?.dataset ?? null,
            logDir: run?.relativePath ?? null,
            nodes: [],
          };
        }),
      };
      return Promise.resolve(responseBody).then((payload) => jsonResponse(payload));
    }
    return jsonResponse({ detail: `Unhandled ${url}` }, 404);
  };

  const replayedMutations = new Map<
    string,
    { fingerprint: string; response: Response }
  >();

  function requestBodyFingerprint(body: BodyInit | null | undefined) {
    if (body instanceof FormData) {
      return JSON.stringify(
        Array.from(body.entries(), ([name, value]) => [
          name,
          value instanceof File
            ? { name: value.name, size: value.size, type: value.type }
            : value,
        ]),
      );
    }
    return body === undefined || body === null ? "" : String(body);
  }

  function declaredMutation(method: string, pathname: string) {
    return (
      (method === "POST" && pathname === "/training/jobs") ||
      (method === "POST" &&
        /^\/training\/jobs\/[^/]+\/(cancel|reconcile)$/.test(pathname)) ||
      (method === "POST" && pathname === "/config-snapshots") ||
      ((method === "PATCH" || method === "DELETE") &&
        pathname.startsWith("/config-snapshots/")) ||
      (method === "POST" && pathname === "/logs/import") ||
      (method === "DELETE" && pathname.startsWith("/logs/experiments/")) ||
      (method === "POST" &&
        (pathname === "/logs/runs/delete" ||
          pathname === "/logs/runs/preset-delete"))
    );
  }

  function fixtureApiPath(pathname: string) {
    const starts = ["/training/", "/config-snapshots", "/logs/"]
      .map((prefix) => pathname.indexOf(prefix))
      .filter((index) => index >= 0);
    return starts.length > 0 ? pathname.slice(Math.min(...starts)) : pathname;
  }

  function oversizedRequest(pathname: string, init?: RequestInit) {
    if (String(init?.method ?? "GET").toUpperCase() !== "POST") return null;
    if (!init?.body || init.body instanceof FormData) return null;
    let body: Record<string, unknown>;
    try {
      body = JSON.parse(String(init.body)) as Record<string, unknown>;
    } catch {
      return null;
    }
    const exceeds = (key: string, limit: number) =>
      Array.isArray(body[key]) && body[key].length > limit;
    const runLimited = new Set([
      "/logs/tags",
      "/logs/scalars",
      "/logs/media",
      "/logs/checkpoints",
      "/logs/parameter-status",
    ]);
    if (runLimited.has(pathname) && exceeds("runIds", 50)) {
      return "runIds exceeds the 50-item backend limit";
    }
    if (pathname === "/logs/media") {
      if (exceeds("imageTags", 20) || exceeds("textTags", 20)) {
        return "media tags exceed the 20-item backend limit";
      }
    }
    if (pathname === "/logs/scalars" && exceeds("tags", 50)) {
      return "scalar tags exceed the 50-item backend limit";
    }
    if (pathname === "/logs/runs/delete") {
      for (const key of ["experiments", "datasets", "models", "presets", "runIds"]) {
        if (exceeds(key, 50)) return `${key} exceeds the 50-item backend limit`;
      }
    }
    return null;
  }

  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    const parsedUrl = new URL(url, "http://testserver");
    const apiPath = fixtureApiPath(parsedUrl.pathname);
    const method = String(init?.method ?? "GET").toUpperCase();
    const isMutation = declaredMutation(method, apiPath);
    const headers = new Headers(init?.headers);
    const mutationHeader = headers.get("X-Workbench-Mutation");
    const idempotencyKey = headers.get("Idempotency-Key");
    if (isMutation && (mutationHeader !== "true" || !idempotencyKey)) {
      return jsonResponse(
        { detail: "Declared mutations require mutation and idempotency headers." },
        428,
      );
    }
    if (!isMutation && (mutationHeader !== null || idempotencyKey !== null)) {
      return jsonResponse(
        { detail: "Read-only requests must not carry mutation headers." },
        400,
      );
    }
    const oversized = oversizedRequest(apiPath, init);
    if (oversized) return jsonResponse({ detail: oversized }, 422);

    if (!isMutation) return dispatchFetch(input, init);

    const replayKey = `${method} ${apiPath} ${idempotencyKey}`;
    const fingerprint = `${url}\n${requestBodyFingerprint(init?.body)}`;
    const replay = replayedMutations.get(replayKey);
    if (replay) {
      if (replay.fingerprint !== fingerprint) {
        return jsonResponse(
          { detail: "Idempotency-Key was already used for a different request." },
          409,
        );
      }
      return Promise.resolve(replay.response.clone());
    }
    return dispatchFetch(input, init).then((response) => {
      replayedMutations.set(replayKey, {
        fingerprint,
        response: response.clone(),
      });
      return response;
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  return {
    fetchMock,
    inspectBodies,
    trainingBodies,
    configSnapshotCreateRequests,
    configSnapshotUpdateRequests,
    logTagRequests,
    logScalarRequests,
    logMediaRequests,
    logCheckpointRequests,
    logArtifactRequests,
    logRunRequests,
    deleteExperimentRequests,
    deletePresetPlanRequests,
    deletePresetRequests,
    deleteRunPlanRequests,
    deleteRunRequests,
    logImportRequests,
    monitorDataRequests,
    logRunMonitorDataRequests,
  };
}

type ScenarioObservations = ReturnType<typeof installFetchMock>;

type ConfigScenarioOptions = Pick<
  FetchScenarioOptions,
  | "configSnapshotLibraryResponse"
  | "configSnapshotsResponse"
  | "schemaResponse"
>;

type GraphScenarioOptions = Pick<
  FetchScenarioOptions,
  "inspectResponse" | "inspectResponseFactory" | "modelsResponse"
>;

type LogsScenarioOptions = Pick<
  FetchScenarioOptions,
  | "capabilitiesResponse"
  | "deleteLogExperimentError"
  | "deleteLogRunPlanErrorFactory"
  | "deleteLogRunsBlockers"
  | "deleteLogRunsErrorFactory"
  | "logCheckpointsByRun"
  | "logExperimentsResponse"
  | "logMediaResponse"
  | "logRunArtifactsByRun"
  | "logRunsResponse"
  | "logScalarResponseFactory"
  | "logScalarSeries"
  | "logTagsByRun"
  | "logTagsResponseFactory"
>;

type MonitorScenarioOptions = Pick<
  FetchScenarioOptions,
  | "inspectError"
  | "inspectResponse"
  | "logParameterStatusResponse"
  | "logRunMonitorDataResponse"
  | "logRunsResponse"
  | "logTagsByRun"
  | "monitorDataResponse"
>;

type OverviewScenarioOptions = Pick<
  FetchScenarioOptions,
  | "capabilitiesResponse"
  | "configSnapshotsResponse"
  | "datasetsResponse"
  | "inspectResponseFactory"
  | "logImportError"
  | "logImportResponse"
  | "logImportResponseFactory"
  | "logRunsResponse"
  | "logTagsByRun"
  | "schemaResponse"
>;

type TrainingScenarioOptions = Pick<
  FetchScenarioOptions,
  | "cancelTrainingJobError"
  | "capabilitiesResponse"
  | "configSnapshotsResponse"
  | "createTrainingJobResponseFactory"
  | "datasetsResponse"
  | "inspectResponseFactory"
  | "logRunsResponse"
  | "presetsResponse"
  | "schemaResponse"
  | "searchSpaceResponse"
  | "searchSpaceResponseFactory"
  | "trainingJobResponseFactory"
  | "trainingJobStatus"
  | "trainingRunPlanResponseFactory"
>;

function setupConfigScenario(
  options: ConfigScenarioOptions = {},
): Pick<
  ScenarioObservations,
  | "configSnapshotCreateRequests"
  | "configSnapshotUpdateRequests"
  | "inspectBodies"
> {
  return installFetchMock(options);
}

function setupGraphScenario(
  options: GraphScenarioOptions = {},
): Pick<ScenarioObservations, "inspectBodies"> {
  return installFetchMock(options);
}

function setupLogsScenario(
  options: LogsScenarioOptions = {},
): Pick<
  ScenarioObservations,
  | "deleteExperimentRequests"
  | "deletePresetPlanRequests"
  | "deletePresetRequests"
  | "deleteRunPlanRequests"
  | "deleteRunRequests"
  | "fetchMock"
  | "logArtifactRequests"
  | "logCheckpointRequests"
  | "logMediaRequests"
  | "logRunRequests"
  | "logScalarRequests"
  | "logTagRequests"
> {
  return installFetchMock(options);
}

function setupMonitorScenario(
  options: MonitorScenarioOptions = {},
): Pick<
  ScenarioObservations,
  | "fetchMock"
  | "logRunMonitorDataRequests"
  | "logRunRequests"
  | "monitorDataRequests"
> {
  return installFetchMock(options);
}

function setupOverviewScenario(
  options: OverviewScenarioOptions = {},
): Pick<
  ScenarioObservations,
  "fetchMock" | "inspectBodies" | "logImportRequests" | "trainingBodies"
> {
  return installFetchMock(options);
}

function setupTrainingScenario(
  options: TrainingScenarioOptions = {},
): Pick<ScenarioObservations, "fetchMock" | "inspectBodies" | "trainingBodies"> {
  return installFetchMock(options);
}

function renderWorkbench() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  const rendered = render(
    <QueryClientProvider client={queryClient}>
      <WorkbenchApp />
    </QueryClientProvider>,
  );
  return { ...rendered, queryClient };
}

async function waitForOpenFullConfigButton(
  user: ReturnType<typeof userEvent.setup> = userEvent.setup(),
) {
  if (!screen.queryByRole("button", { name: /open full config/i })) {
    const modelWorkspaceButton = screen.queryByRole("button", {
      name: /^model\b/i,
    });
    if (modelWorkspaceButton) {
      await user.click(modelWorkspaceButton);
    }
  }
  await waitFor(() =>
    expect(screen.getByRole("button", { name: /open full config/i })).toBeEnabled(),
  );
  return screen.getByRole("button", { name: /open full config/i });
}

type TargetDropdownLabel =
  | "model type"
  | "model"
  | "preset"
  | "snapshot"
  | "experiment"
  | "dataset";

async function findTargetCombobox(label: TargetDropdownLabel) {
  return screen.findByRole("combobox", {
    name: new RegExp(`^${label}$`, "i"),
  });
}

async function waitForTargetValue(label: TargetDropdownLabel, value: string) {
  const control = await findTargetCombobox(label);
  await waitFor(() => expect(control).toHaveTextContent(value));
  return control;
}

function targetListboxName(label: TargetDropdownLabel) {
  return new RegExp(`^${label} options$`, "i");
}

async function openTargetDropdown(
  user: ReturnType<typeof userEvent.setup>,
  label: TargetDropdownLabel,
) {
  const control = await findTargetCombobox(label);
  await user.click(control);
  const listbox = await screen.findByRole("listbox", {
    name: targetListboxName(label),
  });
  return { control, listbox };
}

async function selectSearchableDropdownOption(
  user: ReturnType<typeof userEvent.setup>,
  control: HTMLElement,
  optionName: string | RegExp,
  searchText?: string,
) {
  const root = control.parentElement;

  if (!(root instanceof HTMLElement)) {
    throw new Error("Expected searchable dropdown control to have a root element");
  }

  await user.click(control);
  let listbox = await waitFor(() => {
    const openListbox = within(root).queryByRole("listbox");
    if (!openListbox) {
      throw new Error("Dropdown is not open yet");
    }
    return openListbox;
  }, { timeout: 150 }).catch(() => null);

  if (!listbox) {
    control.focus();
    await user.keyboard("{Enter}");
    listbox = await within(root).findByRole("listbox");
  }

  if (searchText) {
    const search = await within(root).findByRole("searchbox");
    await user.type(search, searchText);
  }

  await user.click(within(listbox).getByRole("option", { name: optionName }));
  await waitFor(() => {
    expect(within(root).queryByRole("listbox")).not.toBeInTheDocument();
  });
}

async function selectTargetOption(
  user: ReturnType<typeof userEvent.setup>,
  label: TargetDropdownLabel,
  optionName: string,
) {
  const { control } = await openTargetDropdown(user, label);
  const root = control.parentElement;

  if (!(root instanceof HTMLElement)) {
    throw new Error("Expected target dropdown control to have a root element");
  }

  const search = within(root).getByRole("searchbox");
  await user.clear(search);
  await user.type(search, optionName);
  const listbox = within(root).getByRole("listbox", {
    name: targetListboxName(label),
  });
  await user.click(within(listbox).getByRole("option", { name: optionName }));
  await waitFor(() => {
    expect(screen.queryByRole("listbox", { name: targetListboxName(label) }))
      .not.toBeInTheDocument();
  });
  return control;
}

async function openFullConfig(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await waitForOpenFullConfigButton(user));
  return screen.findByRole("dialog", { name: /full configuration/i });
}

async function typeConfigFieldValue(
  user: ReturnType<typeof userEvent.setup>,
  dialog: HTMLElement,
  label: RegExp,
  value: string,
) {
  const input = within(dialog).getByLabelText(label);
  await user.clear(input);
  await user.type(input, value);
  return input;
}

function fullConfigSearchPopup(dialog: HTMLElement) {
  return within(dialog).getByRole("dialog", {
    name: /matching config fields/i,
  });
}

function fullConfigSearchResultRow(popup: HTMLElement, name: RegExp) {
  const row = within(popup)
    .getAllByRole("group", { name: /config search result/i })
    .find((candidate) => {
      const title = candidate.querySelector<HTMLElement>(
        "button[data-config-field-label]",
      );
      name.lastIndex = 0;
      return title ? name.test(title.textContent ?? "") : false;
    });

  if (!(row instanceof HTMLElement)) {
    throw new Error(`Expected a full config search result matching ${name}`);
  }

  return row;
}

function configFieldRowFor(control: HTMLElement) {
  const controlGrid = control.closest(".grid");
  const row = controlGrid?.parentElement;

  if (!(row instanceof HTMLElement)) {
    throw new Error("Expected config field control to render inside a field row");
  }

  return row;
}

function configFieldGridFor(control: HTMLElement) {
  const row = configFieldRowFor(control);
  const grid = row.parentElement;

  if (!(grid instanceof HTMLElement)) {
    throw new Error("Expected config field row to render inside a field grid");
  }

  return grid;
}

function expectResponsiveConfigFieldGrid(grid: HTMLElement) {
  expect(Array.from(grid.classList)).toEqual(
    expect.arrayContaining([
      "grid",
      "gap-x-3",
      "gap-y-3",
      "md:grid-cols-2",
      "2xl:grid-cols-3",
    ]),
  );
  expect(grid.classList).not.toContain("grid-cols-1");
}

function fullConfigSectionFor(accordion: HTMLElement) {
  const section = accordion.closest("section");

  if (!(section instanceof HTMLElement)) {
    throw new Error("Expected full config accordion to render inside a section");
  }

  return section;
}

function fullConfigSectionNavRowFor(sectionNav: HTMLElement, name: RegExp) {
  const jump = within(sectionNav).getByRole("button", { name });
  const row = jump.parentElement?.parentElement;

  if (!(row instanceof HTMLElement)) {
    throw new Error("Expected full config section jump to render inside a row");
  }

  return row;
}

async function openTrainingWorkspace(user: ReturnType<typeof userEvent.setup>) {
  const existingWorkspace = document.getElementById("training-workspace");
  let workspaceNav = screen.queryByRole("navigation", { name: "Workspace" });
  if (!workspaceNav && existingWorkspace instanceof HTMLElement) {
    return existingWorkspace;
  }
  workspaceNav ??= await screen.findByRole("navigation", {
    name: "Workspace",
  });
  const trainingWorkspaceButton = await within(workspaceNav).findByRole("button", {
    name: /^training\b/i,
  });
  if (trainingWorkspaceButton.getAttribute("aria-current") !== "page") {
    await user.click(trainingWorkspaceButton);
  }
  if (existingWorkspace instanceof HTMLElement) {
    await waitFor(() => expect(existingWorkspace).toBeVisible());
    return existingWorkspace;
  }
  const workspace = await screen.findByRole("region", {
    name: "Training workspace",
  });
  if (!(workspace instanceof HTMLElement)) {
    throw new Error("Expected Training workspace to render");
  }
  return workspace;
}

async function expandTrainingPanel(user: ReturnType<typeof userEvent.setup>) {
  await openTrainingWorkspace(user);
}

async function expandedTrainingDetails(user: ReturnType<typeof userEvent.setup>) {
  const details = await openTrainingWorkspace(user);
  if (!(details instanceof HTMLElement)) {
    throw new Error("Expected Training workspace details to render");
  }
  return details;
}

async function expandedTrainingDetailsReady(user: ReturnType<typeof userEvent.setup>) {
  const details = await expandedTrainingDetails(user);
  await waitFor(() => {
    expect(
      within(details).getByRole("combobox", { name: /^training model$/i }),
    ).toBeEnabled();
    expect(within(details).getByText("Grid Search")).toBeInTheDocument();
  });
  return details;
}

async function setTargetHiddenDimOverride(
  user: ReturnType<typeof userEvent.setup>,
  value: string,
) {
  const workspaceNav = screen.queryByRole("navigation", { name: "Workspace" });
  const trainingWorkspaceButton = workspaceNav
    ? within(workspaceNav).queryByRole("button", { name: /^training\b/i })
    : null;
  const restoreTrainingWorkspace =
    trainingWorkspaceButton?.getAttribute("aria-current") === "page";
  const dialog = await openFullConfig(user);
  await typeConfigFieldValue(user, dialog, /hidden dim/i, value);
  await user.click(within(dialog).getByRole("button", { name: /^close$/i }));
  if (restoreTrainingWorkspace) {
    return expandedTrainingDetailsReady(user);
  }
  return undefined;
}

async function selectTrainingTargetOption(
  user: ReturnType<typeof userEvent.setup>,
  label: "model type" | "model" | "preset",
  optionName: string,
) {
  const details = await expandedTrainingDetails(user);
  const control = within(details).getByRole("combobox", {
    name: new RegExp(`^training ${label}$`, "i"),
  });
  await selectSearchableDropdownOption(user, control, optionName, optionName);
  return control;
}

type TrainingMultiSelectLabel =
  | "Presets"
  | "Config snapshots"
  | "Training datasets"
  | "Training monitors";

function trainingMultiSelectName(label: TrainingMultiSelectLabel) {
  return new RegExp(`^${label}\\b`, "i");
}

function trainingMultiSelectOptionsName(label: TrainingMultiSelectLabel) {
  return new RegExp(`^${label} options$`, "i");
}

async function openTrainingMultiSelect(
  user: ReturnType<typeof userEvent.setup>,
  details: HTMLElement,
  label: TrainingMultiSelectLabel,
) {
  const control = within(details).getByRole("combobox", {
    name: trainingMultiSelectName(label),
  });
  let listbox = within(details).queryByRole("listbox", {
    name: trainingMultiSelectOptionsName(label),
  });
  if (!listbox) {
    await user.click(control);
    listbox = await within(details).findByRole("listbox", {
      name: trainingMultiSelectOptionsName(label),
    });
  }
  return { control, listbox };
}

async function setTrainingMultiSelectOption(
  user: ReturnType<typeof userEvent.setup>,
  details: HTMLElement,
  label: TrainingMultiSelectLabel,
  optionName: RegExp,
  selected = true,
) {
  const { listbox } = await openTrainingMultiSelect(user, details, label);
  const option = within(listbox).getByRole("option", { name: optionName });
  if ((option.getAttribute("aria-selected") === "true") !== selected) {
    await user.click(option);
    await waitFor(() => {
      expect(option).toHaveAttribute("aria-selected", String(selected));
    });
  }
  return option;
}

async function selectTrainingMonitorOption(
  user: ReturnType<typeof userEvent.setup>,
  optionName: RegExp,
) {
  const details = await expandedTrainingDetails(user);
  await setTrainingMultiSelectOption(
    user,
    details,
    "Training monitors",
    optionName,
  );
  return details;
}

async function selectNewTrainingLogFolder(
  user: ReturnType<typeof userEvent.setup>,
  name = "my_experiment",
) {
  await expandTrainingPanel(user);
  await user.click(screen.getByRole("radio", { name: /new folder/i }));
  const input = screen.getByLabelText(/^new log folder$/i);
  await user.clear(input);
  await user.type(input, name);
}

async function selectExistingTrainingLogFolder(
  user: ReturnType<typeof userEvent.setup>,
  name = "test_model",
) {
  await expandTrainingPanel(user);
  const control = screen.getByRole("combobox", {
    name: /^log experiment folder$/i,
  });
  await user.click(control);
  const listbox = await screen.findByRole("listbox", {
    name: /^log experiment folder options$/i,
  });
  const root = control.parentElement;

  if (!(root instanceof HTMLElement)) {
    throw new Error("Expected log folder dropdown control to have a root element");
  }

  const search = within(root).getByRole("searchbox");
  await user.clear(search);
  await user.type(search, name);
  const escapedName = name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  await user.click(
    within(listbox).getByRole("option", {
      name: new RegExp(`^${escapedName}\\s*\\(`),
    }),
  );
  await waitFor(() => {
    expect(
      screen.queryByRole("listbox", {
        name: /^log experiment folder options$/i,
      }),
    ).not.toBeInTheDocument();
  });
}

function fullConfigSectionGridFor(accordion: HTMLElement) {
  const section = fullConfigSectionFor(accordion);
  if (!section.parentElement) {
    throw new Error("Expected full config accordion to render inside the section grid");
  }
  return section.parentElement;
}

function expectFullConfigSectionGrid(grid: HTMLElement) {
  expect(Array.from(grid.classList)).toEqual(
    expect.arrayContaining([
      "grid",
      "auto-rows-max",
      "items-start",
      "gap-3",
    ]),
  );
  expect(grid.className).not.toMatch(/auto-(fit|fill)/);
  expect(grid.classList).not.toContain("grid-cols-1");
  expect(grid.classList).not.toContain("md:grid-cols-2");
  expect(grid.classList).not.toContain("2xl:grid-cols-3");
}

async function openTrainingCommand(
  user: ReturnType<typeof userEvent.setup>,
  dialog: HTMLElement,
) {
  await user.click(within(dialog).getByRole("button", { name: /training command/i }));
  return screen.findByRole("dialog", { name: /training command/i });
}

function commandField(dialog: HTMLElement) {
  return within(dialog).getByRole("textbox", { name: /^training command$/i });
}

function expectLogsChecklistRowSizing(control: HTMLElement) {
  const optionList = control.closest('[role="listbox"]');

  if (!(optionList instanceof HTMLElement)) {
    throw new Error("Expected the logs filter option to render in a grid row");
  }

  expect(Array.from(optionList.classList)).toEqual(
    expect.arrayContaining([
      "min-h-0",
      "overflow-y-auto",
    ]),
  );
  expect(control).toHaveClass(
    "grid",
    "min-w-0",
    "grid-cols-[auto_minmax(0,1fr)_auto]",
  );
}

function scalarChartGridFor(chart: HTMLElement) {
  const chartSection = chart.closest("section");
  const grid = chartSection?.parentElement;

  if (!(grid instanceof HTMLElement)) {
    throw new Error("Expected scalar chart to render inside the scalar chart grid");
  }

  return grid;
}

function logMetricGroupToggle(label: string) {
  return screen.getByRole("button", {
    name: new RegExp(`^${label}\\s+\\d+\\s+metrics?$`, "i"),
  });
}

function logValidationExamplesToggle() {
  return screen.getByRole("button", { name: /^Validation Examples\b/i });
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolver) => {
    resolve = resolver;
  });
  return { promise, resolve };
}

let scrollIntoViewMock: ReturnType<typeof vi.fn>;


function resetWorkbenchAppTestState() {
  vi.restoreAllMocks();
  try {
    window.localStorage?.clear?.();
    window.sessionStorage?.clear?.();
    loadWorkbenchConnectionRuntime();
  } catch {
    // Storage cleanup is best-effort in test environments.
  }
  scrollIntoViewMock = vi.fn();
  Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
    configurable: true,
    writable: true,
    value: scrollIntoViewMock,
  });
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: undefined,
  });
}

const appDriver = {
  render: renderWorkbench,
  reset: resetWorkbenchAppTestState,
};

const targetDriver = {
  selectOption: selectTargetOption,
  waitForValue: waitForTargetValue,
};

export const configHarness = {
  setup: setupConfigScenario,
  app: appDriver,
  fixtures: {
    schema: schemaResponse,
    schemaWithDescriptions: schemaResponseWithDescriptions,
  },
  config: {
    commandField,
    fieldGridFor: configFieldGridFor,
    fieldRowFor: configFieldRowFor,
    fullSectionGridFor: fullConfigSectionGridFor,
    open: openFullConfig,
    openTrainingCommand,
    responsiveFieldGrid: expectResponsiveConfigFieldGrid,
    searchPopup: fullConfigSearchPopup,
    searchResultRow: fullConfigSearchResultRow,
    sectionFor: fullConfigSectionFor,
    sectionGrid: expectFullConfigSectionGrid,
    sectionNavRowFor: fullConfigSectionNavRowFor,
    typeFieldValue: typeConfigFieldValue,
    waitForOpenButton: waitForOpenFullConfigButton,
  },
  dropdown: { selectOption: selectSearchableDropdownOption },
  target: targetDriver,
  get scrollIntoViewMock() {
    return scrollIntoViewMock;
  },
};

export const graphHarness = {
  setup: setupGraphScenario,
  app: appDriver,
  fixtures: {
    defaultInspection: inspectResponse,
    locations: locationInspectResponse,
    manyRepeatedLayers: manyRepeatedLayersInspectResponse,
    mechanismChildren: mechanismChildrenInspectResponse,
    mechanismMetadata: mechanismMetadataInspectResponse,
    neuronModels: neuronModelsResponse,
    parameterShapes: parameterShapeInspectResponse,
    repeatedLayers: repeatedLayersInspectResponse,
    stackContainer: stackContainerInspectResponse,
    tallSummary: tallSummaryInspectResponse,
  },
  config: {
    open: openFullConfig,
    typeFieldValue: typeConfigFieldValue,
  },
  target: { selectOption: selectTargetOption },
  tools: { deferred },
};

export const logsHarness = {
  setup: setupLogsScenario,
  app: appDriver,
  fixtures: {
    buildKaggleLinear: buildKaggleLinearLogFixture,
    buildLarge: buildLargeLogFixture,
    buildSubsetDelete: buildSubsetDeleteFixture,
    capabilities: capabilitiesResponse,
    runs: logRunsResponse,
    scalarSeries: logScalarSeries,
    tagsByRun: logTagsByRun,
  },
  ui: {
    expectChecklistRowSizing: expectLogsChecklistRowSizing,
    metricGroupToggle: logMetricGroupToggle,
    scalarChartGridFor,
    validationExamplesToggle: logValidationExamplesToggle,
  },
  tools: { deferred },
};

export const monitorHarness = {
  setup: setupMonitorScenario,
  app: appDriver,
  fixtures: {
    buildHistorical: buildHistoricalMonitorFixture,
    defaultLogRunPayload: defaultLogRunMonitorPayload,
    longSelectedNodeId,
    longSelectedNodeInspection: longSelectedNodeInspectResponse,
    monitorScopeInspection: monitorScopeInspectResponse,
    parameterShapeInspection: parameterShapeInspectResponse,
    repeatedLayersInspection: repeatedLayersInspectResponse,
    semanticPayload: semanticMonitorPayload,
  },
  dropdown: { selectOption: selectSearchableDropdownOption },
  training: {
    selectMonitorOption: selectTrainingMonitorOption,
    selectNewLogFolder: selectNewTrainingLogFolder,
  },
  tools: { deferred },
};

export const overviewHarness = {
  setup: setupOverviewScenario,
  app: appDriver,
  fixtures: {
    capabilities: capabilitiesResponse,
    implementedFeatures: IMPLEMENTED_FEATURES,
    inspection: inspectResponse,
    logRuns: logRunsResponse,
    schema: schemaResponse,
  },
  config: {
    commandField,
    open: openFullConfig,
    typeFieldValue: typeConfigFieldValue,
  },
  network: { jsonResponse },
  target: targetDriver,
  training: {
    expandedDetails: expandedTrainingDetails,
    openMultiSelect: openTrainingMultiSelect,
    setMultiSelectOption: setTrainingMultiSelectOption,
  },
};

export const trainingHarness = {
  setup: setupTrainingScenario,
  app: appDriver,
  fixtures: {
    capabilities: capabilitiesResponse,
    inspection: inspectResponse,
    presets: presetsResponse,
    schema: schemaResponse,
    schemaWithDescriptions: schemaResponseWithDescriptions,
    searchSpace: searchSpaceResponse,
  },
  config: {
    commandField,
    open: openFullConfig,
    searchPopup: fullConfigSearchPopup,
    searchResultRow: fullConfigSearchResultRow,
    typeFieldValue: typeConfigFieldValue,
  },
  target: {
    selectOption: selectTargetOption,
    setHiddenDimOverride: setTargetHiddenDimOverride,
    waitForValue: waitForTargetValue,
  },
  training: {
    expandedDetails: expandedTrainingDetails,
    expandedDetailsReady: expandedTrainingDetailsReady,
    mockJobPayload: mockTrainingJobPayload,
    openMultiSelect: openTrainingMultiSelect,
    selectExistingLogFolder: selectExistingTrainingLogFolder,
    selectMonitorOption: selectTrainingMonitorOption,
    selectNewLogFolder: selectNewTrainingLogFolder,
    selectTargetOption: selectTrainingTargetOption,
    setMultiSelectOption: setTrainingMultiSelectOption,
  },
  tools: { deferred },
};
