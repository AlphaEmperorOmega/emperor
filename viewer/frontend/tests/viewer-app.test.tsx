import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ViewerApp } from "@/components/features/viewer/viewer-app";
import { IMPLEMENTED_FEATURES } from "@/lib/feature-catalog";

type MockNodeData = {
  nodeId: string;
  label: string;
  subtitle: string;
  path: string;
  parameterCount: number;
  details: Record<string, unknown>;
  config: {
    typeName: string;
    fields: Array<{ key: string; value: unknown }>;
  } | null;
  childCount: number;
  childSummaries: Array<{
    label: string;
    nestedLabel?: string;
    dims?: string;
    count?: number;
    kind: "child" | "mechanism" | "overflow";
    title?: string;
  }>;
  clusterDiagram?: {
    instantiated: number;
    capacityTotal: number;
    planes: Array<{
      z: number;
      cells: Array<{ filled: boolean; title: string }>;
    }>;
  };
  stackDiagram?: {
    cells: Array<{
      label: string;
      title: string;
      dims?: string;
      kind: "layer" | "overflow" | "total";
    }>;
    dims?: string;
    totalLayers: number;
    hasOverflow: boolean;
  };
  graphDetailMode: "simple" | "basic" | "full";
  height: number;
  isExpanded: boolean;
  canToggleExpansion: boolean;
  canOpenMonitor?: boolean;
  isDetailsExpanded: boolean;
  onActivateNode: () => void;
  onToggleExpansion: () => void;
  onOpenMonitor?: () => void;
  onToggleDetails: () => void;
};

function detailText(value: unknown) {
  if (value === null || value === undefined) {
    return "";
  }
  if (typeof value === "object") {
    return JSON.stringify(value);
  }
  return String(value);
}

function configDetailText(value: unknown) {
  if (value === null || value === undefined) {
    return "None";
  }
  return detailText(value);
}

function nodeDetailRows(
  details: Record<string, unknown>,
  config: MockNodeData["config"],
) {
  if (config) {
    return config.fields.map((field) => ({
      key: field.key,
      value: field.value,
      text: configDetailText(field.value),
    }));
  }

  const previewOnlyKeys = new Set([
    "weightShape",
    "biasShape",
    "dims",
    "inputDim",
    "inputShape",
    "hiddenDim",
    "outputDim",
    "outputShape",
    "shapeTransition",
    "cluster",
    "terminalReach",
  ]);
  return Object.entries(details)
    .filter(([key]) => !previewOnlyKeys.has(key))
    .map(([key, value]) => ({ key, value, text: detailText(value) }));
}

function formatExactCount(count: number) {
  return new Intl.NumberFormat("en-US").format(count);
}

function formatCompactCount(count: number) {
  const absoluteCount = Math.abs(count);
  if (absoluteCount < 1000) {
    return formatExactCount(count);
  }
  const units = [
    { suffix: "B", value: 1_000_000_000 },
    { suffix: "M", value: 1_000_000 },
    { suffix: "K", value: 1_000 },
  ];
  const unit = units.find((candidate) => absoluteCount >= candidate.value);
  if (!unit) {
    return formatExactCount(count);
  }
  const value = count / unit.value;
  const formatted = value >= 100 ? value.toFixed(0) : value.toFixed(1);
  return `${formatted.replace(/\.0$/, "")}${unit.suffix}`;
}

function simpleGraphParamText(parameterCount: number) {
  return parameterCount > 0 ? `${formatCompactCount(parameterCount)} params` : undefined;
}

function dimensionValueText(value: unknown) {
  if (typeof value === "string") {
    const text = value.trim();
    if (text.length === 0) {
      return undefined;
    }
    const numericValue = Number(text);
    return Number.isFinite(numericValue) && numericValue > 0 ? text : undefined;
  }
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    return String(value);
  }
  return undefined;
}

function dimRange(inputDim: string | undefined, outputDim: string | undefined) {
  return inputDim && outputDim ? `${inputDim} -> ${outputDim}` : undefined;
}

function dimsFromText(value: unknown) {
  if (typeof value !== "string") {
    return undefined;
  }

  const parts = value.trim().split(/\s*->\s*/);
  if (parts.length !== 2) {
    return undefined;
  }

  return dimRange(dimensionValueText(parts[0]), dimensionValueText(parts[1]));
}

function configFieldValue(config: MockNodeData["config"], key: string) {
  return config?.fields.find((field) => field.key === key)?.value;
}

function nodeDimsText(details: Record<string, unknown>, config: MockNodeData["config"]) {
  const dims = dimsFromText(details.dims);
  if (dims) {
    return dims;
  }

  return (
    dimRange(dimensionValueText(details.inputDim), dimensionValueText(details.outputDim)) ??
    dimRange(
      dimensionValueText(configFieldValue(config, "input_dim")),
      dimensionValueText(configFieldValue(config, "output_dim")),
    )
  );
}

function parameterShapeRows(details: Record<string, unknown>) {
  return [
    ["weightShape", "W"],
    ["biasShape", "b"],
  ].flatMap(([key, label]) => {
    const value = details[key];
    return typeof value === "string" && value.length > 0
      ? [{ key, label, shape: value }]
      : [];
  });
}

vi.mock("@xyflow/react", () => ({
  ReactFlow: ({
    nodes,
    edges,
    onNodeClick,
    nodesDraggable,
    nodesConnectable,
    elementsSelectable,
    nodesFocusable,
    nodeClickDistance,
    children,
  }: {
    nodes: Array<{
      id: string;
      position: { x: number; y: number };
      style?: { height?: number };
      data: MockNodeData;
    }>;
    edges: Array<{ id: string; source: string; target: string }>;
    onNodeClick?: (event: unknown, node: { id: string }) => void;
    nodesDraggable?: boolean;
    nodesConnectable?: boolean;
    elementsSelectable?: boolean;
    nodesFocusable?: boolean;
    nodeClickDistance?: number;
    children: React.ReactNode;
  }) => {
    expect(nodesDraggable).toBe(false);
    expect(nodesConnectable).toBe(false);
    expect(elementsSelectable).toBe(false);
    expect(nodesFocusable).toBe(false);
    expect(nodeClickDistance).toBe(4);

    return (
      <div data-testid="flow">
        {nodes.map((node) => (
          <div
            key={node.id}
            data-testid={`node-${node.id}`}
            data-x={node.position.x}
            data-y={node.position.y}
            data-height={node.style?.height ?? node.data.height}
          >
            {(() => {
              const isSimpleMode = node.data.graphDetailMode === "simple";
              const parameterShapes = parameterShapeRows(node.data.details);
              const detailRows = nodeDetailRows(node.data.details, node.data.config);
              const simpleParamText = isSimpleMode
                ? simpleGraphParamText(node.data.parameterCount)
                : undefined;
              const simpleDimsText = isSimpleMode
                ? nodeDimsText(node.data.details, node.data.config) ?? node.data.stackDiagram?.dims
                : undefined;
              const detailToggleLabel = node.data.config ? "Config options" : "Details";
              const cardLabel = node.data.canToggleExpansion
                ? `Select and ${node.data.isExpanded ? "collapse" : "expand"} ${node.data.path}`
                : `Select ${node.data.path}`;

              return (
                <>
                  {!isSimpleMode && parameterShapes.length > 0 && (
                    <div data-testid={`parameter-shapes-${node.id}`}>
                      {parameterShapes.map((entry) => (
                        <div
                          key={entry.key}
                          aria-label={`${entry.label} shape ${entry.shape}`}
                        >
                          <span>{entry.label}</span>
                          <span>{entry.shape}</span>
                        </div>
                      ))}
                    </div>
                  )}
                  <div
                    role="button"
                    tabIndex={0}
                    aria-label={cardLabel}
                    aria-expanded={node.data.canToggleExpansion ? node.data.isExpanded : undefined}
                    onClick={() => {
                      node.data.onActivateNode();
                      onNodeClick?.({}, node);
                    }}
                  >
                    <div>
                      <span>{node.data.label}</span>
                      {isSimpleMode && simpleParamText && (
                        <span title={`${formatExactCount(node.data.parameterCount)} parameters`}>
                          {simpleParamText}
                        </span>
                      )}
                      {isSimpleMode && simpleDimsText && (
                        <span title={`input/output: ${simpleDimsText}`}>{simpleDimsText}</span>
                      )}
                      {!isSimpleMode && node.data.childCount > 0 && (
                        <span>
                          {node.data.childCount}{" "}
                          {node.data.childCount === 1 ? "child" : "children"}
                        </span>
                      )}
                      {!isSimpleMode && node.data.parameterCount > 0 && (
                        <span title={`${formatExactCount(node.data.parameterCount)} parameters`}>
                          {formatCompactCount(node.data.parameterCount)}
                        </span>
                      )}
                    </div>
                    {!isSimpleMode && <span>{node.data.subtitle}</span>}
                    {!isSimpleMode && node.data.clusterDiagram ? (
                      <div data-testid={`cluster-diagram-${node.id}`}>
                        <span>Cluster map</span>
                        <span>
                          {node.data.clusterDiagram.instantiated} / {node.data.clusterDiagram.capacityTotal}
                        </span>
                        {node.data.clusterDiagram.planes.flatMap((plane) =>
                          plane.cells.map((cell, index) => (
                            <span
                              key={`${plane.z}-${cell.title}-${index}`}
                              aria-label={cell.title}
                            >
                              {cell.filled ? "filled" : "empty"}
                            </span>
                          )),
                        )}
                      </div>
                    ) : !isSimpleMode && node.data.stackDiagram ? (
                      <div data-testid={`stack-diagram-${node.id}`}>
                        {node.data.stackDiagram.cells.map((cell, index) => (
                          <div
                            key={`${cell.kind}-${cell.label}-${index}`}
                            aria-label={cell.title}
                            title={cell.title}
                          >
                            <span>{cell.label}</span>
                            {cell.dims && <span>{cell.dims}</span>}
                          </div>
                        ))}
                      </div>
                    ) : !isSimpleMode && (
                      <div data-testid={`child-summaries-${node.id}`}>
                        {node.data.childSummaries.map((summary, index) => {
                          const summaryLabel = summary.nestedLabel
                            ? `${summary.label} ${summary.nestedLabel}`
                            : summary.label;
                          const summaryAccessibleLabel = summary.dims
                            ? `${summaryLabel} ${summary.dims}`
                            : summaryLabel;
                          const summaryTitle = summary.title
                            ? summary.dims
                              ? `${summary.title} ${summary.dims}`
                              : summary.title
                            : summaryAccessibleLabel;

                          return (
                            <div
                              key={`${summary.kind}-${summary.label}-${index}`}
                              aria-label={summaryAccessibleLabel}
                              title={summaryTitle}
                            >
                              {summary.kind === "overflow" ? (
                                <span>{summary.label}</span>
                              ) : summary.nestedLabel ? (
                                <>
                                  <span>{summary.label}</span>
                                  <span aria-hidden>›</span>
                                  <span>{summary.nestedLabel}</span>
                                  {summary.dims && <span>{summary.dims}</span>}
                                </>
                              ) : (
                                <>
                                  <span>
                                    {summary.count ? `${summary.label} x${summary.count}` : summary.label}
                                  </span>
                                  {summary.dims && <span>{summary.dims}</span>}
                                </>
                              )}
                            </div>
                          );
                        })}
                      </div>
                    )}
                    {node.data.canToggleExpansion && (
                      <button
                        type="button"
                        aria-label={`${node.data.isExpanded ? "Collapse" : "Expand"} tree ${node.data.path}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          node.data.onToggleExpansion();
                        }}
                      >
                        toggle
                      </button>
                    )}
                    {!isSimpleMode && node.data.canOpenMonitor && node.data.onOpenMonitor && (
                      <button
                        type="button"
                        aria-label={`Open monitor charts for ${node.data.path}`}
                        onClick={(event) => {
                          event.stopPropagation();
                          node.data.onOpenMonitor?.();
                        }}
                      >
                        monitor
                      </button>
                    )}
                    {!isSimpleMode && detailRows.length > 0 && (
                      <button
                        type="button"
                        aria-label={`${detailToggleLabel} for ${node.data.path}`}
                        aria-expanded={node.data.isDetailsExpanded}
                        onClick={(event) => {
                          event.stopPropagation();
                          node.data.onToggleDetails();
                        }}
                      >
                        {detailToggleLabel}
                      </button>
                    )}
                    {!isSimpleMode && node.data.isDetailsExpanded && detailRows.length > 0 && (
                      <div>
                        {detailRows.map((entry) => (
                          <div key={entry.key}>
                            <span>{entry.key}</span>
                            <span>{entry.text}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </>
              );
            })()}
          </div>
        ))}
        {edges.map((edge) => (
          <div key={edge.id} data-testid={`edge-${edge.id}`}>
            {edge.source} to {edge.target}
          </div>
        ))}
        {children}
      </div>
    );
  },
  Background: () => null,
  Controls: () => null,
  Panel: ({
    position,
    className,
    style,
    children,
  }: {
    position?: string;
    className?: string;
    style?: React.CSSProperties;
    children: React.ReactNode;
  }) => (
    <div
      data-testid={`flow-panel-${position ?? "default"}`}
      data-position={position}
      className={className}
      style={style}
    >
      {children}
    </div>
  ),
  Handle: () => null,
  MarkerType: { ArrowClosed: "arrowclosed" },
  Position: { Left: "left", Right: "right" },
}));

const modelsResponse = { models: ["linear", "bert_linear"] };
const presetsResponse = {
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
  model: "bert_linear",
  presets: [{ name: "bert-baseline", label: "BERT_BASELINE", description: "Bert baseline" }],
};
const datasetsResponse = {
  model: "linear",
  datasets: [
    { name: "Mnist", label: "Mnist", inputDim: 784, outputDim: 10 },
    { name: "Cifar10", label: "Cifar 10", inputDim: 3072, outputDim: 10 },
  ],
};
const bertDatasetsResponse = {
  model: "bert_linear",
  datasets: [{ name: "ToyText", label: "Toy Text", inputDim: 128, outputDim: 2 }],
};
const monitorsResponse = {
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
const capabilitiesResponse = {
  authMode: "none",
  trainingEnabled: true,
  logDeletionEnabled: true,
  historicalLogsEnabled: true,
  liveMonitorDataEnabled: true,
  historicalMonitorDataEnabled: true,
  uploadsEnabled: false,
  maxUploadSize: null,
  dataSourcesEnabled: false,
  dataSources: [],
};
const logRunsResponse = {
  runs: [
    {
      id: "log-mnist",
      group: "test_model",
      experiment: "test_model",
      model: "linear",
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
      model: "linear",
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
    };

function logTagsPayload(tags: MockLogTags | undefined) {
  if (Array.isArray(tags)) {
    return {
      scalarTags: tags,
      histogramTags: [],
      imageTags: [],
    };
  }
  return {
    scalarTags: tags?.scalarTags ?? [],
    histogramTags: tags?.histogramTags ?? [],
    imageTags: tags?.imageTags ?? [],
  };
}

const logTagsByRun: Record<string, MockLogTags> = {
  "log-mnist": [
    "train/loss",
    "validation/accuracy",
    "test/accuracy",
    "main_model.0.model/weights/mean",
  ],
  "log-cifar": [
    "train/loss",
    "validation/accuracy",
    "test/accuracy",
    "main_model.0.model/weights/mean",
  ],
};
const logScalarSeries = [
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

function buildLargeLogFixture(count = 42) {
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
            "main_model.1.model/output/mean",
          ],
          histogramTags: [],
          imageTags: [],
        },
      ]),
    ) as Record<string, MockLogTags>,
  };
}

const schemaResponse = {
  model: "linear",
  fields: [
    {
      key: "hidden_dim",
      configKey: "HIDDEN_DIM",
      flag: "--hidden-dim",
      label: "hidden dim",
      section: "Layer Stack Options",
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
      section: "Gate Stack Options",
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
      type: "enum",
      default: "GELU",
      nullable: false,
      choices: ["RELU", "GELU"],
    },
  ],
};
const searchSpaceResponse = {
  model: "linear",
  preset: "baseline",
  axes: [
    {
      key: "hidden_dim",
      configKey: "HIDDEN_DIM",
      searchKey: "SEARCH_SPACE_HIDDEN_DIM",
      label: "hidden dim",
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
      label: "Sequential",
      typeName: "Sequential",
      path: "main_model",
      graphRole: "architecture",
      parameterCount: 65792,
      details: {},
    },
    {
      id: "main_model.0",
      label: "Layer",
      typeName: "Layer",
      path: "main_model.0",
      graphRole: "architecture",
      parameterCount: 33024,
      details: { inputDim: 256, outputDim: 256 },
    },
    {
      id: "main_model.1",
      label: "Layer",
      typeName: "Layer",
      path: "main_model.1",
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
      id: "main_model-main_model.0",
      source: "main_model",
      target: "main_model.0",
    },
    {
      id: "main_model-main_model.1",
      source: "main_model",
      target: "main_model.1",
    },
  ],
};

const manyRepeatedLayersInspectResponse = {
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
          instantiated: 1,
          coordinates: [[1, 1, 1]],
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
  model?: string;
  preset?: string;
  presets?: string[];
  datasets?: string[];
  overrides?: Record<string, unknown>;
  logFolder?: string;
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
  model: string;
  preset: string;
  dataset: string;
  logFolder: string;
  overrides: Record<string, unknown>;
}) {
  const parts = [
    "source",
    "experiment.sh",
    input.model,
    "--preset",
    input.preset,
    "--datasets",
    input.dataset,
  ];
  if (input.logFolder) {
    parts.push("--logdir", input.logFolder);
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

function mockTrainingRunPlan(request: MockTrainingPlanRequest) {
  const model = request.model ?? "linear";
  const preset = request.preset ?? "baseline";
  const presets = request.presets?.length ? request.presets : [preset];
  const datasets = request.datasets?.length ? request.datasets : ["Cifar10"];
  const overrides = request.overrides ?? {};
  const fixedChanges = Object.entries(overrides).map(([key, value]) => ({
    key,
    label: key.replaceAll("_", " "),
    value,
    source: "override",
  }));
  const combinations = mockTrainingSearchCombinations(request);
  const runs = presets.flatMap((runPreset) =>
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
            model,
            preset: runPreset,
            dataset,
            logFolder: request.logFolder ?? "",
            overrides: rowOverrides,
          }),
          totalEpochs: Number(overrides.num_epochs ?? 30),
          currentEpoch: 0,
          metrics: {},
          logDir: null,
          error: null,
        };
      }),
    ),
  ).map((run, index) => ({
    ...run,
    id: `run-${String(index + 1).padStart(4, "0")}`,
    index: index + 1,
  }));
  const totalEpochs = runs.reduce((total, run) => total + run.totalEpochs, 0);
  return {
    model,
    preset: presets[0],
    presets,
    datasets,
    overrides,
    search: request.search ?? null,
    logFolder: request.logFolder ?? "",
    isRandomSearch: request.search?.mode === "random",
    runs,
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

function completedMockTrainingRunPlan(request: MockTrainingPlanRequest) {
  const plan = mockTrainingRunPlan(request);
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
      completedRuns: runs.length,
      pendingRuns: 0,
      completedEpochs: plan.summary.totalEpochs,
      remainingEpochs: 0,
    },
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
    const graphNode = node as { parameterCount?: unknown; [key: string]: unknown };
    return {
      ...graphNode,
      config:
        graphNode.config && typeof graphNode.config === "object"
          ? graphNode.config
          : null,
      parameterCount:
        typeof graphNode.parameterCount === "number" ? graphNode.parameterCount : 0,
    };
  });
  const firstNode = nodes[0] as { parameterCount?: unknown } | undefined;

  return {
    ...payload,
    parameterCount:
      typeof payload.parameterCount === "number"
        ? payload.parameterCount
        : typeof firstNode?.parameterCount === "number"
          ? firstNode.parameterCount
          : 0,
    nodes,
  };
}

function installFetchMock(
  options: {
    inspectError?: boolean;
    inspectResponse?: unknown;
    inspectResponseFactory?: (requestIndex: number) => unknown | Promise<unknown>;
    logRunsResponse?: typeof logRunsResponse;
    logExperimentsResponse?: typeof logExperimentsResponse;
    logScalarSeries?: typeof logScalarSeries;
    logTagsByRun?: Record<string, MockLogTags>;
    deleteLogExperimentError?: string;
    deleteLogRunsError?: string;
    deleteLogRunsBlockers?: Array<{ id: string; logFolder: string; status: string }>;
    capabilitiesResponse?: typeof capabilitiesResponse;
    schemaResponse?: unknown;
    searchSpaceResponse?: typeof searchSpaceResponse;
    datasetsResponse?: typeof datasetsResponse;
    monitorDataResponse?: (context: MockMonitorRequestContext) => MockMonitorPayload;
    logRunMonitorDataResponse?: (context: MockMonitorRequestContext) => MockMonitorPayload;
    parameterStatusResponse?: (context: {
      jobId: string;
      preset: string | null;
      dataset: string | null;
      logDir: string | null;
    }) => unknown;
    logParameterStatusResponse?: (context: { runIds: string[] }) => unknown;
  } = {},
) {
  const inspectBodies: unknown[] = [];
  const trainingBodies: unknown[] = [];
  const logScalarRequests: Array<{ runIds: string[]; tags: string[] }> = [];
  const deleteExperimentRequests: string[] = [];
  const deleteRunPlanRequests: Array<{
    experiments: string[];
    datasets: string[];
    models: string[];
    presets: string[];
    runIds: string[];
  }> = [];
  const deleteRunRequests: Array<{
    experiments: string[];
    datasets: string[];
    models: string[];
    presets: string[];
    runIds: string[];
  }> = [];
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

  function uniqueSorted(values: string[]) {
    return Array.from(new Set(values)).sort((a, b) => a.localeCompare(b));
  }

  function matchingDeleteRuns(filters: {
    experiments: string[];
    datasets: string[];
    models: string[];
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
    const models = new Set(filters.models);
    const presets = new Set(filters.presets);
    const runIds = new Set(filters.runIds);
    return logResponse.runs.filter(
      (run) =>
        experiments.has(run.experiment) &&
        datasets.has(run.dataset) &&
        models.has(run.model) &&
        presets.has(run.preset) &&
        runIds.has(run.id),
    );
  }

  function deletePlanPayload(filters: {
    experiments: string[];
    datasets: string[];
    models: string[];
    presets: string[];
    runIds: string[];
  }) {
    const candidates = matchingDeleteRuns(filters);
    const blockers = options.deleteLogRunsBlockers ?? [];
    return {
      candidateCount: candidates.length,
      counts: {
        runs: candidates.length,
        experiments: uniqueSorted(candidates.map((run) => run.experiment)).length,
        datasets: uniqueSorted(candidates.map((run) => run.dataset)).length,
        models: uniqueSorted(candidates.map((run) => run.model)).length,
        presets: uniqueSorted(candidates.map((run) => run.preset)).length,
      },
      affected: {
        experiments: uniqueSorted(candidates.map((run) => run.experiment)),
        datasets: uniqueSorted(candidates.map((run) => run.dataset)),
        models: uniqueSorted(candidates.map((run) => run.model)),
        presets: uniqueSorted(candidates.map((run) => run.preset)),
        runIds: uniqueSorted(candidates.map((run) => run.id)),
      },
      candidates: candidates.map((run) => ({
        id: run.id,
        experiment: run.experiment,
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

  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
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
      return jsonResponse(modelsResponse);
    }
    if (endsWithAny(["/models/linear/presets", "/models/linears/linear/presets"])) {
      return jsonResponse(presetsResponse);
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
      return jsonResponse(options.searchSpaceResponse ?? searchSpaceResponse);
    }
    if (
      endsWithAny([
        "/models/bert_linear/presets",
        "/models/transformer_encoder/bert_linear/presets",
      ])
    ) {
      return jsonResponse(bertPresetsResponse);
    }
    if (
      endsWithAny([
        "/models/bert_linear/datasets",
        "/models/transformer_encoder/bert_linear/datasets",
      ])
    ) {
      return jsonResponse(bertDatasetsResponse);
    }
    if (
      endsWithAny([
        "/models/bert_linear/monitors",
        "/models/transformer_encoder/bert_linear/monitors",
      ])
    ) {
      return jsonResponse({ model: "bert_linear", monitors: [] });
    }
    if (
      includesAny([
        "/models/bert_linear/config-schema",
        "/models/transformer_encoder/bert_linear/config-schema",
      ])
    ) {
      const schemaPayload = options.schemaResponse ?? schemaResponse;
      return jsonResponse(
        typeof schemaPayload === "object" && schemaPayload !== null
          ? { ...schemaPayload, model: "bert_linear" }
          : schemaPayload,
      );
    }
    if (
      includesAny([
        "/models/bert_linear/search-space",
        "/models/transformer_encoder/bert_linear/search-space",
      ])
    ) {
      return jsonResponse({ model: "bert_linear", preset: "bert-baseline", axes: [] });
    }
    if (url.endsWith("/training/run-plan")) {
      const request = JSON.parse(String(init?.body)) as MockTrainingPlanRequest;
      return jsonResponse(mockTrainingRunPlan(request));
    }
    if (url.endsWith("/inspect")) {
      inspectBodies.push(JSON.parse(String(init?.body)));
      if (options.inspectError) {
        return jsonResponse({ detail: "invalid override value" }, 400);
      }
      const inspectRequestIndex = inspectBodies.length - 1;
      const responseBody = options.inspectResponseFactory
        ? options.inspectResponseFactory(inspectRequestIndex)
        : options.inspectResponse;
      return Promise.resolve(responseBody ?? inspectResponse).then((body) =>
        jsonResponse(withParameterCounts(body)),
      );
    }
    if (url.endsWith("/training/jobs")) {
      latestTrainingRequest = JSON.parse(String(init?.body));
      trainingBodies.push(latestTrainingRequest);
      const runPlan =
        (latestTrainingRequest?.runPlan as
          | ReturnType<typeof mockTrainingRunPlan>
          | undefined) ??
        mockTrainingRunPlan(latestTrainingRequest as MockTrainingPlanRequest);
      const logFolder = String(latestTrainingRequest?.logFolder ?? "test_model");
      if (!experimentResponse.experiments.some((entry) => entry.experiment === logFolder)) {
        experimentResponse = {
          experiments: [
            ...experimentResponse.experiments,
            { experiment: logFolder, runCount: 0, relativePath: logFolder },
          ],
        };
      }
      return jsonResponse({
        id: "job-1",
        status: "running",
        model: "linear",
        preset: "baseline",
        presets: latestTrainingRequest?.presets ?? ["baseline"],
        datasets: ["Mnist"],
        overrides: {},
        runPlan,
        monitors: latestTrainingRequest?.monitors ?? [],
        logFolder,
        createdAt: "2026-06-01T00:00:00Z",
        updatedAt: "2026-06-01T00:00:00Z",
        exitCode: null,
        pid: 123,
        currentPreset: "baseline",
        currentDataset: "Mnist",
        epoch: 0,
        step: 0,
        metrics: {},
        logDir: null,
        events: [],
        logTail: [],
        resultLinks: [],
      });
    }
    if (url.endsWith("/training/jobs/job-1")) {
      const logFolder = String(latestTrainingRequest?.logFolder ?? "test_model");
      const runPlan = completedMockTrainingRunPlan(
        (latestTrainingRequest ?? { logFolder }) as MockTrainingPlanRequest,
      );
      return jsonResponse({
        id: "job-1",
        status: "completed",
        model: "linear",
        preset: "baseline",
        presets: latestTrainingRequest?.presets ?? ["baseline"],
        datasets: ["Mnist"],
        overrides: {},
        runPlan,
        monitors: latestTrainingRequest?.monitors ?? [],
        logFolder,
        createdAt: "2026-06-01T00:00:00Z",
        updatedAt: "2026-06-01T00:00:01Z",
        exitCode: 0,
        pid: 123,
        currentPreset: "baseline",
        currentDataset: "Mnist",
        epoch: 1,
        step: 4,
        metrics: { validation_accuracy: 0.9 },
        logDir: `logs/${logFolder}`,
        events: [],
        logTail: ["done"],
        resultLinks: [{ preset: "baseline", dataset: "Mnist", logDir: `logs/${logFolder}` }],
      });
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
      return jsonResponse({
        jobId: "job-1",
        nodePath,
        preset,
        dataset,
        logDir,
        ...monitorPayload,
      });
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
    if (url.endsWith("/logs/runs/delete-plan")) {
      const body = JSON.parse(String(init?.body)) as {
        experiments: string[];
        datasets: string[];
        models: string[];
        presets: string[];
        runIds: string[];
      };
      deleteRunPlanRequests.push(body);
      return jsonResponse(deletePlanPayload(body));
    }
    if (url.endsWith("/logs/runs/delete")) {
      const body = JSON.parse(String(init?.body)) as {
        experiments: string[];
        datasets: string[];
        models: string[];
        presets: string[];
        runIds: string[];
      };
      deleteRunRequests.push(body);
      if (options.deleteLogRunsError) {
        return jsonResponse({ detail: options.deleteLogRunsError }, 400);
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
    if (url.endsWith("/logs/runs")) {
      return jsonResponse(logResponse);
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
      return jsonResponse({
        jobId: runId,
        nodePath,
        dataset: run?.dataset ?? null,
        logDir,
        ...monitorPayload,
      });
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
      return jsonResponse({
        runs: body.runIds.map((runId) => ({
          runId,
          ...logTagsPayload(tagsByRun[runId]),
        })),
      });
    }
    if (url.endsWith("/logs/scalars")) {
      const body = JSON.parse(String(init?.body)) as {
        runIds: string[];
        tags: string[];
      };
      logScalarRequests.push(body);
      return jsonResponse({
        series: scalarSeries.filter(
          (series) => body.runIds.includes(series.runId) && body.tags.includes(series.tag),
        ),
      });
    }
    if (url.endsWith("/logs/parameter-status")) {
      const body = JSON.parse(String(init?.body)) as { runIds: string[] };
      return jsonResponse(
        options.logParameterStatusResponse?.({ runIds: body.runIds }) ?? {
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
        },
      );
    }
    return jsonResponse({ detail: `Unhandled ${url}` }, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return {
    fetchMock,
    inspectBodies,
    trainingBodies,
    logScalarRequests,
    deleteExperimentRequests,
    deleteRunPlanRequests,
    deleteRunRequests,
    monitorDataRequests,
    logRunMonitorDataRequests,
  };
}

function renderViewer() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ViewerApp />
    </QueryClientProvider>,
  );
}

async function waitForOpenFullConfigButton() {
  await waitFor(() =>
    expect(screen.getByRole("button", { name: /open full config/i })).toBeEnabled(),
  );
  return screen.getByRole("button", { name: /open full config/i });
}

async function openDatasetSelector(user: ReturnType<typeof userEvent.setup>) {
  const trigger = await screen.findByRole("button", {
    name: /datasets\s+\d+\s*\/\s*\d+/i,
  });
  await user.click(trigger);
  return screen.findByRole("dialog", { name: /dataset selector/i });
}

async function findTargetCombobox(label: "model" | "preset") {
  return screen.findByRole("combobox", {
    name: new RegExp(`^${label}$`, "i"),
  });
}

async function waitForTargetValue(label: "model" | "preset", value: string) {
  const control = await findTargetCombobox(label);
  await waitFor(() => expect(control).toHaveTextContent(value));
  return control;
}

function targetListboxName(label: "model" | "preset") {
  return new RegExp(`^${label} options$`, "i");
}

async function openTargetDropdown(
  user: ReturnType<typeof userEvent.setup>,
  label: "model" | "preset",
) {
  const control = await findTargetCombobox(label);
  await user.click(control);
  const listbox = await screen.findByRole("listbox", {
    name: targetListboxName(label),
  });
  return { control, listbox };
}

async function selectTargetOption(
  user: ReturnType<typeof userEvent.setup>,
  label: "model" | "preset",
  optionName: string,
) {
  const { control, listbox } = await openTargetDropdown(user, label);
  await user.click(within(listbox).getByRole("option", { name: optionName }));
  await waitFor(() => {
    expect(screen.queryByRole("listbox", { name: targetListboxName(label) }))
      .not.toBeInTheDocument();
  });
  return control;
}

async function openFullConfig(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await waitForOpenFullConfigButton());
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
    .find((candidate) =>
      within(candidate).queryByRole("button", { name }),
    );

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

async function expandTrainingPanel(user: ReturnType<typeof userEvent.setup>) {
  if (!screen.queryByRole("tab", { name: /new folder/i })) {
    await user.click(await screen.findByRole("button", { name: /^training/i }));
  }
}

async function expandedTrainingDetails(user: ReturnType<typeof userEvent.setup>) {
  await expandTrainingPanel(user);
  const details = document.getElementById("training-panel-details");
  if (!(details instanceof HTMLElement)) {
    throw new Error("Expected expanded training panel details to render");
  }
  return details;
}

async function expandedTrainingDetailsWithConfig(user: ReturnType<typeof userEvent.setup>) {
  const details = await expandedTrainingDetails(user);
  await waitFor(() =>
    expect(trainingFullConfigButton(details)).toBeEnabled(),
  );
  return details;
}

function trainingFullConfigButton(details: HTMLElement) {
  return within(details).getByRole("button", { name: /open full config/i });
}

async function openTrainingFullConfig(
  user: ReturnType<typeof userEvent.setup>,
  details: HTMLElement,
) {
  await user.click(trainingFullConfigButton(details));
  return screen.findByRole("dialog", { name: /full configuration/i });
}

async function setTrainingHiddenDimOverride(
  user: ReturnType<typeof userEvent.setup>,
  details: HTMLElement,
  value: string,
) {
  const dialog = await openTrainingFullConfig(user, details);
  await typeConfigFieldValue(user, dialog, /hidden dim/i, value);
  await user.click(within(dialog).getByRole("button", { name: /^close$/i }));
}

async function selectTrainingTargetOption(
  user: ReturnType<typeof userEvent.setup>,
  label: "model" | "preset",
  optionName: string,
) {
  const details = await expandedTrainingDetails(user);
  const control = within(details).getByRole("combobox", {
    name: new RegExp(`^training ${label}$`, "i"),
  });
  await user.click(control);
  const listbox = await within(details).findByRole("listbox", {
    name: new RegExp(`^training ${label} options$`, "i"),
  });
  await user.click(within(listbox).getByRole("option", { name: optionName }));
  await waitFor(() => {
    expect(
      within(details).queryByRole("listbox", {
        name: new RegExp(`^training ${label} options$`, "i"),
      }),
    ).not.toBeInTheDocument();
  });
  return control;
}

type TrainingMultiSelectLabel =
  | "Presets"
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
  await user.click(screen.getByRole("tab", { name: /new folder/i }));
  const input = screen.getByLabelText(/^new log folder$/i);
  await user.clear(input);
  await user.type(input, name);
}

async function selectExistingTrainingLogFolder(
  user: ReturnType<typeof userEvent.setup>,
  name = "test_model",
) {
  await expandTrainingPanel(user);
  await user.selectOptions(screen.getByLabelText(/^log experiment folder$/i), name);
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
  const label = control.closest("label");
  const row = label?.parentElement;
  const optionList = row?.parentElement;

  if (
    !(label instanceof HTMLElement) ||
    !(row instanceof HTMLElement) ||
    !(optionList instanceof HTMLElement)
  ) {
    throw new Error("Expected the logs checklist option to render in a grid row");
  }

  expect(Array.from(optionList.classList)).toEqual(
    expect.arrayContaining([
      "grid",
      "max-h-64",
      "auto-rows-max",
      "content-start",
      "overflow-y-auto",
    ]),
  );
  expect(row).toHaveClass("min-h-[44px]");
  expect(label).toHaveClass("min-h-[44px]");
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

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolver) => {
    resolve = resolver;
  });
  return { promise, resolve };
}

let scrollIntoViewMock: ReturnType<typeof vi.fn>;

describe("ViewerApp", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
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
  });

  it("renders model and preset selectors from API data", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const model = await waitForTargetValue("model", "linear");
    const preset = await waitForTargetValue("preset", "baseline");

    expect(model).toHaveTextContent("linear");
    expect(preset).toHaveTextContent("baseline");
    expect(model).toHaveAttribute("aria-expanded", "false");
    expect(preset).toHaveAttribute("aria-expanded", "false");

    await user.click(model);

    const modelOptions = await screen.findByRole("listbox", {
      name: /model options/i,
    });
    expect(model).toHaveAttribute("aria-expanded", "true");
    expect(within(modelOptions).getByRole("option", { name: "linear" }))
      .toBeInTheDocument();
    expect(within(modelOptions).getByRole("option", { name: "bert_linear" }))
      .toBeInTheDocument();

    await user.keyboard("{Escape}");

    await waitFor(() => {
      expect(screen.queryByRole("listbox", { name: /model options/i }))
        .not.toBeInTheDocument();
    });
    expect(model).toHaveAttribute("aria-expanded", "false");
    expect(model).toHaveFocus();

    await user.click(preset);

    const presetOptions = await screen.findByRole("listbox", {
      name: /preset options/i,
    });
    expect(preset).toHaveAttribute("aria-expanded", "true");
    expect(within(presetOptions).getByRole("option", { name: "baseline" }))
      .toBeInTheDocument();
    expect(
      within(presetOptions).getByRole("option", {
        name: "recurrent-gating-halting",
      }),
    ).toBeInTheDocument();
  });

  it("shows config status pills while keeping graph count pills out of the header", async () => {
    installFetchMock();
    renderViewer();

    const header = document.querySelector("header");
    if (!header) {
      throw new Error("Expected app header to render");
    }

    expect(within(header).getByText(/^API$/)).toBeInTheDocument();
    expect(await within(header).findByText("online")).toBeInTheDocument();
    const overrideLabel = within(header).getByText(/^overrides$/);
    const presetLabel = within(header).getByText(/^presets$/);
    expect(overrideLabel.nextElementSibling).toHaveTextContent("0");
    expect(presetLabel.nextElementSibling).toHaveTextContent("0");
    expect(within(header).queryByText(/^nodes$/i)).not.toBeInTheDocument();
    expect(within(header).queryByText(/^edges$/i)).not.toBeInTheDocument();
  });

  it("shows preset-owned field count in the top header", async () => {
    installFetchMock({
      schemaResponse: {
        ...schemaResponse,
        fields: schemaResponse.fields.map((field) =>
          field.key === "gate_flag"
            ? {
                ...field,
                locked: true,
                lockedValue: true,
                lockedReason:
                  "Locked by the GATING preset because this preset enables stack gating.",
              }
            : field,
        ),
      },
    });
    renderViewer();

    const header = document.querySelector("header");
    if (!header) {
      throw new Error("Expected app header to render");
    }
    const presetLabel = within(header).getByText(/^presets$/);
    const presetPill = presetLabel.parentElement;
    if (!(presetPill instanceof HTMLElement)) {
      throw new Error("Expected presets status pill to render");
    }

    await waitFor(() => {
      expect(presetLabel.nextElementSibling).toHaveTextContent("1");
      expect(presetPill).toHaveClass("text-amber");
    });
  });

  it("supports keyboard selection and Escape on target dropdowns", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const preset = await waitForTargetValue("preset", "baseline");

    await user.click(preset);
    expect(await screen.findByRole("listbox", { name: /preset options/i }))
      .toBeInTheDocument();
    expect(preset).toHaveAttribute("aria-expanded", "true");

    await user.keyboard("{Escape}");

    await waitFor(() => {
      expect(screen.queryByRole("listbox", { name: /preset options/i }))
        .not.toBeInTheDocument();
    });
    expect(preset).toHaveAttribute("aria-expanded", "false");
    expect(preset).toHaveFocus();

    await user.keyboard("{ArrowDown}{Enter}");

    await waitFor(() => {
      expect(preset).toHaveTextContent("recurrent-gating-halting");
    });
    expect(preset).toHaveAttribute("aria-expanded", "false");
  });

  it("requests the initial preview for the first selected preset", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    expect(inspectBodies[0]).toEqual({
      model: "linear",
      preset: "baseline",
      dataset: "Mnist",
      overrides: {},
    });
  });

  it("shows the selected preset description in a compact popup", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await waitForTargetValue("preset", "baseline");
    const trigger = screen.getByRole("button", { name: /show preset description/i });
    expect(trigger).toHaveAttribute("aria-haspopup", "dialog");
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(trigger).toBeEnabled();
    expect(screen.queryByText("Baseline")).not.toBeInTheDocument();

    await user.click(trigger);

    const dialog = await screen.findByRole("dialog", { name: /preset description/i });
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getByRole("heading", { name: "baseline" })).toBeInTheDocument();
    expect(within(dialog).getByText("Baseline")).toBeInTheDocument();

    await user.keyboard("{Escape}");

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /preset description/i }))
        .not.toBeInTheDocument();
    });
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(trigger).toHaveFocus();

    await user.click(trigger);
    expect(await screen.findByRole("dialog", { name: /preset description/i }))
      .toBeInTheDocument();
    await selectTargetOption(user, "preset", "recurrent-gating-halting");

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /preset description/i }))
        .not.toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /show preset description/i }))
      .toHaveAttribute("aria-expanded", "false");
  });

  it("opens and closes the implemented features dialog without API side effects", async () => {
    const { inspectBodies, trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    const initialInspectRequestCount = inspectBodies.length;
    const featuresButton = screen.getByRole("button", { name: /^features$/i });

    await user.click(featuresButton);

    const dialog = await screen.findByRole("dialog", { name: /implemented features/i });
    expect(within(dialog).getByText(`${IMPLEMENTED_FEATURES.length} features`))
      .toBeInTheDocument();
    expect(within(dialog).getByText("Model inspection")).toBeInTheDocument();
    expect(
      within(dialog).getByText("Graph canvas, modes, scopes, and layout"),
    ).toBeInTheDocument();
    expect(
      within(dialog).getByText("Training job creation, polling, and cancellation"),
    ).toBeInTheDocument();
    expect(within(dialog).getByText("TensorBoard monitor data")).toBeInTheDocument();

    await user.click(
      within(dialog).getByRole("button", { name: /close implemented features/i }),
    );

    expect(screen.queryByRole("dialog", { name: /implemented features/i }))
      .not.toBeInTheDocument();
    expect(inspectBodies).toHaveLength(initialInspectRequestCount);
    expect(trainingBodies).toHaveLength(0);
  });

  it("uses the first selected dataset for preview requests", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    const datasetDialog = await openDatasetSelector(user);
    await user.click(within(datasetDialog).getByLabelText(/dataset Cifar10/i));
    await user.click(within(datasetDialog).getByLabelText(/dataset Mnist/i));
    await user.click(screen.getByRole("button", { name: /update preview/i }));

    await waitFor(() => {
      expect(inspectBodies.at(-1)).toMatchObject({ dataset: "Cifar10" });
    });
  });

  it("opens dataset selection from the compact sidebar trigger", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const trigger = await screen.findByRole("button", {
      name: /datasets\s+1\s*\/\s*2/i,
    });
    expect(trigger).toHaveAttribute("aria-haspopup", "dialog");
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByLabelText(/dataset Mnist/i)).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/dataset Cifar10/i)).not.toBeInTheDocument();

    await user.click(trigger);

    const dialog = await screen.findByRole("dialog", { name: /dataset selector/i });
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getByLabelText(/dataset Mnist/i)).not.toBeChecked();
    expect(within(dialog).getByLabelText(/dataset Cifar10/i)).toBeChecked();

    await user.click(within(dialog).getByLabelText(/dataset Mnist/i));

    expect(screen.getByRole("button", { name: /datasets\s+2\s*\/\s*2/i }))
      .toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /^first$/i }));

    expect(screen.getByRole("button", { name: /datasets\s+1\s*\/\s*2/i }))
      .toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /^all$/i }));

    expect(screen.getByRole("button", { name: /datasets\s+2\s*\/\s*2/i }))
      .toBeInTheDocument();

    await user.keyboard("{Escape}");

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /dataset selector/i }))
        .not.toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /datasets\s+2\s*\/\s*2/i }))
      .toHaveFocus();
  });

  it("renders the experiments preset filter and selects a run on click", async () => {
    const { inspectBodies } = installFetchMock({
      logRunsResponse: {
        runs: [
          logRunsResponse.runs[1],
          logRunsResponse.runs[0],
          {
            ...logRunsResponse.runs[0],
            id: "bert-run",
            group: "bert_experiment",
            experiment: "bert_experiment",
            model: "bert_linear",
            relativePath:
              "bert_experiment/bert_linear/BASELINE/Mnist/ccc_20260601_030405/version_0",
          },
        ],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByRole("heading", { name: "Experiments" }))
      .toBeInTheDocument();
    // The preset filter is independent of the model's build preset; it defaults to
    // "All presets" and lists the presets present in the model's runs.
    const presetFilter = (
      await screen.findByRole("option", { name: "BASELINE (2)" })
    ).closest("select") as HTMLSelectElement;
    expect(presetFilter).toHaveValue("");
    expect(
      within(presetFilter).getByRole("option", { name: "All presets" }),
    ).toBeInTheDocument();

    // Both layer-data runs are listed; nothing is auto-selected.
    const newestRun = await screen.findByRole("button", {
      name: /select experiment run test_model_2 BASELINE Cifar10 2026-06-01 02:03:04/i,
    });
    const olderRun = screen.getByRole("button", {
      name: /select experiment run test_model BASELINE Mnist 2026-06-01 01:02:03/i,
    });
    expect(newestRun).toHaveAttribute("aria-pressed", "false");
    expect(olderRun).toHaveAttribute("aria-pressed", "false");
    expect(screen.queryByText("bert_experiment")).not.toBeInTheDocument();

    // Selecting a run drives the preview; selecting it again clears the selection.
    await user.click(newestRun);
    await waitFor(() => expect(newestRun).toHaveAttribute("aria-pressed", "true"));
    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        model: "linear",
        preset: "baseline",
        dataset: "Cifar10",
        overrides: {},
      });
    });

    await user.click(newestRun);
    await waitFor(() => expect(newestRun).toHaveAttribute("aria-pressed", "false"));
  });

  it("filtering by preset narrows the visible run list", async () => {
    installFetchMock({
      logRunsResponse: {
        runs: [
          {
            ...logRunsResponse.runs[0],
            id: "fast-mnist",
            experiment: "exp_alpha",
            group: "exp_alpha",
            preset: "RECURRENT_GATING_HALTING",
            dataset: "Mnist",
            timestamp: "2026-06-01 01:00:00",
            runName: "fast_20260601_010000",
            relativePath:
              "exp_alpha/linear/RECURRENT_GATING_HALTING/Mnist/fast_20260601_010000/version_0",
          },
          {
            ...logRunsResponse.runs[1],
            id: "base-cifar",
            experiment: "exp_beta",
            group: "exp_beta",
            preset: "BASELINE",
            dataset: "Cifar10",
            timestamp: "2026-06-01 03:00:00",
            runName: "base_20260601_030000",
            relativePath:
              "exp_beta/linear/BASELINE/Cifar10/base_20260601_030000/version_0",
          },
        ],
      },
      logTagsByRun: {
        "fast-mnist": ["main_model.0.model/weights/mean"],
        "base-cifar": ["main_model.0.model/weights/mean"],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const presetFilter = (
      await screen.findByRole("option", { name: "All presets" })
    ).closest("select") as HTMLSelectElement;

    const fastRunName =
      /select experiment run exp_alpha RECURRENT_GATING_HALTING Mnist 2026-06-01 01:00:00/i;
    const baseRunName =
      /select experiment run exp_beta BASELINE Cifar10 2026-06-01 03:00:00/i;

    // All presets: both runs visible.
    expect(await screen.findByRole("button", { name: fastRunName })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: baseRunName })).toBeInTheDocument();

    // Narrow to BASELINE.
    await user.selectOptions(presetFilter, "BASELINE");
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: fastRunName })).not.toBeInTheDocument(),
    );
    expect(screen.getByRole("button", { name: baseRunName })).toBeInTheDocument();

    // Narrow to the other preset.
    await user.selectOptions(presetFilter, "RECURRENT_GATING_HALTING");
    await waitFor(() =>
      expect(screen.queryByRole("button", { name: baseRunName })).not.toBeInTheDocument(),
    );
    expect(screen.getByRole("button", { name: fastRunName })).toBeInTheDocument();

    // Back to all presets.
    await user.selectOptions(presetFilter, "");
    expect(await screen.findByRole("button", { name: baseRunName })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: fastRunName })).toBeInTheDocument();
  });

  it("selecting a historical run syncs preset and dataset, clears overrides, and refreshes preview", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));
    expect(screen.getByText(/1 overrides?/i)).toBeInTheDocument();

    const mnistRun = await screen.findByRole("button", {
      name: /select experiment run test_model BASELINE Mnist 2026-06-01 01:02:03/i,
    });
    await user.click(mnistRun);

    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        model: "linear",
        preset: "baseline",
        dataset: "Mnist",
        overrides: {},
      });
    });
    expect(mnistRun).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("0 overrides")).toBeInTheDocument();
  });

  it("switches to logs workspace and renders historical scalar runs", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    expect(await screen.findByText("Historical Scalars")).toBeInTheDocument();
    expect(screen.getByLabelText("Experiments test_model")).toBeChecked();
    expect(screen.getByLabelText("Experiments test_model_2")).toBeChecked();
    expect(await screen.findByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();
    expect(logMetricGroupToggle("Train")).toHaveAttribute("aria-expanded", "true");
    expect(logMetricGroupToggle("Validation")).toHaveAttribute("aria-expanded", "true");
    expect(logMetricGroupToggle("Test")).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /test\/accuracy scalar chart/i }))
      .not.toBeInTheDocument();
    const accuracyLeaderboard = await screen.findByRole("table", {
      name: /test\/accuracy test leaderboard/i,
    });
    const accuracyRows = within(accuracyLeaderboard).getAllByRole("row").slice(1);
    expect(accuracyRows).toHaveLength(2);
    expect(accuracyRows[0]).toHaveTextContent("0.9");
    expect(within(accuracyRows[0]).getByText("aaa_20260601_010203"))
      .toBeInTheDocument();
    expect(accuracyRows[1]).toHaveTextContent("0.62");
    expect(within(accuracyRows[1]).getByText("bbb_20260601_020304"))
      .toBeInTheDocument();
    expect(
      screen.getAllByText(/test_model · Mnist · linear · BASELINE · 2026-06-01 01:02:03/).length,
    ).toBeGreaterThan(0);
    const cifarLine = within(accuracyLeaderboard).getByRole("button", {
      name: /open run details for test_model_2 · Cifar10 · linear · BASELINE · 2026-06-01 02:03:04/i,
    });

    await user.click(cifarLine);

    const detailsPanel = screen.getByRole("heading", { name: "Run Details" }).closest("aside");
    expect(detailsPanel).not.toBeNull();
    expect(within(detailsPanel as HTMLElement).getByText("Experiment")).toBeInTheDocument();
    expect(within(detailsPanel as HTMLElement).getByText("test_model_2")).toBeInTheDocument();
    expect(screen.getAllByText("No result.json").length).toBeGreaterThan(0);
    expect(screen.queryByRole("button", { name: /start training/i })).not.toBeInTheDocument();
  });

  it("collapses logs metric groups without changing selected tags or refetching scalars", async () => {
    const { logScalarRequests } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    expect(await screen.findByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();
    const trainToggle = logMetricGroupToggle("Train");
    const testToggle = logMetricGroupToggle("Test");
    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(1);
    });

    await user.click(trainToggle);

    expect(trainToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("img", { name: /train\/loss scalar chart/i }))
      .not.toBeInTheDocument();
    expect(screen.getByLabelText("Scalar Tags train/loss")).toBeChecked();
    expect(logScalarRequests).toHaveLength(1);

    await user.click(trainToggle);

    expect(trainToggle).toHaveAttribute("aria-expanded", "true");
    expect(await screen.findByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();
    expect(logScalarRequests).toHaveLength(1);

    await user.click(testToggle);

    expect(testToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("table", { name: /test\/accuracy test leaderboard/i }))
      .not.toBeInTheDocument();
    expect(screen.getByLabelText("Scalar Tags test/accuracy")).toBeChecked();
    expect(logScalarRequests).toHaveLength(1);

    await user.click(testToggle);

    expect(testToggle).toHaveAttribute("aria-expanded", "true");
    expect(await screen.findByRole("table", { name: /test\/accuracy test leaderboard/i }))
      .toBeInTheDocument();
    expect(logScalarRequests).toHaveLength(1);
  });

  it("renders non-standard scalar tags in the Other metric group", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    expect(await screen.findByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();

    const otherTag = screen.getByLabelText("Scalar Tags main_model.0.model/weights/mean");
    expect(otherTag).not.toBeChecked();
    await user.click(otherTag);

    const otherToggle = await screen.findByRole("button", {
      name: /^Other\s+1\s+metric$/i,
    });
    expect(otherToggle).toHaveAttribute("aria-expanded", "true");
    expect(
      await screen.findByRole("img", {
        name: /main_model\.0\.model\/weights\/mean scalar chart/i,
      }),
    ).toBeInTheDocument();
  });

  it("keeps collapsed logs metric groups collapsed after switching workspaces", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    expect(await screen.findByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();

    await user.click(logMetricGroupToggle("Train"));
    expect(screen.queryByRole("img", { name: /train\/loss scalar chart/i }))
      .not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^model$/i }));
    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    const trainToggle = await screen.findByRole("button", {
      name: /^Train\s+1\s+metric$/i,
    });
    expect(trainToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("img", { name: /train\/loss scalar chart/i }))
      .not.toBeInTheDocument();
    expect(screen.getByLabelText("Scalar Tags train/loss")).toBeChecked();
  });

  it("sorts test loss leaderboards ascending", async () => {
    installFetchMock({
      logTagsByRun: {
        "log-mnist": ["test/loss"],
        "log-cifar": ["test/loss"],
      },
      logScalarSeries: [
        {
          runId: "log-mnist",
          tag: "test/loss",
          points: [{ step: 3, wallTime: 1780000003, value: 0.42 }],
        },
        {
          runId: "log-cifar",
          tag: "test/loss",
          points: [{ step: 3, wallTime: 1780000003, value: 0.27 }],
        },
      ],
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    const lossLeaderboard = await screen.findByRole("table", {
      name: /test\/loss test leaderboard/i,
    });
    const lossRows = within(lossLeaderboard).getAllByRole("row").slice(1);
    expect(lossRows).toHaveLength(2);
    expect(lossRows[0]).toHaveTextContent("0.27");
    expect(within(lossRows[0]).getByText("bbb_20260601_020304"))
      .toBeInTheDocument();
    expect(lossRows[1]).toHaveTextContent("0.42");
    expect(within(lossRows[1]).getByText("aaa_20260601_010203"))
      .toBeInTheDocument();
  });

  it("switches historical scalar chart layouts without refetching scalars", async () => {
    const { logScalarRequests } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    const chart = await screen.findByRole("img", { name: /train\/loss scalar chart/i });
    const chartGrid = scalarChartGridFor(chart);
    const layoutControl = screen.getByRole("tablist", { name: /scalar chart layout/i });
    const fullTab = within(layoutControl).getByRole("tab", { name: /^full$/i });
    const twoColumnTab = within(layoutControl).getByRole("tab", { name: /^2 col$/i });
    const threeColumnTab = within(layoutControl).getByRole("tab", { name: /^3 col$/i });

    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(1);
    });
    expect(fullTab).toHaveAttribute("aria-selected", "true");
    expect(twoColumnTab).toHaveAttribute("aria-selected", "false");
    expect(threeColumnTab).toHaveAttribute("aria-selected", "false");
    expect(chartGrid).toHaveClass("grid", "gap-4");
    expect(chartGrid).not.toHaveClass("xl:grid-cols-2");
    expect(chartGrid).not.toHaveClass("2xl:grid-cols-3");

    await user.click(twoColumnTab);

    expect(fullTab).toHaveAttribute("aria-selected", "false");
    expect(twoColumnTab).toHaveAttribute("aria-selected", "true");
    expect(chartGrid).toHaveClass("xl:grid-cols-2");
    expect(chartGrid).not.toHaveClass("2xl:grid-cols-3");
    expect(logScalarRequests).toHaveLength(1);

    await user.click(threeColumnTab);

    expect(twoColumnTab).toHaveAttribute("aria-selected", "false");
    expect(threeColumnTab).toHaveAttribute("aria-selected", "true");
    expect(chartGrid).toHaveClass("xl:grid-cols-2", "2xl:grid-cols-3");
    expect(logScalarRequests).toHaveLength(1);

    await user.click(screen.getByRole("button", { name: /refresh scalar charts/i }));

    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(2);
    });
  });

  it("logs workspace experiment and tag checkboxes hide chart content", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
    expect(
      await screen.findByRole("table", { name: /test\/accuracy test leaderboard/i }),
    ).toBeInTheDocument();
    expect(
      screen.getAllByText(/test_model_2 · Cifar10 · linear · BASELINE · 2026-06-01 02:03:04/).length,
    ).toBeGreaterThan(0);

    await user.click(screen.getByLabelText("Experiments test_model"));

    await waitFor(() => {
      expect(screen.queryByText(/test_model · Mnist · linear · BASELINE · 2026-06-01 01:02:03/))
        .not.toBeInTheDocument();
    });
    const datasetSection = screen.getByLabelText("Datasets Cifar10").closest("section");
    expect(datasetSection).not.toBeNull();
    expect(within(datasetSection as HTMLElement).getByText("1 / 1")).toBeInTheDocument();
    expect(screen.queryByLabelText("Datasets Mnist")).not.toBeInTheDocument();
    expect(
      screen.getAllByText(/test_model_2 · Cifar10 · linear · BASELINE · 2026-06-01 02:03:04/).length,
    ).toBeGreaterThan(0);

    await user.click(screen.getByLabelText("Scalar Tags validation/accuracy"));

    await waitFor(() => {
      expect(screen.queryByRole("img", { name: /validation\/accuracy scalar chart/i }))
        .not.toBeInTheDocument();
    });
    expect(screen.getByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();

    await user.click(screen.getByLabelText("Scalar Tags test/accuracy"));

    await waitFor(() => {
      expect(screen.queryByRole("table", { name: /test\/accuracy test leaderboard/i }))
        .not.toBeInTheDocument();
    });

    const experimentSection = screen.getByLabelText("Experiments test_model_2").closest("section");
    expect(experimentSection).not.toBeNull();
    await user.click(within(experimentSection as HTMLElement).getByRole("button", { name: /^none$/i }));

    expect(await screen.findByText("No runs selected")).toBeInTheDocument();
  });

  it("deletes a log experiment after exact-name confirmation", async () => {
    const { fetchMock, deleteExperimentRequests, logScalarRequests } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    expect(await screen.findByLabelText("Experiments test_model")).toBeChecked();

    await user.click(
      screen.getByRole("button", { name: /^delete experiment test_model$/i }),
    );

    const dialog = screen.getByRole("dialog", { name: /^delete experiment$/i });
    expect(within(dialog).getByText(/permanently deletes 1 run/i)).toBeInTheDocument();
    const deleteButton = within(dialog).getByRole("button", {
      name: /^delete experiment$/i,
    });
    expect(deleteButton).toBeDisabled();

    await user.type(within(dialog).getByLabelText(/type experiment name/i), "test");
    expect(deleteButton).toBeDisabled();
    await user.clear(within(dialog).getByLabelText(/type experiment name/i));
    await user.type(within(dialog).getByLabelText(/type experiment name/i), "test_model");
    expect(deleteButton).toBeEnabled();

    await user.click(deleteButton);

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /^delete experiment$/i }))
        .not.toBeInTheDocument();
    });
    expect(deleteExperimentRequests).toEqual(["test_model"]);
    const deleteCall = fetchMock.mock.calls.find(([url]) =>
      String(url).endsWith("/logs/experiments/test_model"),
    );
    expect(deleteCall?.[1]?.method).toBe("DELETE");
    expect(screen.queryByLabelText("Experiments test_model")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Experiments test_model_2")).toBeChecked();

    await waitFor(() => {
      expect(logScalarRequests.at(-1)?.runIds).toEqual(["log-cifar"]);
    });
  });

  it("does not render the sidebar-level delete visible runs action", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    expect(await screen.findByText("Historical Scalars")).toBeInTheDocument();

    expect(
      screen.queryByRole("button", { name: /^delete visible runs$/i }),
    ).not.toBeInTheDocument();
  });

  it("hides destructive log deletion actions when capabilities disable them", async () => {
    const { deleteExperimentRequests, deleteRunPlanRequests, deleteRunRequests } =
      installFetchMock({
        capabilitiesResponse: {
          ...capabilitiesResponse,
          authMode: "bearer",
          logDeletionEnabled: false,
        },
      });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    expect(await screen.findByLabelText("Experiments test_model")).toBeChecked();

    expect(
      screen.queryByRole("button", { name: /^delete experiment test_model$/i }),
    ).not.toBeInTheDocument();

    await user.click(screen.getByLabelText("Experiments test_model_2"));

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /^delete dataset/i }))
        .not.toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: /^delete preset/i }))
      .not.toBeInTheDocument();
    expect(screen.getByText(/log deletion is disabled/i)).toBeInTheDocument();
    expect(deleteExperimentRequests).toHaveLength(0);
    expect(deleteRunPlanRequests).toHaveLength(0);
    expect(deleteRunRequests).toHaveLength(0);
  });

  it("shows dataset row delete actions only for a single selected experiment", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    expect(await screen.findByLabelText("Datasets Mnist")).toBeChecked();
    expect(screen.queryByRole("button", { name: /^delete dataset/i }))
      .not.toBeInTheDocument();

    await user.click(screen.getByLabelText("Experiments test_model_2"));

    expect(
      await screen.findByRole("button", {
        name: /^delete dataset Mnist from experiment test_model$/i,
      }),
    ).toBeInTheDocument();

    const experimentSection = screen.getByLabelText("Experiments test_model").closest("section");
    expect(experimentSection).not.toBeNull();
    await user.click(
      within(experimentSection as HTMLElement).getByRole("button", { name: /^none$/i }),
    );

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /^delete dataset/i }))
        .not.toBeInTheDocument();
    });
  });

  it("deletes dataset row runs without typed confirmation using the target run set", async () => {
    const { deleteRunPlanRequests, deleteRunRequests } = installFetchMock(
      buildSubsetDeleteFixture(),
    );
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(await screen.findByLabelText("Experiments test_model_2"));

    await user.click(screen.getByRole("button", { name: /presets\s+2\s*\/\s*2/i }));
    await user.click(screen.getByLabelText("Presets ALT"));
    expect(screen.getByLabelText("Presets ALT")).not.toBeChecked();

    await user.click(
      await screen.findByRole("button", {
        name: /^delete dataset Mnist from experiment test_model$/i,
      }),
    );

    const dialog = await screen.findByRole("dialog", { name: /^delete dataset$/i });
    expect(await within(dialog).findByText(/2 matched runs/i)).toBeInTheDocument();
    expect(within(dialog).getAllByText("test_model").length).toBeGreaterThan(0);
    expect(within(dialog).getAllByText("Mnist").length).toBeGreaterThan(0);
    expect(within(dialog).getByText("ALT")).toBeInTheDocument();
    expect(within(dialog).getByText("BASELINE")).toBeInTheDocument();
    expect(
      within(dialog).getByText(
        "test_model/linear/BASELINE/Mnist/mnist_baseline_20260601_010203/version_0",
      ),
    ).toBeInTheDocument();
    expect(within(dialog).getByText("version_*")).toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/type/i)).not.toBeInTheDocument();

    const deleteButton = within(dialog).getByRole("button", {
      name: /^delete dataset$/i,
    });
    expect(deleteButton).toBeEnabled();
    await user.click(deleteButton);

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /^delete dataset$/i }))
        .not.toBeInTheDocument();
    });
    expect(deleteRunPlanRequests).toEqual([
      {
        experiments: ["test_model"],
        datasets: ["Mnist"],
        models: ["linear"],
        presets: ["ALT", "BASELINE"],
        runIds: ["log-mnist-alt", "log-mnist-baseline"],
      },
    ]);
    expect(deleteRunRequests).toEqual(deleteRunPlanRequests);
    await waitFor(() => {
      expect(screen.queryByLabelText("Datasets Mnist")).not.toBeInTheDocument();
    });
  });

  it("deletes preset row runs using the target experiment and preset", async () => {
    const { deleteRunPlanRequests, deleteRunRequests } = installFetchMock(
      buildSubsetDeleteFixture(),
    );
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(await screen.findByLabelText("Experiments test_model_2"));
    await user.click(screen.getByLabelText("Datasets Cifar10"));
    expect(screen.getByLabelText("Datasets Cifar10")).not.toBeChecked();

    await user.click(screen.getByRole("button", { name: /presets\s+2\s*\/\s*2/i }));
    await user.click(
      await screen.findByRole("button", {
        name: /^delete preset BASELINE from experiment test_model$/i,
      }),
    );

    const dialog = await screen.findByRole("dialog", { name: /^delete preset$/i });
    expect(await within(dialog).findByText(/2 matched runs/i)).toBeInTheDocument();
    expect(within(dialog).getAllByText("test_model").length).toBeGreaterThan(0);
    expect(within(dialog).getAllByText("BASELINE").length).toBeGreaterThan(0);
    expect(within(dialog).getByText("Cifar10")).toBeInTheDocument();
    expect(within(dialog).getByText("Mnist")).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /^delete preset$/i }));

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /^delete preset$/i }))
        .not.toBeInTheDocument();
    });
    expect(deleteRunPlanRequests).toEqual([
      {
        experiments: ["test_model"],
        datasets: ["Cifar10", "Mnist"],
        models: ["linear"],
        presets: ["BASELINE"],
        runIds: ["log-cifar-baseline", "log-mnist-baseline"],
      },
    ]);
    expect(deleteRunRequests).toEqual(deleteRunPlanRequests);
  });

  it("blocks dataset row deletion when an active training job uses an affected folder", async () => {
    const { deleteRunRequests } = installFetchMock({
      deleteLogRunsBlockers: [
        { id: "job-1", logFolder: "test_model", status: "running" },
      ],
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(await screen.findByLabelText("Experiments test_model_2"));
    await user.click(
      await screen.findByRole("button", {
        name: /^delete dataset Mnist from experiment test_model$/i,
      }),
    );

    const dialog = await screen.findByRole("dialog", { name: /^delete dataset$/i });
    expect(
      await within(dialog).findByText(
        /A training job is still writing to this log folder/i,
      ),
    ).toBeInTheDocument();
    expect(within(dialog).getByText(/job-1 · logs\/test_model/i)).toBeInTheDocument();
    expect(
      within(dialog).getByRole("button", { name: /^delete dataset$/i }),
    ).toBeDisabled();
    expect(deleteRunRequests).toHaveLength(0);
  });

  it("keeps logs sidebar checklist rows full-height with many options", async () => {
    const fixture = buildLargeLogFixture();
    const { deleteExperimentRequests, logScalarRequests } = installFetchMock(fixture);
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    const firstExperiment = await screen.findByLabelText("Experiments experiment_01");
    expect(firstExperiment).toBeChecked();
    expectLogsChecklistRowSizing(firstExperiment);
    expectLogsChecklistRowSizing(screen.getByLabelText("Experiments experiment_42"));

    const tagOption = await screen.findByLabelText("Scalar Tags custom/tag-42");
    expectLogsChecklistRowSizing(tagOption);
    await user.click(tagOption);
    expect(tagOption).toBeChecked();

    await waitFor(() => {
      expect(logScalarRequests.at(-1)).toEqual({
        runIds: fixture.logRunsResponse.runs.map((run) => run.id),
        tags: ["custom/tag-42"],
      });
    });

    await user.type(screen.getByLabelText(/^search scalar tags$/i), "tag-42");

    const filteredTagOption = screen.getByLabelText("Scalar Tags custom/tag-42");
    expect(filteredTagOption).toBeChecked();
    expectLogsChecklistRowSizing(filteredTagOption);
    expect(screen.queryByLabelText("Scalar Tags custom/tag-01")).not.toBeInTheDocument();

    await user.click(filteredTagOption);
    expect(filteredTagOption).not.toBeChecked();

    await user.click(
      screen.getByRole("button", { name: /^delete experiment experiment_01$/i }),
    );
    const dialog = screen.getByRole("dialog", { name: /^delete experiment$/i });
    await user.type(
      within(dialog).getByLabelText(/type experiment name/i),
      "experiment_01",
    );
    await user.click(
      within(dialog).getByRole("button", { name: /^delete experiment$/i }),
    );

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /^delete experiment$/i }))
        .not.toBeInTheDocument();
    });
    expect(deleteExperimentRequests).toEqual(["experiment_01"]);
    expect(screen.queryByLabelText("Experiments experiment_01")).not.toBeInTheDocument();

    const secondExperiment = screen.getByLabelText("Experiments experiment_02");
    expect(secondExperiment).toBeChecked();
    expectLogsChecklistRowSizing(secondExperiment);
    await user.click(secondExperiment);
    expect(secondExperiment).not.toBeChecked();
    expect(screen.getByLabelText("Experiments experiment_42")).toBeChecked();
  });

  it("does not delete a log experiment when the dialog is cancelled", async () => {
    const { deleteExperimentRequests } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(
      screen.getByRole("button", { name: /^delete experiment test_model$/i }),
    );
    let dialog = screen.getByRole("dialog", { name: /^delete experiment$/i });
    await user.click(within(dialog).getByRole("button", { name: /^cancel$/i }));

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /^delete experiment$/i }))
        .not.toBeInTheDocument();
    });

    await user.click(
      screen.getByRole("button", { name: /^delete experiment test_model$/i }),
    );
    dialog = screen.getByRole("dialog", { name: /^delete experiment$/i });
    await user.click(
      within(dialog).getByRole("button", { name: /^close delete experiment$/i }),
    );

    expect(deleteExperimentRequests).toHaveLength(0);
    expect(screen.getByLabelText("Experiments test_model")).toBeChecked();
  });

  it("keeps the delete dialog open and shows backend errors", async () => {
    const { deleteExperimentRequests } = installFetchMock({
      deleteLogExperimentError: "Refusing to delete symlink log experiment: test_model",
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(
      screen.getByRole("button", { name: /^delete experiment test_model$/i }),
    );

    const dialog = screen.getByRole("dialog", { name: /^delete experiment$/i });
    await user.type(within(dialog).getByLabelText(/type experiment name/i), "test_model");
    await user.click(
      within(dialog).getByRole("button", { name: /^delete experiment$/i }),
    );

    expect(
      await within(dialog).findByText(/refusing to delete symlink log experiment/i),
    ).toBeInTheDocument();
    expect(deleteExperimentRequests).toEqual(["test_model"]);
    expect(screen.getByLabelText("Experiments test_model")).toBeChecked();
  });

  it("omits stale scalar tags from chart requests after experiment filtering", async () => {
    const { logScalarRequests } = installFetchMock({
      logTagsByRun: {
        "log-mnist": ["train/loss", "validation/accuracy"],
        "log-cifar": ["train/loss"],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    await waitFor(() => {
      expect(logScalarRequests.at(-1)).toEqual({
        runIds: ["log-mnist", "log-cifar"],
        tags: ["train/loss", "validation/accuracy"],
      });
    });

    await user.click(screen.getByLabelText("Experiments test_model"));

    await waitFor(() => {
      expect(logScalarRequests.at(-1)).toEqual({
        runIds: ["log-cifar"],
        tags: ["train/loss"],
      });
    });
    expect(screen.getByText(/1 runs · 1 selected tags/i)).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .not.toBeInTheDocument();

    await user.click(screen.getByLabelText("Experiments test_model"));

    await waitFor(() => {
      expect(screen.getByText(/2 runs · 2 selected tags/i)).toBeInTheDocument();
    });
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
  });

  it("shows the scalar empty state without scalar fetches when event runs have no tags", async () => {
    const { logScalarRequests } = installFetchMock({
      logRunsResponse: {
        runs: [
          {
            ...logRunsResponse.runs[0],
            id: "log-empty",
            eventFileCount: 1,
            hasResult: false,
            metrics: {},
          },
        ],
      },
      logTagsByRun: { "log-empty": [] },
      logScalarSeries: [],
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    expect(await screen.findByText("No TensorBoard scalars")).toBeInTheDocument();
    expect(
      screen.getByText("The selected runs do not contain scalar event data."),
    ).toBeInTheDocument();
    expect(logScalarRequests).toHaveLength(0);
  });

  it("renders preset-locked fields disabled with their reason", async () => {
    installFetchMock({
      schemaResponse: {
        ...schemaResponse,
        fields: schemaResponse.fields.map((field) =>
          field.key === "gate_flag"
            ? {
                ...field,
                locked: true,
                lockedValue: true,
                lockedReason:
                  "Locked by the GATING preset because this preset enables stack gating.",
              }
            : field,
        ),
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const closeButton = within(dialog).getByRole("button", {
      name: /close full config/i,
    });
    const headerActions = closeButton.parentElement;
    if (!(headerActions instanceof HTMLElement)) {
      throw new Error("Expected full config close button to render in the header actions");
    }
    const headerPresetBadge = within(headerActions).getByText("1 preset");
    expect(headerPresetBadge).toHaveClass("text-amber");
    expect(closeButton.previousElementSibling).toBe(headerPresetBadge);

    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const gateAccordion = within(dialog).getByRole("button", {
      name: /gate stack options section, 1 field, 0 overrides, 1 preset/i,
    });
    const layerSection = layerAccordion.closest("section");
    const gateSection = gateAccordion.closest("section");
    const layerJump = within(sectionNav).getByRole("button", {
      name: /jump to layer stack options/i,
    });
    const gateJump = within(sectionNav).getByRole("button", {
      name: /jump to gate stack options/i,
    });
    const layerNavRow = layerJump.parentElement?.parentElement;
    const gateNavRow = gateJump.parentElement?.parentElement;
    if (
      !(layerSection instanceof HTMLElement) ||
      !(gateSection instanceof HTMLElement) ||
      !(layerNavRow instanceof HTMLElement) ||
      !(gateNavRow instanceof HTMLElement)
    ) {
      throw new Error("Expected full config sections and sidebar rows to render");
    }

    expect(gateNavRow).toHaveClass(
      "border-amber/30",
      "bg-amber/[0.055]",
      "hover:bg-amber/[0.09]",
    );
    expect(within(gateNavRow).getByText("1 preset")).toHaveClass("text-amber");
    expect(layerNavRow).not.toHaveClass("border-amber/30", "bg-amber/[0.055]");
    expect(within(layerNavRow).queryByText("1 preset")).not.toBeInTheDocument();
    expect(gateAccordion).toHaveClass("bg-amber/[0.08]", "hover:bg-amber/[0.12]");
    expect(gateSection).toHaveClass("border-amber/35", "bg-amber/[0.045]");
    expect(within(gateAccordion).getByText("1 preset")).toHaveClass("text-amber");
    expect(layerAccordion).not.toHaveClass("bg-amber/[0.08]");
    expect(layerSection).not.toHaveClass("border-amber/35", "bg-amber/[0.045]");
    expect(within(layerAccordion).queryByText("1 preset")).not.toBeInTheDocument();

    const gateSwitch = within(dialog).getByRole("switch", { name: /gate flag/i });
    const gateRow = configFieldRowFor(gateSwitch);
    const presetBadge = within(gateRow).getByText("preset");

    expect(gateSwitch).toBeDisabled();
    expect(gateRow).toHaveClass("border-amber/55", "bg-amber/[0.055]");
    expect(gateRow).not.toHaveClass("border-violet/40");
    expect(presetBadge).toHaveClass("text-amber");
    expect(within(gateRow).queryByText("override")).not.toBeInTheDocument();
    expect(within(dialog).getByText(/locked by the GATING preset/i)).toBeInTheDocument();

    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });
    await user.type(search, "gate");
    const searchPopup = fullConfigSearchPopup(dialog);
    const gateSearchRow = fullConfigSearchResultRow(searchPopup, /gate flag/i);
    const searchPresetBadge = within(gateSearchRow).getByText("preset");
    const searchGateSwitch = within(gateSearchRow).getByRole("switch", {
      name: /current value/i,
    });

    expect(gateSearchRow).toHaveTextContent(/current\s*true/i);
    expect(gateSearchRow).toHaveTextContent(/default\s*false/i);
    expect(searchPresetBadge).toHaveClass("text-amber");
    expect(searchGateSwitch).toBeDisabled();
    expect(
      within(gateSearchRow).getByRole("button", { name: /reset search result override/i }),
    ).toBeDisabled();
  });

  it("expanded training panel shows the flattened setup flow in order", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    const footerFieldBoxClasses = [
      "grid",
      "content-start",
      "gap-1.5",
      "rounded-[10px]",
      "border",
      "border-line",
      "bg-white/[0.018]",
      "px-2.5",
      "py-2",
    ];
    const footerFieldGridClasses = [
      "grid",
      "gap-3",
      "sm:grid-cols-2",
      "xl:grid-cols-3",
    ];
    const footerFieldHeaderClasses = [
      "flex",
      "min-h-[38px]",
      "flex-wrap",
      "items-center",
      "justify-between",
      "gap-2",
    ];
    const footerIconClasses = ["h-[15px]", "w-[15px]", "text-violet"];
    function expectBoxedField(element: HTMLElement) {
      expect(element).toHaveClass(...footerFieldBoxClasses);
    }
    function expectFieldHeader(element: HTMLElement) {
      const header = element.parentElement;
      if (!(header instanceof HTMLElement)) {
        throw new Error("Expected field heading to render inside a header row");
      }
      expect(header).toHaveClass(...footerFieldHeaderClasses);
      return header;
    }
    function closestWithClasses(element: HTMLElement, classNames: string[]) {
      let current: HTMLElement | null = element;
      while (current && current !== details) {
        const candidate = current;
        if (
          classNames.every((className) => candidate.classList.contains(className))
        ) {
          return candidate;
        }
        current = current.parentElement;
      }
      return null;
    }
    function closestFooterFieldBox(element: HTMLElement) {
      const fieldBox = closestWithClasses(element, footerFieldBoxClasses);
      if (!fieldBox) {
        throw new Error("Expected control to render inside a footer field box");
      }
      return fieldBox;
    }
    function expectHeadingIcon(label: string) {
      const heading = within(details).getByText(label);
      const icon = heading.querySelector("svg");
      if (!(icon instanceof SVGElement)) {
        throw new Error(`Expected ${label} heading to render an icon`);
      }
      expect(icon).toHaveClass(...footerIconClasses);
      return heading;
    }

    const modelHeading = expectHeadingIcon("Model");
    const presetsHeading = expectHeadingIcon("Presets");
    const modelSelector = within(details).getByRole("combobox", {
      name: /^training model$/i,
    });
    const presetSelector = within(details).getByRole("combobox", {
      name: /^presets\s+1\s*\/\s*2 selected$/i,
    });
    const datasetsHeading = expectHeadingIcon("Datasets");
    const monitorsHeading = expectHeadingIcon("Monitors");
    const gridSearchHeading = expectHeadingIcon("Grid Search");
    const searchModeControl = within(details).getByRole("tablist", {
      name: /training search mode/i,
    });
    const logFolderSelect = within(details).getByLabelText("Log experiment folder");
    const logFolderField = closestFooterFieldBox(logFolderSelect);
    const logFolderModeControl = within(details).getByRole("tablist", {
      name: /log folder mode/i,
    });
    const modelField = closestFooterFieldBox(modelSelector);
    const presetField = closestFooterFieldBox(presetSelector);
    const datasetSelector = within(details).getByRole("combobox", {
      name: /^training datasets\s+1\s*\/\s*2 selected$/i,
    });
    const datasetBox = closestFooterFieldBox(datasetSelector);
    const monitorSelector = within(details).getByRole("combobox", {
      name: /^training monitors\s+0\s*\/\s*2 selected$/i,
    });
    const monitorBox = closestFooterFieldBox(monitorSelector);
    const fullConfigButton = trainingFullConfigButton(details);
    const configAction = closestFooterFieldBox(fullConfigButton);
    const configHeading = within(configAction).getByText(/^Overrides$/);
    const configHeadingIcon = configHeading.querySelector("svg");
    if (!(configHeadingIcon instanceof SVGElement)) {
      throw new Error("Expected Config action heading to render an icon");
    }
    expect(configHeadingIcon).toHaveClass(...footerIconClasses);
    const resetButton = within(configAction).getByRole("button", { name: /^reset$/i });
    const searchBox = closestFooterFieldBox(searchModeControl);
    const fieldGrid = closestWithClasses(logFolderField, footerFieldGridClasses);
    if (!fieldGrid) {
      throw new Error("Expected setup fields to render inside the footer field grid");
    }
    const fieldGridItems = Array.from(fieldGrid.children);

    function expectBefore(before: HTMLElement, after: HTMLElement) {
      expect(
        before.compareDocumentPosition(after) &
          Node.DOCUMENT_POSITION_FOLLOWING,
      ).toBeTruthy();
    }

    expectBefore(modelHeading, modelSelector);
    expectBefore(modelSelector, presetSelector);
    expectBefore(presetsHeading, presetSelector);
    expectBefore(presetSelector, datasetsHeading);
    expectBefore(datasetsHeading, monitorsHeading);
    expectBefore(monitorsHeading, monitorSelector);
    expectBefore(monitorSelector, configHeading);
    expectBefore(configHeading, fullConfigButton);
    expectBefore(fullConfigButton, gridSearchHeading);
    expect(fieldGrid).toHaveClass(...footerFieldGridClasses);
    expect(fieldGridItems).toHaveLength(6);
    expect(fieldGridItems[0]).toContainElement(logFolderField);
    expect(fieldGridItems[1]).toContainElement(modelSelector);
    expect(fieldGridItems[2]).toContainElement(presetSelector);
    expect(fieldGridItems[3]).toContainElement(datasetSelector);
    expect(fieldGridItems[4]).toContainElement(monitorSelector);
    expect(fieldGridItems[5]).toBe(configAction);
    expect(configAction).toContainElement(fullConfigButton);
    expect(fullConfigButton).toHaveAttribute("aria-label", "Open Full Config");
    expect(fullConfigButton).toHaveTextContent(/^Config$/);
    expectBoxedField(logFolderField);
    expectBoxedField(modelField);
    expectBoxedField(presetField);
    expectBoxedField(datasetBox);
    expectBoxedField(monitorBox);
    expectBoxedField(configAction);
    expectBoxedField(searchBox);
    expectFieldHeader(modelHeading);
    expectFieldHeader(presetsHeading);
    expectFieldHeader(datasetsHeading);
    expectFieldHeader(monitorsHeading);
    expectFieldHeader(configHeading);
    expectFieldHeader(gridSearchHeading);
    expect(closestWithClasses(fullConfigButton, footerFieldBoxClasses)).toBe(configAction);
    expect(searchBox).toContainElement(gridSearchHeading);
    expect(searchBox).toContainElement(searchModeControl);
    expect(fieldGrid).not.toContainElement(searchBox);
    expect(closestFooterFieldBox(logFolderModeControl)).toBe(logFolderField);
    expect(logFolderField).toContainElement(logFolderModeControl);
    expect(logFolderField).toContainElement(
      within(logFolderField).getByRole("combobox", {
        name: "Log experiment folder",
      }),
    );
    const activeLogFolderLabel = within(logFolderField).getByText("Existing folder", {
      selector: "span",
    });
    const activeLogFolderIcon = activeLogFolderLabel.querySelector("svg");
    if (!(activeLogFolderIcon instanceof SVGElement)) {
      throw new Error("Expected active log folder field label to render an icon");
    }
    expect(activeLogFolderIcon).toHaveClass(...footerIconClasses);
    expectFieldHeader(activeLogFolderLabel);
    expect(within(details).queryByText("Experiment Setup")).not.toBeInTheDocument();
    expect(
      within(details).queryByRole("combobox", { name: /^training preset$/i }),
    ).not.toBeInTheDocument();
    expect(modelSelector).toHaveTextContent("linear");
    expect(presetSelector).toHaveTextContent("baseline");
    expect(
      within(details).getAllByRole("combobox", { name: /^presets\b/i }),
    ).toHaveLength(1);
    expect(datasetSelector).toHaveTextContent("Cifar 10");
    expect(within(details).getByRole("button", { name: /^Primary only$/i }))
      .toBeInTheDocument();
    expect(monitorBox).toContainElement(monitorSelector);
    expect(within(monitorBox).getByText("0 / 2")).toBeInTheDocument();
    expect(within(monitorBox).queryByRole("button", { name: /^(all|none)$/i }))
      .not.toBeInTheDocument();
    expect(within(details).queryByLabelText(/monitor Linear layers/i))
      .not.toBeInTheDocument();
    expect(within(details).queryByText(/^Metrics$/)).not.toBeInTheDocument();
    expect(within(details).queryByText(/^Runs$/)).not.toBeInTheDocument();

    const { listbox: datasetList } = await openTrainingMultiSelect(
      user,
      details,
      "Training datasets",
    );
    expect(within(datasetList).getByRole("option", { name: /Mnist/i }))
      .toBeInTheDocument();
    expect(within(datasetList).getByRole("option", { name: /Cifar 10/i }))
      .toHaveAttribute("aria-selected", "true");
    await user.keyboard("{Escape}");

    const allDatasetsButton = within(datasetBox).getByRole("button", { name: /^All$/i });
    const firstDatasetButton = within(datasetBox).getByRole("button", {
      name: /^First$/i,
    });
    expect(datasetBox).toContainElement(allDatasetsButton);
    expect(datasetBox).toContainElement(firstDatasetButton);
    expect(allDatasetsButton.parentElement).toBe(firstDatasetButton.parentElement);
    expect(allDatasetsButton.parentElement).toHaveClass("grid", "grid-cols-2", "gap-2");
    expect(allDatasetsButton).toHaveClass("h-9", "text-[13px]");
    expect(firstDatasetButton).toHaveClass(
      "h-9",
      "border",
      "border-line",
      "bg-white/[0.025]",
      "text-[13px]",
    );
    await user.click(allDatasetsButton);
    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^training datasets\s+2\s*\/\s*2 selected$/i,
        }),
      ).toBeInTheDocument();
    });
    await user.click(firstDatasetButton);
    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^training datasets\s+1\s*\/\s*2 selected$/i,
        }),
      ).toBeInTheDocument();
    });
    expect(within(details).getAllByText(/^Overrides$/)).toHaveLength(1);
    expect(within(details).getAllByText("0 overrides").length).toBeGreaterThan(0);
    expect(within(details).getAllByText("4 fields").length).toBeGreaterThan(0);
    expect(configAction).toContainElement(configHeading);
    expect(within(configAction).getByText("4 fields")).toBeInTheDocument();
    expect(within(configAction).getByText("0 overrides")).toBeInTheDocument();
    expect(resetButton).toBeDisabled();
    expect(fullConfigButton).toBeEnabled();
    expect(closestWithClasses(resetButton, [
      "edge",
      "grid",
      "gap-2",
      "rounded-card",
      "p-3",
    ])).toBeNull();
    expect(within(configAction).queryByText("Config fields")).not.toBeInTheDocument();
    expect(
      within(details).queryByRole("combobox", { name: /search config fields/i }),
    ).not.toBeInTheDocument();
    expect(
      within(details).queryByRole("navigation", { name: /training override sections/i }),
    ).not.toBeInTheDocument();
    expect(
      within(details).queryByRole("button", {
        name: /layer stack options section/i,
      }),
    ).not.toBeInTheDocument();
    expect(within(details).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(details).getByRole("tab", { name: /new folder/i })).toBeInTheDocument();
    await user.click(within(details).getByRole("tab", { name: /new folder/i }));
    const newLogFolderModeControl = within(details).getByRole("tablist", {
      name: /log folder mode/i,
    });
    const newLogFolderField = closestFooterFieldBox(
      within(details).getByLabelText("New log folder"),
    );
    expectBoxedField(newLogFolderField);
    expect(closestFooterFieldBox(newLogFolderModeControl)).toBe(newLogFolderField);
    expect(newLogFolderField).toContainElement(newLogFolderModeControl);
    expect(newLogFolderField).toContainElement(
      within(newLogFolderField).getByRole("textbox", {
        name: "New log folder",
      }),
    );
    const newLogFolderLabel = within(newLogFolderField).getByText("New folder", {
      selector: "span",
    });
    const newLogFolderIcon = newLogFolderLabel.querySelector("svg");
    if (!(newLogFolderIcon instanceof SVGElement)) {
      throw new Error("Expected new log folder field label to render an icon");
    }
    expect(newLogFolderIcon).toHaveClass(...footerIconClasses);
    expectFieldHeader(newLogFolderLabel);
    const { listbox: monitorList } = await openTrainingMultiSelect(
      user,
      details,
      "Training monitors",
    );
    expect(within(monitorList).getByRole("option", { name: /Linear layers/i }))
      .toBeInTheDocument();
    expect(within(monitorList).getByRole("option", { name: /Sampler usage/i }))
      .toBeInTheDocument();
  });

  it("training setup opens the shared full config dialog and reflects popup edits", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    let dialog = await openTrainingFullConfig(user, details);

    expect(dialog).toBeInTheDocument();
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));

    expect(screen.getAllByText(/1 overrides?/i).length).toBeGreaterThan(0);
    expect(within(details).getByRole("button", { name: /^reset$/i })).toBeEnabled();

    await user.click(within(details).getByRole("button", { name: /^reset$/i }));

    await waitFor(() => {
      expect(screen.getAllByText("0 overrides").length).toBeGreaterThan(0);
      expect(within(details).getByRole("button", { name: /^reset$/i })).toBeDisabled();
    });

    dialog = await openTrainingFullConfig(user, details);

    expect(within(dialog).getByLabelText(/hidden dim/i)).toHaveValue(256);
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));
  });

  it("training setup selectors update shared target state and clear overrides", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    let details = await expandedTrainingDetailsWithConfig(user);
    await setTrainingHiddenDimOverride(user, details, "128");
    expect(screen.getAllByText(/1 overrides?/i).length).toBeGreaterThan(0);

    const { listbox } = await openTrainingMultiSelect(user, details, "Presets");
    await user.click(
      within(listbox).getByRole("option", {
        name: /recurrent-gating-halting/i,
      }),
    );
    await user.click(
      within(listbox).getByRole("button", {
        name: /make recurrent-gating-halting primary/i,
      }),
    );
    await user.keyboard("{Escape}");

    expect(await waitForTargetValue("preset", "recurrent-gating-halting"))
      .toHaveTextContent("recurrent-gating-halting");
    expect(screen.getAllByText("0 overrides").length).toBeGreaterThan(0);
    details = await expandedTrainingDetailsWithConfig(user);
    let dialog = await openTrainingFullConfig(user, details);
    await waitFor(() => {
      expect(within(dialog).getByLabelText(/hidden dim/i)).toHaveValue(256);
    });
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));

    await setTrainingHiddenDimOverride(user, details, "192");
    expect(screen.getAllByText(/1 overrides?/i).length).toBeGreaterThan(0);

    await selectTrainingTargetOption(user, "model", "bert_linear");

    expect(await waitForTargetValue("model", "bert_linear")).toHaveTextContent("bert_linear");
    expect(screen.getAllByText("0 overrides").length).toBeGreaterThan(0);
    details = await expandedTrainingDetailsWithConfig(user);
    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^presets\s+1\s*\/\s*1 selected$/i,
        }),
      ).toHaveTextContent("bert-baseline");
    });
    dialog = await openTrainingFullConfig(user, details);
    await waitFor(() => {
      expect(within(dialog).getByLabelText(/hidden dim/i)).toHaveValue(256);
    });
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));
    expect(
      within(details).getByRole("combobox", {
        name: /^training datasets\s+1\s*\/\s*1 selected$/i,
      }),
    ).toHaveTextContent("Toy Text");
  });

  it("training panel posts selected model, preset, datasets, and overrides", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await setTrainingHiddenDimOverride(user, details, "128");
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Mnist/i,
    );
    await selectNewTrainingLogFolder(user, "my_experiment");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        model: "linear",
        preset: "baseline",
        presets: ["baseline"],
        datasets: ["Cifar10", "Mnist"],
        overrides: { hidden_dim: "128" },
        logFolder: "my_experiment",
        monitors: [],
      });
      expect(trainingBodies[0]).toHaveProperty("runPlan.summary.totalRuns", 2);
      expect(trainingBodies[0]).toHaveProperty(
        "runPlan.summary.remainingEpochs",
        60,
      );
    });
  });

  it("training panel posts selected presets and multiplies planned runs", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    const { listbox: presetListbox } = await openTrainingMultiSelect(
      user,
      details,
      "Presets",
    );
    const baselinePreset = within(presetListbox).getByRole("option", {
      name: /baseline/i,
    });
    expect(baselinePreset).toHaveAttribute("aria-selected", "true");
    expect(baselinePreset).toHaveAttribute("aria-disabled", "true");
    await user.click(
      within(presetListbox).getByRole("option", {
        name: /recurrent-gating-halting/i,
      }),
    );
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Mnist/i,
    );
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Cifar 10/i,
    );

    expect(within(details).getByText("2 presets")).toBeInTheDocument();
    expect(within(details).getByText("4 planned runs")).toBeInTheDocument();

    await selectNewTrainingLogFolder(user, "multi_preset");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        model: "linear",
        preset: "baseline",
        presets: ["baseline", "recurrent-gating-halting"],
        datasets: expect.arrayContaining(["Mnist", "Cifar10"]),
        logFolder: "multi_preset",
      });
    });
  });

  it("shows planned training runs and row commands before training starts", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await expandedTrainingDetailsWithConfig(user);
    const progressButton = await screen.findByRole("button", {
      name: /0\s*\/\s*1 runs.*30 epochs left/i,
    });
    await user.click(progressButton);

    const progressDialog = await screen.findByRole("dialog", {
      name: /training progress/i,
    });
    const progressOverlay = progressDialog.parentElement;
    const progressHeader = progressDialog.querySelector("header");
    const progressBody = progressDialog.querySelector(".full-config-dialog-body");

    if (
      !(progressOverlay instanceof HTMLElement) ||
      !(progressHeader instanceof HTMLElement) ||
      !(progressBody instanceof HTMLElement)
    ) {
      throw new Error("Expected training progress dialog chrome to render");
    }

    expect(progressOverlay).toHaveClass(
      "fixed",
      "inset-0",
      "items-center",
      "justify-center",
      "bg-black/70",
      "p-3",
      "backdrop-blur-sm",
      "sm:p-6",
    );
    expect(progressOverlay.parentElement).toBe(document.body);
    expect(progressDialog).toHaveClass(
      "edge",
      "full-config-dialog-shell",
      "w-full",
      "max-w-[92rem]",
      "rounded-card",
      "max-h-[calc(100vh-1.5rem)]",
      "sm:max-h-[calc(100vh-3rem)]",
    );
    expect(progressDialog).not.toHaveClass(
      "h-full",
      "max-w-none",
      "rounded-none",
    );
    expect(progressDialog).not.toHaveClass("max-w-6xl");
    expect(progressHeader).toHaveClass(
      "full-config-dialog-chrome",
      "full-config-dialog-header",
    );
    expect(progressBody).toHaveClass("full-config-dialog-body");
    expect(progressDialog.querySelector("footer")).not.toBeInTheDocument();
    expect(within(progressDialog).getByText("Pending")).toBeInTheDocument();
    expect(within(progressDialog).getAllByText("baseline").length).toBeGreaterThan(0);
    expect(within(progressDialog).getByText("Mnist")).toBeInTheDocument();
    expect(within(progressDialog).getByText("0 / 30")).toBeInTheDocument();

    await user.click(
      within(progressDialog).getByRole("button", { name: /command for run 1/i }),
    );
    const commandDialog = await screen.findByRole("dialog", {
      name: /training command/i,
    });
    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh linear --preset baseline --datasets Mnist",
    );
  });

  it("shows Resample in the progress popup for random search before start", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await user.click(within(details).getByRole("tab", { name: /^random$/i }));
    await user.click(within(details).getByLabelText(/^search axis hidden_dim$/i));
    await user.click(
      await screen.findByRole("button", {
        name: /0\s*\/\s*2 runs.*60 epochs left/i,
      }),
    );

    const progressDialog = await screen.findByRole("dialog", {
      name: /training progress/i,
    });
    expect(
      within(progressDialog).getByRole("button", { name: /^resample$/i }),
    ).toBeInTheDocument();
  });

  it("keeps completed job progress visible after the draft config changes", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await setTrainingHiddenDimOverride(user, details, "128");
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Mnist/i,
    );
    await selectNewTrainingLogFolder(user, "completed_plan");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(
        screen.getByRole("button", {
          name: /2\s*\/\s*2 runs.*0 epochs left/i,
        }),
      ).toBeEnabled();
    });

    await user.click(within(details).getByRole("button", { name: /^reset$/i }));
    await waitFor(() => {
      expect(screen.getAllByText("0 overrides").length).toBeGreaterThan(0);
    });
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Mnist/i,
      false,
    );
    await user.keyboard("{Escape}");
    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^training datasets\s+1\s*\/\s*2 selected$/i,
        }),
      ).toBeInTheDocument();
    });

    const progressButton = await screen.findByRole("button", {
      name: /2\s*\/\s*2 runs.*0 epochs left/i,
    });
    await user.click(progressButton);

    const progressDialog = await screen.findByRole("dialog", {
      name: /training progress/i,
    });
    expect(within(progressDialog).getAllByText("Completed")).toHaveLength(2);
    expect(within(progressDialog).getAllByText("hidden_dim=128")).toHaveLength(2);
    expect(within(progressDialog).getByText("Cifar10")).toBeInTheDocument();
    expect(within(progressDialog).getByText("Mnist")).toBeInTheDocument();
    expect(within(progressDialog).getByText("2 runs")).toBeInTheDocument();
    expect(within(progressDialog).getByText("0 epochs left")).toBeInTheDocument();
    expect(
      within(progressDialog).queryByRole("button", { name: /^resample$/i }),
    ).not.toBeInTheDocument();
  });

  it("starts the next training run from the changed draft plan after completion", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await setTrainingHiddenDimOverride(user, details, "128");
    await selectNewTrainingLogFolder(user, "first_plan");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies).toHaveLength(1);
      expect(
        screen.getByRole("button", {
          name: /1\s*\/\s*1 runs.*0 epochs left/i,
        }),
      ).toBeEnabled();
    });

    await setTrainingHiddenDimOverride(user, details, "192");
    await selectNewTrainingLogFolder(user, "second_plan");
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /start training/i }))
        .toBeEnabled();
    });
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies).toHaveLength(2);
      expect(trainingBodies[1]).toMatchObject({
        logFolder: "second_plan",
        overrides: { hidden_dim: "192" },
      });
      expect(trainingBodies[1]).toHaveProperty(
        "runPlan.runs.0.overrides.hidden_dim",
        "192",
      );
      expect(trainingBodies[1]).toHaveProperty(
        "runPlan.runs.0.command",
        "source experiment.sh linear --preset baseline --datasets Cifar10 --logdir second_plan --config --hidden-dim 192",
      );
    });
  });

  it("making a selected preset primary updates the primary target and resets setup state", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await setTrainingHiddenDimOverride(user, details, "128");
    await user.click(within(details).getByRole("tab", { name: /^grid$/i }));
    await user.click(within(details).getByLabelText(/^search axis hidden_dim$/i));
    expect(within(details).getByText(/1 fixed override replaced by search axes/i))
      .toBeInTheDocument();

    const initialInspectRequestCount = inspectBodies.length;
    const { listbox } = await openTrainingMultiSelect(user, details, "Presets");
    await user.click(
      within(listbox).getByRole("option", {
        name: /recurrent-gating-halting/i,
      }),
    );
    expect(within(details).getByText("2 presets")).toBeInTheDocument();

    await user.click(
      await within(listbox).findByRole("button", {
        name: /make recurrent-gating-halting primary/i,
      }),
    );
    await user.keyboard("{Escape}");

    expect(await waitForTargetValue("preset", "recurrent-gating-halting"))
      .toHaveTextContent("recurrent-gating-halting");
    expect(screen.getAllByText("0 overrides").length).toBeGreaterThan(0);
    await waitFor(() => {
      expect(within(details).getByRole("tab", { name: /^off$/i }))
        .toHaveAttribute("aria-selected", "true");
    });
    expect(inspectBodies).toHaveLength(initialInspectRequestCount);

    await user.click(screen.getByRole("button", { name: /update preview/i }));
    await waitFor(() =>
      expect(inspectBodies).toHaveLength(initialInspectRequestCount + 1),
    );
    expect(inspectBodies.at(-1)).toEqual({
      model: "linear",
      preset: "recurrent-gating-halting",
      dataset: "Cifar10",
      overrides: {},
    });
    expect(
      within(details).getByRole("combobox", {
        name: /^presets\s+2\s*\/\s*2 selected$/i,
      }),
    ).toHaveTextContent("recurrent-gating-halting");
  });

  it("removing the current primary preset promotes another selected preset", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    const { listbox } = await openTrainingMultiSelect(user, details, "Presets");
    await user.click(
      within(listbox).getByRole("option", {
        name: /recurrent-gating-halting/i,
      }),
    );

    const currentPrimary = within(listbox).getByRole("option", { name: /baseline/i });
    expect(currentPrimary).toHaveAttribute("aria-selected", "true");
    expect(currentPrimary).not.toHaveAttribute("aria-disabled", "true");
    await user.click(currentPrimary);

    expect(await waitForTargetValue("preset", "recurrent-gating-halting"))
      .toHaveTextContent("recurrent-gating-halting");
    await waitFor(() => {
      expect(
        within(details).getByRole("combobox", {
          name: /^presets\s+1\s*\/\s*2 selected$/i,
        }),
      ).toHaveTextContent("recurrent-gating-halting");
    });
    expect(within(listbox).getByRole("option", { name: /baseline/i }))
      .toHaveAttribute("aria-selected", "false");
  });

  it("keeps the last selected preset selected in the preset multiselect", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    const { listbox } = await openTrainingMultiSelect(user, details, "Presets");
    const onlySelectedOption = within(listbox).getByRole("option", { name: /baseline/i });

    expect(onlySelectedOption).toHaveAttribute("aria-selected", "true");
    expect(onlySelectedOption).toHaveAttribute("aria-disabled", "true");
    await user.click(onlySelectedOption);

    expect(onlySelectedOption).toHaveAttribute("aria-selected", "true");
    expect(
      within(details).getByRole("combobox", {
        name: /^presets\s+1\s*\/\s*2 selected$/i,
      }),
    ).toHaveTextContent("baseline");
  });

  it("keeps the last dataset selected in the dataset multiselect", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    const { listbox } = await openTrainingMultiSelect(
      user,
      details,
      "Training datasets",
    );
    const selectedDataset = within(listbox).getByRole("option", { name: /Mnist/i });

    expect(selectedDataset).toHaveAttribute("aria-selected", "true");
    expect(selectedDataset).toHaveAttribute("aria-disabled", "true");
    await user.click(selectedDataset);

    expect(selectedDataset).toHaveAttribute("aria-selected", "true");
    expect(
      within(details).getByRole("combobox", {
        name: /^training datasets\s+1\s*\/\s*2 selected$/i,
      }),
    ).toHaveTextContent("Mnist");
  });

  it("filters training multiselect options and selects matching results", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    let { listbox } = await openTrainingMultiSelect(user, details, "Presets");
    await user.type(
      within(details).getByRole("searchbox", { name: /^search presets$/i }),
      "recurrent",
    );

    expect(within(listbox).queryByRole("option", { name: /baseline/i }))
      .not.toBeInTheDocument();
    await user.click(
      within(listbox).getByRole("option", { name: /recurrent-gating-halting/i }),
    );
    expect(
      within(details).getByRole("combobox", {
        name: /^presets\s+2\s*\/\s*2 selected$/i,
      }),
    ).toHaveTextContent("recurrent-gating-halting");

    await user.keyboard("{Escape}");
    ({ listbox } = await openTrainingMultiSelect(
      user,
      details,
      "Training datasets",
    ));
    await user.type(
      within(details).getByRole("searchbox", { name: /^search training datasets$/i }),
      "mnist",
    );

    expect(within(listbox).queryByRole("option", { name: /Cifar 10/i }))
      .not.toBeInTheDocument();
    await user.click(within(listbox).getByRole("option", { name: /Mnist/i }));
    expect(
      within(details).getByRole("combobox", {
        name: /^training datasets\s+2\s*\/\s*2 selected$/i,
      }),
    ).toHaveTextContent("Mnist");
  });

  it("multiplies grid planned runs by selected presets and datasets", async () => {
    installFetchMock({
      searchSpaceResponse: {
        ...searchSpaceResponse,
        axes: [
          {
            ...searchSpaceResponse.axes[0],
            values: [64, 128, 256],
          },
        ],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await setTrainingMultiSelectOption(
      user,
      details,
      "Presets",
      /recurrent-gating-halting/i,
    );
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Mnist/i,
    );
    await setTrainingMultiSelectOption(
      user,
      details,
      "Training datasets",
      /Cifar 10/i,
    );

    await user.click(within(details).getByRole("tab", { name: /^grid$/i }));
    await user.click(within(details).getByLabelText(/^search axis hidden_dim$/i));

    expect(within(details).getByText("3 combinations")).toBeInTheDocument();
    expect(within(details).getAllByText("12 planned runs").length).toBeGreaterThan(0);
  });

  it("resetting overrides from training setup clears posted overrides", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await setTrainingHiddenDimOverride(user, details, "128");
    expect(screen.getAllByText(/1 overrides?/i).length).toBeGreaterThan(0);

    await user.click(within(details).getByRole("button", { name: /^reset$/i }));

    await waitFor(() => {
      expect(screen.getAllByText("0 overrides").length).toBeGreaterThan(0);
      expect(within(details).getByRole("button", { name: /^reset$/i })).toBeDisabled();
    });
    const dialog = await openTrainingFullConfig(user, details);
    expect(within(dialog).getByLabelText(/hidden dim/i)).toHaveValue(256);
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));
    await selectNewTrainingLogFolder(user, "reset_overrides");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        logFolder: "reset_overrides",
        overrides: {},
      });
    });
  });

  it("starts grid search with selected axis values and omits conflicting overrides", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await setTrainingHiddenDimOverride(user, details, "128");
    await selectNewTrainingLogFolder(user, "grid_search");

    await user.click(within(details).getByRole("tab", { name: /^grid$/i }));
    expect(screen.getByRole("button", { name: /start training/i })).toBeDisabled();

    await user.click(within(details).getByLabelText(/^search axis hidden_dim$/i));
    await user.click(within(details).getByLabelText(/^search value hidden_dim 128$/i));

    expect(within(details).getByText(/1 fixed override replaced by search axes/i))
      .toBeInTheDocument();
    expect(screen.getByRole("button", { name: /start training/i })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        model: "linear",
        preset: "baseline",
        presets: ["baseline"],
        datasets: ["Cifar10"],
        overrides: {},
        logFolder: "grid_search",
        monitors: [],
        search: {
          mode: "grid",
          values: { hidden_dim: [64] },
        },
      });
      expect(trainingBodies[0]).toHaveProperty("runPlan.summary.totalRuns", 1);
      expect(trainingBodies[0]).toHaveProperty(
        "runPlan.runs.0.overrides.hidden_dim",
        64,
      );
    });
  });

  it("posts random search with the configured sample count", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await selectNewTrainingLogFolder(user, "random_search");
    await user.click(within(details).getByRole("tab", { name: /^random$/i }));
    expect(within(details).getByLabelText(/^random search samples$/i)).toHaveValue(10);

    await user.click(within(details).getByLabelText(/^search axis stack_activation$/i));
    const samplesInput = within(details).getByLabelText(/^random search samples$/i);
    await user.clear(samplesInput);
    await user.type(samplesInput, "7");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        logFolder: "random_search",
        search: {
          mode: "random",
          values: { stack_activation: ["RELU", "GELU"] },
          randomSamples: 7,
        },
      });
    });
  });

  it("selects all search axes and values from the grid setup", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await selectNewTrainingLogFolder(user, "all_axes_search");
    await user.click(within(details).getByRole("tab", { name: /^grid$/i }));
    await user.click(within(details).getByRole("button", { name: /^all axes$/i }));

    expect(within(details).getByText("3 axes")).toBeInTheDocument();
    expect(within(details).getByText("8 combinations")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        logFolder: "all_axes_search",
        search: {
          mode: "grid",
          values: {
            hidden_dim: [64, 128],
            stack_num_layers: [2, 4],
            stack_activation: ["RELU", "GELU"],
          },
        },
      });
    });
  });

  it("confirms large grid searches before posting", async () => {
    const largeSearchSpace = {
      ...searchSpaceResponse,
      axes: [
        {
          ...searchSpaceResponse.axes[0],
          values: Array.from({ length: 11 }, (_, index) => index + 1),
        },
        {
          ...searchSpaceResponse.axes[1],
          values: Array.from({ length: 10 }, (_, index) => index + 1),
        },
      ],
    };
    const { trainingBodies } = installFetchMock({
      searchSpaceResponse: largeSearchSpace,
    });
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetailsWithConfig(user);
    await selectNewTrainingLogFolder(user, "large_grid_search");
    await user.click(within(details).getByRole("tab", { name: /^grid$/i }));
    await user.click(within(details).getByRole("button", { name: /^all axes$/i }));

    expect(within(details).getAllByText("110 planned runs").length).toBeGreaterThan(0);

    await user.click(screen.getByRole("button", { name: /start training/i }));

    const dialog = await screen.findByRole("dialog", { name: /confirm grid search/i });
    expect(trainingBodies).toHaveLength(0);
    expect(within(dialog).getByText(/110 training runs/i)).toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /^cancel$/i }));
    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /confirm grid search/i }))
        .not.toBeInTheDocument();
    });
    expect(trainingBodies).toHaveLength(0);

    await user.click(screen.getByRole("button", { name: /start training/i }));
    await user.click(
      within(await screen.findByRole("dialog", { name: /confirm grid search/i }))
        .getByRole("button", { name: /start training/i }),
    );

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        logFolder: "large_grid_search",
        search: {
          mode: "grid",
        },
      });
    });
  });

  it("training panel posts selected monitor names", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await selectTrainingMonitorOption(user, /Linear layers/i);
    await selectNewTrainingLogFolder(user, "monitor_run");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({
        logFolder: "monitor_run",
        monitors: ["linear"],
      });
    });
    expect(screen.getAllByText(/1 monitors/i).length).toBeGreaterThan(0);
  });

  it("requires a valid log folder before starting training", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const startButton = await screen.findByRole("button", { name: /start training/i });
    expect(startButton).toBeDisabled();

    await expandTrainingPanel(user);
    await user.click(screen.getByRole("tab", { name: /new folder/i }));
    const input = screen.getByLabelText(/^new log folder$/i);

    for (const value of [
      "my experiment",
      "my-experiment",
      "my.folder",
      "my/folder",
      "_my_folder",
      "my__folder",
    ]) {
      await user.clear(input);
      await user.type(input, value);
      expect(screen.getByRole("button", { name: /start training/i })).toBeDisabled();
      expect(screen.getByRole("alert")).toHaveTextContent(/single underscores/i);
    }

    await user.clear(input);
    await user.type(input, "my_experiment");
    expect(screen.getByRole("button", { name: /start training/i })).toBeEnabled();

    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({ logFolder: "my_experiment" });
    });
  });

  it("disables training planning and submission when capabilities disable training", async () => {
    const { fetchMock, trainingBodies } = installFetchMock({
      capabilitiesResponse: {
        ...capabilitiesResponse,
        authMode: "bearer",
        trainingEnabled: false,
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const startButton = await screen.findByRole("button", { name: /start training/i });
    expect(startButton).toBeDisabled();

    const details = await expandedTrainingDetails(user);
    expect(within(details).getByText(/training is disabled/i)).toBeInTheDocument();

    await user.click(within(details).getByRole("tab", { name: /new folder/i }));
    await user.type(within(details).getByLabelText(/^new log folder$/i), "hosted_disabled");

    expect(startButton).toBeDisabled();
    expect(trainingBodies).toHaveLength(0);
    expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/training/run-plan")))
      .toBe(false);
  });

  it("keeps Start Training disabled when the selected model has no datasets", async () => {
    const { trainingBodies } = installFetchMock({
      datasetsResponse: { model: "linear", datasets: [] },
    });
    renderViewer();
    const user = userEvent.setup();

    const details = await expandedTrainingDetails(user);
    expect(within(details).getByText("No datasets for this model")).toBeInTheDocument();

    await user.click(within(details).getByRole("tab", { name: /new folder/i }));
    await user.type(within(details).getByLabelText(/^new log folder$/i), "no_dataset_run");

    expect(screen.getByRole("button", { name: /start training/i })).toBeDisabled();
    expect(trainingBodies).toHaveLength(0);
  });

  it("posts the selected existing log folder", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await selectExistingTrainingLogFolder(user, "test_model_2");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({ logFolder: "test_model_2" });
    });
  });

  it("checks the started experiment when switching to logs", async () => {
    const { trainingBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await selectNewTrainingLogFolder(user, "fresh_run");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(trainingBodies[0]).toMatchObject({ logFolder: "fresh_run" });
    });

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    expect(await screen.findByLabelText("Experiments fresh_run")).toBeChecked();
  });

  it("Update Preview sends a new inspect request for the same selection", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    const initialRequestCount = inspectBodies.length;

    await user.click(screen.getByRole("button", { name: /update preview/i }));
    await waitFor(() => expect(inspectBodies).toHaveLength(initialRequestCount + 1));

    await user.click(screen.getByRole("button", { name: /update preview/i }));
    await waitFor(() => expect(inspectBodies).toHaveLength(initialRequestCount + 2));
  });

  it("changing presets clears overrides without auto-updating the preview", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    await user.click(within(dialog).getByRole("button", { name: /^close$/i }));
    const initialRequestCount = inspectBodies.length;

    await selectTargetOption(user, "preset", "recurrent-gating-halting");

    expect(screen.getByText("0 overrides")).toBeInTheDocument();
    expect(screen.queryByText("No overrides set")).not.toBeInTheDocument();
    expect(inspectBodies).toHaveLength(initialRequestCount);

    await user.click(screen.getByRole("button", { name: /update preview/i }));

    await waitFor(() => expect(inspectBodies).toHaveLength(initialRequestCount + 1));
    expect(inspectBodies.at(-1)).toEqual({
      model: "linear",
      preset: "recurrent-gating-halting",
      dataset: "Cifar10",
      overrides: {},
    });
  });

  it("resetting overrides refreshes the preview when a target is selected", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    const requestCountBeforeReset = inspectBodies.length;

    await user.click(within(dialog).getByRole("button", { name: /reset overrides/i }));

    await waitFor(() => expect(inspectBodies).toHaveLength(requestCountBeforeReset + 1));
    expect(inspectBodies.at(-1)).toEqual({
      model: "linear",
      preset: "baseline",
      dataset: "Cifar10",
      overrides: {},
    });
  });

  it("clears the displayed graph while a preview refresh is pending", async () => {
    const nextPreview = deferred<unknown>();
    installFetchMock({
      logRunsResponse: { runs: [] },
      inspectResponseFactory: (requestIndex) =>
        requestIndex === 1 ? nextPreview.promise : inspectResponse,
    });
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /update preview/i }));

    await waitFor(() => {
      expect(screen.queryByText("main_model.0")).not.toBeInTheDocument();
    });
    expect(screen.getByText("building")).toBeInTheDocument();

    nextPreview.resolve(inspectResponse);

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
  });

  it("keeps config controls compact in the left sidebar", async () => {
    installFetchMock();
    renderViewer();

    expect(await waitForOpenFullConfigButton()).toBeInTheDocument();
    expect(screen.queryByText("Sections")).not.toBeInTheDocument();
    expect(screen.queryByText("Fields")).not.toBeInTheDocument();
    expect(screen.queryByText("Changed")).not.toBeInTheDocument();
    expect(screen.queryByText("Layer Stack Options")).not.toBeInTheDocument();
    expect(screen.queryByText("Gate Stack Options")).not.toBeInTheDocument();
    expect(screen.queryByText("Modified Overrides")).not.toBeInTheDocument();
    expect(screen.queryByText("No overrides set")).not.toBeInTheDocument();
    expect(screen.queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
  });

  it("opens the full config popup with section accordions expanded by default", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const dialogHeader = dialog.querySelector("header");
    const dialogBody = dialog.querySelector(".full-config-dialog-body");
    const dialogFooter = dialog.querySelector("footer");
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const gateAccordion = within(dialog).getByRole("button", {
      name: /gate stack options section, 1 field, 0 overrides/i,
    });
    const layerNavToggle = within(sectionNav).getByRole("button", {
      name: /close layer stack options/i,
    });
    const gateNavToggle = within(sectionNav).getByRole("button", {
      name: /close gate stack options/i,
    });
    const layerSection = layerAccordion.closest("section");
    const gateSection = gateAccordion.closest("section");
    const sectionGrid = fullConfigSectionGridFor(layerAccordion);
    const closeButton = within(dialog).getByRole("button", {
      name: /close full config/i,
    });

    if (
      !(dialogHeader instanceof HTMLElement) ||
      !(dialogBody instanceof HTMLElement) ||
      !(dialogFooter instanceof HTMLElement)
    ) {
      throw new Error("Expected full config dialog chrome to render");
    }

    expect(dialog).toHaveClass("edge", "full-config-dialog-shell");
    expect(dialogHeader).toHaveClass(
      "full-config-dialog-chrome",
      "full-config-dialog-header",
      "border-line-soft",
    );
    expect(dialogHeader).not.toHaveClass("bg-panel/85");
    expect(dialogHeader).not.toHaveClass("border-line");
    expect(dialogBody).toHaveClass("full-config-dialog-body");
    expect(dialogBody).not.toHaveClass("bg-bg-2/80");
    expect(dialogFooter).toHaveClass(
      "full-config-dialog-chrome",
      "full-config-dialog-footer",
      "border-line-soft",
    );
    expect(dialogFooter).not.toHaveClass("bg-panel/85");
    expect(dialogFooter).not.toHaveClass("border-line");
    expect(closeButton).toHaveClass("border-line-soft", "bg-white/[0.025]");
    expect(closeButton).not.toHaveClass("bg-white/[0.035]");
    expect(sectionGrid).toBe(fullConfigSectionGridFor(gateAccordion));
    expectFullConfigSectionGrid(sectionGrid);
    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(layerAccordion).toHaveAttribute("aria-controls");
    expect(gateAccordion).toHaveAttribute("aria-expanded", "true");
    expect(gateAccordion).toHaveAttribute("aria-controls");
    expect(layerAccordion).toHaveClass("overflow-hidden", "bg-white/[0.055]");
    expect(gateAccordion).toHaveClass("overflow-hidden", "bg-white/[0.055]");
    expect(layerSection).toHaveClass(
      "overflow-hidden",
      "rounded-[12px]",
      "border",
      "border-line",
      "bg-panel/80",
      "shadow-[0_16px_40px_-30px_rgba(0,0,0,0.95)]",
    );
    expect(gateSection).toHaveClass(
      "overflow-hidden",
      "rounded-[12px]",
      "border",
      "border-line",
      "bg-panel/80",
      "shadow-[0_16px_40px_-30px_rgba(0,0,0,0.95)]",
    );
    expect(layerNavToggle).toHaveAttribute("aria-expanded", "true");
    expect(layerNavToggle).toHaveAttribute("aria-controls");
    expect(gateNavToggle).toHaveAttribute("aria-expanded", "true");
    expect(gateNavToggle).toHaveAttribute("aria-controls");
    expect(layerAccordion).not.toHaveTextContent(/3 fields|0 overrides/i);
    expect(within(layerAccordion).getByLabelText("3 fields")).not.toHaveAttribute("tabindex");
    expect(within(layerAccordion).getByLabelText("0 overrides")).not.toHaveAttribute("tabindex");
    expect(within(gateAccordion).getByLabelText("1 field")).not.toHaveAttribute("tabindex");
    expect(within(gateAccordion).getByLabelText("0 overrides")).not.toHaveAttribute("tabindex");
    expect(within(dialog).getByLabelText("4 fields")).toHaveTextContent("4");
    expect(within(dialog).getByLabelText("4 fields")).not.toHaveTextContent("4 fields");
    const layerJump = within(sectionNav).getByRole("button", {
      name: /jump to layer stack options/i,
    });
    const gateJump = within(sectionNav).getByRole("button", {
      name: /jump to gate stack options/i,
    });
    const layerNavRow = layerJump.parentElement?.parentElement;
    expect(layerNavRow).toHaveClass("group/section-row");
    expect(layerNavRow).toHaveClass("focus-within:ring-2", "hover:bg-violet/10");
    expect(layerJump).not.toHaveClass("peer/title", "group/title");
    expect(layerJump).not.toHaveClass("pr-[7.75rem]");
    expect(gateJump).not.toHaveClass("pr-[7.75rem]");
    expect(within(sectionNav).getByLabelText("3 fields")).toHaveTextContent("3");
    expect(within(sectionNav).getByLabelText("3 fields")).not.toHaveAttribute("tabindex");
    within(sectionNav)
      .getAllByLabelText("0 overrides")
      .forEach((metric) => expect(metric).not.toHaveAttribute("tabindex"));
    expect(layerJump).not.toHaveTextContent(/3 fields|0 overrides/i);
    expect(within(sectionNav).getByLabelText("1 field")).toHaveTextContent("1");
    expect(within(sectionNav).getByLabelText("1 field")).not.toHaveAttribute("tabindex");
    expect(gateJump).not.toHaveTextContent(/1 field|0 overrides/i);
    const hiddenDimControl = within(dialog).getByLabelText(/hidden dim/i);
    const gateSwitch = within(dialog).getByRole("switch", { name: /gate flag/i });

    expect(hiddenDimControl).toBeInTheDocument();
    expectResponsiveConfigFieldGrid(configFieldGridFor(hiddenDimControl));
    expect(hiddenDimControl).toHaveClass("h-10", "px-3", "py-2");
    expect(within(dialog).getByLabelText(/stack activation/i)).toBeInTheDocument();
    expect(gateSwitch).toBeInTheDocument();
    expect(gateSwitch.parentElement).toHaveClass("h-10", "px-3");
  });

  it("shows an accessible full config field search", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    expect(search).toHaveAttribute("aria-expanded", "false");
    expect(search).toHaveAttribute("aria-controls");
    expect(search).toHaveAttribute("aria-haspopup", "dialog");
    expect(search).not.toHaveAttribute("aria-activedescendant");

    await user.type(search, "hidden");

    expect(search).toHaveAttribute("aria-expanded", "true");
    expect(search).not.toHaveAttribute("aria-activedescendant");
    const searchPopup = fullConfigSearchPopup(dialog);
    const hiddenDimRow = fullConfigSearchResultRow(searchPopup, /hidden dim/i);

    expect(hiddenDimRow).toHaveTextContent(/default\s*256/i);
    expect(hiddenDimRow).not.toHaveTextContent(/current\s*256/i);
    expect(within(hiddenDimRow).getByRole("button", { name: /hidden dim/i }))
      .toBeInTheDocument();
    expect(
      within(hiddenDimRow).getByRole("spinbutton", { name: /current value/i }),
    ).toHaveValue(256);
  });

  it("filters full config cards and sidebar sections while typing", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "hidden");

    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/stack activation/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByRole("switch", { name: /gate flag/i })).not.toBeInTheDocument();
    expect(
      within(dialog).getByRole("button", {
        name: /layer stack options section, 1 field, 0 overrides/i,
      }),
    ).toBeInTheDocument();
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 1 field, 0 overrides/i,
    });
    const sectionGrid = fullConfigSectionGridFor(layerAccordion);
    expectFullConfigSectionGrid(sectionGrid);
    expect(sectionGrid.children).toHaveLength(1);
    expect(
      within(sectionNav).getByRole("button", { name: /jump to layer stack options/i }),
    ).toBeInTheDocument();
    expect(
      within(sectionNav).queryByRole("button", { name: /jump to gate stack options/i }),
    ).not.toBeInTheDocument();
    expect(within(sectionNav).getByLabelText("1 field")).toHaveTextContent("1");
  });

  it("finds full config fields by flag and key", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "--gate-flag");

    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();

    await user.clear(search);
    await user.type(search, "gate_flag");

    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
    expect(
      within(dialog).getByRole("button", {
        name: /gate stack options section, 1 field, 0 overrides/i,
      }),
    ).toBeInTheDocument();
  });

  it("does not match full config fields by current or default value text", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "256");
    const searchPopup = fullConfigSearchPopup(dialog);

    expect(within(searchPopup).getByText("No matching fields")).toBeInTheDocument();
    expect(
      within(searchPopup).queryByRole("group", { name: /hidden dim/i }),
    ).not.toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(sectionNav).getByText("No matching sections")).toBeInTheDocument();
  });

  it("selects a dropdown field and filters to exactly that field", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "activation");
    const searchPopup = fullConfigSearchPopup(dialog);
    const stackActivationRow = fullConfigSearchResultRow(searchPopup, /stack activation/i);
    await user.click(
      within(stackActivationRow).getByRole("button", { name: /stack activation/i }),
    );

    expect(search).toHaveValue("stack activation");
    expect(within(dialog).getByLabelText(/stack activation/i)).toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByRole("switch", { name: /gate flag/i })).not.toBeInTheDocument();
    expect(
      within(dialog).queryByRole("dialog", { name: /matching config fields/i }),
    ).not.toBeInTheDocument();
    expect(scrollIntoViewMock).toHaveBeenCalledWith({
      block: "start",
      behavior: "smooth",
    });
  });

  it("edits a numeric field from full config search results", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "hidden");
    const searchPopup = fullConfigSearchPopup(dialog);
    const hiddenDimRow = fullConfigSearchResultRow(searchPopup, /hidden dim/i);
    const hiddenDimSearchInput = within(hiddenDimRow).getByRole("spinbutton", {
      name: /current value/i,
    });

    expect(hiddenDimRow).toHaveTextContent(/default\s*256/i);
    expect(hiddenDimRow).not.toHaveTextContent(/current\s*256/i);

    await user.clear(hiddenDimSearchInput);
    await user.type(hiddenDimSearchInput, "128");

    expect(fullConfigSearchPopup(dialog)).toBeInTheDocument();
    expect(hiddenDimRow).toHaveTextContent(/current\s*128/i);
    expect(hiddenDimRow).toHaveTextContent(/default\s*256/i);
    expect(within(hiddenDimRow).getByText("override")).toHaveClass("text-violet");
    expect(within(dialog).getAllByLabelText("1 override").length).toBeGreaterThan(0);
    expect(within(dialog).getByLabelText(/hidden dim/i)).toHaveValue(128);

    await user.click(within(dialog).getByRole("button", { name: /update preview/i }));

    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        model: "linear",
        preset: "baseline",
        dataset: "Cifar10",
        overrides: { hidden_dim: "128" },
      });
    });
  });

  it("edits enum and bool fields from full config search results", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "activation");
    let searchPopup = fullConfigSearchPopup(dialog);
    const stackActivationRow = fullConfigSearchResultRow(searchPopup, /stack activation/i);
    const stackActivationSelect = within(stackActivationRow).getByRole("combobox", {
      name: /current value/i,
    });

    await user.selectOptions(stackActivationSelect, "RELU");

    expect(stackActivationRow).toHaveTextContent(/current\s*RELU/i);
    expect(within(dialog).getByLabelText(/stack activation/i)).toHaveValue("RELU");

    await user.click(within(dialog).getByRole("button", { name: /clear config search/i }));
    await user.type(search, "gate");
    searchPopup = fullConfigSearchPopup(dialog);
    const gateFlagRow = fullConfigSearchResultRow(searchPopup, /gate flag/i);
    const gateFlagSwitch = within(gateFlagRow).getByRole("switch", {
      name: /current value/i,
    });

    await user.click(gateFlagSwitch);

    expect(gateFlagRow).toHaveTextContent(/current\s*true/i);
    expect(within(dialog).getByRole("switch", { name: /gate flag/i }))
      .toHaveAttribute("aria-checked", "true");
    expect(within(dialog).getAllByLabelText("2 overrides").length).toBeGreaterThan(0);
  });

  it("clears full config search and restores all sections and fields", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "hidden");
    expect(within(dialog).queryByRole("switch", { name: /gate flag/i })).not.toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: /clear config search/i }));

    expect(search).toHaveValue("");
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/stack activation/i)).toBeInTheDocument();
    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
    expect(
      within(sectionNav).getByRole("button", { name: /jump to layer stack options/i }),
    ).toBeInTheDocument();
    expect(
      within(sectionNav).getByRole("button", { name: /jump to gate stack options/i }),
    ).toBeInTheDocument();
  });

  it("shows empty states when full config search has no matches", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "missing option");

    const noResults = within(dialog).getByText('No config fields match "missing option".');
    expect(noResults).toBeInTheDocument();
    expect(noResults).not.toHaveClass("md:col-span-2");
    expect(noResults).not.toHaveClass("2xl:col-span-3");
    const sectionGrid = noResults.parentElement;
    expect(sectionGrid).toBeInstanceOf(HTMLElement);
    expectFullConfigSectionGrid(sectionGrid as HTMLElement);
    expect(within(sectionNav).getByText("No matching sections")).toBeInTheDocument();
    expect(
      within(sectionNav).queryByRole("button", { name: /jump to layer stack options/i }),
    ).not.toBeInTheDocument();
    expect(
      within(sectionNav).queryByRole("button", { name: /jump to gate stack options/i }),
    ).not.toBeInTheDocument();
  });

  it("keeps sidebar jumps and toggles working while full config search is filtered", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "hidden");
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 1 field, 0 overrides/i,
    });

    await user.click(within(sectionNav).getByRole("button", { name: /^close all$/i }));

    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByRole("switch", { name: /gate flag/i })).not.toBeInTheDocument();

    await user.click(within(sectionNav).getByRole("button", { name: /^open all$/i }));

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(within(dialog).queryByRole("switch", { name: /gate flag/i })).not.toBeInTheDocument();

    await user.click(
      within(sectionNav).getByRole("button", { name: /close layer stack options/i }),
    );

    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();

    await user.click(
      within(sectionNav).getByRole("button", { name: /jump to layer stack options/i }),
    );

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(scrollIntoViewMock).toHaveBeenCalledWith({
      block: "start",
      behavior: "smooth",
    });
  });

  it("keeps filtered accordion metric pills non-tabbable and number-only", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const search = within(dialog).getByRole("combobox", {
      name: /search config fields/i,
    });

    await user.type(search, "hidden");
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 1 field, 0 overrides/i,
    });

    expect(layerAccordion).not.toHaveTextContent(/1 field|0 overrides/i);
    expect(within(layerAccordion).getByLabelText("1 field")).toHaveTextContent("1");
    expect(within(layerAccordion).getByLabelText("1 field")).not.toHaveAttribute("tabindex");
    expect(within(layerAccordion).getByLabelText("0 overrides")).toHaveTextContent("0");
    expect(within(layerAccordion).getByLabelText("0 overrides")).not.toHaveAttribute("tabindex");
  });

  it("shows popup config metric tooltips on hover and focus", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const fieldMetric = within(dialog).getByLabelText("4 fields");
    const overrideMetric = within(dialog).getAllByLabelText("0 overrides")[0];

    expect(fieldMetric).toHaveAttribute("tabindex", "0");
    expect(overrideMetric).toHaveAttribute("tabindex", "0");

    await user.click(fieldMetric);
    expect(within(dialog).getByRole("tooltip")).toHaveTextContent("Fields");

    await user.click(within(dialog).getByLabelText(/hidden dim/i));
    expect(within(dialog).queryByRole("tooltip")).not.toBeInTheDocument();

    await user.hover(overrideMetric);
    expect(within(dialog).getByRole("tooltip")).toHaveTextContent("Overrides");

    await user.unhover(overrideMetric);
    expect(within(dialog).queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("global section toggle closes and opens every popup config accordion", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const gateAccordion = within(dialog).getByRole("button", {
      name: /gate stack options section, 1 field, 0 overrides/i,
    });

    const closeAllButton = within(sectionNav).getByRole("button", { name: /^close all$/i });
    expect(closeAllButton).toBeInTheDocument();
    expect(closeAllButton).toHaveClass("whitespace-nowrap", "min-w-[5.75rem]");

    await user.click(closeAllButton);

    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");
    expect(gateAccordion).toHaveAttribute("aria-expanded", "false");
    expect(within(sectionNav).getByRole("button", { name: /^open all$/i })).toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/stack activation/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByRole("switch", { name: /gate flag/i })).not.toBeInTheDocument();

    await user.click(within(sectionNav).getByRole("button", { name: /^open all$/i }));

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(gateAccordion).toHaveAttribute("aria-expanded", "true");
    expect(within(sectionNav).getByRole("button", { name: /^close all$/i })).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/stack activation/i)).toBeInTheDocument();
    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
  });

  it("global section toggle opens every popup config accordion from a partial state", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const gateAccordion = within(dialog).getByRole("button", {
      name: /gate stack options section, 1 field, 0 overrides/i,
    });

    await user.click(layerAccordion);

    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");
    expect(gateAccordion).toHaveAttribute("aria-expanded", "true");
    expect(within(sectionNav).getByRole("button", { name: /^open all$/i })).toBeInTheDocument();

    await user.click(within(sectionNav).getByRole("button", { name: /^open all$/i }));

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(gateAccordion).toHaveAttribute("aria-expanded", "true");
    expect(within(sectionNav).getByRole("button", { name: /^close all$/i })).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(within(dialog).getByLabelText(/stack activation/i)).toBeInTheDocument();
    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
  });

  it("collapsing a popup config section hides its field controls", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });

    await user.click(layerAccordion);

    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");
    expect(layerAccordion).toHaveClass("bg-white/[0.025]");
    expect(layerAccordion.closest("section")).toHaveClass(
      "rounded-[12px]",
      "border-line-soft",
      "bg-panel/70",
    );
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(within(dialog).queryByLabelText(/stack activation/i)).not.toBeInTheDocument();
    expect(within(dialog).getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
  });

  it("sidebar section clicks reopen collapsed sections and scroll to them", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });

    await user.click(layerAccordion);
    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");

    await user.click(
      within(sectionNav).getByRole("button", { name: /jump to layer stack options/i }),
    );

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(scrollIntoViewMock).toHaveBeenCalledWith({
      block: "start",
      behavior: "smooth",
    });
  });

  it("sidebar section triggers toggle their matching accordion", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });

    await user.click(
      within(sectionNav).getByRole("button", { name: /close layer stack options/i }),
    );

    expect(layerAccordion).toHaveAttribute("aria-expanded", "false");
    expect(
      within(sectionNav).getByRole("button", { name: /open layer stack options/i }),
    ).toHaveAttribute("aria-expanded", "false");
    expect(within(dialog).queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
    expect(scrollIntoViewMock).not.toHaveBeenCalled();

    await user.click(
      within(sectionNav).getByRole("button", { name: /open layer stack options/i }),
    );

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(
      within(sectionNav).getByRole("button", { name: /close layer stack options/i }),
    ).toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(scrollIntoViewMock).not.toHaveBeenCalled();
  });

  it("editing a popup field updates overrides", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });

    const hiddenDimInput = await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    const hiddenDimRow = configFieldRowFor(hiddenDimInput);
    const overrideBadge = within(hiddenDimRow).getByText("override");
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 1 override/i,
    });
    const layerSection = fullConfigSectionFor(layerAccordion);
    const layerNavRow = fullConfigSectionNavRowFor(
      sectionNav,
      /jump to layer stack options/i,
    );

    expect(within(dialog).getAllByLabelText("1 override").length).toBeGreaterThan(0);
    expect(within(dialog).queryByText("1 override")).not.toBeInTheDocument();
    expect(layerAccordion).toHaveClass("bg-violet/[0.08]", "hover:bg-violet/[0.12]");
    expect(within(layerAccordion).getByLabelText("1 override")).toHaveClass("text-violet");
    expect(within(layerAccordion).queryByText("1 preset")).not.toBeInTheDocument();
    expect(layerSection).toHaveClass("border-violet/35", "bg-violet/[0.06]");
    expect(layerSection).not.toHaveClass("border-amber/35", "bg-amber/[0.045]");
    expect(layerNavRow).toHaveClass(
      "border-violet/30",
      "bg-violet/[0.055]",
      "hover:bg-violet/15",
    );
    expect(layerNavRow).not.toHaveClass("border-amber/30", "bg-amber/[0.055]");
    expect(within(layerNavRow).getByLabelText("1 override")).toHaveClass("text-violet");
    expect(within(layerNavRow).queryByText("1 preset")).not.toBeInTheDocument();
    expect(overrideBadge).toBeInTheDocument();
    expect(overrideBadge).toHaveClass("text-violet");
    expect(hiddenDimRow).toHaveClass("border-violet/40");
    expect(hiddenDimRow).not.toHaveClass("border-amber/55", "bg-amber/[0.055]");
    expect(within(hiddenDimRow).queryByText("preset")).not.toBeInTheDocument();
    expect(within(dialog).queryByText(/\d+ preset/i)).not.toBeInTheDocument();
    expect(screen.getAllByText("hidden dim")).toHaveLength(1);
  });

  it("highlights a section gradient when it has an override and a preset-owned field", async () => {
    installFetchMock({
      schemaResponse: {
        ...schemaResponse,
        fields: schemaResponse.fields.map((field) =>
          field.key === "stack_activation"
            ? {
                ...field,
                locked: true,
                lockedValue: "GELU",
                lockedReason:
                  "Locked by the ACTIVATION preset because this preset fixes stack activation.",
              }
            : field,
        ),
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const sectionNav = within(dialog).getByRole("navigation", {
      name: /full config sections/i,
    });

    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");

    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 1 override, 1 preset/i,
    });
    const layerSection = fullConfigSectionFor(layerAccordion);
    const layerNavRow = fullConfigSectionNavRowFor(
      sectionNav,
      /jump to layer stack options/i,
    );

    expect(layerAccordion).toHaveClass(
      "bg-[linear-gradient(90deg,rgba(255,209,102,0.12),rgba(167,139,250,0.13))]",
      "hover:bg-[linear-gradient(90deg,rgba(255,209,102,0.16),rgba(167,139,250,0.17))]",
    );
    expect(within(layerAccordion).getByLabelText("1 override")).toHaveClass("text-violet");
    expect(within(layerAccordion).getByText("1 preset")).toHaveClass("text-amber");
    expect(layerSection).toHaveClass(
      "border-amber/35",
      "bg-[linear-gradient(135deg,rgba(255,209,102,0.075),rgba(167,139,250,0.105))]",
      "ring-violet/25",
    );
    expect(layerSection).not.toHaveClass("bg-amber/[0.045]", "bg-violet/[0.06]");
    expect(layerNavRow).toHaveClass(
      "border-amber/35",
      "bg-[linear-gradient(90deg,rgba(255,209,102,0.075),rgba(167,139,250,0.095))]",
      "hover:bg-[linear-gradient(90deg,rgba(255,209,102,0.11),rgba(167,139,250,0.13))]",
      "ring-violet/20",
    );
    expect(within(layerNavRow).getByLabelText("1 override")).toHaveClass("text-violet");
    expect(within(layerNavRow).getByText("1 preset")).toHaveClass("text-amber");
  });

  it("renders stack layer count as an editable numeric default", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const layerAccordion = within(dialog).getByRole("button", {
      name: /layer stack options section, 3 fields, 0 overrides/i,
    });
    const stackLayersInput = within(dialog).getByLabelText(/stack num layers/i);

    expect(layerAccordion).toHaveAttribute("aria-expanded", "true");
    expect(stackLayersInput).toHaveAttribute("type", "number");
    expect(stackLayersInput).toHaveValue(5);

    await user.clear(stackLayersInput);
    await user.type(stackLayersInput, "7");

    expect(stackLayersInput).toHaveValue(7);
    expect(within(dialog).getAllByLabelText("1 override").length).toBeGreaterThan(0);
  });

  it("popup Update Preview posts selected model, preset, and overrides", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    await user.click(within(dialog).getByRole("button", { name: /update preview/i }));

    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        model: "linear",
        preset: "baseline",
        dataset: "Cifar10",
        overrides: { hidden_dim: "128" },
      });
    });
    expect(screen.getByRole("dialog", { name: /full configuration/i })).toBeInTheDocument();
  });

  it("opens a training command popup without closing full config", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const commandDialog = await openTrainingCommand(user, dialog);

    expect(commandDialog).toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: /full configuration/i })).toBeInTheDocument();
    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh linear --preset baseline",
    );
  });

  it("uses the current selected preset and omits --config when there are no overrides", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await waitForTargetValue("preset", "baseline");
    await selectTargetOption(user, "preset", "recurrent-gating-halting");
    const dialog = await openFullConfig(user);
    const commandDialog = await openTrainingCommand(user, dialog);

    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh linear --preset recurrent-gating-halting",
    );
    expect((commandField(commandDialog) as HTMLTextAreaElement).value).not.toContain("--config");
  });

  it("includes live overrides in display order before Update Preview", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    await user.selectOptions(within(dialog).getByLabelText(/stack activation/i), "RELU");
    await user.click(within(dialog).getByRole("switch", { name: /gate flag/i }));

    const commandDialog = await openTrainingCommand(user, dialog);

    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh linear --preset baseline --config --hidden-dim 128 --stack-activation RELU --gate-flag true",
    );
  });

  it("updates the training command after resetting overrides", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    let commandDialog = await openTrainingCommand(user, dialog);
    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh linear --preset baseline --config --hidden-dim 128",
    );

    await user.click(within(commandDialog).getByRole("button", { name: /close training command/i }));
    await user.click(within(dialog).getByRole("button", { name: /reset overrides/i }));
    commandDialog = await openTrainingCommand(user, dialog);

    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh linear --preset baseline",
    );
  });

  it("shell-quotes override values and serializes nullable empty overrides as None", async () => {
    installFetchMock({
      schemaResponse: {
        ...schemaResponse,
        fields: [
          ...schemaResponse.fields,
          {
            key: "dropout_schedule",
            configKey: "DROPOUT_SCHEDULE",
            flag: "--dropout-schedule",
            label: "dropout schedule",
            section: "Layer Stack Options",
            type: "enum",
            default: null,
            nullable: true,
            choices: ["cosine decay"],
          },
        ],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const scheduleInput = within(dialog).getByLabelText(/dropout schedule/i);
    await user.selectOptions(scheduleInput, "cosine decay");
    let commandDialog = await openTrainingCommand(user, dialog);
    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh linear --preset baseline --config --dropout-schedule 'cosine decay'",
    );

    await user.click(within(commandDialog).getByRole("button", { name: /close training command/i }));
    await user.selectOptions(scheduleInput, "");
    commandDialog = await openTrainingCommand(user, dialog);

    expect(commandField(commandDialog)).toHaveValue(
      "source experiment.sh linear --preset baseline --config --dropout-schedule None",
    );
  });

  it("copies the exact training command to the clipboard", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    Object.defineProperty(navigator, "clipboard", {
      configurable: true,
      value: { writeText },
    });
    installFetchMock();
    renderViewer();

    const dialog = await openFullConfig(user);
    await typeConfigFieldValue(user, dialog, /hidden dim/i, "128");
    const commandDialog = await openTrainingCommand(user, dialog);
    const expectedCommand = "source experiment.sh linear --preset baseline --config --hidden-dim 128";

    await user.click(within(commandDialog).getByRole("button", { name: /copy command/i }));

    expect(writeText).toHaveBeenCalledWith(expectedCommand);
    expect(within(commandDialog).getByRole("status")).toHaveTextContent("Command copied");
  });

  it("closing the training command popup leaves the full config dialog open", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const dialog = await openFullConfig(user);
    const commandDialog = await openTrainingCommand(user, dialog);

    await user.click(within(commandDialog).getByRole("button", { name: /close training command/i }));

    expect(screen.queryByRole("dialog", { name: /training command/i })).not.toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: /full configuration/i })).toBeInTheDocument();
  });

  it("popup Reset Overrides clears override state", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });
    const hiddenInput = within(dialog).getByLabelText(/hidden dim/i);

    await user.clear(hiddenInput);
    await user.type(hiddenInput, "128");
    expect(hiddenInput).toHaveValue(128);

    await user.click(within(dialog).getByRole("button", { name: /reset overrides/i }));

    expect(hiddenInput).toHaveValue(256);
    expect(within(dialog).getAllByLabelText("0 overrides").length).toBeGreaterThan(0);
    expect(screen.queryByText("No overrides set")).not.toBeInTheDocument();
  });

  it("close button removes the full config dialog", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await waitForOpenFullConfigButton());
    const dialog = await screen.findByRole("dialog", { name: /full configuration/i });

    await user.click(within(dialog).getByRole("button", { name: /close full config/i }));

    expect(screen.queryByRole("dialog", { name: /full configuration/i })).not.toBeInTheDocument();
  });

  it("renders graph nodes progressively by default", async () => {
    installFetchMock();
    renderViewer();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    expect(screen.getByTestId("edge-model-main_model.0")).toBeInTheDocument();
    expect(screen.queryByText("main_model.0.model")).not.toBeInTheDocument();
    expect(screen.queryByText("Dropout")).not.toBeInTheDocument();
    expect(screen.queryByText("LayerNorm")).not.toBeInTheDocument();
    expect(screen.queryByText("CrossEntropyLoss")).not.toBeInTheDocument();
  });

  it("shows child count beside the graph card title", async () => {
    installFetchMock({ inspectResponse: repeatedLayersInspectResponse });
    renderViewer();

    const modelNode = await screen.findByTestId("node-model");
    expect(within(modelNode).getByText("3 children")).toBeInTheDocument();
  });

  it("renders total params in the right sidebar summary", async () => {
    installFetchMock();
    renderViewer();

    const heading = await screen.findByRole("heading", { name: /node details/i });
    const sidebar = heading.closest("aside");
    if (!sidebar) {
      throw new Error("Expected Node Details heading to render inside the sidebar");
    }

    const paramsValue = await within(sidebar).findByTitle("65,792 parameters");
    const paramsCard = paramsValue.parentElement;
    if (!paramsCard) {
      throw new Error("Expected total params value to render inside a summary card");
    }

    expect(within(paramsCard).getByText(/^Params$/)).toBeInTheDocument();
    expect(paramsValue).toHaveTextContent("65.8K");
    expect(within(sidebar).queryByText(/^Nodes$/)).not.toBeInTheDocument();
    expect(within(sidebar).queryByText(/^Edges$/)).not.toBeInTheDocument();
  });

  it("renders graph parameter badges only for nodes with parameters", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const layerNode = await screen.findByTestId("node-main_model.0");
    expect(within(layerNode).getByTitle("33,024 parameters")).toHaveTextContent("33K");

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );

    const gateNode = await screen.findByTestId("node-main_model.0.gate_model");
    expect(within(gateNode).queryByTitle(/parameters/)).not.toBeInTheDocument();
  });

  it("renders layer dims on the inner-model graph-card row", async () => {
    installFetchMock();
    renderViewer();

    const layerNode = await screen.findByTestId("node-main_model.0");
    const matchingSummaries = within(layerNode).getAllByLabelText("LinearLayer 128 -> 128");
    const summary = matchingSummaries[0];
    expect(matchingSummaries).toHaveLength(2);
    expect(summary).toHaveAttribute("title", "LinearLayer 128 -> 128");
    expect(within(summary).getByText("LinearLayer")).toBeInTheDocument();
    expect(within(summary).getByText("128 -> 128")).toBeInTheDocument();
    expect(within(layerNode).queryByText("shapeTransition")).not.toBeInTheDocument();
  });

  it("renders weight and bias shapes only on the graph cards that own them", async () => {
    installFetchMock({ inspectResponse: parameterShapeInspectResponse });
    renderViewer();
    const user = userEvent.setup();

    const parentNode = await screen.findByTestId("node-main_model.0");
    expect(within(parentNode).queryByTestId("parameter-shapes-main_model.0")).not.toBeInTheDocument();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );

    const ownerNode = await screen.findByTestId("node-main_model.0.model");
    const shapes = within(ownerNode).getByTestId("parameter-shapes-main_model.0.model");
    expect(within(shapes).getByLabelText("W shape 128 x 128")).toBeInTheDocument();
    expect(within(shapes).getByLabelText("b shape 128")).toBeInTheDocument();
    expect(within(screen.getByTestId("node-main_model.0")).queryByText("128 x 128")).not.toBeInTheDocument();
  });

  it("renders repeated layer children as separate graph card pills", async () => {
    installFetchMock({ inspectResponse: repeatedLayersInspectResponse });
    renderViewer();

    const modelNode = await screen.findByTestId("node-model");
    expect(within(modelNode).getByLabelText("Layer 0 LinearLayer")).toBeInTheDocument();
    expect(within(modelNode).getByLabelText("Layer 1 LinearLayer")).toBeInTheDocument();
    expect(within(modelNode).getByLabelText("Layer 2 LinearLayer")).toBeInTheDocument();
    expect(within(modelNode).queryByText("Layer -> LinearLayer")).not.toBeInTheDocument();
  });

  it("summarizes long layer stacks with an ellipsis and total count", async () => {
    installFetchMock({ inspectResponse: manyRepeatedLayersInspectResponse });
    renderViewer();

    const modelNode = await screen.findByTestId("node-model");
    expect(within(modelNode).getByLabelText("Layer 0 LinearLayer")).toBeInTheDocument();
    expect(within(modelNode).getByLabelText("Layer 4 LinearLayer")).toBeInTheDocument();
    expect(within(modelNode).queryByLabelText("Layer 5 LinearLayer")).not.toBeInTheDocument();
    expect(within(modelNode).getByTitle("4 more layers")).toHaveTextContent("...");
    expect(within(modelNode).getByTitle("9 layers total")).toHaveTextContent("9 layers");
  });

  it("builds basic-mode child summaries from the filtered graph", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const layerNode = await screen.findByTestId("node-main_model.0");
    const summaries = within(layerNode).getByTestId("child-summaries-main_model.0");
    expect(within(summaries).getAllByText("LinearLayer")).toHaveLength(2);
    expect(within(layerNode).getByText("Gate")).toBeInTheDocument();
    expect(within(layerNode).queryByText("Dropout")).not.toBeInTheDocument();
    expect(within(layerNode).queryByText("LayerNorm")).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: /full/i }));

    const fullLayerNode = screen.getByTestId("node-main_model.0");
    expect(within(fullLayerNode).getByText("Dropout")).toBeInTheDocument();
    expect(within(fullLayerNode).getByText("LayerNorm")).toBeInTheDocument();
    expect(within(fullLayerNode).getByText("SelfAttentionProcessor")).toBeInTheDocument();
  });

  it("renders simple graph cards with inline metrics only", async () => {
    installFetchMock({ inspectResponse: parameterShapeInspectResponse });
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    expect(screen.getByText("dims: 128 -> 128")).toBeInTheDocument();
    const expandedLayerNode = await screen.findByTestId("node-main_model.0");
    await user.click(
      within(expandedLayerNode).getByRole("button", { name: /details for main_model\.0/i }),
    );
    expect(within(screen.getByTestId("node-main_model.0")).getByText("activation"))
      .toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: /simple/i }));

    const layerNode = await screen.findByTestId("node-main_model.0");
    expect(within(layerNode).getByText("Layer")).toBeInTheDocument();
    expect(
      within(layerNode).getByRole("button", { name: /^collapse tree main_model\.0$/i }),
    ).toBeInTheDocument();
    expect(within(layerNode).queryByText("main_model.0")).not.toBeInTheDocument();
    expect(within(layerNode).getByTitle("33,024 parameters")).toHaveTextContent("33K params");
    expect(within(layerNode).getByTitle("input/output: 128 -> 128")).toHaveTextContent(
      "128 -> 128",
    );
    expect(within(layerNode).queryByText("3 children")).not.toBeInTheDocument();
    expect(within(layerNode).queryByTestId("child-summaries-main_model.0")).not.toBeInTheDocument();
    expect(within(layerNode).queryByTestId("parameter-shapes-main_model.0")).not.toBeInTheDocument();
    expect(
      within(layerNode).queryByRole("button", { name: /details for main_model\.0/i }),
    ).not.toBeInTheDocument();
    expect(within(layerNode).queryByText("activation")).not.toBeInTheDocument();
    expect(screen.getByText("dims: 128 -> 128")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: /basic/i }));

    const basicLayerNode = screen.getByTestId("node-main_model.0");
    expect(within(basicLayerNode).getByRole("button", { name: /details for main_model\.0/i }))
      .toBeInTheDocument();
    expect(within(basicLayerNode).getByText("activation")).toBeInTheDocument();
    expect(within(basicLayerNode).getByTestId("child-summaries-main_model.0"))
      .toBeInTheDocument();
  });

  it("renders stack-derived dims on simple stack container cards", async () => {
    installFetchMock({ inspectResponse: stackContainerInspectResponse });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("tab", { name: /simple/i }));

    const stackNode = await screen.findByTestId("node-main_model");
    expect(within(stackNode).getByText("Main Model")).toBeInTheDocument();
    expect(within(stackNode).getByTitle("65,792 parameters")).toHaveTextContent("65.8K params");
    expect(within(stackNode).getByTitle("input/output: 256 -> 10")).toHaveTextContent(
      "256 -> 10",
    );
    expect(within(stackNode).queryByTestId("stack-diagram-main_model")).not.toBeInTheDocument();
    expect(within(stackNode).queryByTestId("child-summaries-main_model")).not.toBeInTheDocument();
    expect(
      within(stackNode).queryByRole("button", { name: /details for main_model/i }),
    ).not.toBeInTheDocument();
  });

  it("expands simple-mode cards using the basic graph topology", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("tab", { name: /simple/i }));

    const layerNode = await screen.findByTestId("node-main_model.0");
    await user.click(
      within(layerNode).getByRole("button", { name: /^expand tree main_model\.0$/i }),
    );

    const modelNode = await screen.findByTestId("node-main_model.0.model");
    const projectedNode = await screen.findByTestId("node-main_model.0.processor.projection");
    expect(within(modelNode).getByText("LinearLayer")).toBeInTheDocument();
    expect(within(modelNode).queryByText("main_model.0.model")).not.toBeInTheDocument();
    expect(within(modelNode).getByTitle("16,512 parameters")).toHaveTextContent("16.5K params");
    expect(within(modelNode).getByTitle("input/output: 128 -> 128")).toHaveTextContent(
      "128 -> 128",
    );
    expect(within(projectedNode).getByText("LinearLayer")).toBeInTheDocument();
    expect(within(projectedNode).getByTitle("16,512 parameters")).toHaveTextContent(
      "16.5K params",
    );
    expect(screen.queryByTestId("node-main_model.0.processor")).not.toBeInTheDocument();
    expect(screen.queryByTestId("node-main_model.0.dropout_module")).not.toBeInTheDocument();
    expect(screen.queryByTestId("node-main_model.0.layer_norm_module")).not.toBeInTheDocument();
  });

  it("renders enabled gate and halting metadata as child summary rows", async () => {
    installFetchMock({ inspectResponse: mechanismMetadataInspectResponse });
    renderViewer();

    const controllerNode = await screen.findByTestId("node-controller");
    expect(within(controllerNode).getByText("Gate")).toBeInTheDocument();
    expect(within(controllerNode).getByText("Halting mechanism")).toBeInTheDocument();
  });

  it("does not duplicate mechanism summaries when matching child nodes exist", async () => {
    installFetchMock({ inspectResponse: mechanismChildrenInspectResponse });
    renderViewer();

    const controllerNode = await screen.findByTestId("node-controller");
    const summaries = within(controllerNode).getByTestId("child-summaries-controller");
    expect(within(summaries).getAllByText("Gate")).toHaveLength(1);
    expect(within(summaries).getAllByText("Halting mechanism")).toHaveLength(1);
  });

  it("uses child summary rows when computing graph card height", async () => {
    installFetchMock({ inspectResponse: tallSummaryInspectResponse });
    renderViewer();

    const blockNode = await screen.findByTestId("node-block");
    expect(Number(blockNode.getAttribute("data-height"))).toBeGreaterThan(132);
    expect(within(blockNode).getByText("LinearLayer")).toBeInTheDocument();
    expect(within(blockNode).getByText("AttentionLayer")).toBeInTheDocument();
    expect(within(blockNode).getByText("Embedding")).toBeInTheDocument();
    expect(within(blockNode).getByText("OutputHead")).toBeInTheDocument();
  });

  it("expands graph nodes and collapses the opened graph", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const layerCard = await screen.findByRole("button", {
      name: /select and expand main_model\.0/i,
    });
    await user.click(layerCard);
    expect(layerCard).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("dims: 128 -> 128")).toBeInTheDocument();
    expect(await screen.findByText("main_model.0.model")).toBeInTheDocument();
    expect(screen.getByTestId("edge-main_model.0-main_model.0.model")).toBeInTheDocument();
    expect(await screen.findByText("Gate Model")).toBeInTheDocument();
    expect(screen.getByText("Sequential · main_model.0.gate_model")).toBeInTheDocument();
    expect(await screen.findByText("main_model.0.processor.projection")).toBeInTheDocument();
    expect(
      screen.getByTestId("edge-main_model.0-main_model.0.processor.projection"),
    ).toBeInTheDocument();
    expect(screen.queryByText("SelfAttentionProcessor")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /collapse all/i }));
    expect(screen.queryByText("main_model.0.model")).not.toBeInTheDocument();
    expect(screen.queryByText("main_model.0.processor.projection")).not.toBeInTheDocument();
  });

  it("collapses an expanded graph card when its body is clicked again", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    expect(await screen.findByTestId("node-main_model.0.model")).toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: /select and collapse main_model\.0/i }),
    ).toHaveAttribute("aria-expanded", "true");

    await user.click(screen.getByRole("button", { name: /select and collapse main_model\.0/i }));

    expect(screen.queryByTestId("node-main_model.0.model")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /select and expand main_model\.0/i }))
      .toHaveAttribute("aria-expanded", "false");
    expect(screen.getByText("dims: 128 -> 128")).toBeInTheDocument();
  });

  it("keeps root graph cards expanded when their body is clicked", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const rootCard = await screen.findByRole("button", { name: /^select model$/i });
    expect(await screen.findByTestId("node-main_model.0")).toBeInTheDocument();

    await user.click(rootCard);

    expect(screen.getByTestId("node-main_model.0")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /select and collapse model/i }),
    ).not.toBeInTheDocument();
  });

  it("selects leaf graph cards without changing expansion", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );

    await user.click(
      await screen.findByRole("button", { name: /^select main_model\.0\.model$/i }),
    );

    expect(screen.getByTestId("node-main_model.0.model")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /select and collapse main_model\.0/i }))
      .toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("16,512")).toBeInTheDocument();
  });

  it("expands another card on the first body click after one card is already open", async () => {
    installFetchMock({ inspectResponse: repeatedLayersInspectResponse });
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    expect(await screen.findByTestId("node-main_model.0.model")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /select and expand main_model\.1/i }));

    expect(await screen.findByTestId("node-main_model.1.model")).toBeInTheDocument();
  });

  it("expands and collapses the current-mode subtree from the chevron", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(screen.getByRole("tab", { name: /full/i }));
    expect(await screen.findByTestId("node-loss_fn")).toBeInTheDocument();

    const layerNode = await screen.findByTestId("node-main_model.0");
    await user.click(
      within(layerNode).getByRole("button", { name: /^expand tree main_model\.0$/i }),
    );
    expect(await screen.findByTestId("node-main_model.0.model")).toBeInTheDocument();
    expect(await screen.findByTestId("node-main_model.0.processor")).toBeInTheDocument();
    expect(
      await screen.findByTestId("node-main_model.0.processor.projection"),
    ).toBeInTheDocument();

    const expandedLayerNode = screen.getByTestId("node-main_model.0");
    await user.click(
      within(expandedLayerNode).getByRole("button", { name: /^collapse tree main_model\.0$/i }),
    );
    expect(screen.queryByTestId("node-main_model.0.model")).not.toBeInTheDocument();
    expect(screen.queryByTestId("node-main_model.0.processor")).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("node-main_model.0.processor.projection"),
    ).not.toBeInTheDocument();

    const collapsedLayerNode = screen.getByTestId("node-main_model.0");
    await user.click(
      within(collapsedLayerNode).getByRole("button", {
        name: /select and expand main_model\.0/i,
      }),
    );
    expect(await screen.findByTestId("node-main_model.0.model")).toBeInTheDocument();
    expect(await screen.findByTestId("node-main_model.0.processor")).toBeInTheDocument();
    expect(
      screen.queryByTestId("node-main_model.0.processor.projection"),
    ).not.toBeInTheDocument();
  });

  it("keeps chevron expansion in basic detail mode and does not select the card", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const layerNode = await screen.findByTestId("node-main_model.0");
    await user.click(
      within(layerNode).getByRole("button", { name: /^expand tree main_model\.0$/i }),
    );

    expect(await screen.findByTestId("node-main_model.0.model")).toBeInTheDocument();
    expect(
      await screen.findByTestId("node-main_model.0.processor.projection"),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("node-main_model.0.processor")).not.toBeInTheDocument();
    expect(screen.queryByTestId("node-main_model.0.dropout_module")).not.toBeInTheDocument();
    expect(screen.queryByTestId("node-main_model.0.layer_norm_module")).not.toBeInTheDocument();
    expect(screen.queryByText("BEFORE")).not.toBeInTheDocument();
  });

  it("keeps opened graph cards visibly separated", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );

    const directChildY = Number(
      screen.getByTestId("node-main_model.0.model").getAttribute("data-y"),
    );
    const projectedChildY = Number(
      screen.getByTestId("node-main_model.0.processor.projection").getAttribute("data-y"),
    );

    expect(Math.abs(projectedChildY - directChildY)).toBeGreaterThanOrEqual(148);
  });

  it("opens card details without expanding the subgraph", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    const layerNode = screen.getByTestId("node-main_model.0");
    await user.click(
      within(layerNode).getByRole("button", { name: /details for main_model\.0/i }),
    );

    expect(within(layerNode).queryByText("dims")).not.toBeInTheDocument();
    expect(within(layerNode).getByText("activation")).toBeInTheDocument();
    expect(within(layerNode).getByText("dropout")).toBeInTheDocument();
    expect(screen.queryByText("main_model.0.model")).not.toBeInTheDocument();
  });

  it("relayouts graph cards when details change card height", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );

    const layerNode = screen.getByTestId("node-main_model.0");
    const initialHeight = Number(layerNode.getAttribute("data-height"));
    const initialY = Number(layerNode.getAttribute("data-y"));

    await user.click(
      within(layerNode).getByRole("button", { name: /details for main_model\.0/i }),
    );

    const expandedLayerNode = screen.getByTestId("node-main_model.0");
    expect(Number(expandedLayerNode.getAttribute("data-height"))).toBeGreaterThan(initialHeight);
    expect(Number(expandedLayerNode.getAttribute("data-y"))).not.toBe(initialY);
  });

  it("collapse all keeps card detail accordions open", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const layerNode = await screen.findByTestId("node-main_model.0");
    await user.click(
      within(layerNode).getByRole("button", { name: /details for main_model\.0/i }),
    );
    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    expect(await screen.findByText("main_model.0.model")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /collapse all/i }));

    const visibleLayerNode = screen.getByTestId("node-main_model.0");
    expect(within(visibleLayerNode).getByText("activation")).toBeInTheDocument();
    expect(screen.queryByText("main_model.0.model")).not.toBeInTheDocument();
  });

  it("can switch between opened and entire graph scopes", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    expect(screen.queryByText("main_model.0.model")).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: /entire/i }));
    expect(await screen.findByText("main_model.0.model")).toBeInTheDocument();
    expect(await screen.findByText("main_model.0.processor.projection")).toBeInTheDocument();
    expect(
      screen.getByTestId("edge-main_model.0-main_model.0.processor.projection"),
    ).toBeInTheDocument();
    expect(screen.queryByText("Dropout")).not.toBeInTheDocument();
    expect(screen.queryByText("LayerNorm")).not.toBeInTheDocument();
    expect(screen.queryByText("CrossEntropyLoss")).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: /opened/i }));
    expect(screen.queryByText("main_model.0.model")).not.toBeInTheDocument();
  });

  it("selects cards in entire scope without changing opened-scope expansion", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    await user.click(screen.getByRole("tab", { name: /entire/i }));

    await user.click(await screen.findByRole("button", { name: /^select main_model\.0$/i }));
    expect(screen.getByText("dims: 128 -> 128")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: /opened/i }));
    expect(screen.queryByText("main_model.0.model")).not.toBeInTheDocument();
  });

  it("switches between basic and full graph detail", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    expect(await screen.findByText("main_model.0.model")).toBeInTheDocument();
    expect(screen.queryByText("Dropout")).not.toBeInTheDocument();
    expect(screen.queryByText("LayerNorm")).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: /full/i }));
    expect(await screen.findByTestId("node-main_model.0.dropout_module")).toBeInTheDocument();
    expect(await screen.findByTestId("node-main_model.0.layer_norm_module")).toBeInTheDocument();
    expect(await screen.findByTestId("node-main_model.0.processor")).toBeInTheDocument();
    expect(
      screen.queryByTestId("node-main_model.0.processor.projection"),
    ).not.toBeInTheDocument();
    expect(await screen.findByTestId("node-loss_fn")).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: /basic/i }));
    expect(screen.queryByText("Dropout")).not.toBeInTheDocument();
    expect(screen.queryByText("LayerNorm")).not.toBeInTheDocument();
    expect(screen.queryByText("SelfAttentionProcessor")).not.toBeInTheDocument();
    expect(screen.queryByText("CrossEntropyLoss")).not.toBeInTheDocument();
  });

  it("shows the complete graph only in full entire mode", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("tab", { name: /entire/i }));
    expect(await screen.findByText("main_model.0.model")).toBeInTheDocument();
    expect(screen.queryByText("Dropout")).not.toBeInTheDocument();
    expect(screen.queryByText("LayerNorm")).not.toBeInTheDocument();
    expect(screen.queryByText("CrossEntropyLoss")).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: /full/i }));
    expect(await screen.findByTestId("node-main_model.0.dropout_module")).toBeInTheDocument();
    expect(await screen.findByTestId("node-main_model.0.layer_norm_module")).toBeInTheDocument();
    expect(await screen.findByTestId("node-loss_fn")).toBeInTheDocument();
    expect(await screen.findByTestId("node-metrics")).toBeInTheDocument();
    expect(await screen.findByText("main_model.0.processor.projection")).toBeInTheDocument();
  });

  it("collapse all keeps the selected graph detail mode", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    await user.click(screen.getByRole("tab", { name: /full/i }));
    expect(await screen.findByTestId("node-main_model.0.dropout_module")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /collapse all/i }));
    expect(screen.getByRole("tab", { name: /full/i })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    expect(screen.queryByTestId("node-main_model.0.dropout_module")).not.toBeInTheDocument();
    expect(
      within(screen.getByTestId("node-main_model.0")).getByText("Dropout"),
    ).toBeInTheDocument();
  });

  it("renders graph-only preview controls with the locations card", async () => {
    installFetchMock();
    renderViewer();

    expect(await screen.findByTestId("flow")).toBeInTheDocument();
    expect(screen.queryByRole("tablist", { name: /visualization mode/i })).not.toBeInTheDocument();
    expect(screen.getAllByRole("tab").map((tab) => tab.textContent)).toEqual([
      "Simple",
      "Basic",
      "Full",
      "Opened",
      "Entire",
    ]);
    expect(screen.getByRole("tablist", { name: /graph detail/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /basic/i })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tablist", { name: /graph scope/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /opened/i })).toHaveAttribute("aria-selected", "true");
    const panel = screen.getByTestId("flow-panel-bottom-right");

    expect(screen.queryByTestId("model-structure-card")).not.toBeInTheDocument();
    expect(screen.queryByText("Model Structure")).not.toBeInTheDocument();
    expect(within(panel).getByTestId("graph-locations-card")).toBeInTheDocument();
    expect(panel).toHaveClass("nodrag", "nopan", "hidden", "w-[312px]", "xl:block");
    expect(panel).toHaveStyle({ right: "28px", bottom: "24px" });
    expect(screen.getByText("No analysed locations found.")).toBeInTheDocument();
  });

  it("updates the locations card with full-mode-only location nodes", async () => {
    installFetchMock({ inspectResponse: locationInspectResponse });
    renderViewer();
    const user = userEvent.setup();

    const card = await screen.findByTestId("graph-locations-card");
    const clusterDiagram = await screen.findByTestId("cluster-diagram-model.cluster");
    expect(within(clusterDiagram).getByText("Cluster map")).toBeInTheDocument();
    expect(within(clusterDiagram).getByText("1 / 4")).toBeInTheDocument();
    expect(
      within(clusterDiagram).getByLabelText(/Neuron \(1, 1, 1\).*active/i),
    ).toBeInTheDocument();
    expect(
      await within(card).findByRole("group", { name: /^model\.cluster locations$/i }),
    ).toBeInTheDocument();
    expect(
      within(card).queryByRole("group", {
        name: /^model\.cluster\.terminal locations$/i,
      }),
    ).not.toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: /full/i }));

    await waitFor(() => {
      expect(
        within(card).getByRole("group", {
          name: /^model\.cluster\.terminal locations$/i,
        }),
      ).toBeInTheDocument();
    });
    expect(
      within(card).getByRole("button", {
        name: /terminal position \(4, 4, 2\)/i,
      }),
    ).toBeInTheDocument();
  });

  it("reveals a graph node from a location and keeps node details selected", async () => {
    installFetchMock({ inspectResponse: locationInspectResponse });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("tab", { name: /full/i }));
    const card = await screen.findByTestId("graph-locations-card");
    const terminal = await within(card).findByRole("group", {
      name: /^model\.cluster\.terminal locations$/i,
    });
    await user.click(
      within(terminal).getByRole("button", {
        name: /reachable coordinate \(5, 4, 2\)/i,
      }),
    );

    expect(await screen.findByTestId("node-model.cluster.terminal")).toBeInTheDocument();
    expect(await screen.findByText("Sampler reach")).toBeInTheDocument();
  });

  it("displays selected node metadata", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    expect(screen.getByText("dims: 128 -> 128")).toBeInTheDocument();
    expect(screen.getByText("33,024")).toBeInTheDocument();
    expect(screen.getByText("BEFORE")).toBeInTheDocument();
  });

  it("opens selected-node monitor charts for the active training job", async () => {
    const { monitorDataRequests } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    await selectTrainingMonitorOption(user, /Linear layers/i);
    await selectNewTrainingLogFolder(user, "monitor_charts");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /^monitor charts$/i })).toBeEnabled(),
    );
    await user.click(screen.getByRole("button", { name: /^monitor charts$/i }));

    const dialog = await screen.findByRole("dialog", { name: /monitor charts/i });
    expect(dialog).toBeInTheDocument();
    await waitFor(() => {
      expect(monitorDataRequests[0]).toEqual({
        jobId: "job-1",
        nodePath: "main_model.0.model",
        preset: "baseline",
        dataset: "Mnist",
      });
    });
    expect(within(dialog).queryByLabelText(/^compare$/i)).not.toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: /Activations\s+1 chart/i }))
      .toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText("output/mean")).toBeInTheDocument();
    const visualSummaries = within(dialog).getByRole("button", {
      name: /Visual summaries\s+2 charts/i,
    });
    expect(visualSummaries).toHaveAttribute("aria-expanded", "false");
    await user.click(visualSummaries);
    expect(
      screen.getByAltText(
        "Monitor image for main_model.0.model/heatmap/usage_fraction at step 2",
      ),
    ).toBeInTheDocument();
  });

  it("groups selected-node monitor charts into semantic accordions", async () => {
    const { monitorDataRequests } = installFetchMock({
      monitorDataResponse: ({ nodePath }) => semanticMonitorPayload(nodePath),
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    await selectTrainingMonitorOption(user, /Linear layers/i);
    await selectNewTrainingLogFolder(user, "semantic_monitor_charts");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /^monitor charts$/i })).toBeEnabled(),
    );
    await user.click(screen.getByRole("button", { name: /^monitor charts$/i }));

    const dialog = await screen.findByRole("dialog", { name: /monitor charts/i });
    await waitFor(() => {
      expect(monitorDataRequests[0]).toEqual({
        jobId: "job-1",
        nodePath: "main_model.0.model",
        preset: "baseline",
        dataset: "Mnist",
      });
    });

    const activations = within(dialog).getByRole("button", {
      name: /Activations\s+1 chart/i,
    });
    const gradients = within(dialog).getByRole("button", {
      name: /Gradients\s+1 chart/i,
    });
    const bias = within(dialog).getByRole("button", { name: /Bias\s+1 chart/i });
    const weights = within(dialog).getByRole("button", {
      name: /Weights\s+1 chart/i,
    });
    const attention = within(dialog).getByRole("button", {
      name: /Attention\s+1 chart/i,
    });
    const recurrent = within(dialog).getByRole("button", {
      name: /Recurrent\s+1 chart/i,
    });
    const controllers = within(dialog).getByRole("button", {
      name: /Controllers\s+1 chart/i,
    });
    const parametric = within(dialog).getByRole("button", {
      name: /Parametric\s+1 chart/i,
    });
    const routing = within(dialog).getByRole("button", {
      name: /Routing\s+1 chart/i,
    });
    const visualSummaries = within(dialog).getByRole("button", {
      name: /Visual summaries\s+2 charts/i,
    });

    expect(activations).toHaveAttribute("aria-expanded", "true");
    expect(gradients).toHaveAttribute("aria-expanded", "false");
    expect(bias).toHaveAttribute("aria-expanded", "false");
    expect(weights).toHaveAttribute("aria-expanded", "false");
    expect(attention).toHaveAttribute("aria-expanded", "false");
    expect(recurrent).toHaveAttribute("aria-expanded", "false");
    expect(controllers).toHaveAttribute("aria-expanded", "false");
    expect(parametric).toHaveAttribute("aria-expanded", "false");
    expect(routing).toHaveAttribute("aria-expanded", "false");
    expect(visualSummaries).toHaveAttribute("aria-expanded", "false");
    expect(within(dialog).queryByRole("button", { name: /Other/i })).not.toBeInTheDocument();
    expect(within(dialog).getByText("input/mean")).toBeInTheDocument();
    expect(within(dialog).queryByText("bias/grad_mean")).not.toBeInTheDocument();

    await user.click(gradients);
    const gradientSection = gradients.closest("section");
    expect(gradientSection).not.toBeNull();
    expect(within(gradientSection as HTMLElement).getByText("bias/grad_mean"))
      .toBeInTheDocument();

    await user.click(bias);
    const biasSection = bias.closest("section");
    expect(biasSection).not.toBeNull();
    expect(within(biasSection as HTMLElement).getByText("bias/mean")).toBeInTheDocument();
    expect(within(biasSection as HTMLElement).queryByText("bias/grad_mean"))
      .not.toBeInTheDocument();

    await user.click(visualSummaries);
    const visualSection = visualSummaries.closest("section");
    expect(visualSection).not.toBeNull();
    expect(within(visualSection as HTMLElement).getByText("histogram/usage_fraction"))
      .toBeInTheDocument();
    expect(
      within(visualSection as HTMLElement).getByAltText(
        "Monitor image for main_model.0.model/heatmap/usage_fraction at step 2",
      ),
    ).toBeInTheDocument();
  });

  it("compares selected-node monitor charts with a numeric sibling layer", async () => {
    const { monitorDataRequests } = installFetchMock({
      inspectResponse: repeatedLayersInspectResponse,
      monitorDataResponse: ({ nodePath }) => semanticMonitorPayload(nodePath),
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    await selectTrainingMonitorOption(user, /Linear layers/i);
    await selectNewTrainingLogFolder(user, "monitor_compare");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /^monitor charts$/i })).toBeEnabled(),
    );
    await user.click(screen.getByRole("button", { name: /^monitor charts$/i }));
    const dialog = await screen.findByRole("dialog", { name: /monitor charts/i });
    expect(within(dialog).getByLabelText(/^scope$/i)).toHaveValue("same-stack");
    await user.selectOptions(within(dialog).getByLabelText(/^compare$/i), "main_model.1.model");

    await waitFor(() => {
      expect(monitorDataRequests).toEqual(
        expect.arrayContaining([
          { jobId: "job-1", nodePath: "main_model.0.model", preset: "baseline", dataset: "Mnist" },
          { jobId: "job-1", nodePath: "main_model.1.model", preset: "baseline", dataset: "Mnist" },
        ]),
      );
    });
    expect(within(dialog).getByText("Primary layer")).toBeInTheDocument();
    expect(within(dialog).getByText("Comparison layer")).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: /Activations\s+1 pair/i }))
      .toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getAllByText("input/mean").length).toBeGreaterThanOrEqual(2);

    const gradients = within(dialog).getByRole("button", { name: /Gradients\s+1 pair/i });
    expect(gradients).toHaveAttribute("aria-expanded", "false");
    expect(within(dialog).queryByText("bias/grad_mean")).not.toBeInTheDocument();
    await user.click(gradients);
    const gradientSection = gradients.closest("section");
    expect(gradientSection).not.toBeNull();
    expect(within(gradientSection as HTMLElement).getByText("main_model.0.model/bias/grad_mean"))
      .toBeInTheDocument();
    expect(within(gradientSection as HTMLElement).getByText("main_model.1.model/bias/grad_mean"))
      .toBeInTheDocument();
  });

  it("switches selected-node comparison scope to input and output linear layers", async () => {
    const { monitorDataRequests } = installFetchMock({
      inspectResponse: monitorScopeInspectResponse,
      monitorDataResponse: ({ nodePath }) => semanticMonitorPayload(nodePath),
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    await selectTrainingMonitorOption(user, /Linear layers/i);
    await selectNewTrainingLogFolder(user, "monitor_scope_select");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() =>
      expect(screen.getByRole("button", { name: /^monitor charts$/i })).toBeEnabled(),
    );
    await user.click(screen.getByRole("button", { name: /^monitor charts$/i }));
    const dialog = await screen.findByRole("dialog", { name: /monitor charts/i });
    const scopeSelect = within(dialog).getByLabelText(/^scope$/i);
    const compareSelect = within(dialog).getByLabelText(/^compare$/i);

    expect(scopeSelect).toHaveValue("same-stack");
    expect(within(compareSelect).queryByRole("option", { name: "input_model.model" }))
      .not.toBeInTheDocument();
    expect(within(compareSelect).queryByRole("option", { name: "output_model.model" }))
      .not.toBeInTheDocument();

    await user.selectOptions(scopeSelect, "all-layers");

    expect(scopeSelect).toHaveValue("all-layers");
    expect(within(compareSelect).getByRole("option", { name: "input_model.model" }))
      .toBeInTheDocument();
    expect(within(compareSelect).getByRole("option", { name: "output_model.model" }))
      .toBeInTheDocument();

    await user.selectOptions(compareSelect, "input_model.model");
    await waitFor(() => {
      expect(monitorDataRequests).toEqual(
        expect.arrayContaining([
          { jobId: "job-1", nodePath: "main_model.0.model", preset: "baseline", dataset: "Mnist" },
          { jobId: "job-1", nodePath: "input_model.model", preset: "baseline", dataset: "Mnist" },
        ]),
      );
    });

    await user.selectOptions(scopeSelect, "same-stack");
    await waitFor(() => expect(compareSelect).toHaveValue(""));

    await user.selectOptions(scopeSelect, "all-layers");
    await user.selectOptions(compareSelect, "output_model.model");
    await waitFor(() => {
      expect(monitorDataRequests).toEqual(
        expect.arrayContaining([
          { jobId: "job-1", nodePath: "output_model.model", preset: "baseline", dataset: "Mnist" },
        ]),
      );
    });
  });

  it("opens graph-card monitor charts for the wrapped linear target", async () => {
    const { monitorDataRequests } = installFetchMock({
      inspectResponse: repeatedLayersInspectResponse,
    });
    renderViewer();
    const user = userEvent.setup();

    await selectTrainingMonitorOption(user, /Linear layers/i);
    await selectNewTrainingLogFolder(user, "graph_card_monitor");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await user.click(
      await screen.findByRole("button", {
        name: /^open monitor charts for main_model\.0$/i,
      }),
    );

    const dialog = await screen.findByRole("dialog", { name: /monitor charts/i });
    expect(dialog).toBeInTheDocument();
    await user.selectOptions(within(dialog).getByLabelText(/^compare$/i), "main_model.1.model");
    await waitFor(() => {
      expect(monitorDataRequests).toEqual(
        expect.arrayContaining([
          { jobId: "job-1", nodePath: "main_model.0.model", preset: "baseline", dataset: "Mnist" },
          { jobId: "job-1", nodePath: "main_model.1.model", preset: "baseline", dataset: "Mnist" },
        ]),
      );
    });
  });

  it("opens input/output graph-card monitor modals with all-layers scope selected", async () => {
    const { monitorDataRequests } = installFetchMock({
      inspectResponse: monitorScopeInspectResponse,
      monitorDataResponse: ({ nodePath }) => semanticMonitorPayload(nodePath),
    });
    renderViewer();
    const user = userEvent.setup();

    await selectTrainingMonitorOption(user, /Linear layers/i);
    await selectNewTrainingLogFolder(user, "input_output_monitor_scope");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await user.click(
      await screen.findByRole("button", {
        name: /^open monitor charts for input_model$/i,
      }),
    );

    const dialog = await screen.findByRole("dialog", { name: /monitor charts/i });
    const scopeSelect = within(dialog).getByLabelText(/^scope$/i);
    const compareSelect = within(dialog).getByLabelText(/^compare$/i);

    expect(scopeSelect).toHaveValue("all-layers");
    expect(within(compareSelect).getByRole("option", { name: "output_model.model" }))
      .toBeInTheDocument();

    await user.selectOptions(compareSelect, "output_model.model");

    await waitFor(() => {
      expect(monitorDataRequests).toEqual(
        expect.arrayContaining([
          { jobId: "job-1", nodePath: "input_model.model", preset: "baseline", dataset: "Mnist" },
          { jobId: "job-1", nodePath: "output_model.model", preset: "baseline", dataset: "Mnist" },
        ]),
      );
    });
  });

  it("hides graph-card monitor buttons when the active job lacks the linear monitor", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await selectTrainingMonitorOption(user, /Sampler usage/i);
    await selectNewTrainingLogFolder(user, "sampler_only_monitor");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /training (running|completed) mnist/i }),
      ).toBeInTheDocument();
    });
    expect(
      screen.queryByRole("button", {
        name: /^open monitor charts for main_model\.0$/i,
      }),
    ).not.toBeInTheDocument();
  });

  it("opens selected-node monitor charts for the latest five filtered historical runs", async () => {
    const fixture = buildHistoricalMonitorFixture();
    const { logRunMonitorDataRequests } = installFetchMock({
      ...fixture,
    });
    renderViewer();
    const user = userEvent.setup();

    // Pick a run so its experiment/dataset drives the historical monitor group.
    await user.click(
      await screen.findByRole("button", {
        name: /select experiment run monitor_exp BASELINE Mnist 2026-06-01 06:00:00/i,
      }),
    );
    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    await user.click(
      await screen.findByRole("button", { name: /^select main_model\.0\.model$/i }),
    );
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /^monitor charts$/i })).toBeEnabled();
    });
    await user.click(screen.getByRole("button", { name: /^monitor charts$/i }));

    const dialog = await screen.findByRole("dialog", { name: /monitor charts/i });
    await waitFor(() => {
      expect(logRunMonitorDataRequests).toEqual([
        { runId: "historical-06", nodePath: "main_model.0.model" },
        { runId: "historical-05", nodePath: "main_model.0.model" },
        { runId: "historical-04", nodePath: "main_model.0.model" },
        { runId: "historical-03", nodePath: "main_model.0.model" },
        { runId: "historical-02", nodePath: "main_model.0.model" },
      ]);
    });
    expect(within(dialog).getByText("historical group")).toBeInTheDocument();
    expect(within(dialog).getByText("monitor_exp")).toBeInTheDocument();
    expect(within(dialog).getByText("5 runs")).toBeInTheDocument();
    expect(
      within(dialog).getByRole("img", {
        name: /output\/mean multi-run scalar chart/i,
      }),
    ).toBeInTheDocument();
    expect(within(dialog).getByText(/monitor_run_06_20260601_060000/))
      .toBeInTheDocument();
    expect(within(dialog).getByText(/monitor_run_02_20260601_020000/))
      .toBeInTheDocument();
    expect(within(dialog).queryByText(/monitor_run_01_20260601_010000/))
      .not.toBeInTheDocument();
  });

  it("compares historical monitor charts through log-run monitor-data requests", async () => {
    const fixture = buildHistoricalMonitorFixture(2);
    const { logRunMonitorDataRequests } = installFetchMock({
      inspectResponse: repeatedLayersInspectResponse,
      ...fixture,
    });
    renderViewer();
    const user = userEvent.setup();

    // Pick a run so its experiment/dataset drives the historical monitor group.
    await user.click(
      await screen.findByRole("button", {
        name: /select experiment run monitor_exp BASELINE Mnist 2026-06-01 02:00:00/i,
      }),
    );
    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /^monitor charts$/i })).toBeEnabled();
    });
    await user.click(screen.getByRole("button", { name: /^monitor charts$/i }));
    const dialog = await screen.findByRole("dialog", { name: /monitor charts/i });
    await user.selectOptions(within(dialog).getByLabelText(/^compare$/i), "main_model.1.model");

    await waitFor(() => {
      expect(logRunMonitorDataRequests).toEqual(
        expect.arrayContaining([
          { runId: "historical-02", nodePath: "main_model.0.model" },
          { runId: "historical-01", nodePath: "main_model.0.model" },
          { runId: "historical-02", nodePath: "main_model.1.model" },
          { runId: "historical-01", nodePath: "main_model.1.model" },
        ]),
      );
    });
    expect(within(dialog).getByText("Primary layer")).toBeInTheDocument();
    expect(within(dialog).getByText("Comparison layer")).toBeInTheDocument();
  });

  it("prefers an active linear training job over filtered historical monitor runs", async () => {
    const fixture = buildHistoricalMonitorFixture(2);
    const { monitorDataRequests, logRunMonitorDataRequests } = installFetchMock({
      ...fixture,
      monitorDataResponse: ({ nodePath }) => semanticMonitorPayload(nodePath),
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    await selectTrainingMonitorOption(user, /Linear layers/i);
    await selectNewTrainingLogFolder(user, "active_precedence");
    await user.click(screen.getByRole("button", { name: /start training/i }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /^monitor charts$/i })).toBeEnabled();
    });
    await user.click(screen.getByRole("button", { name: /^monitor charts$/i }));

    await waitFor(() => {
      expect(monitorDataRequests).toEqual(
        expect.arrayContaining([
          { jobId: "job-1", nodePath: "main_model.0.model", preset: "baseline", dataset: "Mnist" },
        ]),
      );
    });
    expect(logRunMonitorDataRequests).toHaveLength(0);
  });

  it("keeps long selected node ids from squeezing the path column", async () => {
    installFetchMock({ inspectResponse: longSelectedNodeInspectResponse });
    renderViewer();

    const idRow = (await screen.findAllByTitle(longSelectedNodeId)).find((element) =>
      element.classList.contains("grid"),
    );
    expect(idRow).toBeDefined();
    const resolvedIdRow = idRow as HTMLElement;
    expect(resolvedIdRow).toHaveClass("grid");
    expect(within(resolvedIdRow).getByText("ID")).toBeInTheDocument();
    expect(within(resolvedIdRow).getByText(longSelectedNodeId)).toHaveClass("truncate");

    const pathText = screen
      .getAllByText(longSelectedNodeId)
      .find((element) => element.classList.contains("break-words"));
    expect(pathText).toBeDefined();
  });

  it("renders a non-crashing API error panel", async () => {
    installFetchMock({ inspectError: true });
    renderViewer();

    expect(await screen.findByText("Preview failed")).toBeInTheDocument();
    expect(screen.getByText(/invalid override value/i)).toBeInTheDocument();
  });
});
