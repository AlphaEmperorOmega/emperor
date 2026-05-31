import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { ViewerApp } from "@/components/viewer-app";

type MockNodeData = {
  nodeId: string;
  label: string;
  subtitle: string;
  path: string;
  parameterCount: number;
  details: Record<string, unknown>;
  childCount: number;
  childSummaries: Array<{
    label: string;
    nestedLabel?: string;
    count?: number;
    kind: "child" | "mechanism" | "overflow";
    title?: string;
  }>;
  height: number;
  isExpanded: boolean;
  canToggleExpansion: boolean;
  isDetailsExpanded: boolean;
  onActivateNode: () => void;
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

vi.mock("@xyflow/react", () => ({
  ReactFlow: ({
    nodes,
    edges,
    onNodeClick,
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
    children: React.ReactNode;
  }) => (
    <div data-testid="flow">
      {nodes.map((node) => (
        <div
          key={node.id}
          data-testid={`node-${node.id}`}
          data-x={node.position.x}
          data-y={node.position.y}
          data-height={node.style?.height ?? node.data.height}
        >
          <div
            role="button"
            tabIndex={0}
            aria-label={
              node.data.canToggleExpansion
                ? `${node.data.isExpanded ? "Select and collapse" : "Select and expand"} ${node.data.path}`
                : `Select ${node.data.path}`
            }
            aria-expanded={node.data.canToggleExpansion ? node.data.isExpanded : undefined}
            onClick={() => {
              node.data.onActivateNode();
              onNodeClick?.({}, node);
            }}
          >
            <div>
              <span>{node.data.label}</span>
              {node.data.childCount > 0 && (
                <span>
                  {node.data.childCount}{" "}
                  {node.data.childCount === 1 ? "child" : "children"}
                </span>
              )}
              {node.data.parameterCount > 0 && (
                <span title={`${formatExactCount(node.data.parameterCount)} parameters`}>
                  {formatCompactCount(node.data.parameterCount)}
                </span>
              )}
            </div>
            <span>{node.data.subtitle}</span>
            <div data-testid={`child-summaries-${node.id}`}>
              {node.data.childSummaries.map((summary, index) => {
                const summaryLabel = summary.nestedLabel
                  ? `${summary.label} ${summary.nestedLabel}`
                  : summary.label;

                return (
                  <div
                    key={`${summary.kind}-${summary.label}-${index}`}
                    aria-label={summaryLabel}
                    title={summary.title ?? summaryLabel}
                  >
                    {summary.kind === "overflow" ? (
                      <span>{summary.label}</span>
                    ) : summary.nestedLabel ? (
                      <>
                        <span>{summary.label}</span>
                        <span aria-hidden>›</span>
                        <span>{summary.nestedLabel}</span>
                      </>
                    ) : (
                      <span>
                        {summary.count ? `${summary.label} x${summary.count}` : summary.label}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
            {node.data.canToggleExpansion && <span>toggle</span>}
            {Object.keys(node.data.details).length > 0 && (
              <button
                type="button"
                aria-label={`Details for ${node.data.path}`}
                aria-expanded={node.data.isDetailsExpanded}
                onClick={(event) => {
                  event.stopPropagation();
                  node.data.onToggleDetails();
                }}
              >
                Details
              </button>
            )}
            {node.data.isDetailsExpanded && Object.keys(node.data.details).length > 0 && (
              <div>
                {Object.entries(node.data.details).map(([key, value]) => (
                  <div key={key}>
                    <span>{key}</span>
                    <span>{detailText(value)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      ))}
      {edges.map((edge) => (
        <div key={edge.id} data-testid={`edge-${edge.id}`}>
          {edge.source} to {edge.target}
        </div>
      ))}
      {children}
    </div>
  ),
  Background: () => null,
  Controls: () => null,
  Handle: () => null,
  MarkerType: { ArrowClosed: "arrowclosed" },
  Position: { Left: "left", Right: "right" },
}));

const modelsResponse = { models: ["linear", "bert"] };
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
      searchChoices: [128, 256],
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
      searchChoices: [],
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
      searchChoices: [],
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

function jsonResponse(body: unknown, status = 200) {
  return Promise.resolve(
    new Response(JSON.stringify(body), {
      status,
      headers: { "content-type": "application/json" },
    }),
  );
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

function installFetchMock(options: { inspectError?: boolean; inspectResponse?: unknown } = {}) {
  const inspectBodies: unknown[] = [];
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith("/health")) {
      return jsonResponse({ status: "ok" });
    }
    if (url.endsWith("/models")) {
      return jsonResponse(modelsResponse);
    }
    if (url.endsWith("/models/linear/presets")) {
      return jsonResponse(presetsResponse);
    }
    if (url.endsWith("/models/linear/config-schema")) {
      return jsonResponse(schemaResponse);
    }
    if (url.endsWith("/inspect")) {
      inspectBodies.push(JSON.parse(String(init?.body)));
      if (options.inspectError) {
        return jsonResponse({ detail: "invalid override value" }, 400);
      }
      return jsonResponse(withParameterCounts(options.inspectResponse ?? inspectResponse));
    }
    return jsonResponse({ detail: `Unhandled ${url}` }, 404);
  });
  vi.stubGlobal("fetch", fetchMock);
  return { fetchMock, inspectBodies };
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

describe("ViewerApp", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders model and preset selectors from API data", async () => {
    installFetchMock();
    renderViewer();

    expect(await screen.findByDisplayValue("linear")).toBeInTheDocument();
    expect(await screen.findByDisplayValue("baseline")).toBeInTheDocument();
  });

  it("renders config controls from schema", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /layer stack options/i }));
    expect(await screen.findByLabelText(/hidden dim/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/stack activation/i)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /gate stack options/i }));
    expect(screen.getByRole("switch", { name: /gate flag/i })).toBeInTheDocument();
  });

  it("groups config controls under config section titles", async () => {
    installFetchMock();
    renderViewer();

    expect(await screen.findByRole("heading", { name: /layer stack options/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /gate stack options/i })).toBeInTheDocument();
  });

  it("renders config sections collapsed by default", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    const layerSection = await screen.findByRole("button", { name: /layer stack options/i });
    expect(screen.queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();

    await user.click(layerSection);
    expect(await screen.findByLabelText(/hidden dim/i)).toBeInTheDocument();

    await user.click(layerSection);
    expect(screen.queryByLabelText(/hidden dim/i)).not.toBeInTheDocument();
  });

  it("posts selected model, preset, and overrides on update", async () => {
    const { inspectBodies } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /layer stack options/i }));
    const hiddenInput = await screen.findByLabelText(/hidden dim/i);
    await user.selectOptions(hiddenInput, "128");
    await user.click(screen.getByRole("button", { name: /update preview/i }));

    await waitFor(() => {
      expect(inspectBodies.at(-1)).toEqual({
        model: "linear",
        preset: "baseline",
        overrides: { hidden_dim: "128" },
      });
    });
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

    expect((await screen.findAllByText("Params")).length).toBeGreaterThan(0);
    expect(screen.getAllByText("65.8K").length).toBeGreaterThan(0);
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
    expect(screen.getByText("128 -> 128")).toBeInTheDocument();
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

    expect(within(layerNode).getByText("dims")).toBeInTheDocument();
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
    expect(screen.getByText("128 -> 128")).toBeInTheDocument();

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

  it("renders graph data as a nested hierarchy view", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(screen.getByRole("tab", { name: /hierarchy/i }));
    expect(await screen.findByText("main_model.0")).toBeInTheDocument();
    expect(screen.queryByText("main_model.0.model")).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /main_model\.0/ }));
    expect(await screen.findByText("main_model.0.model")).toBeInTheDocument();
  });

  it("displays selected node metadata", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(
      await screen.findByRole("button", { name: /select and expand main_model\.0/i }),
    );
    expect(screen.getByText("128 -> 128")).toBeInTheDocument();
    expect(screen.getByText("33,024")).toBeInTheDocument();
    expect(screen.getByText("BEFORE")).toBeInTheDocument();
  });

  it("keeps long selected node ids from squeezing the path column", async () => {
    installFetchMock({ inspectResponse: longSelectedNodeInspectResponse });
    renderViewer();

    const idRow = await screen.findByTitle(longSelectedNodeId);
    expect(idRow).toHaveClass("grid");
    expect(within(idRow).getByText("ID")).toBeInTheDocument();
    expect(within(idRow).getByText(longSelectedNodeId)).toHaveClass("truncate");

    const pathText = screen
      .getAllByText(longSelectedNodeId)
      .find((element) => element.classList.contains("break-words"));
    expect(pathText).toBeDefined();
  });

  it("renders a non-crashing API error panel", async () => {
    installFetchMock({ inspectError: true });
    renderViewer();

    expect(await screen.findByText("Preview failed")).toBeInTheDocument();
    expect(screen.getByText("invalid override value")).toBeInTheDocument();
  });
});
