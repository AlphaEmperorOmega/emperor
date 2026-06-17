import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  useCompareTargetState: vi.fn(),
  fetchPresets: vi.fn(),
  fetchDatasets: vi.fn(),
  fetchMonitors: vi.fn(),
  fetchConfigSchema: vi.fn(),
  inspectModel: vi.fn(),
}));

vi.mock("@/features/viewer/providers/viewer-providers", () => ({
  useCompareTargetState: mocks.useCompareTargetState,
}));

vi.mock("@/lib/api", () => ({
  fetchPresets: mocks.fetchPresets,
  fetchDatasets: mocks.fetchDatasets,
  fetchMonitors: mocks.fetchMonitors,
  fetchConfigSchema: mocks.fetchConfigSchema,
  inspectModel: mocks.inspectModel,
}));

import { CompareWorkspace } from "@/features/viewer/components/compare-workspace";

const selectModel = vi.fn();
const selectPreset = vi.fn();
const onUseTarget = vi.fn();

function renderCompareWorkspace() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  return render(
    <QueryClientProvider client={client}>
      <CompareWorkspace onUseTarget={onUseTarget} />
    </QueryClientProvider>,
  );
}

function modelKey(identity: { modelType: string; model: string }) {
  return `${identity.modelType}/${identity.model}`;
}

function graph(
  identity: { modelType: string; model: string },
  preset: string,
  parameterCount: number,
) {
  return {
    modelType: identity.modelType,
    model: identity.model,
    preset,
    parameterCount,
    parameterSizeBytes: parameterCount * 4,
    nodes: [
      {
        id: `${modelKey(identity)}-root`,
        label: "Root",
        typeName: "Model",
        path: "main_model",
        graphRole: "architecture",
        parameterCount,
        parameterSizeBytes: parameterCount * 4,
        details: {},
        config: null,
      },
      {
        id: `${modelKey(identity)}-runtime`,
        label: "Runtime",
        typeName: "RuntimeState",
        path: "main_model.runtime",
        graphRole: "runtime",
        parameterCount: 0,
        parameterSizeBytes: 0,
        details: {},
        config: null,
      },
    ],
    edges: [
      {
        id: `${modelKey(identity)}-edge`,
        source: `${modelKey(identity)}-root`,
        target: `${modelKey(identity)}-runtime`,
      },
    ],
  };
}

beforeEach(() => {
  selectModel.mockReset();
  selectPreset.mockReset();
  onUseTarget.mockReset();
  mocks.useCompareTargetState.mockReset().mockReturnValue({
    selectedModelType: "linears",
    selectedModel: "linear",
    selectedPreset: "baseline",
    selectModel,
    selectPreset,
    catalog: {
      models: [
        { modelType: "linears", model: "linear" },
        { modelType: "experts", model: "experts_linear" },
      ],
      isLoading: false,
      isError: false,
      error: null,
    },
  });
  mocks.fetchPresets.mockReset().mockImplementation((identity) =>
    Promise.resolve({
      ...identity,
      presets:
        modelKey(identity) === "experts/experts_linear"
          ? [{ name: "expert-baseline", label: "Expert baseline", description: "" }]
          : [{ name: "baseline", label: "Baseline", description: "" }],
    }),
  );
  mocks.fetchDatasets.mockReset().mockImplementation((identity) =>
    Promise.resolve({
      ...identity,
      datasets:
        modelKey(identity) === "experts/experts_linear"
          ? [{ name: "ExpertToy", label: "Expert Toy", inputDim: 64, outputDim: 4 }]
          : [{ name: "Mnist", label: "MNIST", inputDim: 784, outputDim: 10 }],
    }),
  );
  mocks.fetchMonitors.mockReset().mockImplementation((identity) =>
    Promise.resolve({
      ...identity,
      monitors:
        modelKey(identity) === "experts/experts_linear"
          ? [
              {
                name: "experts",
                label: "Experts",
                description: "",
                kinds: ["scalar"],
                defaultEnabled: false,
              },
            ]
          : [
              {
                name: "linear_layers",
                label: "Linear layers",
                description: "",
                kinds: ["scalar"],
                defaultEnabled: false,
              },
              {
                name: "recurrent_layers",
                label: "Recurrent layers",
                description: "",
                kinds: ["scalar"],
                defaultEnabled: false,
              },
              {
                name: "layer_controllers",
                label: "Layer controllers",
                description: "",
                kinds: ["scalar"],
                defaultEnabled: false,
              },
              {
                name: "gradient_norm",
                label: "Gradient norm",
                description: "",
                kinds: ["scalar"],
                defaultEnabled: false,
              },
            ],
    }),
  );
  mocks.fetchConfigSchema.mockReset().mockImplementation((identity) =>
    Promise.resolve({
      ...identity,
      fields: [
        {
          key: "hidden_size",
          configKey: "hidden_size",
          flag: "--hidden-size",
          label: "Hidden size",
          section: "Architecture",
          type: "int",
          default: modelKey(identity) === "experts/experts_linear" ? 128 : 64,
          nullable: false,
          choices: [],
        },
      ],
    }),
  );
  mocks.inspectModel.mockReset().mockImplementation(
    (request: { modelType: string; model: string; preset: string }) =>
      Promise.resolve(
        graph(
          request,
          request.preset,
          modelKey(request) === "experts/experts_linear" ? 2048 : 1024,
        ),
      ),
  );
});

describe("CompareWorkspace", () => {
  it("compares selected model/preset targets with split public model identity", async () => {
    renderCompareWorkspace();

    expect(
      await screen.findByRole("heading", { name: /model comparison/i }),
    ).toBeInTheDocument();

    await waitFor(() => {
      expect(mocks.inspectModel).toHaveBeenCalledWith({
        modelType: "linears",
        model: "linear",
        preset: "baseline",
        dataset: "Mnist",
        overrides: {},
      });
      expect(mocks.inspectModel).toHaveBeenCalledWith({
        modelType: "experts",
        model: "experts_linear",
        preset: "expert-baseline",
        dataset: "ExpertToy",
        overrides: {},
      });
    });

    expect(screen.getByText("Changed Summary Metrics")).toBeInTheDocument();
    expect(screen.getByText("Changed Config Values")).toBeInTheDocument();
    expect(screen.getByText("Hidden size")).toBeInTheDocument();
    expect(screen.getByText("64")).toBeInTheDocument();
    expect(screen.getByText("128")).toBeInTheDocument();
    expect(screen.getByText(/\+1,024/)).toBeInTheDocument();
    expect(
      screen.getByText("Linear layers, Recurrent layers, Layer controllers, +1"),
    ).toHaveAttribute(
      "title",
      "Linear layers, Recurrent layers, Layer controllers, Gradient norm",
    );
  });

  it("applies a comparison target to the main viewer target", async () => {
    renderCompareWorkspace();
    const user = userEvent.setup();

    await waitFor(() => {
      expect(screen.getByText("experts_linear")).toBeInTheDocument();
    });

    const targetCards = screen.getAllByText(/^Target \d$/).map((label) => {
      const card = label.closest(".edge");
      expect(card).toBeInstanceOf(HTMLElement);
      return card as HTMLElement;
    });
    await user.click(
      within(targetCards[1]).getByRole("button", { name: /use as target/i }),
    );

    expect(selectModel).toHaveBeenCalledWith("experts_linear", "experts");
    expect(selectPreset).toHaveBeenCalledWith("expert-baseline");
    expect(onUseTarget).toHaveBeenCalled();
  });

  it("adds, removes, and resets comparison targets around the four-target limit", async () => {
    mocks.useCompareTargetState.mockReturnValue({
      selectedModelType: "linears",
      selectedModel: "linear",
      selectedPreset: "baseline",
      selectModel,
      selectPreset,
      catalog: {
        models: [
          { modelType: "linears", model: "linear" },
          { modelType: "experts", model: "experts_linear" },
          { modelType: "linears", model: "linear_small" },
          { modelType: "linears", model: "linear_wide" },
        ],
        isLoading: false,
        isError: false,
        error: null,
      },
    });
    renderCompareWorkspace();
    const user = userEvent.setup();

    await waitFor(() => {
      expect(screen.getAllByText(/^Target \d$/)).toHaveLength(2);
    });

    const addTarget = screen.getByRole("button", { name: /add target/i });
    await user.click(addTarget);
    await user.click(addTarget);

    await waitFor(() => {
      expect(screen.getAllByText(/^Target \d$/)).toHaveLength(4);
    });
    expect(addTarget).toBeDisabled();

    await user.click(
      screen.getByRole("button", { name: /remove comparison target 4/i }),
    );

    await waitFor(() => {
      expect(screen.getAllByText(/^Target \d$/)).toHaveLength(3);
    });
    expect(addTarget).toBeEnabled();

    await user.click(screen.getByRole("button", { name: /^reset$/i }));

    await waitFor(() => {
      expect(screen.getAllByText(/^Target \d$/)).toHaveLength(2);
    });
    expect(screen.getByText("linear")).toBeInTheDocument();
    expect(screen.getByText("experts_linear")).toBeInTheDocument();
  });
});
