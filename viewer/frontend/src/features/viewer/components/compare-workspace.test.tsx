import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  useCompareTargetState: vi.fn(),
  useTargetConfig: vi.fn(),
  fetchPresets: vi.fn(),
  fetchDatasets: vi.fn(),
  fetchMonitors: vi.fn(),
  fetchConfigSchema: vi.fn(),
  inspectModel: vi.fn(),
  fetchLogRuns: vi.fn(),
  fetchLogTags: vi.fn(),
  fetchLogScalars: vi.fn(),
}));

vi.mock("@/features/viewer/providers/viewer-providers", () => ({
  useCompareTargetState: mocks.useCompareTargetState,
  useTargetConfig: mocks.useTargetConfig,
}));

vi.mock("@/lib/api", () => ({
  fetchPresets: mocks.fetchPresets,
  fetchDatasets: mocks.fetchDatasets,
  fetchMonitors: mocks.fetchMonitors,
  fetchConfigSchema: mocks.fetchConfigSchema,
  inspectModel: mocks.inspectModel,
  fetchLogRuns: mocks.fetchLogRuns,
  fetchLogTags: mocks.fetchLogTags,
  fetchLogScalars: mocks.fetchLogScalars,
  DEFAULT_LOG_SCALAR_MAX_POINTS: 500,
  LOG_SCALAR_SAMPLING: "tail",
}));

import { CompareWorkspace } from "@/features/viewer/components/compare-workspace";

const selectModel = vi.fn();
const selectPreset = vi.fn();
const onUseTarget = vi.fn();
const onOpenLogs = vi.fn();

function renderCompareWorkspace() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });

  return render(
    <QueryClientProvider client={client}>
      <CompareWorkspace onOpenLogs={onOpenLogs} onUseTarget={onUseTarget} />
    </QueryClientProvider>,
  );
}

async function openConfigsMode(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByRole("tab", { name: /configs/i }));
}

async function selectSearchableOption(
  user: ReturnType<typeof userEvent.setup>,
  control: HTMLElement,
  optionName: string | RegExp,
  searchText: string,
) {
  await user.click(control);
  const root = control.parentElement;

  if (!(root instanceof HTMLElement)) {
    throw new Error("Expected dropdown root");
  }

  const search = within(root).getByRole("searchbox");
  await user.clear(search);
  await user.type(search, searchText);
  const listbox = within(root).getByRole("listbox");
  await user.click(within(listbox).getByRole("option", { name: optionName }));
  await waitFor(() => {
    expect(within(root).queryByRole("listbox")).not.toBeInTheDocument();
  });
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function optionName(label: string) {
  return new RegExp(`^${escapeRegExp(label)}(?:\\b|$)`, "i");
}

async function selectDefaultCompareRuns(user: ReturnType<typeof userEvent.setup>) {
  await screen.findByText("Run Target 1");
  await user.click(await screen.findByRole("button", { name: /add target/i }));
  await screen.findByText("Run Target 2");

  return screen.findByRole("combobox", { name: /^Scalar Tags\b/i });
}

function runTargetCards() {
  return screen.getAllByText(/^Run Target \d$/).map((label) => {
    const card = label.closest("article");
    expect(card).toBeInstanceOf(HTMLElement);
    return card as HTMLElement;
  });
}

async function openCompareScalarTags(user: ReturnType<typeof userEvent.setup>) {
  const trigger = await screen.findByRole("combobox", {
    name: /^Scalar Tags\b/i,
  });
  if (trigger.getAttribute("aria-expanded") !== "true") {
    await user.click(trigger);
  }
  return screen.findByRole("listbox", { name: "Scalar Tags options" });
}

function mockCompareScalarTags(tags: string[]) {
  mocks.fetchLogTags.mockResolvedValue({
    runs: [
      {
        runId: "run-a",
        scalarTags: tags,
        histogramTags: [],
        imageTags: [],
        textTags: [],
      },
      {
        runId: "run-b",
        scalarTags: tags,
        histogramTags: [],
        imageTags: [],
        textTags: [],
      },
      {
        runId: "run-c",
        scalarTags: [],
        histogramTags: [],
        imageTags: [],
        textTags: [],
      },
    ],
  });
  mocks.fetchLogScalars.mockResolvedValue({
    series: tags.flatMap((tag, index) => [
      {
        runId: "run-a",
        tag,
        points: [
          { step: 1, wallTime: 1780000000 + index, value: 0.2 + index },
          { step: 2, wallTime: 1780000100 + index, value: 0.4 + index },
        ],
      },
      {
        runId: "run-b",
        tag,
        points: [
          { step: 1, wallTime: 1780000200 + index, value: 0.1 + index },
          { step: 2, wallTime: 1780000300 + index, value: 0.3 + index },
        ],
      },
    ]),
  });
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
  onOpenLogs.mockReset();
  mocks.useTargetConfig.mockReset().mockReturnValue({
    selectedModelType: "linears",
    selectedModel: "linear",
    selectedPreset: "baseline",
    selectedPresetMeta: { name: "baseline", label: "BASELINE", description: "" },
    selectedDatasets: ["Mnist"],
  });
  mocks.fetchLogRuns.mockReset().mockResolvedValue({
    runs: [
      {
        id: "run-a",
        group: "exp-a",
        experiment: "exp-a",
        modelType: "linears",
        model: "linear",
        preset: "BASELINE",
        dataset: "Mnist",
        runName: "aaa_20260601_010203",
        timestamp: "2026-06-01 01:02:03",
        version: "version_0",
        relativePath: "exp-a/linear/BASELINE/Mnist/aaa/version_0",
        hasResult: true,
        eventFileCount: 1,
        checkpointCount: 1,
        hasHparams: true,
        metrics: {},
      },
      {
        id: "run-b",
        group: "exp-b",
        experiment: "exp-b",
        modelType: "linears",
        model: "linear",
        preset: "BASELINE",
        dataset: "Cifar10",
        runName: "bbb_20260601_020304",
        timestamp: "2026-06-01 02:03:04",
        version: "version_0",
        relativePath: "exp-b/linear/BASELINE/Cifar10/bbb/version_0",
        hasResult: true,
        eventFileCount: 1,
        checkpointCount: 0,
        hasHparams: true,
        metrics: {},
      },
      {
        id: "run-f",
        group: "exp-a",
        experiment: "exp-a",
        modelType: "linears",
        model: "linear",
        preset: "BASELINE",
        dataset: "FashionMnist",
        runName: "fff_20260601_013000",
        timestamp: "2026-06-01 01:30:00",
        version: "version_0",
        relativePath: "exp-a/linear/BASELINE/FashionMnist/fff/version_0",
        hasResult: true,
        eventFileCount: 1,
        checkpointCount: 0,
        hasHparams: true,
        metrics: {},
      },
      {
        id: "run-c",
        group: "exp-c",
        experiment: "exp-c",
        modelType: "linears",
        model: "linear",
        preset: "BASELINE",
        dataset: "Mnist",
        runName: "ccc_20260601_030405",
        timestamp: "2026-06-01 03:04:05",
        version: "version_0",
        relativePath: "exp-c/linear/BASELINE/Mnist/ccc/version_0",
        hasResult: false,
        eventFileCount: 1,
        checkpointCount: 0,
        hasHparams: false,
        metrics: {},
      },
      {
        id: "run-d",
        group: "exp-a",
        experiment: "exp-a",
        modelType: "experts",
        model: "experts_linear",
        preset: "EXPERT",
        dataset: "ExpertToy",
        runName: "ddd_20260601_003000",
        timestamp: "2026-06-01 00:30:00",
        version: "version_0",
        relativePath: "exp-a/experts_linear/EXPERT/ExpertToy/ddd/version_0",
        hasResult: true,
        eventFileCount: 1,
        checkpointCount: 0,
        hasHparams: true,
        metrics: {},
      },
      {
        id: "run-e",
        group: "exp-a",
        experiment: "exp-a",
        modelType: "linears",
        model: "wide_linear",
        preset: "WIDE",
        dataset: "FashionMnist",
        runName: "eee_20260601_001500",
        timestamp: "2026-06-01 00:15:00",
        version: "version_0",
        relativePath: "exp-a/wide_linear/WIDE/FashionMnist/eee/version_0",
        hasResult: true,
        eventFileCount: 1,
        checkpointCount: 0,
        hasHparams: true,
        metrics: {},
      },
    ],
    total: 6,
    limit: 500,
    offset: 0,
    hasMore: false,
  });
  mocks.fetchLogTags.mockReset().mockResolvedValue({
    runs: [
      {
        runId: "run-a",
        scalarTags: ["train/loss", "validation/accuracy"],
        histogramTags: [],
        imageTags: [],
        textTags: [],
      },
      {
        runId: "run-b",
        scalarTags: ["train/loss", "validation/accuracy"],
        histogramTags: [],
        imageTags: [],
        textTags: [],
      },
      {
        runId: "run-f",
        scalarTags: ["train/loss", "validation/accuracy"],
        histogramTags: [],
        imageTags: [],
        textTags: [],
      },
      {
        runId: "run-c",
        scalarTags: [],
        histogramTags: [],
        imageTags: [],
        textTags: [],
      },
      {
        runId: "run-d",
        scalarTags: ["train/loss", "validation/accuracy"],
        histogramTags: [],
        imageTags: [],
        textTags: [],
      },
      {
        runId: "run-e",
        scalarTags: ["train/loss", "validation/accuracy"],
        histogramTags: [],
        imageTags: [],
        textTags: [],
      },
    ],
  });
  mocks.fetchLogScalars.mockReset().mockResolvedValue({
    series: [
      { runId: "run-a", firstAccuracy: 0.6, lastAccuracy: 0.8, firstLoss: 0.7, lastLoss: 0.3 },
      { runId: "run-b", firstAccuracy: 0.4, lastAccuracy: 0.55, firstLoss: 0.9, lastLoss: 0.5 },
      { runId: "run-d", firstAccuracy: 0.5, lastAccuracy: 0.74, firstLoss: 0.8, lastLoss: 0.38 },
      { runId: "run-e", firstAccuracy: 0.52, lastAccuracy: 0.76, firstLoss: 0.82, lastLoss: 0.36 },
      { runId: "run-f", firstAccuracy: 0.58, lastAccuracy: 0.79, firstLoss: 0.76, lastLoss: 0.32 },
    ].flatMap((fixture, index) => [
      {
        runId: fixture.runId,
        tag: "validation/accuracy",
        points: [
          { step: 1, wallTime: 1780000000 + index, value: fixture.firstAccuracy },
          { step: 2, wallTime: 1780000100 + index, value: fixture.lastAccuracy },
        ],
      },
      {
        runId: fixture.runId,
        tag: "train/loss",
        points: [
          { step: 1, wallTime: 1780000200 + index, value: fixture.firstLoss },
          { step: 2, wallTime: 1780000300 + index, value: fixture.lastLoss },
        ],
      },
    ]),
  });
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
  it("constrains the compare content to a shell scrollport", async () => {
    renderCompareWorkspace();

    const heading = await screen.findByRole("heading", {
      name: /training run comparison/i,
    });
    const scrollRoot = heading.closest(".overflow-y-auto");

    expect(scrollRoot).toBeInstanceOf(HTMLElement);
    expect(scrollRoot).toHaveClass("h-full", "min-h-0", "overflow-y-auto");
  });

  it("opens in Training Run comparison mode", async () => {
    renderCompareWorkspace();

    expect(
      await screen.findByRole("heading", { name: /training run comparison/i }),
    ).toBeInTheDocument();
    expect(await screen.findByText("Training Run Targets")).toBeInTheDocument();
    expect(await screen.findByText("Run Target 1")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /add target/i }))
      .toBeInTheDocument();
    expect(screen.getByText("1 target")).toBeInTheDocument();
    expect(screen.getByText("aaa_20260601_010203 · version_0"))
      .toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /runs/i })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });

  it("renders target cards with searchable single-select filters and no run picker", async () => {
    renderCompareWorkspace();
    const user = userEvent.setup();

    await screen.findByText("Run Target 1");
    const [card] = runTargetCards();

    expect(within(card).getByRole("combobox", { name: "Experiment" }))
      .toHaveTextContent("exp-a");
    expect(within(card).getByRole("combobox", { name: "Model Type" }))
      .toHaveTextContent("linears");
    expect(within(card).getByRole("combobox", { name: "Model" }))
      .toHaveTextContent("linear");
    expect(within(card).getByRole("combobox", { name: "Model" }))
      .not.toHaveTextContent(/linear\s*·\s*linears/i);
    expect(within(card).getByRole("combobox", { name: "Preset" }))
      .toHaveTextContent("BASELINE");
    expect(within(card).getByRole("combobox", { name: "Dataset" }))
      .toHaveTextContent("Mnist");

    await user.click(within(card).getByRole("combobox", { name: "Experiment" }));
    expect(
      within(card).getByRole("searchbox", { name: /search experiment/i }),
    ).toBeInTheDocument();
    expect(
      await within(card).findByRole("option", { name: optionName("exp-b") }),
    ).toBeInTheDocument();

    expect(screen.queryByRole("dialog", { name: /add run target/i }))
      .not.toBeInTheDocument();
    expect(screen.queryByText("Direct Training Run Search")).not.toBeInTheDocument();
  });

  it("filters run target models by the selected model type", async () => {
    renderCompareWorkspace();
    const user = userEvent.setup();

    await screen.findByText("Run Target 1");
    const [card] = runTargetCards();
    await selectSearchableOption(
      user,
      within(card).getByRole("combobox", { name: "Model Type" }),
      optionName("experts"),
      "experts",
    );

    expect(within(card).getByRole("combobox", { name: "Model Type" }))
      .toHaveTextContent("experts");
    expect(within(card).getByRole("combobox", { name: "Model" }))
      .toHaveTextContent("experts_linear");
    expect(within(card).getByRole("combobox", { name: "Preset" }))
      .toHaveTextContent("EXPERT");
    expect(within(card).getByRole("combobox", { name: "Dataset" }))
      .toHaveTextContent("ExpertToy");
    expect(screen.getByText("ddd_20260601_003000 · version_0"))
      .toBeInTheDocument();

    await user.click(within(card).getByRole("combobox", { name: "Model" }));
    const listbox = within(card).getByRole("listbox", { name: "Model options" });
    expect(
      within(listbox).getByRole("option", { name: optionName("experts_linear") }),
    ).toBeInTheDocument();
    expect(
      within(listbox).queryByRole("option", { name: optionName("linear") }),
    ).not.toBeInTheDocument();
  });

  it("cascades preset and dataset when the run target model changes", async () => {
    renderCompareWorkspace();
    const user = userEvent.setup();

    await screen.findByText("Run Target 1");
    const [card] = runTargetCards();
    await selectSearchableOption(
      user,
      within(card).getByRole("combobox", { name: "Model" }),
      optionName("wide_linear"),
      "wide",
    );

    expect(within(card).getByRole("combobox", { name: "Model Type" }))
      .toHaveTextContent("linears");
    expect(within(card).getByRole("combobox", { name: "Model" }))
      .toHaveTextContent("wide_linear");
    expect(within(card).getByRole("combobox", { name: "Preset" }))
      .toHaveTextContent("WIDE");
    expect(within(card).getByRole("combobox", { name: "Dataset" }))
      .toHaveTextContent("FashionMnist");
    expect(screen.getByText("eee_20260601_001500 · version_0"))
      .toBeInTheDocument();
  });

  it("adds Training Run target cards from unused scalar-capable combinations", async () => {
    renderCompareWorkspace();
    const user = userEvent.setup();

    await screen.findByText("Run Target 1");
    await user.click(await screen.findByRole("button", { name: /add target/i }));

    expect(await screen.findByText("Run Target 1")).toBeInTheDocument();
    expect(screen.getByText("Run Target 2")).toBeInTheDocument();
    expect(screen.getByText("aaa_20260601_010203 · version_0")).toBeInTheDocument();
    expect(screen.getByText("bbb_20260601_020304 · version_0")).toBeInTheDocument();
    expect(screen.getByText("2 targets")).toBeInTheDocument();
  });

  it("keeps duplicate run target combinations disabled at the dataset level", async () => {
    renderCompareWorkspace();
    const user = userEvent.setup();

    await screen.findByText("Run Target 1");
    await user.click(await screen.findByRole("button", { name: /add target/i }));
    const [, secondCard] = runTargetCards();

    await selectSearchableOption(
      user,
      within(secondCard).getByRole("combobox", { name: "Experiment" }),
      optionName("exp-a"),
      "exp-a",
    );

    expect(within(secondCard).getByRole("combobox", { name: "Experiment" }))
      .toHaveTextContent("exp-a");
    expect(within(secondCard).getByRole("combobox", { name: "Dataset" }))
      .toHaveTextContent("FashionMnist");
    expect(screen.getByText("fff_20260601_013000 · version_0"))
      .toBeInTheDocument();

    await user.click(
      within(secondCard).getByRole("combobox", { name: "Dataset" }),
    );
    const listbox = within(secondCard).getByRole("listbox", {
      name: "Dataset options",
    });
    expect(within(listbox).getByRole("option", { name: optionName("Mnist") }))
      .toHaveAttribute("aria-disabled", "true");
    expect(
      within(listbox).getByRole("option", { name: optionName("FashionMnist") }),
    ).toHaveAttribute("aria-selected", "true");
  });

  it("changes upstream card filters and cascades downstream values to valid defaults", async () => {
    renderCompareWorkspace();
    const user = userEvent.setup();

    await screen.findByText("Run Target 1");
    const [card] = runTargetCards();
    await selectSearchableOption(
      user,
      within(card).getByRole("combobox", { name: "Experiment" }),
      optionName("exp-b"),
      "exp-b",
    );

    expect(within(card).getByRole("combobox", { name: "Experiment" }))
      .toHaveTextContent("exp-b");
    expect(within(card).getByRole("combobox", { name: "Model Type" }))
      .toHaveTextContent("linears");
    expect(within(card).getByRole("combobox", { name: "Model" }))
      .toHaveTextContent("linear");
    expect(within(card).getByRole("combobox", { name: "Preset" }))
      .toHaveTextContent("BASELINE");
    expect(within(card).getByRole("combobox", { name: "Dataset" }))
      .toHaveTextContent("Cifar10");
    expect(screen.getByText("bbb_20260601_020304 · version_0"))
      .toBeInTheDocument();
  });

  it("resolves a target card to the latest matching scalar-capable run", async () => {
    mocks.fetchLogRuns.mockResolvedValue({
      runs: [
        {
          id: "run-old",
          group: "exp-a",
          experiment: "exp-a",
          modelType: "linears",
          model: "linear",
          preset: "BASELINE",
          dataset: "Mnist",
          runName: "old_20260601_010203",
          timestamp: "2026-06-01 01:02:03",
          version: "version_0",
          relativePath: "exp-a/linear/BASELINE/Mnist/old/version_0",
          hasResult: true,
          eventFileCount: 1,
          checkpointCount: 0,
          hasHparams: true,
          metrics: {},
        },
        {
          id: "run-new",
          group: "exp-a",
          experiment: "exp-a",
          modelType: "linears",
          model: "linear",
          preset: "BASELINE",
          dataset: "Mnist",
          runName: "new_20260601_030405",
          timestamp: "2026-06-01 03:04:05",
          version: "version_0",
          relativePath: "exp-a/linear/BASELINE/Mnist/new/version_0",
          hasResult: true,
          eventFileCount: 1,
          checkpointCount: 0,
          hasHparams: true,
          metrics: {},
        },
      ],
      total: 2,
      limit: 500,
      offset: 0,
      hasMore: false,
    });
    mocks.fetchLogTags.mockResolvedValue({
      runs: [
        {
          runId: "run-old",
          scalarTags: ["validation/accuracy"],
          histogramTags: [],
          imageTags: [],
          textTags: [],
        },
        {
          runId: "run-new",
          scalarTags: ["validation/accuracy"],
          histogramTags: [],
          imageTags: [],
          textTags: [],
        },
      ],
    });

    renderCompareWorkspace();

    expect(await screen.findByText("new_20260601_030405 · version_0"))
      .toBeInTheDocument();
    expect(screen.queryByText("old_20260601_010203 · version_0"))
      .not.toBeInTheDocument();
  });

  it("renders Compare scalar tags as a searchable multi-select instead of a checkbox grid", async () => {
    renderCompareWorkspace();
    const user = userEvent.setup();

    const scalarTagsControl = await selectDefaultCompareRuns(user);

    expect(
      screen.getByRole("heading", { name: "Scalar Tags" }),
    ).toBeInTheDocument();
    expect(scalarTagsControl).toHaveAttribute("role", "combobox");
    expect(
      screen.queryByLabelText(/select metric validation\/accuracy/i),
    ).not.toBeInTheDocument();

    const listbox = await openCompareScalarTags(user);
    expect(
      within(listbox).getByRole("option", {
        name: optionName("validation/accuracy"),
      }),
    ).toHaveAttribute("aria-selected", "true");
    expect(
      within(listbox).getByRole("option", { name: optionName("train/loss") }),
    ).toHaveAttribute("aria-selected", "true");
  });

  it("filters Compare scalar tags and updates graph/data output from dropdown selections", async () => {
    mockCompareScalarTags([
      "validation/accuracy",
      "train/loss",
      "test/accuracy",
      "debug/latency",
      "debug/memory",
      "debug/throughput",
    ]);
    renderCompareWorkspace();
    const user = userEvent.setup();

    await selectDefaultCompareRuns(user);
    const listbox = await openCompareScalarTags(user);
    const dropdownRoot = listbox.closest(".relative");
    expect(dropdownRoot).toBeInstanceOf(HTMLElement);

    const search = within(dropdownRoot as HTMLElement).getByRole("searchbox", {
      name: /search scalar tags/i,
    });
    await user.type(search, "memory");

    expect(
      within(listbox).getByRole("option", { name: optionName("debug/memory") }),
    ).toHaveAttribute("aria-selected", "false");
    expect(
      within(listbox).queryByRole("option", {
        name: optionName("validation/accuracy"),
      }),
    ).not.toBeInTheDocument();

    await user.click(
      within(listbox).getByRole("option", { name: optionName("debug/memory") }),
    );

    expect(
      await screen.findByRole("img", {
        name: /debug\/memory training run comparison chart/i,
      }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("tab", { name: /^data$/i }));
    expect(await screen.findByText("Metric Summary")).toBeInTheDocument();
    expect(screen.getAllByText("debug/memory").length).toBeGreaterThan(0);

    await openCompareScalarTags(user);
    await user.click(
      within(screen.getByRole("listbox", { name: "Scalar Tags options" }))
        .getByRole("option", { name: optionName("debug/memory") }),
    );
    await user.click(
      screen.getByRole("combobox", { name: /^Scalar Tags\b/i }),
    );

    await waitFor(() => {
      expect(screen.queryByText("debug/memory")).not.toBeInTheDocument();
    });
  });

  it("updates Compare scalar tag counts with Defaults, All, and None actions", async () => {
    mockCompareScalarTags([
      "validation/accuracy",
      "train/loss",
      "test/accuracy",
      "debug/latency",
      "debug/memory",
      "debug/throughput",
    ]);
    renderCompareWorkspace();
    const user = userEvent.setup();

    await selectDefaultCompareRuns(user);

    expect(
      await screen.findByRole("combobox", {
        name: /^Scalar Tags 4 \/ 6 selected$/i,
      }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^all$/i }));
    expect(
      await screen.findByRole("combobox", {
        name: /^Scalar Tags 6 \/ 6 selected$/i,
      }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^none$/i }));
    expect(
      await screen.findByRole("combobox", {
        name: /^Scalar Tags 0 \/ 6 selected$/i,
      }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^defaults$/i }));
    expect(
      await screen.findByRole("combobox", {
        name: /^Scalar Tags 4 \/ 6 selected$/i,
      }),
    ).toBeInTheDocument();
  });

  it("adds, removes, and resets run target cards around the eight-target cap", async () => {
    const runs = Array.from({ length: 10 }, (_, index) => {
      const number = String(index + 1).padStart(2, "0");
      const experiment = `exp-${number}`;
      return {
        id: `quick-run-${number}`,
        group: experiment,
        experiment,
        modelType: "linears",
        model: "linear",
        preset: "BASELINE",
        dataset: "Mnist",
        runName: `quick_${number}_20260601_010203`,
        timestamp: `2026-06-01 01:${number}:03`,
        version: "version_0",
        relativePath: `${experiment}/linear/BASELINE/Mnist/quick_${number}/version_0`,
        hasResult: true,
        eventFileCount: 1,
        checkpointCount: 0,
        hasHparams: true,
        metrics: {},
      };
    });
    mocks.fetchLogRuns.mockResolvedValue({
      runs,
      total: runs.length,
      limit: 500,
      offset: 0,
      hasMore: false,
    });
    mocks.fetchLogTags.mockResolvedValue({
      runs: runs.map((run) => ({
        runId: run.id,
        scalarTags: ["validation/accuracy"],
        histogramTags: [],
        imageTags: [],
        textTags: [],
      })),
    });

    renderCompareWorkspace();
    const user = userEvent.setup();

    await screen.findByText("Run Target 1");
    const addTarget = screen.getByRole("button", { name: /add target/i });
    for (let index = 0; index < 7; index += 1) {
      await user.click(addTarget);
    }

    await waitFor(() => {
      expect(screen.getAllByText(/^Run Target \d$/)).toHaveLength(8);
    });
    expect(addTarget).toBeDisabled();

    await user.click(
      screen.getByRole("button", { name: /remove run target 8/i }),
    );

    await waitFor(() => {
      expect(screen.getAllByText(/^Run Target \d$/)).toHaveLength(7);
    });
    expect(addTarget).toBeEnabled();

    await user.click(screen.getByRole("button", { name: /^reset$/i }));
    await waitFor(() => {
      expect(screen.getAllByText(/^Run Target \d$/)).toHaveLength(1);
    });
  });

  it("shows compact unresolved status when card filters match no scalar-capable run", async () => {
    renderCompareWorkspace();
    const user = userEvent.setup();

    await screen.findByText("Run Target 1");
    const [card] = runTargetCards();
    await selectSearchableOption(
      user,
      within(card).getByRole("combobox", { name: "Experiment" }),
      optionName("exp-c"),
      "exp-c",
    );

    expect(await screen.findByText("No scalar-capable run matches these filters."))
      .toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /training run comparison chart/i }))
      .not.toBeInTheDocument();
  });

  it("compares selected model/preset targets with split public model identity", async () => {
    renderCompareWorkspace();
    const user = userEvent.setup();
    await openConfigsMode(user);

    expect(
      await screen.findByRole("heading", { name: /model config comparison/i }),
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
    await openConfigsMode(user);

    await waitFor(() => {
      expect(screen.getByText("experts_linear")).toBeInTheDocument();
    });

    const targetCards = screen.getAllByText(/^Target \d$/).map((label) => {
      const card = label.closest("article");
      expect(card).toBeInstanceOf(HTMLElement);
      expect(card).toHaveClass(
        "rounded-[10px]",
        "border",
        "border-line",
        "bg-white/[0.018]",
        "p-4",
      );
      expect(card).not.toHaveClass("edge", "rounded-card");
      return card as HTMLElement;
    });
    await user.click(
      within(targetCards[1]).getByRole("button", { name: /use as target/i }),
    );

    expect(selectModel).toHaveBeenCalledWith("experts_linear", "experts");
    expect(selectPreset).toHaveBeenCalledWith("expert-baseline");
    expect(onUseTarget).toHaveBeenCalled();
  });

  it("searches comparison model and preset selectors", async () => {
    const consoleWarn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});

    renderCompareWorkspace();
    const user = userEvent.setup();
    await openConfigsMode(user);

    try {
      await waitFor(() => {
        expect(screen.getByText("experts_linear")).toBeInTheDocument();
      });

      const targetCards = screen.getAllByText(/^Target \d$/).map((label) => {
        const card = label.closest("article");
        expect(card).toBeInstanceOf(HTMLElement);
        expect(card).toHaveClass("rounded-[10px]", "border-line", "bg-white/[0.018]");
        expect(card).not.toHaveClass("edge", "rounded-card");
        return card as HTMLElement;
      });
      const firstTarget = targetCards[0];
      const modelControl = within(firstTarget).getByRole("combobox", {
        name: "Model",
      });

      await selectSearchableOption(user, modelControl, /experts_linear/i, "experts");

      const presetControl = within(firstTarget).getByRole("combobox", {
        name: "Preset",
      });
      await waitFor(() => {
        expect(modelControl).toHaveTextContent("experts_linear");
        expect(presetControl).toHaveTextContent("expert-baseline");
      });

      await selectSearchableOption(
        user,
        presetControl,
        "expert-baseline",
        "expert",
      );

      expect(presetControl).toHaveTextContent("expert-baseline");

      const consoleMessages = [...consoleWarn.mock.calls, ...consoleError.mock.calls]
        .flat()
        .map(String);
      expect(
        consoleMessages.some((message) =>
          message.includes("[QueriesObserver]: Duplicate Queries found"),
        ),
      ).toBe(false);
    } finally {
      consoleWarn.mockRestore();
      consoleError.mockRestore();
    }
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
    await openConfigsMode(user);

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
