import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { logsHarness } from "./support";

const {
  setup: setupLogsScenario,
  app: { render: renderWorkbench, reset: resetWorkbenchAppTestState },
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
} = logsHarness;

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function logOptionName(label: string) {
  return new RegExp(`^${escapeRegExp(label)}(?:\\b|$)`, "i");
}

const DEFAULT_EPOCH_SCALAR_TAGS = [
  "validation/accuracy_epoch",
  "validation/loss_epoch",
  "train/loss_epoch",
  "train/accuracy_epoch",
];

function scalarTagsWithEpochDefaults(...tags: string[]) {
  return Array.from(new Set([...DEFAULT_EPOCH_SCALAR_TAGS, ...tags]));
}

function epochScalarSeriesForRun(runId: string, offset = 0) {
  return [
    {
      runId,
      tag: "validation/accuracy_epoch",
      points: [{ step: 1, wallTime: 1780000000 + offset, value: 0.75 + offset / 1000 }],
    },
    {
      runId,
      tag: "validation/loss_epoch",
      points: [{ step: 1, wallTime: 1780000001 + offset, value: 0.5 - offset / 1000 }],
    },
    {
      runId,
      tag: "train/loss_epoch",
      points: [{ step: 1, wallTime: 1780000002 + offset, value: 0.6 - offset / 1000 }],
    },
    {
      runId,
      tag: "train/accuracy_epoch",
      points: [{ step: 1, wallTime: 1780000003 + offset, value: 0.7 + offset / 1000 }],
    },
  ];
}

async function openLogFilter(user: ReturnType<typeof userEvent.setup>, title: string) {
  const trigger = await screen.findByRole("combobox", {
    name: new RegExp(`^${escapeRegExp(title)}\\b`, "i"),
  });
  if (trigger.getAttribute("aria-expanded") !== "true") {
    await user.click(trigger);
  }
  return screen.findByRole("listbox", { name: `${title} options` });
}

function logFilterSection(title: string) {
  const trigger = screen.getByRole("combobox", {
    name: new RegExp(`^${escapeRegExp(title)}\\b`, "i"),
  });
  const section = trigger.closest("section");
  if (!(section instanceof HTMLElement)) {
    throw new Error(`Expected ${title} filter to render inside a section`);
  }
  return section;
}

async function findLogOption(
  user: ReturnType<typeof userEvent.setup>,
  title: string,
  label: string,
) {
  const listbox = await openLogFilter(user, title);
  return within(listbox).findByRole("option", { name: logOptionName(label) });
}

function queryOpenLogOption(title: string, label: string) {
  const listbox = screen.queryByRole("listbox", { name: `${title} options` });
  return listbox
    ? within(listbox).queryByRole("option", { name: logOptionName(label) })
    : null;
}

function expectLogOptionSelected(option: HTMLElement, selected = true) {
  expect(option).toHaveAttribute("aria-selected", selected ? "true" : "false");
}

function scalarChartSection(name: RegExp) {
  const chart = screen.getByRole("img", { name });
  const section = chart.closest("section");
  if (!(section instanceof HTMLElement)) {
    throw new Error(`Expected ${name} scalar chart to render inside a section`);
  }
  return section;
}

function accordionSectionGridFor(toggle: HTMLElement) {
  const section = toggle.closest("section");
  const grid = section?.parentElement;

  if (!(grid instanceof HTMLElement)) {
    throw new Error("Expected accordion section to render inside the section grid");
  }

  return grid;
}

function metricGroupBody(group: "train" | "validation" | "other") {
  const body = document.getElementById(`logs-metric-group-${group}`);
  if (!(body instanceof HTMLElement)) {
    throw new Error(`Expected ${group} metric group body to render`);
  }
  return body;
}

function testScoresSection() {
  const section = screen
    .getByRole("heading", { name: "Test Metric Leaderboards" })
    .closest("section");
  if (!(section instanceof HTMLElement)) {
    throw new Error("Expected Test Metric Leaderboards to render inside a section");
  }
  return section;
}

function testScoresSplitCheckbox() {
  return within(testScoresSection()).getByRole("checkbox", {
    name: /split by experiment/i,
  });
}

function expectElementsInDocumentOrder(elements: HTMLElement[]) {
  for (let index = 1; index < elements.length; index += 1) {
    const previous = elements[index - 1];
    const current = elements[index];
    expect(previous.compareDocumentPosition(current)).toBe(
      Node.DOCUMENT_POSITION_FOLLOWING,
    );
  }
}

function logScalarLegendButton(card: HTMLElement, runLabel: RegExp) {
  return within(card).getAllByRole("button", { name: runLabel })[0];
}

function expectLegendOpacity(button: HTMLElement, opacity: "normal" | "dimmed") {
  if (opacity === "dimmed") {
    expect(button).toHaveClass("opacity-30");
    expect(button).not.toHaveClass("opacity-100");
    return;
  }
  expect(button).toHaveClass("opacity-100");
  expect(button).not.toHaveClass("opacity-30");
}

function logRunsWithSharedDataset(dataset = "Mnist") {
  return {
    runs: logRunsResponse.runs.map((run) => ({
      ...run,
      dataset,
      relativePath: run.relativePath.replace(/\/(Mnist|Cifar10)\//, `/${dataset}/`),
    })),
  };
}

async function expectLogFilterSelection(
  user: ReturnType<typeof userEvent.setup>,
  title: string,
  label: string,
  selected: boolean,
) {
  expectLogOptionSelected(await findLogOption(user, title, label), selected);
}

async function clickLogOption(
  user: ReturnType<typeof userEvent.setup>,
  title: string,
  label: string,
) {
  const option = await findLogOption(user, title, label);
  await user.click(option);
  return option;
}

async function activateLogOptionToolbar(
  user: ReturnType<typeof userEvent.setup>,
  title: string,
  label: string,
) {
  const option = await findLogOption(user, title, label);
  await user.hover(option);
  return screen.findByRole("toolbar", { name: `${label} actions` });
}

async function setLogOptionSelection(
  user: ReturnType<typeof userEvent.setup>,
  title: string,
  label: string,
  selected: boolean,
) {
  const option = await findLogOption(user, title, label);
  const isSelected = option.getAttribute("aria-selected") === "true";
  if (isSelected !== selected) {
    await user.click(option);
  }
}

async function openMetricPlotSelector(
  user: ReturnType<typeof userEvent.setup>,
  group: "Train" | "Validation" | "Train vs Validation",
) {
  const trigger = await screen.findByRole("combobox", {
    name: new RegExp(`^${escapeRegExp(group)} plots\\b`, "i"),
  });
  if (trigger.getAttribute("aria-expanded") !== "true") {
    await user.click(trigger);
  }
  return screen.findByRole("listbox", { name: `${group} plots options` });
}

async function findMetricPlotOption(
  user: ReturnType<typeof userEvent.setup>,
  group: "Train" | "Validation" | "Train vs Validation",
  label: string,
) {
  const listbox = await openMetricPlotSelector(user, group);
  return within(listbox).findByRole("option", { name: logOptionName(label) });
}

async function queryMetricPlotOption(
  user: ReturnType<typeof userEvent.setup>,
  group: "Train" | "Validation" | "Train vs Validation",
  label: string,
) {
  const listbox = await openMetricPlotSelector(user, group);
  return within(listbox).queryByRole("option", { name: logOptionName(label) });
}

function trainValidationComparisonToggle() {
  return screen.getByRole("button", {
    name: /^Train vs Validation\s+\d+\s+plots?$/i,
  });
}

function findTrainValidationComparisonToggle() {
  return screen.findByRole("button", {
    name: /^Train vs Validation\s+\d+\s+plots?$/i,
  });
}

function trainValidationComparisonSection() {
  const section = trainValidationComparisonToggle().closest("section");
  if (!(section instanceof HTMLElement)) {
    throw new Error("Expected Train vs Validation accordion to render inside a section");
  }
  return section;
}

function hasCombinedTrainValidationRequest(
  requests: Array<{ tags: string[] }>,
  suffix: string,
) {
  return requests.some(
    (request) =>
      request.tags.includes(`train/${suffix}`) &&
      request.tags.includes(`validation/${suffix}`),
  );
}

function makeScrollable(element: HTMLElement) {
  Object.defineProperty(element, "clientHeight", {
    configurable: true,
    value: 180,
  });
  Object.defineProperty(element, "scrollHeight", {
    configurable: true,
    value: 900,
  });
  element.scrollTop = 760;
}

async function selectAllLogExperiments(user: ReturnType<typeof userEvent.setup>) {
  await screen.findByRole("combobox", { name: /^Experiments\b/i });
  const section = logFilterSection("Experiments");
  await user.click(within(section).getByRole("button", { name: /^all$/i }));
}

async function selectAllLogDatasets(user: ReturnType<typeof userEvent.setup>) {
  await screen.findByRole("combobox", { name: /^Datasets\b/i });
  const section = logFilterSection("Datasets");
  await user.click(within(section).getByRole("button", { name: /^all$/i }));
}

async function selectAllLogPresets(user: ReturnType<typeof userEvent.setup>) {
  await screen.findByRole("combobox", { name: /^Presets\b/i });
  const section = logFilterSection("Presets");
  await user.click(within(section).getByRole("button", { name: /^all$/i }));
}

async function selectLogExperiments(
  user: ReturnType<typeof userEvent.setup>,
  experiments: string[],
) {
  for (const experiment of experiments) {
    const option = await findLogOption(user, "Experiments", experiment);
    if (option.getAttribute("aria-selected") !== "true") {
      await user.click(option);
    }
  }
}

describe("WorkbenchApp Logs Workspace", () => {
  beforeEach(resetWorkbenchAppTestState);

  it("opens logs without selecting runs or loading TensorBoard data", async () => {
    const { logScalarRequests, logTagRequests } = setupLogsScenario();
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    expect(await screen.findByText("Historical Scalars")).toBeInTheDocument();
    await expectLogFilterSelection(user, "Experiments", "test_model", false);
    expect(await screen.findByRole("combobox", { name: /^Datasets\b/i }))
      .toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: /^Runs\b/i })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Datasets Mnist")).not.toBeInTheDocument();
    expect(await screen.findByText("No runs selected")).toBeInTheDocument();
    expect(logTagRequests).toHaveLength(0);
    expect(logScalarRequests).toHaveLength(0);
  });

  it("moves explicitly between the current target and all-runs scope", async () => {
    const { logRunRequests } = setupLogsScenario();
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await screen.findByRole("combobox", { name: /^Experiments\b/i });

    const experimentSection = logFilterSection("Experiments");
    const currentTarget = within(experimentSection).getByRole("button", {
      name: /current target/i,
    });
    const allRuns = within(experimentSection).getByRole("button", {
      name: /all runs/i,
    });
    expect(currentTarget).toHaveAttribute("aria-pressed", "true");
    expect(allRuns).toHaveAttribute("aria-pressed", "false");

    await user.click(allRuns);

    await waitFor(() => {
      expect(logRunRequests).toContainEqual(
        expect.objectContaining({
          experiments: ["test_model", "test_model_2"],
          limit: 100,
          offset: 0,
        }),
      );
      const updatedSection = logFilterSection("Experiments");
      expect(
        within(updatedSection).getByRole("button", { name: /current target/i }),
      ).toHaveAttribute("aria-pressed", "false");
      expect(
        within(updatedSection).getByRole("button", { name: /all runs/i }),
      ).toHaveAttribute("aria-pressed", "true");
    });

    await user.click(
      within(logFilterSection("Experiments")).getByRole("button", {
        name: /current target/i,
      }),
    );

    await waitFor(() => {
      const updatedSection = logFilterSection("Experiments");
      expect(
        within(updatedSection).getByRole("button", { name: /current target/i }),
      ).toHaveAttribute("aria-pressed", "true");
      expect(
        within(updatedSection).getByRole("button", { name: /all runs/i }),
      ).toHaveAttribute("aria-pressed", "false");
    });
    const finalExperimentSection = logFilterSection("Experiments");
    expect(within(finalExperimentSection).getByRole("button", { name: /^all$/i }))
      .toBeInTheDocument();
    expect(within(finalExperimentSection).getByRole("button", { name: /^none$/i }))
      .toBeInTheDocument();
    expect(screen.queryByText(/^Runs$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^Tags$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^linear · BASELINE · Mnist$/)).not.toBeInTheDocument();
  });

  it("opens logs scoped to the current target dataset and applies common filters from experiment selection", async () => {
    const { logRunRequests, logScalarRequests } = setupLogsScenario();
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    expect(await screen.findByText("Historical Scalars")).toBeInTheDocument();
    await waitFor(() => {
      expect(logRunRequests).toEqual(
        expect.arrayContaining([
          {
            experiments: [],
            modelTypes: ["linears"],
            models: ["linear"],
            presets: ["BASELINE"],
            datasets: ["Mnist"],
            experimentTask: null,
            hasEventFiles: "true",
            limit: 5,
            offset: 0,
            projection: null,
          },
        ]),
      );
    });
    expect(screen.getByRole("button", { name: /current target/i }))
      .toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: /all runs/i }))
      .toHaveAttribute("aria-pressed", "false");
    await expectLogFilterSelection(user, "Experiments", "test_model", false);
    await selectLogExperiments(user, ["test_model"]);
    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();
    expect(logMetricGroupToggle("Train")).toHaveAttribute("aria-expanded", "true");
    expect(logMetricGroupToggle("Validation")).toHaveAttribute("aria-expanded", "true");
    expect(screen.queryByRole("button", { name: /^Test\s+\d+\s+metrics?$/i }))
      .not.toBeInTheDocument();
    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(3);
      expect(logScalarRequests).toEqual(expect.arrayContaining([
        {
          runIds: ["log-mnist"],
          tags: ["train/accuracy_epoch", "train/loss_epoch"],
          maxPoints: 500,
          sampling: "tail",
        },
        {
          runIds: ["log-mnist"],
          tags: ["validation/accuracy_epoch", "validation/loss_epoch"],
          maxPoints: 500,
          sampling: "tail",
        },
        {
          runIds: ["log-mnist"],
          tags: ["validation/accuracy"],
          maxPoints: 500,
          sampling: "tail",
        },
      ]));
    });
    expect(await screen.findByRole("img", { name: /train\/loss_epoch scalar chart/i }))
      .toBeInTheDocument();
    expect(screen.queryByRole("table", { name: /test\/accuracy test leaderboard/i }))
      .not.toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /test\/accuracy scalar chart/i }))
      .not.toBeInTheDocument();

    await selectAllLogExperiments(user);
    await waitFor(() => {
      expect(logRunRequests).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            experiments: ["test_model", "test_model_2"],
            datasets: [],
            limit: 100,
            offset: 0,
          }),
        ]),
      );
    });
    await waitFor(() => {
      expect(screen.getByRole("combobox", { name: /^Datasets\b/i })).toBeDisabled();
      expect(screen.getByRole("combobox", { name: /^Models\b/i })).toBeEnabled();
      expect(screen.getByRole("combobox", { name: /^Presets\b/i })).toBeEnabled();
    });
    expect(await screen.findByText("No runs selected")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /^Test\s+\d+\s+metrics?$/i }))
      .not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /start training/i }))
      .not.toBeInTheDocument();
  });

  it("renders the train-vs-validation accordion after best run and keeps it cold while collapsed", async () => {
    const { logScalarRequests } = setupLogsScenario();
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);
    await clickLogOption(user, "Scalar Tags", "test/accuracy");

    const bestRunSection = (await screen.findByRole("heading", {
      name: "Best Run by Selected Metric",
    })).closest("section");
    expect(bestRunSection).toBeInstanceOf(HTMLElement);
    const comparisonToggle = trainValidationComparisonToggle();
    const comparisonSection = trainValidationComparisonSection();
    const testSection = testScoresSection();

    expectElementsInDocumentOrder([
      bestRunSection as HTMLElement,
      comparisonSection,
      testSection,
    ]);
    expect(comparisonToggle).toHaveAttribute("aria-expanded", "false");
    await waitFor(() => {
      expect(logScalarRequests.length).toBeGreaterThan(0);
    });
    expect(hasCombinedTrainValidationRequest(logScalarRequests, "accuracy_epoch"))
      .toBe(false);

    await user.click(comparisonToggle);

    expect(
      await screen.findByRole("combobox", {
        name: /^Train vs Validation plots\b/i,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("radiogroup", {
        name: /Train vs Validation chart layout/i,
      }),
    ).toBeInTheDocument();
  });

  it("renders a selected train-vs-validation pair and omits incomplete pair options", async () => {
    const { logScalarRequests } = setupLogsScenario({
      logTagsByRun: {
        "log-mnist": scalarTagsWithEpochDefaults(
          "validation/precision",
          "train/recall",
        ),
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);

    await findTrainValidationComparisonToggle();
    const comparisonSection = trainValidationComparisonSection();
    await user.click(
      within(comparisonSection).getByRole("button", {
        name: /select no Train vs Validation plots/i,
      }),
    );
    await user.click(trainValidationComparisonToggle());
    expect(await screen.findByText("No paired plots selected")).toBeInTheDocument();

    const accuracyOption = await findMetricPlotOption(
      user,
      "Train vs Validation",
      "accuracy_epoch",
    );
    expect(
      await queryMetricPlotOption(user, "Train vs Validation", "validation/precision"),
    ).not.toBeInTheDocument();
    expect(
      await queryMetricPlotOption(user, "Train vs Validation", "train/recall"),
    ).not.toBeInTheDocument();
    await user.click(accuracyOption);

    expect(
      await screen.findByRole("img", {
        name: /accuracy_epoch train vs validation scalar chart/i,
      }),
    ).toBeInTheDocument();
    await waitFor(() => {
      expect(hasCombinedTrainValidationRequest(logScalarRequests, "accuracy_epoch"))
        .toBe(true);
    });
    expect(screen.getByText(/test_model · Mnist .* · Train/)).toBeInTheDocument();
    expect(screen.getByText(/test_model · Mnist .* · Validation/))
      .toBeInTheDocument();
  });

  it("renders separate train-vs-validation legend entries for each visible run", async () => {
    const runs = [
      logRunsResponse.runs[0],
      {
        ...logRunsResponse.runs[0],
        id: "log-mnist-wide",
        runName: "wide_20260601_030405",
        timestamp: "2026-06-01 03:04:05",
        relativePath:
          "test_model/linear/WIDE/Mnist/wide_20260601_030405/version_0",
      },
    ];
    setupLogsScenario({
      logRunsResponse: { runs },
      logExperimentsResponse: {
        experiments: [
          { experiment: "test_model", runCount: 2, relativePath: "test_model" },
        ],
      },
      logTagsByRun: {
        "log-mnist": scalarTagsWithEpochDefaults(),
        "log-mnist-wide": scalarTagsWithEpochDefaults(),
      },
      logScalarSeries: [
        ...epochScalarSeriesForRun("log-mnist"),
        ...epochScalarSeriesForRun("log-mnist-wide", 100),
      ],
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);
    await user.click(await findTrainValidationComparisonToggle());

    const chart = await screen.findByRole("img", {
      name: /accuracy_epoch train vs validation scalar chart/i,
    });
    const card = chart.closest("section");
    if (!(card instanceof HTMLElement)) {
      throw new Error("Expected combined scalar chart to render inside a section");
    }
    expect(await within(card).findAllByText(/· Train$/)).toHaveLength(2);
    expect(within(card).getAllByText(/· Validation$/)).toHaveLength(2);
  });

  it("ranks best runs independently from selected scalar chart tags", async () => {
    const runs = [
      logRunsResponse.runs[0],
      {
        ...logRunsResponse.runs[1],
        group: "test_model",
        experiment: "test_model",
        relativePath:
          "test_model/linear/BASELINE/Cifar10/bbb_20260601_020304/version_0",
      },
    ];
    setupLogsScenario({
      logRunsResponse: { runs },
      logExperimentsResponse: {
        experiments: [
          { experiment: "test_model", runCount: 2, relativePath: "test_model" },
        ],
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectAllLogExperiments(user);
    await clickLogOption(user, "Scalar Tags", "train/loss");
    await clickLogOption(user, "Scalar Tags", "validation/accuracy");
    await selectAllLogDatasets(user);

    const bestRunHeading = await screen.findByRole("heading", {
      name: "Best Run by Selected Metric",
    });
    const bestRunPanel = bestRunHeading.closest("section");
    expect(bestRunPanel).toBeInstanceOf(HTMLElement);
    const panel = bestRunPanel as HTMLElement;
    expect(within(panel).getByText(/best run per visible dataset/i))
      .toBeInTheDocument();
    expect(within(panel).queryByRole("combobox", { name: /best run dataset/i }))
      .not.toBeInTheDocument();

    const bestRunTable = await within(panel).findByRole("table", {
      name: /validation\/accuracy best run leaderboard/i,
    });
    expect(within(bestRunTable).getByRole("columnheader", { name: "Dataset" }))
      .toBeInTheDocument();
    expect(within(bestRunTable).queryByRole("columnheader", { name: "Rank" }))
      .not.toBeInTheDocument();
    await waitFor(() => {
      expect(within(bestRunTable).getByText("Mnist")).toBeInTheDocument();
      expect(within(bestRunTable).getByText("Cifar10")).toBeInTheDocument();
      expect(within(bestRunTable).getByText("0.8")).toBeInTheDocument();
      expect(within(bestRunTable).getByText("aaa_20260601_010203"))
        .toBeInTheDocument();
      expect(within(bestRunTable).getByText("0.55")).toBeInTheDocument();
      expect(within(bestRunTable).getByText("bbb_20260601_020304"))
        .toBeInTheDocument();
    });

    await clickLogOption(user, "Scalar Tags", "validation/accuracy_epoch");

    await waitFor(() => {
      expect(screen.queryByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
        .not.toBeInTheDocument();
    });
    expect(
      within(panel).getByRole("table", {
        name: /validation\/accuracy best run leaderboard/i,
      }),
    ).toBeInTheDocument();

    await user.click(
      within(panel).getAllByRole("button", {
        name: /open details for test_model · Cifar10/i,
      })[0],
    );

    const detailsPanel = screen.getByRole("heading", { name: "Run Details" })
      .closest('[data-workbench-region="details"]');
    expect(detailsPanel).not.toBeNull();
    expect(within(detailsPanel as HTMLElement).getByText("Cifar10"))
      .toBeInTheDocument();
  });

  it("keeps scalar tag options available but checks only epoch defaults", async () => {
    setupLogsScenario({
      logTagsByRun: {
        "log-mnist": scalarTagsWithEpochDefaults(
          "train/loss",
          "train/accuracy",
          "train/calibration/ece",
          "train/confidence/mean",
          "train/f1_score",
          "train/per_class/class_0/f1_score",
          "train/kaggle_logloss",
          "validation/loss",
          "validation/accuracy",
          "validation/calibration/ece",
          "validation/confidence/mean",
          "validation/f1_score",
          "validation/per_class/class_0/f1_score",
          "validation/kaggle_auc",
        ),
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);
    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();

    const tagList = await openLogFilter(user, "Scalar Tags");
    for (const tag of DEFAULT_EPOCH_SCALAR_TAGS) {
      expectLogOptionSelected(
        await within(tagList).findByRole("option", { name: logOptionName(tag) }),
        true,
      );
    }
    for (const tag of [
      "train/loss",
      "train/accuracy",
      "train/calibration/ece",
      "train/confidence/mean",
      "train/f1_score",
      "train/per_class/class_0/f1_score",
      "train/kaggle_logloss",
      "validation/loss",
      "validation/accuracy",
      "validation/calibration/ece",
      "validation/confidence/mean",
      "validation/f1_score",
      "validation/per_class/class_0/f1_score",
      "validation/kaggle_auc",
    ]) {
      expectLogOptionSelected(
        await within(tagList).findByRole("option", { name: logOptionName(tag) }),
        false,
      );
    }

    expect(logMetricGroupToggle("Train")).toHaveAccessibleName("Train 7 metrics");
    expect(logMetricGroupToggle("Validation")).toHaveAccessibleName(
      "Validation 7 metrics",
    );
    expect(
      screen.getByRole("combobox", { name: /^Train plots\s+2 \/ 7 selected$/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("combobox", {
        name: /^Validation plots\s+2 \/ 7 selected$/i,
      }),
    ).toBeInTheDocument();
    for (const tag of [
      "train/loss",
      "train/accuracy",
      "train/calibration/ece",
      "train/confidence/mean",
      "train/f1_score",
    ]) {
      expectLogOptionSelected(await findMetricPlotOption(user, "Train", tag), false);
    }
    for (const tag of ["train/per_class/class_0/f1_score", "train/kaggle_logloss"]) {
      expect(await queryMetricPlotOption(user, "Train", tag)).not.toBeInTheDocument();
    }
    for (const tag of [
      "validation/loss",
      "validation/accuracy",
      "validation/calibration/ece",
      "validation/confidence/mean",
      "validation/f1_score",
    ]) {
      expectLogOptionSelected(
        await findMetricPlotOption(user, "Validation", tag),
        false,
      );
    }
    for (const tag of [
      "validation/per_class/class_0/f1_score",
      "validation/kaggle_auc",
    ]) {
      expect(await queryMetricPlotOption(user, "Validation", tag))
        .not.toBeInTheDocument();
    }
  });

  it("orders Train immediately after Validation before validation diagnostics", async () => {
    const matrixRateTags = [
      "validation/confusion_matrix/true_class_0/predicted_class_0/rate",
      "validation/confusion_matrix/true_class_0/predicted_class_1/rate",
      "validation/confusion_matrix/true_class_1/predicted_class_0/rate",
      "validation/confusion_matrix/true_class_1/predicted_class_1/rate",
    ];
    const defaultMnistTags = logTagsByRun["log-mnist"];
    if (!Array.isArray(defaultMnistTags)) {
      throw new Error("Expected default Mnist log tags to be scalar tag names");
    }
    setupLogsScenario({
      logTagsByRun: {
        "log-mnist": {
          scalarTags: [...defaultMnistTags, ...matrixRateTags],
          histogramTags: [],
          imageTags: ["validation/examples/predictions"],
          textTags: ["validation/examples/predictions/text_summary"],
        },
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);
    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();

    await clickLogOption(user, "Scalar Tags", "test/accuracy");
    await clickLogOption(user, "Scalar Tags", "main_model.0.model/weights/mean");

    const bestRunSection = screen
      .getByRole("heading", { name: "Best Run by Selected Metric" })
      .closest("section");
    expect(bestRunSection).toBeInstanceOf(HTMLElement);

    expectElementsInDocumentOrder([
      bestRunSection as HTMLElement,
      testScoresSection(),
      logMetricGroupToggle("Validation"),
      logMetricGroupToggle("Train"),
      logValidationExamplesToggle(),
      screen.getByRole("button", { name: /^Confusion Matrix\b/i }),
      logMetricGroupToggle("Other"),
    ]);
  });

  it("filters logs by dataset across tags, scalars, charts, and details", async () => {
    const runs = [
      {
        ...logRunsResponse.runs[0],
        id: "multi-mnist",
        group: "multi_dataset",
        experiment: "multi_dataset",
        dataset: "Mnist",
        runName: "multi_mnist_20260601_010203",
        timestamp: "2026-06-01 01:02:03",
        relativePath:
          "multi_dataset/linear/BASELINE/Mnist/multi_mnist_20260601_010203/version_0",
        metrics: { "test/accuracy": 0.91 },
      },
      {
        ...logRunsResponse.runs[0],
        id: "multi-cifar",
        group: "multi_dataset",
        experiment: "multi_dataset",
        dataset: "Cifar10",
        runName: "multi_cifar_20260601_020304",
        timestamp: "2026-06-01 02:03:04",
        relativePath:
          "multi_dataset/linear/BASELINE/Cifar10/multi_cifar_20260601_020304/version_0",
        metrics: { "test/accuracy": 0.73 },
      },
    ];
    const { logScalarRequests, logTagRequests } = setupLogsScenario({
      logRunsResponse: { runs },
      logExperimentsResponse: {
        experiments: [
          { experiment: "multi_dataset", runCount: 2, relativePath: "multi_dataset" },
        ],
      },
      logTagsByRun: {
        "multi-mnist": scalarTagsWithEpochDefaults("train/loss", "validation/accuracy"),
        "multi-cifar": scalarTagsWithEpochDefaults("train/loss", "validation/accuracy"),
      },
      logScalarSeries: [
        ...epochScalarSeriesForRun("multi-mnist"),
        ...epochScalarSeriesForRun("multi-cifar", 100),
        {
          runId: "multi-mnist",
          tag: "train/loss",
          points: [{ step: 1, wallTime: 1780000000, value: 0.44 }],
        },
        {
          runId: "multi-mnist",
          tag: "validation/accuracy",
          points: [{ step: 1, wallTime: 1780000000, value: 0.81 }],
        },
        {
          runId: "multi-cifar",
          tag: "train/loss",
          points: [{ step: 1, wallTime: 1780000100, value: 0.66 }],
        },
        {
          runId: "multi-cifar",
          tag: "validation/accuracy",
          points: [{ step: 1, wallTime: 1780000100, value: 0.59 }],
        },
      ],
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectAllLogExperiments(user);
    await selectAllLogDatasets(user);

    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();
    await waitFor(() => {
      expect(
        screen.getByRole("combobox", { name: /^Datasets\s+2 \/ 2 selected$/i }),
      ).toBeInTheDocument();
    });
    await waitFor(() => {
      expect(logTagRequests.at(-1)).toEqual({
        runIds: ["multi-mnist", "multi-cifar"],
      });
    });

    await clickLogOption(user, "Datasets", "Mnist");

    await waitFor(() => {
      expect(logTagRequests.at(-1)).toEqual({ runIds: ["multi-cifar"] });
    });
    await waitFor(() => {
      expect(logScalarRequests).toContainEqual({
        runIds: ["multi-cifar"],
        tags: ["validation/accuracy_epoch", "validation/loss_epoch"],
        maxPoints: 500,
        sampling: "tail",
      });
      expect(logScalarRequests).toContainEqual({
        runIds: ["multi-cifar"],
        tags: ["train/accuracy_epoch", "train/loss_epoch"],
        maxPoints: 500,
        sampling: "tail",
      });
    });
    expect(within(logFilterSection("Datasets")).getByText("1 / 2")).toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();
    await waitFor(() => {
      expect(screen.queryByText(/multi_mnist_20260601_010203/))
        .not.toBeInTheDocument();
    });
    expect(screen.getAllByText(/multi_dataset · Cifar10 · linear · linears · BASELINE/).length)
      .toBeGreaterThan(0);

    const detailsPanel = screen.getByRole("heading", { name: "Run Details" })
      .closest('[data-workbench-region="details"]');
    expect(detailsPanel).not.toBeNull();
    expect(within(detailsPanel as HTMLElement).getByTitle("multi_cifar_20260601_020304"))
      .toBeInTheDocument();
    expect(within(detailsPanel as HTMLElement).queryByTitle("multi_mnist_20260601_010203"))
      .not.toBeInTheDocument();

    await selectAllLogDatasets(user);

    await waitFor(() => {
      expect(logTagRequests.at(-1)).toEqual({
        runIds: ["multi-mnist", "multi-cifar"],
      });
    });
    await waitFor(() => {
      expect(logScalarRequests).toContainEqual({
        runIds: ["multi-mnist", "multi-cifar"],
        tags: ["validation/accuracy_epoch", "validation/loss_epoch"],
        maxPoints: 500,
        sampling: "tail",
      });
      expect(logScalarRequests).toContainEqual({
        runIds: ["multi-mnist", "multi-cifar"],
        tags: ["train/accuracy_epoch", "train/loss_epoch"],
        maxPoints: 500,
        sampling: "tail",
      });
    });
    expect(screen.getAllByText(/multi_dataset · Mnist · linear · linears · BASELINE/).length)
      .toBeGreaterThan(0);
    expect(screen.getAllByText(/multi_dataset · Cifar10 · linear · linears · BASELINE/).length)
      .toBeGreaterThan(0);
  });

  it("updates scalar charts when preset filters return to All", async () => {
    const runs = [
      {
        ...logRunsResponse.runs[0],
        id: "multi-baseline",
        group: "multi_preset",
        experiment: "multi_preset",
        preset: "BASELINE",
        dataset: "Mnist",
        runName: "multi_baseline_20260601_010203",
        timestamp: "2026-06-01 01:02:03",
        relativePath:
          "multi_preset/linear/BASELINE/Mnist/multi_baseline_20260601_010203/version_0",
        metrics: { "test/accuracy": 0.91 },
      },
      {
        ...logRunsResponse.runs[0],
        id: "multi-wide",
        group: "multi_preset",
        experiment: "multi_preset",
        preset: "WIDE",
        dataset: "Mnist",
        runName: "multi_wide_20260601_020304",
        timestamp: "2026-06-01 02:03:04",
        relativePath:
          "multi_preset/linear/WIDE/Mnist/multi_wide_20260601_020304/version_0",
        metrics: { "test/accuracy": 0.73 },
      },
    ];
    const expandedScalarResponse = deferred<unknown>();
    let delayExpandedScalarReads = false;
    const { logScalarRequests, logTagRequests } = setupLogsScenario({
      logRunsResponse: { runs },
      logExperimentsResponse: {
        experiments: [
          { experiment: "multi_preset", runCount: 2, relativePath: "multi_preset" },
        ],
      },
      logTagsByRun: {
        "multi-baseline": scalarTagsWithEpochDefaults("train/loss", "validation/accuracy"),
        "multi-wide": scalarTagsWithEpochDefaults("train/loss", "validation/accuracy"),
      },
      logScalarSeries: [
        ...epochScalarSeriesForRun("multi-baseline"),
        ...epochScalarSeriesForRun("multi-wide", 100),
        {
          runId: "multi-baseline",
          tag: "train/loss",
          points: [{ step: 1, wallTime: 1780000000, value: 0.44 }],
        },
        {
          runId: "multi-baseline",
          tag: "validation/accuracy",
          points: [{ step: 1, wallTime: 1780000000, value: 0.81 }],
        },
        {
          runId: "multi-wide",
          tag: "train/loss",
          points: [{ step: 1, wallTime: 1780000100, value: 0.66 }],
        },
        {
          runId: "multi-wide",
          tag: "validation/accuracy",
          points: [{ step: 1, wallTime: 1780000100, value: 0.59 }],
        },
      ],
      logScalarResponseFactory: (body) => {
        if (
          delayExpandedScalarReads &&
          body.runIds.includes("multi-baseline") &&
          body.runIds.includes("multi-wide") &&
          body.tags.includes("validation/accuracy_epoch")
        ) {
          return expandedScalarResponse.promise;
        }
        return undefined;
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["multi_preset"]);
    await selectAllLogPresets(user);
    await waitFor(() => {
      expect(
        screen.getByRole("combobox", { name: /^Presets\s+2 \/ 2 selected$/i }),
      ).toBeInTheDocument();
    });

    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();
    await waitFor(() => {
      expect(logTagRequests.at(-1)).toEqual({
        runIds: ["multi-baseline", "multi-wide"],
      });
    });

    await clickLogOption(user, "Presets", "BASELINE");

    await waitFor(() => {
      expect(logTagRequests.at(-1)).toEqual({ runIds: ["multi-wide"] });
    });
    await waitFor(() => {
      expect(logScalarRequests).toContainEqual({
        runIds: ["multi-wide"],
        tags: ["validation/accuracy_epoch", "validation/loss_epoch"],
        maxPoints: 500,
        sampling: "tail",
      });
      expect(logScalarRequests).toContainEqual({
        runIds: ["multi-wide"],
        tags: ["train/accuracy_epoch", "train/loss_epoch"],
        maxPoints: 500,
        sampling: "tail",
      });
    });
    expect(screen.queryByText(/multi_baseline_20260601_010203/))
      .not.toBeInTheDocument();
    expect(screen.getAllByText(/multi_preset · Mnist · linear · linears · WIDE/).length)
      .toBeGreaterThan(0);

    delayExpandedScalarReads = true;
    await selectAllLogPresets(user);

    await waitFor(() => {
      expect(logTagRequests.at(-1)).toEqual({
        runIds: ["multi-baseline", "multi-wide"],
      });
    });
    await waitFor(() => {
      expect(logScalarRequests).toContainEqual({
        runIds: ["multi-baseline", "multi-wide"],
        tags: ["validation/accuracy_epoch", "validation/loss_epoch"],
        maxPoints: 500,
        sampling: "tail",
      });
      expect(logScalarRequests).toContainEqual({
        runIds: ["multi-baseline", "multi-wide"],
        tags: ["train/accuracy_epoch", "train/loss_epoch"],
        maxPoints: 500,
        sampling: "tail",
      });
    });
    const retainedChart = screen.getByRole("img", {
      name: /validation\/accuracy_epoch scalar chart/i,
    });
    const retainedChartSection = retainedChart.closest("section");
    expect(retainedChartSection).toBeInstanceOf(HTMLElement);
    expect(
      within(retainedChartSection as HTMLElement).getByText(/1 line/i),
    ).toBeInTheDocument();

    expandedScalarResponse.resolve({
      series: [
        {
          runId: "multi-baseline",
          tag: "validation/accuracy_epoch",
          points: [{ step: 1, wallTime: 1780000000, value: 0.81 }],
        },
        {
          runId: "multi-wide",
          tag: "validation/accuracy_epoch",
          points: [{ step: 1, wallTime: 1780000100, value: 0.59 }],
        },
      ],
    });

    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();
    expect(screen.getAllByText(/multi_preset · Mnist · linear · linears · BASELINE/).length)
      .toBeGreaterThan(0);
    expect(screen.getAllByText(/multi_preset · Mnist · linear · linears · WIDE/).length)
      .toBeGreaterThan(0);
  });

  it("keeps validation examples collapsed until the accordion is opened", async () => {
    const { logMediaRequests } = setupLogsScenario({
      logTagsByRun: {
        "log-mnist": {
          scalarTags: scalarTagsWithEpochDefaults("validation/accuracy"),
          histogramTags: [],
          imageTags: ["validation/examples/predictions"],
          textTags: ["validation/examples/predictions/text_summary"],
        },
      },
      logMediaResponse: {
        images: [
          {
            runId: "log-mnist",
            tag: "validation/examples/predictions",
            step: 2,
            wallTime: 1780000001,
            mimeType: "image/png",
            dataUrl: "data:image/png;base64,AAAA",
          },
        ],
        texts: [
          {
            runId: "log-mnist",
            tag: "validation/examples/predictions/text_summary",
            step: 2,
            wallTime: 1780000001,
            text: "cat -> dog",
          },
        ],
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);
    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();

    const examplesToggle = logValidationExamplesToggle();
    expect(examplesToggle).toHaveAttribute("aria-expanded", "false");
    expect(logMediaRequests).toHaveLength(0);
    expect(
      screen.queryByAltText(/Most-confident wrong validation predictions/i),
    ).not.toBeInTheDocument();

    await user.click(examplesToggle);

    await waitFor(() => {
      expect(logMediaRequests).toEqual([
        {
          runIds: ["log-mnist"],
          imageTags: ["validation/examples/predictions"],
          textTags: ["validation/examples/predictions/text_summary"],
        },
      ]);
    });
    expect(examplesToggle).toHaveAttribute("aria-expanded", "true");
    expect(
      await screen.findByAltText(/Most-confident wrong validation predictions/i),
    ).toBeInTheDocument();
    expect(screen.getByText("cat -> dog")).toBeInTheDocument();
  });

  it("loads confusion matrix scalars lazily from hidden matrix rate tags", async () => {
    const matrixRateTags = [
      "validation/confusion_matrix/true_class_0/predicted_class_0/rate",
      "validation/confusion_matrix/true_class_0/predicted_class_1/rate",
      "validation/confusion_matrix/true_class_1/predicted_class_0/rate",
      "validation/confusion_matrix/true_class_1/predicted_class_1/rate",
    ];
    const matrixCountTags = [
      "validation/confusion_matrix/true_class_0/predicted_class_0/count",
      "validation/confusion_matrix/true_class_0/predicted_class_1/count",
      "validation/confusion_matrix/true_class_1/predicted_class_0/count",
      "validation/confusion_matrix/true_class_1/predicted_class_1/count",
    ];
    const { logScalarRequests } = setupLogsScenario({
      logTagsByRun: {
        "log-mnist": {
          scalarTags: [...matrixRateTags, ...matrixCountTags],
          histogramTags: [],
          imageTags: [],
          textTags: [],
        },
      },
      logScalarSeries: matrixRateTags.map((tag, index) => ({
        runId: "log-mnist",
        tag,
        points: [
          {
            step: 4,
            wallTime: 1780000010 + index,
            value: [0.8, 0.2, 0.1, 0.9][index],
          },
        ],
      })),
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);

    const matrixToggle = await screen.findByRole("button", {
      name: /^Confusion Matrix\s+Available$/i,
    });
    expect(matrixToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("No scalar tags selected")).not.toBeInTheDocument();
    expect(screen.queryByLabelText(`Scalar Tags ${matrixRateTags[0]}`))
      .not.toBeInTheDocument();
    expect(screen.queryByLabelText(`Scalar Tags ${matrixCountTags[0]}`))
      .not.toBeInTheDocument();
    expect(logScalarRequests).toHaveLength(0);

    await user.click(matrixToggle);

    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(1);
    });
    expect(logScalarRequests[0]).toEqual({
      runIds: ["log-mnist"],
      tags: [...matrixRateTags].sort((a, b) => a.localeCompare(b)),
      maxPoints: 500,
      sampling: "tail",
    });
    const matrixDataToggle = await screen.findByRole("button", {
      name: /view validation confusion matrix for aaa_20260601_010203.*2 classes data/i,
    });
    expect(matrixDataToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("table")).not.toBeInTheDocument();

    await user.click(matrixDataToggle);

    const matrixTable = await screen.findByRole("table", {
      name: /validation confusion matrix for aaa_20260601_010203.*2 classes/i,
    });
    expect(within(matrixTable).getAllByRole("cell").map((cell) => cell.textContent))
      .toEqual(["0.8", "0.2", "0.1", "0.9"]);
    expect(
      await screen.findByRole("button", { name: /^Confusion Matrix\s+1 matrix$/i }),
    ).toHaveAttribute("aria-expanded", "true");
  });

  it("collapses logs metric groups without changing selected tags or refetching scalars", async () => {
    const { logScalarRequests } = setupLogsScenario();
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);

    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /train\/loss_epoch scalar chart/i }))
      .toBeInTheDocument();
    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(3);
    });
    expect(
      screen.queryByRole("button", { name: /^Test\s+\d+\s+metrics?$/i }),
    ).not.toBeInTheDocument();

    await user.click(logMetricGroupToggle("Train"));

    await waitFor(() => {
      expect(logMetricGroupToggle("Train")).toHaveAttribute("aria-expanded", "false");
    });
    expect(screen.queryByRole("img", { name: /train\/loss_epoch scalar chart/i }))
      .not.toBeInTheDocument();
    await expectLogFilterSelection(user, "Scalar Tags", "train/loss_epoch", true);
    expect(logScalarRequests).toHaveLength(3);

    await clickLogOption(user, "Scalar Tags", "main_model.0.model/weights/mean");
    await expectLogFilterSelection(
      user,
      "Scalar Tags",
      "main_model.0.model/weights/mean",
      true,
    );
    const otherToggle = await screen.findByRole("button", {
      name: /^Other\s+1\s+metric$/i,
    });
    expect(otherToggle).toHaveAttribute("aria-expanded", "false");
    expect(logScalarRequests).toHaveLength(3);

    await user.click(otherToggle);
    await waitFor(() => {
      expect(logMetricGroupToggle("Other")).toHaveAttribute("aria-expanded", "true");
    });
    expect(
      await screen.findByRole("img", {
        name: /main_model\.0\.model\/weights\/mean scalar chart/i,
      }),
    )
      .toBeInTheDocument();
    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(4);
    });
    expect(logScalarRequests[3]).toMatchObject({
      runIds: ["log-mnist"],
      tags: ["main_model.0.model/weights/mean"],
    });

    await user.click(logMetricGroupToggle("Other"));

    await waitFor(() => {
      expect(logMetricGroupToggle("Other")).toHaveAttribute("aria-expanded", "false");
    });
    expect(
      screen.queryByRole("img", {
        name: /main_model\.0\.model\/weights\/mean scalar chart/i,
      }),
    )
      .not.toBeInTheDocument();
    expect(logScalarRequests).toHaveLength(4);
  });

  it("filters train and validation plots from accordion plot selectors", async () => {
    const { logScalarRequests } = setupLogsScenario();
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);

    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /train\/loss_epoch scalar chart/i }))
      .toBeInTheDocument();
    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(3);
    });
    const scalarRequestCount = logScalarRequests.length;

    expect(
      screen.getByRole("combobox", { name: /^Train plots\s+2 \/ 4 selected$/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("combobox", { name: /^Validation plots\s+2 \/ 3 selected$/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "Train plot controls" })).toHaveClass(
      "grid",
      "grid-cols-[minmax(0,1fr)_auto_auto]",
    );
    expect(screen.getByRole("group", { name: "Validation plot controls" })).toHaveClass(
      "grid",
      "grid-cols-[minmax(0,1fr)_auto_auto]",
    );
    expect(screen.getByRole("radiogroup", { name: "Train chart layout" }))
      .toBeInTheDocument();
    expect(screen.getByRole("radiogroup", { name: "Validation chart layout" }))
      .toBeInTheDocument();
    expect(screen.queryByRole("radiogroup", { name: "Test chart layout" }))
      .not.toBeInTheDocument();
    expectLogOptionSelected(
      await findMetricPlotOption(user, "Train", "train/loss_epoch"),
      true,
    );
    expectLogOptionSelected(
      await findMetricPlotOption(user, "Train", "train/loss"),
      false,
    );
    expectLogOptionSelected(
      await findMetricPlotOption(user, "Train", "train/accuracy"),
      false,
    );
    expectLogOptionSelected(
      await findMetricPlotOption(user, "Validation", "validation/accuracy_epoch"),
      true,
    );
    expectLogOptionSelected(
      await findMetricPlotOption(user, "Validation", "validation/accuracy"),
      false,
    );

    await user.click(
      screen.getByRole("button", { name: "Select no Validation plots" }),
    );

    await waitFor(() => {
      expect(screen.queryByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
        .not.toBeInTheDocument();
    });
    expect(screen.getByText("No plots selected in this group")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /train\/loss_epoch scalar chart/i }))
      .toBeInTheDocument();
    await expectLogFilterSelection(
      user,
      "Scalar Tags",
      "validation/accuracy_epoch",
      false,
    );
    expect(logScalarRequests).toHaveLength(scalarRequestCount);

    await user.click(
      screen.getByRole("button", { name: "Select all Validation plots" }),
    );

    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();
    expect(
      screen.getByRole("combobox", { name: /^Validation plots\s+3 \/ 3 selected$/i }),
    ).toBeInTheDocument();
    await waitFor(() => {
      expect(logScalarRequests.length).toBeGreaterThan(scalarRequestCount);
    });
  });

  it("opens a metric plot selector without collapsing its accordion", async () => {
    setupLogsScenario();
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);

    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /train\/loss_epoch scalar chart/i }))
      .toBeInTheDocument();
    const validationToggle = logMetricGroupToggle("Validation");
    const trainToggle = logMetricGroupToggle("Train");
    expect(validationToggle).toHaveAttribute("aria-expanded", "true");
    expect(trainToggle).toHaveAttribute("aria-expanded", "true");

    await user.click(
      screen.getByRole("combobox", { name: /^Validation plots\s+2 \/ 3 selected$/i }),
    );

    expect(
      await screen.findByRole("listbox", { name: "Validation plots options" }),
    ).toBeInTheDocument();
    expect(validationToggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: "Select all Validation plots" }),
    );
    expect(validationToggle).toHaveAttribute("aria-expanded", "true");

    await user.click(
      screen.getByRole("button", { name: "Select no Validation plots" }),
    );
    expect(validationToggle).toHaveAttribute("aria-expanded", "true");

    await user.click(
      screen.getByRole("combobox", { name: /^Train plots\s+2 \/ 4 selected$/i }),
    );

    expect(
      await screen.findByRole("listbox", { name: "Train plots options" }),
    ).toBeInTheDocument();
    expect(trainToggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("img", { name: /train\/loss_epoch scalar chart/i }))
      .toBeInTheDocument();
  });

  it("keeps loaded scalar groups visible while the default-open Train group loads", async () => {
    const trainScalarResponse = deferred<unknown>();
    const { logScalarRequests } = setupLogsScenario({
      logScalarResponseFactory: (body) => {
        if (body.tags.includes("train/loss_epoch")) {
          return trainScalarResponse.promise;
        }
        return undefined;
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);

    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(2);
    });
    expect(screen.queryByText(/^Loading scalar points$/i)).not.toBeInTheDocument();
    const trainBody = document.getElementById("logs-metric-group-train");
    expect(trainBody).toBeInstanceOf(HTMLElement);
    expect(within(trainBody as HTMLElement).getByText("Loading train/loss_epoch scalar points…"))
      .toBeInTheDocument();

    trainScalarResponse.resolve({
      series: logScalarSeries.filter(
        (series) => series.runId === "log-mnist" && series.tag === "train/loss_epoch",
      ),
    });

    expect(await screen.findByRole("img", { name: /train\/loss_epoch scalar chart/i }))
      .toBeInTheDocument();
    expect(screen.queryByText("Loading Train scalar points")).not.toBeInTheDocument();
  });

  it("keeps successful scalar chunks visible when a sibling chunk fails", async () => {
    const baseFixture = buildLargeLogFixture(11);
    const runs = baseFixture.logRunsResponse.runs.map((run) => ({
      ...run,
      group: "partial_scalar",
      experiment: "partial_scalar",
      dataset: "Mnist",
      modelType: "linears",
      model: "linear",
      preset: "BASELINE",
      relativePath: `partial_scalar/linear/BASELINE/Mnist/${run.runName}/version_0`,
    }));
    const failedRunId = runs.at(-1)?.id;
    const { logScalarRequests } = setupLogsScenario({
      logRunsResponse: { runs },
      logExperimentsResponse: {
        experiments: [
          {
            experiment: "partial_scalar",
            runCount: runs.length,
            relativePath: "partial_scalar",
          },
        ],
      },
      logTagsByRun: Object.fromEntries(
        runs.map((run) => [run.id, scalarTagsWithEpochDefaults()]),
      ),
      logScalarSeries: runs.flatMap((run, index) =>
        epochScalarSeriesForRun(run.id, index),
      ),
      logScalarResponseFactory: (body) => {
        if (
          failedRunId &&
          body.runIds.includes(failedRunId) &&
          body.tags.includes("train/accuracy_epoch")
        ) {
          return Promise.reject(new Error("scalar chunk unavailable"));
        }
        return undefined;
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["partial_scalar"]);

    await waitFor(() => {
      expect(
        logScalarRequests.filter((request) =>
          request.tags.includes("train/accuracy_epoch"),
        ),
      ).toHaveLength(2);
    });
    expect(await screen.findByText(/scalar read failed/i))
      .toBeInTheDocument();

    const chart = await screen.findByRole("img", {
      name: /train\/accuracy_epoch scalar chart/i,
    });
    const chartSection = chart.closest("section");
    expect(chartSection).toBeInstanceOf(HTMLElement);
    await waitFor(() => {
      expect(within(chartSection as HTMLElement).getByText(/10 lines/i))
        .toBeInTheDocument();
    });
    expect(screen.getByText(/scalar chunk unavailable/i)).toBeInTheDocument();
  });

  it("keeps existing charts visible while a later log tag chunk loads", async () => {
    const extraRuns = buildLargeLogFixture(55).logRunsResponse.runs.map(
      (run, index) => ({
        ...run,
        id: `extra-log-${index + 1}`,
        model: "linear",
        dataset: "Mnist",
        preset: "BASELINE",
        relativePath:
          `${run.experiment}/linear/BASELINE/Mnist/${run.runName}/version_0`,
      }),
    );
    const runs = [...logRunsWithSharedDataset().runs, ...extraRuns];
    const delayedTagChunk = deferred<null>();
    const { logTagRequests } = setupLogsScenario({
      logRunsResponse: { runs },
      logExperimentsResponse: {
        experiments: runs.map((run) => ({
          experiment: run.experiment,
          runCount: 1,
          relativePath: run.experiment,
        })),
      },
      logTagsByRun: {
        "log-mnist": scalarTagsWithEpochDefaults("train/loss", "validation/accuracy"),
        "log-cifar": scalarTagsWithEpochDefaults("validation/accuracy"),
        ...Object.fromEntries(
          extraRuns.map((run) => [
            run.id,
            scalarTagsWithEpochDefaults("validation/accuracy"),
          ]),
        ),
      },
      logScalarSeries: [
        ...logScalarSeries,
        ...extraRuns.flatMap((run, index) =>
          epochScalarSeriesForRun(run.id, index + 10),
        ),
        ...extraRuns.map((run, index) => ({
          runId: run.id,
          tag: "validation/accuracy",
          points: [
            { step: 1, wallTime: 1780000000 + index, value: 0.5 + index / 100 },
            { step: 2, wallTime: 1780000100 + index, value: 0.6 + index / 100 },
          ],
        })),
      ],
      logTagsResponseFactory: (body) => {
        if (!body.runIds.includes("extra-log-49")) {
          return undefined;
        }
        return delayedTagChunk.promise.then(() => ({
          runs: body.runIds.map((runId) => ({
            runId,
            scalarTags: scalarTagsWithEpochDefaults("validation/accuracy"),
            histogramTags: [],
            imageTags: [],
            textTags: [],
          })),
        }));
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await openLogFilter(user, "Experiments");
    await user.type(await screen.findByLabelText(/^search experiments$/i), "test_model");
    await selectLogExperiments(user, ["test_model"]);
    expect(await screen.findByText("Historical Scalars")).toBeInTheDocument();
    expect(await screen.findByText("Run Details")).toBeInTheDocument();
    const loadedChart = await screen.findByRole("img", {
      name: /validation\/accuracy_epoch scalar chart/i,
    });
    const loadedChartSection = loadedChart.closest("section");
    expect(loadedChartSection).toBeInstanceOf(HTMLElement);
    await waitFor(() => {
      expect(
        within(loadedChartSection as HTMLElement).getByText(/1 line/i),
      ).toBeInTheDocument();
    });

    await selectAllLogExperiments(user);

    await waitFor(() => {
      expect(
        logTagRequests.some((request) => request.runIds.includes("extra-log-49")),
      ).toBe(true);
    });
    expect(screen.getByText("Historical Scalars")).toBeInTheDocument();
    expect(
      screen.getByRole("img", {
        name: /validation\/accuracy_epoch scalar chart/i,
      }),
    ).toBeInTheDocument();
    expect(screen.getByText("Refreshing TensorBoard tags…")).toBeInTheDocument();
    expect(screen.queryByText("Reading TensorBoard tags")).not.toBeInTheDocument();

    delayedTagChunk.resolve(null);

    await waitFor(() => {
      expect(screen.queryByText("Refreshing TensorBoard tags…")).not.toBeInTheDocument();
    });
  });

  it("shows checkpoints, params, and artifacts in run details", async () => {
    const { logArtifactRequests, logCheckpointRequests } = setupLogsScenario();
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);

    const detailsPanel = screen.getByRole("heading", { name: "Run Details" })
      .closest('[data-workbench-region="details"]');
    expect(detailsPanel).not.toBeNull();
    await waitFor(() => {
      expect(logArtifactRequests).toEqual(["log-mnist"]);
    });
    await waitFor(() => {
      expect(logCheckpointRequests.at(-1)).toEqual({
        runIds: ["log-mnist"],
      });
    });

    expect(
      await within(detailsPanel as HTMLElement).findByText("epoch=0-step=2.ckpt"),
    ).toBeInTheDocument();
    expect(within(detailsPanel as HTMLElement).getByText(/epoch 0.*step 2/))
      .toBeInTheDocument();
    expect(within(detailsPanel as HTMLElement).getByText("batch_size"))
      .toBeInTheDocument();
    expect(within(detailsPanel as HTMLElement).getByText("learning_rate"))
      .toBeInTheDocument();
    expect(within(detailsPanel as HTMLElement).getByText("checkpoints/epoch=0-step=2.ckpt"))
      .toBeInTheDocument();
    expect(within(detailsPanel as HTMLElement).getByText(/event_file/))
      .toBeInTheDocument();
  });

  it("contains long values in the logs run details sidebar", async () => {
    const longRunName = `run_${"name".repeat(32)}`;
    const longExperiment = `experiment_${"exp".repeat(36)}`;
    const longDataset = `dataset_${"data".repeat(32)}`;
    const longModel = `model_${"model".repeat(32)}`;
    const longPreset = `preset_${"preset".repeat(24)}`;
    const longVersion = `version_${"version".repeat(20)}`;
    const longRelativePath = [
      "logs",
      longExperiment,
      longModel,
      longPreset,
      longDataset,
      longRunName,
      longVersion,
      "leaf_without_breakpoints",
    ].join("/");
    const longCheckpointFilename = `checkpoint_${"ckpt".repeat(34)}.ckpt`;
    const longParamKey = `param_${"key".repeat(38)}`;
    const longParamValue = `value_${"paramvalue".repeat(20)}`;
    const longMetricKey = `metric_${"accuracy".repeat(22)}`;
    const longArtifactLabel = `artifacts/${"artifact".repeat(24)}.json`;
    const longRun = {
      ...logRunsResponse.runs[0],
      id: "log-overflow",
      group: longExperiment,
      experiment: longExperiment,
      model: longModel,
      preset: longPreset,
      dataset: longDataset,
      runName: longRunName,
      version: longVersion,
      relativePath: longRelativePath,
      checkpointCount: 1,
      metrics: { [longMetricKey]: 0.987654321 },
    };
    const longCheckpoint = {
      id: "ckpt-overflow",
      runId: longRun.id,
      filename: longCheckpointFilename,
      relativePath: `${longRelativePath}/checkpoints/${longCheckpointFilename}`,
      epoch: 0,
      step: 1000,
      sizeBytes: 4096,
      modifiedAt: "2026-06-01T01:03:00Z",
    };

    setupLogsScenario({
      logRunsResponse: { runs: [longRun] },
      logExperimentsResponse: {
        experiments: [
          { experiment: longExperiment, runCount: 1, relativePath: longExperiment },
        ],
      },
      logTagsByRun: { [longRun.id]: [] },
      logScalarSeries: [],
      logCheckpointsByRun: { [longRun.id]: [longCheckpoint] },
      logRunArtifactsByRun: {
        [longRun.id]: {
          params: { [longParamKey]: longParamValue },
          metrics: { [longMetricKey]: 0.987654321 },
          checkpoints: [longCheckpoint],
          artifacts: [
            {
              id: "artifact-overflow",
              kind: "result",
              label: longArtifactLabel,
              relativePath: `${longRelativePath}/${longArtifactLabel}`,
              sizeBytes: 2048,
              modifiedAt: "2026-06-01T01:04:00Z",
            },
          ],
        },
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectAllLogExperiments(user);

    const detailsPanel = screen.getByRole("heading", { name: "Run Details" })
      .closest('[data-workbench-region="details"]');
    expect(detailsPanel).not.toBeNull();
    const panel = detailsPanel as HTMLElement;
    expect(panel).toHaveClass("min-w-0", "overflow-x-hidden");

    expect(await within(panel).findByTitle(longRunName)).toHaveClass("min-w-0", "truncate");
    const path = await within(panel).findByTitle(longRelativePath);
    expect(path).toHaveClass("min-w-0", "break-words");
    expect(path.classList.contains("[overflow-wrap:anywhere]")).toBe(true);

    const summaryCardGrid = within(panel).getByTitle(longExperiment).parentElement
      ?.parentElement;
    if (!(summaryCardGrid instanceof HTMLElement)) {
      throw new Error("Expected log run metadata cards to render inside a grid");
    }
    expect(summaryCardGrid).toHaveClass("grid", "w-full", "min-w-0", "grid-cols-1");
    expect(summaryCardGrid).not.toHaveClass("grid-cols-2");

    for (const summaryValue of [
      longExperiment,
      longDataset,
      longModel,
      longPreset,
      longVersion,
    ]) {
      const summaryValueElement = within(panel).getByTitle(summaryValue);
      const summaryCard = summaryValueElement.parentElement;
      if (!(summaryCard instanceof HTMLElement)) {
        throw new Error(`Expected ${summaryValue} to render inside a metric card`);
      }
      expect(summaryValueElement).toHaveClass("min-w-0", "truncate");
      expect(summaryCard).toHaveClass("w-full", "min-w-0");
    }

    const checkpointLabel = await within(panel).findByText(longCheckpointFilename);
    const paramKey = await within(panel).findByText(longParamKey);
    const paramValue = within(panel).getByText(longParamValue);
    const metricKey = within(panel).getByText(longMetricKey);
    const artifactLabel = await within(panel).findByText(longArtifactLabel);

    for (const detailCell of [
      checkpointLabel,
      paramKey,
      paramValue,
      metricKey,
      artifactLabel,
    ]) {
      expect(detailCell).toHaveClass("min-w-0", "whitespace-normal", "break-words");
      expect(detailCell.classList.contains("[overflow-wrap:anywhere]")).toBe(true);
      expect(detailCell.closest("div")).toHaveClass(
        "grid-cols-[minmax(0,1fr)_minmax(0,1fr)]",
      );
    }
  });

  it("renders non-standard scalar tags in the Other metric group", async () => {
    setupLogsScenario();
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);
    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();

    const otherTag = await findLogOption(
      user,
      "Scalar Tags",
      "main_model.0.model/weights/mean",
    );
    expectLogOptionSelected(otherTag, false);
    await user.click(otherTag);

    const otherToggle = await screen.findByRole("button", {
      name: /^Other\s+1\s+metric$/i,
    });
    await user.click(otherToggle);
    await waitFor(() => {
      expect(logMetricGroupToggle("Other")).toHaveAttribute("aria-expanded", "true");
    });
    expect(
      await screen.findByRole("img", {
        name: /main_model\.0\.model\/weights\/mean scalar chart/i,
      }),
    ).toBeInTheDocument();
  });

  it("keeps collapsed logs metric groups collapsed after switching workspaces", async () => {
    setupLogsScenario();
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);
    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /train\/loss_epoch scalar chart/i }))
      .toBeInTheDocument();

    await user.click(logMetricGroupToggle("Train"));
    expect(screen.queryByRole("img", { name: /train\/loss_epoch scalar chart/i }))
      .not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^model$/i }));
    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    const trainToggle = logMetricGroupToggle("Train");
    expect(trainToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("img", { name: /train\/loss_epoch scalar chart/i }))
      .not.toBeInTheDocument();
    await expectLogFilterSelection(user, "Scalar Tags", "train/loss_epoch", true);
  });

  it("keeps log selections when Training is activated after Logs", async () => {
    setupLogsScenario();
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);
    expect(
      await screen.findByRole("img", {
        name: /validation\/accuracy_epoch scalar chart/i,
      }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^training$/i }));
    await user.click(await screen.findByRole("button", { name: /^logs$/i }));

    await expectLogFilterSelection(user, "Experiments", "test_model", true);
    expect(
      await screen.findByRole("img", {
        name: /validation\/accuracy_epoch scalar chart/i,
      }),
    ).toBeInTheDocument();
  });

  it("splits test score leaderboards by experiment without refetching scalars", async () => {
    const { logScalarRequests } = setupLogsScenario({
      logRunsResponse: logRunsWithSharedDataset(),
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectAllLogExperiments(user);
    await clickLogOption(user, "Scalar Tags", "test/accuracy");

    const combinedLeaderboard = await screen.findByRole("table", {
      name: /^test\/accuracy test leaderboard$/i,
    });
    const testScores = testScoresSection();
    const splitCheckbox = testScoresSplitCheckbox();
    expect(screen.getAllByRole("checkbox", { name: /split by experiment/i }))
      .toHaveLength(1);
    expect(splitCheckbox.closest("section")).toBe(testScores);
    expect(splitCheckbox).not.toBeChecked();

    await waitFor(() => {
      expect(within(combinedLeaderboard).getAllByRole("row").slice(1))
        .toHaveLength(2);
    });
    const combinedRows = within(combinedLeaderboard).getAllByRole("row").slice(1);
    expect(combinedRows).toHaveLength(2);
    expect(combinedRows[0]).toHaveTextContent("0.9");
    expect(within(combinedRows[0]).getByText("aaa_20260601_010203"))
      .toBeInTheDocument();
    expect(combinedRows[1]).toHaveTextContent("0.62");
    expect(within(combinedRows[1]).getByText("bbb_20260601_020304"))
      .toBeInTheDocument();

    await waitFor(() => {
      expect(logScalarRequests).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            runIds: ["log-mnist", "log-cifar"],
            tags: ["test/accuracy"],
          }),
        ]),
      );
    });
    const scalarRequestCount = logScalarRequests.length;

    await user.click(splitCheckbox);

    expect(splitCheckbox).toBeChecked();
    expect(
      within(testScores).queryByRole("table", {
        name: /^test\/accuracy test leaderboard$/i,
      }),
    ).not.toBeInTheDocument();
    const mnistLeaderboard = await within(testScores).findByRole("table", {
      name: /test_model · test\/accuracy test leaderboard/i,
    });
    const cifarLeaderboard = await within(testScores).findByRole("table", {
      name: /test_model_2 · test\/accuracy test leaderboard/i,
    });
    expect(within(mnistLeaderboard).getAllByRole("row").slice(1)).toHaveLength(1);
    expect(within(cifarLeaderboard).getAllByRole("row").slice(1)).toHaveLength(1);
    expect(within(mnistLeaderboard).getByText("aaa_20260601_010203"))
      .toBeInTheDocument();
    expect(within(cifarLeaderboard).getByText("bbb_20260601_020304"))
      .toBeInTheDocument();
    expect(logScalarRequests).toHaveLength(scalarRequestCount);

    await user.click(
      within(cifarLeaderboard).getByRole("button", {
        name: /open details for test_model_2 · Mnist/i,
      }),
    );
    const detailsPanel = screen.getByRole("heading", { name: "Run Details" })
      .closest('[data-workbench-region="details"]');
    expect(detailsPanel).toBeInstanceOf(HTMLElement);
    expect(within(detailsPanel as HTMLElement).getByText("test_model_2"))
      .toBeInTheDocument();
    expect(within(detailsPanel as HTMLElement).getByText("bbb_20260601_020304"))
      .toBeInTheDocument();

    await user.click(splitCheckbox);

    expect(splitCheckbox).not.toBeChecked();
    expect(
      await within(testScores).findByRole("table", {
        name: /^test\/accuracy test leaderboard$/i,
      }),
    ).toBeInTheDocument();
    expect(
      within(testScores).queryByRole("table", {
        name: /test_model · test\/accuracy test leaderboard/i,
      }),
    ).not.toBeInTheDocument();
    expect(logScalarRequests).toHaveLength(scalarRequestCount);
  });

  it("sorts test score leaderboards by inferred direction in combined and split modes", async () => {
    const sharedDatasetRuns = logRunsWithSharedDataset().runs;
    const runs = [
      sharedDatasetRuns[0],
      {
        ...sharedDatasetRuns[0],
        id: "log-mnist-alt",
        runName: "ccc_20260601_030405",
        timestamp: "2026-06-01 03:04:05",
        relativePath:
          "test_model/linear/BASELINE/Mnist/ccc_20260601_030405/version_0",
      },
      sharedDatasetRuns[1],
      {
        ...sharedDatasetRuns[1],
        id: "log-cifar-alt",
        runName: "ddd_20260601_040506",
        timestamp: "2026-06-01 04:05:06",
        relativePath:
          "test_model_2/linear/BASELINE/Mnist/ddd_20260601_040506/version_0",
      },
    ];
    const scoreSeries = [
      { runId: "log-mnist", accuracy: 0.91, loss: 0.42 },
      { runId: "log-mnist-alt", accuracy: 0.84, loss: 0.11 },
      { runId: "log-cifar", accuracy: 0.73, loss: 0.27 },
      { runId: "log-cifar-alt", accuracy: 0.95, loss: 0.5 },
    ];
    setupLogsScenario({
      logRunsResponse: { runs },
      logExperimentsResponse: {
        experiments: [
          { experiment: "test_model", runCount: 2, relativePath: "test_model" },
          {
            experiment: "test_model_2",
            runCount: 2,
            relativePath: "test_model_2",
          },
        ],
      },
      logTagsByRun: Object.fromEntries(
        runs.map((run) => [run.id, ["test/accuracy", "test/loss"]]),
      ),
      logScalarSeries: scoreSeries.flatMap(({ runId, accuracy, loss }) => [
        {
          runId,
          tag: "test/accuracy",
          points: [{ step: 3, wallTime: 1780000003, value: accuracy }],
        },
        {
          runId,
          tag: "test/loss",
          points: [{ step: 3, wallTime: 1780000003, value: loss }],
        },
      ]),
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectAllLogExperiments(user);
    await clickLogOption(user, "Scalar Tags", "test/accuracy");
    await clickLogOption(user, "Scalar Tags", "test/loss");

    const lossLeaderboard = await screen.findByRole("table", {
      name: /test\/loss test leaderboard/i,
    });
    const accuracyLeaderboard = await screen.findByRole("table", {
      name: /test\/accuracy test leaderboard/i,
    });
    const testScoresGrid = lossLeaderboard.closest("section")?.parentElement;
    expect(testScoresGrid).toBeInstanceOf(HTMLElement);
    expect(accuracyLeaderboard.closest("section")?.parentElement).toBe(
      testScoresGrid,
    );
    expect(testScoresGrid).toHaveClass("grid", "xl:grid-cols-2");

    const accuracyRows = within(accuracyLeaderboard).getAllByRole("row").slice(1);
    expect(accuracyRows).toHaveLength(4);
    expect(accuracyRows[0]).toHaveTextContent("0.95");
    expect(within(accuracyRows[0]).getByText("ddd_20260601_040506"))
      .toBeInTheDocument();
    expect(accuracyRows[1]).toHaveTextContent("0.91");
    expect(within(accuracyRows[1]).getByText("aaa_20260601_010203"))
      .toBeInTheDocument();

    const lossRows = within(lossLeaderboard).getAllByRole("row").slice(1);
    expect(lossRows).toHaveLength(4);
    expect(lossRows[0]).toHaveTextContent("0.11");
    expect(within(lossRows[0]).getByText("ccc_20260601_030405"))
      .toBeInTheDocument();
    expect(lossRows[1]).toHaveTextContent("0.27");
    expect(within(lossRows[1]).getByText("bbb_20260601_020304"))
      .toBeInTheDocument();

    await user.click(testScoresSplitCheckbox());

    const testModelAccuracyRows = within(
      await screen.findByRole("table", {
        name: /test_model · test\/accuracy test leaderboard/i,
      }),
    ).getAllByRole("row").slice(1);
    expect(testModelAccuracyRows).toHaveLength(2);
    expect(testModelAccuracyRows[0]).toHaveTextContent("0.91");
    expect(within(testModelAccuracyRows[0]).getByText("aaa_20260601_010203"))
      .toBeInTheDocument();
    expect(testModelAccuracyRows[1]).toHaveTextContent("0.84");
    expect(within(testModelAccuracyRows[1]).getByText("ccc_20260601_030405"))
      .toBeInTheDocument();

    const testModelLossRows = within(
      await screen.findByRole("table", {
        name: /test_model · test\/loss test leaderboard/i,
      }),
    ).getAllByRole("row").slice(1);
    expect(testModelLossRows).toHaveLength(2);
    expect(testModelLossRows[0]).toHaveTextContent("0.11");
    expect(within(testModelLossRows[0]).getByText("ccc_20260601_030405"))
      .toBeInTheDocument();
    expect(testModelLossRows[1]).toHaveTextContent("0.42");
    expect(within(testModelLossRows[1]).getByText("aaa_20260601_010203"))
      .toBeInTheDocument();

    const testModel2AccuracyRows = within(
      await screen.findByRole("table", {
        name: /test_model_2 · test\/accuracy test leaderboard/i,
      }),
    ).getAllByRole("row").slice(1);
    expect(testModel2AccuracyRows).toHaveLength(2);
    expect(testModel2AccuracyRows[0]).toHaveTextContent("0.95");
    expect(within(testModel2AccuracyRows[0]).getByText("ddd_20260601_040506"))
      .toBeInTheDocument();
    expect(testModel2AccuracyRows[1]).toHaveTextContent("0.73");
    expect(within(testModel2AccuracyRows[1]).getByText("bbb_20260601_020304"))
      .toBeInTheDocument();

    const testModel2LossRows = within(
      await screen.findByRole("table", {
        name: /test_model_2 · test\/loss test leaderboard/i,
      }),
    ).getAllByRole("row").slice(1);
    expect(testModel2LossRows).toHaveLength(2);
    expect(testModel2LossRows[0]).toHaveTextContent("0.27");
    expect(within(testModel2LossRows[0]).getByText("bbb_20260601_020304"))
      .toBeInTheDocument();
    expect(testModel2LossRows[1]).toHaveTextContent("0.5");
    expect(within(testModel2LossRows[1]).getByText("ddd_20260601_040506"))
      .toBeInTheDocument();
  });

  it("switches historical scalar accordion layouts without refetching scalars", async () => {
    const { logScalarRequests } = setupLogsScenario();
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);
    await clickLogOption(user, "Scalar Tags", "test/accuracy");

    const chart = await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i });
    const validationToggle = logMetricGroupToggle("Validation");
    const chartGrid = scalarChartGridFor(chart);
    const accordionGrid = accordionSectionGridFor(validationToggle);
    expect(within(validationToggle).getByText(/metrics?$/i))
      .toHaveClass("shrink-0", "whitespace-nowrap");
    const bestRunSection = screen
      .getByRole("heading", { name: "Best Run by Selected Metric" })
      .closest("section");
    expect(bestRunSection).toBeInstanceOf(HTMLElement);
    const testScores = testScoresSection();
    expect(testScores.parentElement).not.toBe(accordionGrid);
    expectElementsInDocumentOrder([
      bestRunSection as HTMLElement,
      testScores,
      accordionGrid,
    ]);
    const layoutControl = screen.getByRole("radiogroup", {
      name: /scalar accordion layout/i,
    });
    const fullTab = within(layoutControl).getByRole("radio", { name: /^full$/i });
    const twoColumnTab = within(layoutControl).getByRole("radio", { name: /^2 col$/i });
    const threeColumnTab = within(layoutControl).getByRole("radio", { name: /^3 col$/i });

    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(4);
    });
    expect(fullTab).toHaveAttribute("aria-checked", "false");
    expect(twoColumnTab).toHaveAttribute("aria-checked", "true");
    expect(threeColumnTab).toHaveAttribute("aria-checked", "false");
    expect(accordionGrid).toHaveClass(
      "grid",
      "items-start",
      "gap-5",
      "xl:grid-cols-2",
    );
    expect(accordionGrid).not.toHaveClass("2xl:grid-cols-3");
    expect(chartGrid).toHaveClass("grid", "gap-4");
    expect(chartGrid).not.toHaveClass("xl:grid-cols-2");
    expect(chartGrid).not.toHaveClass("2xl:grid-cols-3");

    await user.click(fullTab);

    expect(fullTab).toHaveAttribute("aria-checked", "true");
    expect(twoColumnTab).toHaveAttribute("aria-checked", "false");
    expect(accordionGrid).toHaveClass("items-start");
    expect(accordionGrid).not.toHaveClass("xl:grid-cols-2");
    expect(accordionGrid).not.toHaveClass("2xl:grid-cols-3");
    expectElementsInDocumentOrder([
      bestRunSection as HTMLElement,
      testScores,
      accordionGrid,
    ]);
    expect(chartGrid).not.toHaveClass("xl:grid-cols-2");
    expect(chartGrid).not.toHaveClass("2xl:grid-cols-3");
    expect(logScalarRequests).toHaveLength(4);

    await user.click(twoColumnTab);

    expect(fullTab).toHaveAttribute("aria-checked", "false");
    expect(twoColumnTab).toHaveAttribute("aria-checked", "true");
    expect(accordionGrid).toHaveClass("items-start");
    expect(accordionGrid).toHaveClass("xl:grid-cols-2");
    expect(accordionGrid).not.toHaveClass("2xl:grid-cols-3");
    expect(chartGrid).not.toHaveClass("xl:grid-cols-2");
    expect(chartGrid).not.toHaveClass("2xl:grid-cols-3");
    expect(logScalarRequests).toHaveLength(4);

    await user.click(threeColumnTab);

    expect(twoColumnTab).toHaveAttribute("aria-checked", "false");
    expect(threeColumnTab).toHaveAttribute("aria-checked", "true");
    expect(accordionGrid).toHaveClass("items-start");
    expect(accordionGrid).toHaveClass("xl:grid-cols-2", "2xl:grid-cols-3");
    expect(chartGrid).not.toHaveClass("xl:grid-cols-2");
    expect(chartGrid).not.toHaveClass("2xl:grid-cols-3");
    expect(logScalarRequests).toHaveLength(4);

    await user.click(screen.getByRole("button", { name: /refresh scalar charts/i }));

    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(8);
    });
  });

  it("switches individual metric chart layouts without refetching scalars", async () => {
    const { logScalarRequests } = setupLogsScenario();
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);
    await clickLogOption(user, "Scalar Tags", "test/accuracy");

    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /train\/loss_epoch scalar chart/i }))
      .toBeInTheDocument();
    expect(
      await screen.findByRole("table", { name: /test\/accuracy test leaderboard/i }),
    ).toBeInTheDocument();

    await clickLogOption(user, "Scalar Tags", "main_model.0.model/weights/mean");
    await user.click(logMetricGroupToggle("Other"));
    expect(
      await screen.findByRole("img", {
        name: /main_model\.0\.model\/weights\/mean scalar chart/i,
      }),
    ).toBeInTheDocument();

    for (const group of ["Train", "Validation", "Other"]) {
      const chartLayoutControl = screen.getByRole("radiogroup", {
        name: `${group} chart layout`,
      });
      expect(
        within(chartLayoutControl).getByRole("radio", { name: /^full$/i }),
      ).toHaveAttribute("aria-checked", "true");
    }
    expect(screen.queryByRole("radiogroup", { name: "Test chart layout" }))
      .not.toBeInTheDocument();

    await waitFor(() => {
      expect(
        logScalarRequests.some((request) => request.tags.includes("test/accuracy")),
      ).toBe(true);
      expect(
        logScalarRequests.some((request) =>
          request.tags.includes("main_model.0.model/weights/mean"),
        ),
      ).toBe(true);
    });

    const scalarRequestCount = logScalarRequests.length;
    const trainGrid = metricGroupBody("train");
    const validationGrid = metricGroupBody("validation");
    const otherGrid = metricGroupBody("other");

    for (const grid of [trainGrid, validationGrid, otherGrid]) {
      expect(grid).toHaveClass("grid", "gap-4");
      expect(grid).not.toHaveClass("xl:grid-cols-2");
      expect(grid).not.toHaveClass("2xl:grid-cols-3");
    }

    const validationLayoutControl = screen.getByRole("radiogroup", {
      name: "Validation chart layout",
    });
    await user.click(
      within(validationLayoutControl).getByRole("radio", { name: /^3 col$/i }),
    );

    expect(validationGrid).toHaveClass("xl:grid-cols-2", "2xl:grid-cols-3");
    for (const grid of [trainGrid, otherGrid]) {
      expect(grid).not.toHaveClass("xl:grid-cols-2");
      expect(grid).not.toHaveClass("2xl:grid-cols-3");
    }
    expect(logScalarRequests).toHaveLength(scalarRequestCount);
  });

  it("links scalar legend hover across charts in the same metric accordion", async () => {
    setupLogsScenario({
      logRunsResponse: logRunsWithSharedDataset(),
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectAllLogExperiments(user);
    await clickLogOption(user, "Scalar Tags", "train/accuracy");

    expect(await screen.findByRole("img", { name: /train\/loss_epoch scalar chart/i }))
      .toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /train\/accuracy scalar chart/i }))
      .toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();

    const mnistRun = /test_model · Mnist · linear · linears · BASELINE · 2026-06-01 01:02:03/i;
    const cifarRun = /test_model_2 · Mnist · linear · linears · BASELINE · 2026-06-01 02:03:04/i;
    const trainLossCard = scalarChartSection(/train\/loss_epoch scalar chart/i);
    const trainAccuracyCard = scalarChartSection(/train\/accuracy scalar chart/i);
    const validationCard = scalarChartSection(/validation\/accuracy_epoch scalar chart/i);
    const mnistLoss = logScalarLegendButton(trainLossCard, mnistRun);
    const cifarLoss = logScalarLegendButton(trainLossCard, cifarRun);
    const mnistTrainAccuracy = logScalarLegendButton(trainAccuracyCard, mnistRun);
    const cifarTrainAccuracy = logScalarLegendButton(trainAccuracyCard, cifarRun);
    const mnistValidation = logScalarLegendButton(validationCard, mnistRun);
    const cifarValidation = logScalarLegendButton(validationCard, cifarRun);

    fireEvent.pointerEnter(mnistLoss);

    await waitFor(() => {
      expectLegendOpacity(cifarLoss, "dimmed");
      expectLegendOpacity(cifarTrainAccuracy, "dimmed");
    });
    expectLegendOpacity(mnistLoss, "normal");
    expectLegendOpacity(mnistTrainAccuracy, "normal");
    expectLegendOpacity(mnistValidation, "normal");
    expectLegendOpacity(cifarValidation, "normal");

    fireEvent.pointerLeave(mnistLoss);

    await waitFor(() => {
      expectLegendOpacity(cifarLoss, "normal");
      expectLegendOpacity(cifarTrainAccuracy, "normal");
    });

    fireEvent.focus(cifarTrainAccuracy);

    await waitFor(() => {
      expectLegendOpacity(mnistLoss, "dimmed");
      expectLegendOpacity(mnistTrainAccuracy, "dimmed");
    });
    expectLegendOpacity(cifarLoss, "normal");
    expectLegendOpacity(cifarTrainAccuracy, "normal");
    expectLegendOpacity(mnistValidation, "normal");
    expectLegendOpacity(cifarValidation, "normal");

    fireEvent.blur(cifarTrainAccuracy);

    await waitFor(() => {
      expectLegendOpacity(mnistLoss, "normal");
      expectLegendOpacity(mnistTrainAccuracy, "normal");
    });
  });

  it("logs workspace experiment and tag checkboxes hide chart content", async () => {
    setupLogsScenario({
      logRunsResponse: logRunsWithSharedDataset(),
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectAllLogExperiments(user);
    await clickLogOption(user, "Scalar Tags", "test/accuracy");
    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /train\/loss_epoch scalar chart/i }))
      .toBeInTheDocument();
    expect(
      await screen.findByRole("table", { name: /test\/accuracy test leaderboard/i }),
    ).toBeInTheDocument();
    await clickLogOption(user, "Experiments", "test_model");

    await waitFor(() => {
      expect(screen.queryByText(/test_model · Mnist · linear · linears · BASELINE · 2026-06-01 01:02:03/))
        .not.toBeInTheDocument();
    });
    const datasetSection = logFilterSection("Datasets");
    expect(within(datasetSection).getByText("1 / 1")).toBeInTheDocument();
    await expectLogFilterSelection(user, "Datasets", "Mnist", true);
    expect(queryOpenLogOption("Datasets", "Cifar10")).not.toBeInTheDocument();
    await setLogOptionSelection(user, "Scalar Tags", "train/loss_epoch", true);
    expect(await screen.findByRole("img", { name: /train\/loss_epoch scalar chart/i }))
      .toBeInTheDocument();
    await setLogOptionSelection(
      user,
      "Scalar Tags",
      "validation/accuracy_epoch",
      true,
    );
    expect(
      await screen.findByRole("img", {
        name: /validation\/accuracy_epoch scalar chart/i,
      }),
    ).toBeInTheDocument();
    await setLogOptionSelection(
      user,
      "Scalar Tags",
      "validation/accuracy_epoch",
      false,
    );

    await waitFor(() => {
      expect(screen.queryByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
        .not.toBeInTheDocument();
    });
    expect(screen.getByRole("img", { name: /train\/loss_epoch scalar chart/i }))
      .toBeInTheDocument();

    await setLogOptionSelection(user, "Scalar Tags", "test/accuracy", true);
    expect(
      await screen.findByRole("table", { name: /test\/accuracy test leaderboard/i }),
    ).toBeInTheDocument();
    await setLogOptionSelection(user, "Scalar Tags", "test/accuracy", false);

    await waitFor(() => {
      expect(screen.queryByRole("table", { name: /test\/accuracy test leaderboard/i }))
        .not.toBeInTheDocument();
    });

    const experimentSection = logFilterSection("Experiments");
    await user.click(within(experimentSection).getByRole("button", { name: /^none$/i }));

    expect(await screen.findByText("No runs selected")).toBeInTheDocument();
  });

  it("requests checkpoint markers only for visible log runs", async () => {
    const { logCheckpointRequests } = setupLogsScenario({
      logRunsResponse: {
        runs: logRunsWithSharedDataset().runs.map((run) =>
          run.id === "log-cifar" ? { ...run, checkpointCount: 1 } : run,
        ),
      },
      logCheckpointsByRun: {
        "log-mnist": [
          {
            id: "ckpt-mnist",
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
        "log-cifar": [
          {
            id: "ckpt-cifar",
            runId: "log-cifar",
            filename: "epoch=0-step=2.ckpt",
            relativePath:
              "test_model_2/linear/BASELINE/Cifar10/bbb_20260601_020304/version_0/checkpoints/epoch=0-step=2.ckpt",
            epoch: 0,
            step: 2,
            sizeBytes: 2048,
            modifiedAt: "2026-06-01T02:04:00Z",
          },
        ],
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectAllLogExperiments(user);
    expect(await screen.findByRole("img", { name: /train\/loss_epoch scalar chart/i }))
      .toBeInTheDocument();

    await waitFor(() => {
      expect(logCheckpointRequests.at(-1)).toEqual({
        runIds: ["log-mnist", "log-cifar"],
      });
    });

    await clickLogOption(user, "Experiments", "test_model");
    await setLogOptionSelection(user, "Scalar Tags", "train/loss_epoch", true);
    expect(await screen.findByRole("img", { name: /train\/loss_epoch scalar chart/i }))
      .toBeInTheDocument();

    await waitFor(() => {
      expect(logCheckpointRequests).toContainEqual({ runIds: ["log-cifar"] });
    });
    expect(screen.queryByText(/test_model · Mnist · linear · linears · BASELINE/))
      .not.toBeInTheDocument();
  });

  it("deletes a log experiment after exact-name confirmation", async () => {
    const { fetchMock, deleteExperimentRequests, logScalarRequests } = setupLogsScenario();
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await expectLogFilterSelection(user, "Experiments", "test_model", false);

    await openLogFilter(user, "Experiments");
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
    await openLogFilter(user, "Experiments");
    expect(queryOpenLogOption("Experiments", "test_model")).not.toBeInTheDocument();
    expectLogOptionSelected(
      await findLogOption(user, "Experiments", "test_model_2"),
      false,
    );

    expect(await screen.findByText("No runs selected")).toBeInTheDocument();
    expect(logScalarRequests).toHaveLength(0);
  });

  it("does not render the sidebar-level delete visible runs action", async () => {
    setupLogsScenario();
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    expect(await screen.findByText("Historical Scalars")).toBeInTheDocument();

    expect(
      screen.queryByRole("button", { name: /^delete visible runs$/i }),
    ).not.toBeInTheDocument();
  });

  it("hides destructive log deletion actions when capabilities disable them", async () => {
    const {
      deleteExperimentRequests,
      deletePresetPlanRequests,
      deletePresetRequests,
    } =
      setupLogsScenario({
        capabilitiesResponse: {
          ...capabilitiesResponse,
          logDeletionEnabled: false,
        },
      });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await expectLogFilterSelection(user, "Experiments", "test_model", false);

    expect(
      screen.queryByRole("button", { name: /^delete experiment test_model$/i }),
    ).not.toBeInTheDocument();

    await clickLogOption(user, "Experiments", "test_model_2");

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /^delete dataset/i }))
        .not.toBeInTheDocument();
    });
    await openLogFilter(user, "Presets");
    expect(screen.queryByRole("button", { name: /^delete preset/i }))
      .not.toBeInTheDocument();
    expect(screen.getByText(/log deletion is disabled/i)).toBeInTheDocument();
    expect(deleteExperimentRequests).toHaveLength(0);
    expect(deletePresetPlanRequests).toHaveLength(0);
    expect(deletePresetRequests).toHaveLength(0);
  });

  it("deletes preset row runs using the target experiment and preset", async () => {
    const { deletePresetPlanRequests, deletePresetRequests } = setupLogsScenario(
      buildSubsetDeleteFixture(),
    );
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await clickLogOption(user, "Experiments", "test_model");

    const presetToolbar = await activateLogOptionToolbar(
      user,
      "Presets",
      "BASELINE",
    );
    await user.click(
      within(presetToolbar).getByRole("button", {
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
    expect(deletePresetPlanRequests).toEqual([
      { experiment: "test_model", preset: "BASELINE" },
    ]);
    expect(deletePresetRequests).toEqual(deletePresetPlanRequests);
  });

  it("cancels a planned preset deletion without issuing a mutation", async () => {
    const { deletePresetPlanRequests, deletePresetRequests } = setupLogsScenario(
      buildSubsetDeleteFixture(),
    );
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await clickLogOption(user, "Experiments", "test_model");
    const presetToolbar = await activateLogOptionToolbar(
      user,
      "Presets",
      "BASELINE",
    );
    await user.click(
      within(presetToolbar).getByRole("button", {
        name: /^delete preset BASELINE from experiment test_model$/i,
      }),
    );

    const dialog = await screen.findByRole("dialog", { name: /^delete preset$/i });
    expect(await within(dialog).findByText(/2 matched runs/i)).toBeInTheDocument();
    await user.click(within(dialog).getByRole("button", { name: /^cancel$/i }));

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /^delete preset$/i }))
        .not.toBeInTheDocument();
    });
    expect(deletePresetPlanRequests).toHaveLength(1);
    expect(deletePresetRequests).toHaveLength(0);
  });

  it("retries preset deletion planning after a scoped plan failure", async () => {
    const { deletePresetPlanRequests, deletePresetRequests } = setupLogsScenario({
      ...buildSubsetDeleteFixture(),
      deleteLogRunPlanErrorFactory: (requestIndex) =>
        requestIndex === 0 ? "delete plan unavailable" : undefined,
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await clickLogOption(user, "Experiments", "test_model");
    const presetToolbar = await activateLogOptionToolbar(
      user,
      "Presets",
      "BASELINE",
    );
    await user.click(
      within(presetToolbar).getByRole("button", {
        name: /^delete preset BASELINE from experiment test_model$/i,
      }),
    );

    const dialog = await screen.findByRole("dialog", { name: /^delete preset$/i });
    expect(await within(dialog).findByText(/delete plan unavailable/i))
      .toBeInTheDocument();
    await user.click(
      within(dialog).getByRole("button", { name: /^retry plan$/i }),
    );

    expect(await within(dialog).findByText(/2 matched runs/i)).toBeInTheDocument();
    expect(deletePresetPlanRequests).toHaveLength(2);
    expect(deletePresetRequests).toHaveLength(0);
  });

  it("retains a preset plan while a failed mutation is retried", async () => {
    const { deletePresetRequests } = setupLogsScenario({
      ...buildSubsetDeleteFixture(),
      deleteLogRunsErrorFactory: (requestIndex) =>
        requestIndex === 0 ? "delete mutation unavailable" : undefined,
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await clickLogOption(user, "Experiments", "test_model");
    const presetToolbar = await activateLogOptionToolbar(
      user,
      "Presets",
      "BASELINE",
    );
    await user.click(
      within(presetToolbar).getByRole("button", {
        name: /^delete preset BASELINE from experiment test_model$/i,
      }),
    );

    const dialog = await screen.findByRole("dialog", { name: /^delete preset$/i });
    expect(await within(dialog).findByText(/2 matched runs/i)).toBeInTheDocument();
    const deleteButton = within(dialog).getByRole("button", {
      name: /^delete preset$/i,
    });
    await user.click(deleteButton);
    expect(await within(dialog).findByText(/delete mutation unavailable/i))
      .toBeInTheDocument();

    await user.click(deleteButton);
    expect(await within(dialog).findByText(/delete mutation unavailable/i))
      .toBeInTheDocument();
    expect(deletePresetRequests).toHaveLength(1);
  });

  it("blocks preset row deletion when an active training job uses an affected folder", async () => {
    const { deletePresetRequests } = setupLogsScenario({
      ...buildSubsetDeleteFixture(),
      deleteLogRunsBlockers: [
        { id: "job-1", logFolder: "test_model", status: "running" },
      ],
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await clickLogOption(user, "Experiments", "test_model");
    const presetToolbar = await activateLogOptionToolbar(
      user,
      "Presets",
      "BASELINE",
    );
    await user.click(
      within(presetToolbar).getByRole("button", {
        name: /^delete preset BASELINE from experiment test_model$/i,
      }),
    );

    const dialog = await screen.findByRole("dialog", { name: /^delete preset$/i });
    expect(
      await within(dialog).findByText(
        /A training job is still writing to this log folder/i,
      ),
    ).toBeInTheDocument();
    expect(within(dialog).getByText(/job-1 · logs\/test_model/i)).toBeInTheDocument();
    expect(
      within(dialog).getByRole("button", { name: /^delete preset$/i }),
    ).toBeDisabled();
    expect(deletePresetRequests).toHaveLength(0);
  });

  it("lazy-loads logs sidebar filters with many options", async () => {
    const fixture = buildLargeLogFixture();
    const sharedRuns = fixture.logRunsResponse.runs.map((run) => ({
      ...run,
      dataset: "Mnist",
      relativePath:
        `${run.experiment}/linear/BASELINE/Mnist/${run.runName}/version_0`,
    }));
    const sharedFixture = {
      ...fixture,
      logRunsResponse: { runs: sharedRuns },
    };
    const { deleteExperimentRequests, logScalarRequests } =
      setupLogsScenario(sharedFixture);
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectAllLogExperiments(user);

    let experimentsList = await openLogFilter(user, "Experiments");
    const firstExperiment = await within(experimentsList).findByRole("option", {
      name: logOptionName("experiment_01"),
    });
    expectLogOptionSelected(firstExperiment);
    expect(firstExperiment).toHaveAccessibleDescription("1 run");
    expect(within(firstExperiment).getByText("experiment_01")).toHaveClass(
      "whitespace-normal",
      "break-words",
      "[overflow-wrap:anywhere]",
    );
    expect(within(firstExperiment).getByRole("tooltip", { name: "1 run" }))
      .toHaveClass("opacity-0", "group-hover:opacity-100");
    expectLogsChecklistRowSizing(firstExperiment);
    expectLogsChecklistRowSizing(
      within(experimentsList).getByRole("option", {
        name: logOptionName("experiment_42"),
      }),
    );
    expect(
      within(experimentsList).queryByRole("option", {
        name: logOptionName("experiment_64"),
      }),
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/Showing 50 of 64\. Search to narrow\./i))
      .not.toBeInTheDocument();

    makeScrollable(experimentsList);
    fireEvent.scroll(experimentsList);
    expect(
      within(experimentsList).getByRole("option", {
        name: /loading more experiments/i,
      }),
    ).toBeInTheDocument();
    const lastExperiment = await within(experimentsList).findByRole("option", {
      name: logOptionName("experiment_64"),
    });
    expectLogOptionSelected(lastExperiment);
    expectLogsChecklistRowSizing(lastExperiment);

    const tagList = await openLogFilter(user, "Scalar Tags");
    const tagOption = await within(tagList).findByRole("option", {
      name: logOptionName("custom/tag-42"),
    });
    expectLogsChecklistRowSizing(tagOption);
    await user.click(tagOption);
    expectLogOptionSelected(tagOption);
    await user.click(logMetricGroupToggle("Other"));

    const expectedRunIds = sharedFixture.logRunsResponse.runs.map((run) => run.id);
    await waitFor(() => {
      const selectedTagRequests = logScalarRequests.filter(
        (request) =>
          request.tags.length === 1 && request.tags[0] === "custom/tag-42",
      );
      expect(
        selectedTagRequests.slice(-Math.ceil(expectedRunIds.length / 10)),
      ).toEqual(
        Array.from(
          { length: Math.ceil(expectedRunIds.length / 10) },
          (_, index) => ({
            runIds: expectedRunIds.slice(index * 10, (index + 1) * 10),
            tags: ["custom/tag-42"],
            maxPoints: 500,
            sampling: "tail",
          }),
        ),
      );
    });

    await openLogFilter(user, "Scalar Tags");
    await user.type(screen.getByLabelText(/^search scalar tags$/i), "tag-42");

    const filteredTagList = screen.getByRole("listbox", {
      name: "Scalar Tags options",
    });
    const filteredTagOption = within(filteredTagList).getByRole("option", {
      name: logOptionName("custom/tag-42"),
    });
    expectLogOptionSelected(filteredTagOption);
    expectLogsChecklistRowSizing(filteredTagOption);
    expect(
      within(filteredTagList).queryByRole("option", {
        name: logOptionName("custom/tag-01"),
      }),
    ).not.toBeInTheDocument();

    await user.click(filteredTagOption);
    expectLogOptionSelected(filteredTagOption, false);

    await openLogFilter(user, "Experiments");
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
    experimentsList = await openLogFilter(user, "Experiments");
    expect(
      within(experimentsList).queryByRole("option", {
        name: logOptionName("experiment_01"),
      }),
    ).not.toBeInTheDocument();

    const secondExperiment = within(experimentsList).getByRole("option", {
      name: logOptionName("experiment_02"),
    });
    expectLogOptionSelected(secondExperiment);
    expectLogsChecklistRowSizing(secondExperiment);
    expectLogOptionSelected(
      within(experimentsList).getByRole("option", {
        name: logOptionName("experiment_42"),
      }),
    );
  });

  it("shows complete server facets before loading the next custom run page", async () => {
    const runs = Array.from({ length: 105 }, (_, index) => {
      const number = String(index + 1).padStart(3, "0");
      const lateRun = index >= 100;
      const dataset = lateRun ? "ZebraSet" : "Mnist";
      const model = lateRun ? "wide_linear" : "linear";
      const preset = lateRun ? "BASELINE" : "AAA_CONTROL";
      const runName = `mega_${number}_20260601_010203`;

      return {
        ...logRunsResponse.runs[0],
        id: `mega-${number}`,
        group: "mega_experiment",
        experiment: "mega_experiment",
        dataset,
        model,
        preset,
        runName,
        timestamp: `2026-06-01 01:${number}:03`,
        relativePath:
          `mega_experiment/${model}/${preset}/${dataset}/${runName}/version_0`,
        metrics: { "test/accuracy": 0.7 + index / 1000 },
      };
    });
    const { logRunRequests } = setupLogsScenario({
      logRunsResponse: { runs },
      logExperimentsResponse: {
        experiments: [
          {
            experiment: "mega_experiment",
            runCount: runs.length,
            relativePath: "mega_experiment",
          },
        ],
      },
      logTagsByRun: Object.fromEntries(
        runs.map((run) => [run.id, ["train/loss"]]),
      ),
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectAllLogExperiments(user);

    await waitFor(() => {
      expect(logRunRequests).toContainEqual(
        expect.objectContaining({
          experiments: ["mega_experiment"],
          limit: 100,
          offset: 0,
        }),
      );
    });
    expect(logRunRequests).not.toContainEqual(
      expect.objectContaining({
        experiments: ["mega_experiment"],
        offset: 100,
      }),
    );
    expect(screen.getByRole("button", { name: /^load more runs$/i }))
      .toBeInTheDocument();

    const datasetOptions = await openLogFilter(user, "Datasets");
    expect(within(datasetOptions).getByRole("option", {
      name: logOptionName("Mnist"),
    })).toBeInTheDocument();
    expect(within(datasetOptions).getByRole("option", {
      name: logOptionName("ZebraSet"),
    })).toBeInTheDocument();

    const modelOptions = await openLogFilter(user, "Models");
    expect(within(modelOptions).getByRole("option", {
      name: /^linear · linears/i,
    })).toBeInTheDocument();
    expect(within(modelOptions).getByRole("option", {
      name: /^wide_linear · linears/i,
    })).toBeInTheDocument();

    const presetOptions = await openLogFilter(user, "Presets");
    expect(within(presetOptions).getByRole("option", {
      name: logOptionName("AAA_CONTROL"),
    })).toBeInTheDocument();
    expect(within(presetOptions).getByRole("option", {
      name: logOptionName("BASELINE"),
    })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^load more runs$/i }));
    await waitFor(() => {
      expect(logRunRequests).toContainEqual(
        expect.objectContaining({
          experiments: ["mega_experiment"],
          limit: 100,
          offset: 100,
        }),
      );
    });
    expect(screen.queryByRole("button", { name: /^load more runs$/i }))
      .not.toBeInTheDocument();
  });

  it("lazy-loads scalar tag discovery without changing non-scalar filters", async () => {
    const fixture = buildLargeLogFixture(105);
    const sharedRuns = fixture.logRunsResponse.runs.map((run) => ({
      ...run,
      dataset: "Mnist",
      relativePath:
        `${run.experiment}/linear/BASELINE/Mnist/${run.runName}/version_0`,
    }));
    const sharedFixture = {
      ...fixture,
      logRunsResponse: { runs: sharedRuns },
    };
    const { logRunRequests, logTagRequests } = setupLogsScenario(sharedFixture);
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectAllLogExperiments(user);

    await waitFor(() => {
      expect(logRunRequests).toContainEqual(
        expect.objectContaining({ limit: 100, offset: 0 }),
      );
    });
    await user.click(screen.getByRole("button", { name: /^load more runs$/i }));
    await waitFor(() => {
      expect(logRunRequests).toContainEqual(
        expect.objectContaining({ limit: 100, offset: 100 }),
      );
    });
    await waitFor(() => {
      expect(screen.getByText("Scalar tags scanned for 100 of 105 visible runs"))
        .toBeInTheDocument();
    });
    expect(screen.queryByRole("button", { name: /^load more runs$/i }))
      .not.toBeInTheDocument();

    const firstScalarTagRunIds = sharedRuns.slice(0, 100).map((run) => run.id);
    await waitFor(() => {
      expect(logTagRequests.flatMap((request) => request.runIds))
        .toEqual(firstScalarTagRunIds);
    });
    expect(logTagRequests.flatMap((request) => request.runIds))
      .not.toContain(sharedRuns[100].id);

    const datasetCount = within(logFilterSection("Datasets")).getByText("1 / 1")
      .textContent;
    const modelCount = within(logFilterSection("Models")).getByText("1 / 1")
      .textContent;
    const presetCount = within(logFilterSection("Presets")).getByText("1 / 1")
      .textContent;

    await user.click(screen.getByRole("button", { name: /^load more scalar tags$/i }));

    await waitFor(() => {
      expect(logTagRequests.at(-1)).toEqual({
        runIds: sharedRuns.slice(100).map((run) => run.id),
      });
    });
    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /^load more scalar tags$/i }))
        .not.toBeInTheDocument();
    });
    expect(within(logFilterSection("Datasets")).getByText("1 / 1").textContent)
      .toBe(datasetCount);
    expect(within(logFilterSection("Models")).getByText("1 / 1").textContent)
      .toBe(modelCount);
    expect(within(logFilterSection("Presets")).getByText("1 / 1").textContent)
      .toBe(presetCount);
  });

  it("does not delete a log experiment when the dialog is cancelled", async () => {
    const { deleteExperimentRequests } = setupLogsScenario();
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await openLogFilter(user, "Experiments");
    await user.click(
      screen.getByRole("button", { name: /^delete experiment test_model$/i }),
    );
    let dialog = screen.getByRole("dialog", { name: /^delete experiment$/i });
    await user.click(within(dialog).getByRole("button", { name: /^cancel$/i }));

    await waitFor(() => {
      expect(screen.queryByRole("dialog", { name: /^delete experiment$/i }))
        .not.toBeInTheDocument();
    });

    await openLogFilter(user, "Experiments");
    await user.click(
      screen.getByRole("button", { name: /^delete experiment test_model$/i }),
    );
    dialog = screen.getByRole("dialog", { name: /^delete experiment$/i });
    await user.click(
      within(dialog).getByRole("button", { name: /^close delete experiment$/i }),
    );

    expect(deleteExperimentRequests).toHaveLength(0);
    await expectLogFilterSelection(user, "Experiments", "test_model", false);
  });

  it("keeps the delete dialog open and shows backend errors", async () => {
    const { deleteExperimentRequests } = setupLogsScenario({
      deleteLogExperimentError: "Refusing to delete symlink log experiment: test_model",
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await openLogFilter(user, "Experiments");
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
    expect(dialog).toBeInTheDocument();
  });

  it("omits stale scalar tags from chart requests after experiment filtering", async () => {
    const { logScalarRequests } = setupLogsScenario({
      logRunsResponse: logRunsWithSharedDataset(),
      logTagsByRun: {
        "log-mnist": ["train/loss", "validation/accuracy"],
        "log-cifar": ["train/loss"],
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectAllLogExperiments(user);
    await setLogOptionSelection(user, "Scalar Tags", "validation/accuracy", true);
    await setLogOptionSelection(user, "Scalar Tags", "train/loss", true);

    await waitFor(() => {
      expect(logScalarRequests).toContainEqual({
        runIds: ["log-mnist", "log-cifar"],
        tags: ["validation/accuracy"],
        maxPoints: 500,
        sampling: "tail",
      });
      expect(logScalarRequests).toContainEqual({
        runIds: ["log-mnist", "log-cifar"],
        tags: ["train/loss"],
        maxPoints: 500,
        sampling: "tail",
      });
    });

    await clickLogOption(user, "Experiments", "test_model");
    await setLogOptionSelection(user, "Scalar Tags", "train/loss", true);

    await waitFor(() => {
      expect(logScalarRequests).toContainEqual({
        runIds: ["log-cifar"],
        tags: ["train/loss"],
        maxPoints: 500,
        sampling: "tail",
      });
    });
    expect(screen.getByText(/1 runs · 1 selected tags/i)).toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .not.toBeInTheDocument();

    await clickLogOption(user, "Experiments", "test_model");
    await setLogOptionSelection(user, "Scalar Tags", "validation/accuracy", true);
    await setLogOptionSelection(user, "Scalar Tags", "train/loss", true);

    await waitFor(() => {
      expect(screen.getByText(/2 runs · 2 selected tags/i)).toBeInTheDocument();
    });
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
  });

  it("recovers scalar tags when selecting Kaggle after a stale monitor tag", async () => {
    const priorRun = {
      ...logRunsResponse.runs[0],
      id: "prior-monitor",
      group: "prior_monitor",
      experiment: "prior_monitor",
      dataset: "Mnist",
      preset: "BASELINE",
      runName: "prior_monitor_20260601_010203",
      timestamp: "2026-06-01 01:02:03",
      relativePath:
        "prior_monitor/linear/BASELINE/Mnist/prior_monitor_20260601_010203/version_0",
      metrics: { "test/accuracy": 0.82 },
    };
    const kaggleRunIds = ["kaggle-linear-all-fold-0", "kaggle-linear-all-fold-1"];
    const kaggleRuns = kaggleRunIds.map((id, index) => ({
      ...logRunsResponse.runs[0],
      id,
      group: "kaggle_linear_all",
      experiment: "kaggle_linear_all",
      preset: "KAGGLE_LINEAR",
      dataset: "KaggleDigits",
      runName: `kaggle_linear_all_fold_${index}_20260601_030405`,
      timestamp: `2026-06-01 0${index + 3}:04:05`,
      relativePath:
        `kaggle_linear_all/linear/KAGGLE_LINEAR/KaggleDigits/kaggle_linear_all_fold_${index}_20260601_030405/version_0`,
      metrics: { "test/accuracy": 0.88 + index / 100 },
    }));
    const runs = [priorRun, ...kaggleRuns];
    const monitorTag = "main_model.0.model/weights/mean";
    const { logScalarRequests, logTagRequests } = setupLogsScenario({
      logRunsResponse: { runs },
      logExperimentsResponse: {
        experiments: [
          { experiment: "prior_monitor", runCount: 1, relativePath: "prior_monitor" },
          {
            experiment: "kaggle_linear_all",
            runCount: kaggleRuns.length,
            relativePath: "kaggle_linear_all",
          },
        ],
      },
      logTagsByRun: {
        [priorRun.id]: ["train/loss", "validation/accuracy", monitorTag],
        ...Object.fromEntries(
          kaggleRuns.map((run) => [
            run.id,
            ["train/loss", "validation/accuracy"],
          ]),
        ),
      },
      logScalarSeries: [
        {
          runId: priorRun.id,
          tag: "train/loss",
          points: [{ step: 1, wallTime: 1780000000, value: 0.7 }],
        },
        {
          runId: priorRun.id,
          tag: "validation/accuracy",
          points: [{ step: 1, wallTime: 1780000000, value: 0.81 }],
        },
        {
          runId: priorRun.id,
          tag: monitorTag,
          points: [{ step: 1, wallTime: 1780000000, value: 0.01 }],
        },
        ...kaggleRuns.flatMap((run, index) => [
          {
            runId: run.id,
            tag: "train/loss",
            points: [
              { step: 1, wallTime: 1780000100 + index, value: 0.6 - index / 100 },
            ],
          },
          {
            runId: run.id,
            tag: "validation/accuracy",
            points: [
              { step: 1, wallTime: 1780000100 + index, value: 0.76 + index / 100 },
            ],
          },
        ]),
      ],
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await screen.findByText("Historical Scalars");
    await selectLogExperiments(user, ["prior_monitor"]);
    await clickLogOption(user, "Scalar Tags", "train/loss");
    await clickLogOption(user, "Scalar Tags", "validation/accuracy");
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();

    await clickLogOption(user, "Scalar Tags", "train/loss");
    await clickLogOption(user, "Scalar Tags", "validation/accuracy");
    await clickLogOption(user, "Scalar Tags", monitorTag);
    await expectLogFilterSelection(user, "Scalar Tags", monitorTag, true);
    await expectLogFilterSelection(user, "Scalar Tags", "train/loss", false);
    await expectLogFilterSelection(user, "Scalar Tags", "validation/accuracy", false);

    await clickLogOption(user, "Experiments", "prior_monitor");
    await clickLogOption(user, "Experiments", "kaggle_linear_all");

    await waitFor(() => {
      expect(logTagRequests.at(-1)).toEqual({ runIds: kaggleRunIds });
    });
    await waitFor(() => {
      expect(logScalarRequests).toContainEqual({
        runIds: kaggleRunIds,
        tags: ["validation/accuracy"],
        maxPoints: 500,
        sampling: "tail",
      });
      expect(logScalarRequests).toContainEqual({
        runIds: kaggleRunIds,
        tags: ["train/loss"],
        maxPoints: 500,
        sampling: "tail",
      });
    });
    const staleKaggleRequests = logScalarRequests.filter(
      (request) =>
        request.runIds.length === kaggleRunIds.length &&
        kaggleRunIds.every((runId) => request.runIds.includes(runId)) &&
        request.tags.includes(monitorTag),
    );
    expect(staleKaggleRequests).toEqual([]);
    expect(screen.queryByText("No scalar points for selection")).not.toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();
  });

  it("shows only common lower filters for selected experiments", async () => {
    const expACifarRun = {
      ...logRunsResponse.runs[0],
      id: "exp-a-cifar-common",
      group: "exp_a",
      experiment: "exp_a",
      preset: "BASELINE",
      dataset: "Cifar10",
      runName: "exp_a_cifar_common_20260601_010203",
      timestamp: "2026-06-01 01:02:03",
      relativePath:
        "exp_a/linear/BASELINE/Cifar10/exp_a_cifar_common_20260601_010203/version_0",
      metrics: { "test/accuracy": 0.82 },
    };
    const expAMnistRun = {
      ...expACifarRun,
      id: "exp-a-mnist-wide",
      model: "wide-linear",
      preset: "WIDE_ONLY",
      dataset: "Mnist",
      runName: "exp_a_mnist_wide_20260601_020304",
      timestamp: "2026-06-01 02:03:04",
      relativePath:
        "exp_a/wide-linear/WIDE_ONLY/Mnist/exp_a_mnist_wide_20260601_020304/version_0",
      metrics: { "test/accuracy": 0.84 },
    };
    const expBCifarRun = {
      ...logRunsResponse.runs[0],
      id: "exp-b-cifar-common",
      group: "exp_b",
      experiment: "exp_b",
      preset: "BASELINE",
      dataset: "Cifar10",
      runName: "exp_b_cifar_common_20260601_030405",
      timestamp: "2026-06-01 03:04:05",
      relativePath:
        "exp_b/linear/BASELINE/Cifar10/exp_b_cifar_common_20260601_030405/version_0",
      metrics: { "test/accuracy": 0.88 },
    };
    const expBConvRun = {
      ...expBCifarRun,
      id: "exp-b-cifar-conv",
      model: "conv",
      preset: "CONV_ONLY",
      runName: "exp_b_cifar_conv_20260601_040506",
      timestamp: "2026-06-01 04:05:06",
      relativePath:
        "exp_b/conv/CONV_ONLY/Cifar10/exp_b_cifar_conv_20260601_040506/version_0",
      metrics: { "test/accuracy": 0.9 },
    };
    const runs = [expACifarRun, expAMnistRun, expBCifarRun, expBConvRun];
    const { logRunRequests, logTagRequests } = setupLogsScenario({
      logRunsResponse: { runs },
      logExperimentsResponse: {
        experiments: [
          { experiment: "exp_a", runCount: 2, relativePath: "exp_a" },
          { experiment: "exp_b", runCount: 2, relativePath: "exp_b" },
        ],
      },
      logTagsByRun: Object.fromEntries(
        runs.map((run) => [
          run.id,
          scalarTagsWithEpochDefaults("train/loss", "validation/accuracy"),
        ]),
      ),
      logScalarSeries: runs.flatMap((run, index) => [
        ...epochScalarSeriesForRun(run.id, index),
        {
          runId: run.id,
          tag: "train/loss",
          points: [
            { step: 1, wallTime: 1780000000 + index, value: 0.7 - index / 100 },
          ],
        },
        {
          runId: run.id,
          tag: "validation/accuracy",
          points: [
            { step: 1, wallTime: 1780000000 + index, value: 0.7 + index / 100 },
          ],
        },
      ]),
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await screen.findByText("Historical Scalars");
    await selectLogExperiments(user, ["exp_a", "exp_b"]);

    await waitFor(() => {
      expect(logRunRequests).toContainEqual(
        expect.objectContaining({
          experiments: ["exp_a", "exp_b"],
          limit: 100,
          offset: 0,
        }),
      );
    });
    await waitFor(() => {
      expect(logTagRequests.at(-1)).toEqual({
        runIds: ["exp-a-cifar-common", "exp-b-cifar-common"],
      });
    });

    const datasetOptions = await openLogFilter(user, "Datasets");
    expect(within(datasetOptions).getByRole("option", {
      name: logOptionName("Cifar10"),
    })).toBeInTheDocument();
    expect(within(datasetOptions).queryByRole("option", {
      name: logOptionName("Mnist"),
    })).not.toBeInTheDocument();

    const modelOptions = await openLogFilter(user, "Models");
    expect(within(modelOptions).getByRole("option", {
      name: /linear · linears/i,
    })).toBeInTheDocument();
    expect(within(modelOptions).queryByRole("option", {
      name: /wide-linear · linears/i,
    })).not.toBeInTheDocument();
    expect(within(modelOptions).queryByRole("option", {
      name: /conv · linears/i,
    })).not.toBeInTheDocument();

    const presetOptions = await openLogFilter(user, "Presets");
    expect(within(presetOptions).getByRole("option", {
      name: logOptionName("BASELINE"),
    })).toBeInTheDocument();
    expect(within(presetOptions).queryByRole("option", {
      name: logOptionName("WIDE_ONLY"),
    })).not.toBeInTheDocument();
    expect(within(presetOptions).queryByRole("option", {
      name: logOptionName("CONV_ONLY"),
    })).not.toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();
  });

  it("shows no lower filters or chart requests for experiments with no overlap", async () => {
    const mnistRun = {
      ...logRunsResponse.runs[0],
      id: "exp-a-mnist",
      group: "exp_a",
      experiment: "exp_a",
      dataset: "Mnist",
      model: "linear",
      preset: "BASELINE",
      runName: "exp_a_mnist_20260601_010203",
      timestamp: "2026-06-01 01:02:03",
      relativePath: "exp_a/linear/BASELINE/Mnist/exp_a_mnist_20260601_010203/version_0",
    };
    const cifarRun = {
      ...logRunsResponse.runs[0],
      id: "exp-b-cifar",
      group: "exp_b",
      experiment: "exp_b",
      dataset: "Cifar10",
      model: "wide-linear",
      preset: "WIDE_ONLY",
      runName: "exp_b_cifar_20260601_020304",
      timestamp: "2026-06-01 02:03:04",
      relativePath:
        "exp_b/wide-linear/WIDE_ONLY/Cifar10/exp_b_cifar_20260601_020304/version_0",
    };
    const { logScalarRequests, logTagRequests } = setupLogsScenario({
      logRunsResponse: { runs: [mnistRun, cifarRun] },
      logExperimentsResponse: {
        experiments: [
          { experiment: "exp_a", runCount: 1, relativePath: "exp_a" },
          { experiment: "exp_b", runCount: 1, relativePath: "exp_b" },
        ],
      },
      logTagsByRun: {
        [mnistRun.id]: ["train/loss"],
        [cifarRun.id]: ["train/loss"],
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await screen.findByText("Historical Scalars");
    await selectAllLogExperiments(user);

    await waitFor(() => {
      expect(screen.getByRole("combobox", { name: /^Datasets\b/i })).toBeDisabled();
      expect(screen.getByRole("combobox", { name: /^Models\b/i })).toBeDisabled();
      expect(screen.getByRole("combobox", { name: /^Presets\b/i })).toBeDisabled();
    });
    expect(await screen.findByText("No runs selected")).toBeInTheDocument();
    expect(logTagRequests).toHaveLength(0);
    expect(logScalarRequests).toHaveLength(0);
  });

  it("renders Kaggle linear scalars after All tags is clicked during tag refresh", async () => {
    const fixture = buildKaggleLinearLogFixture();
    const kaggleTagsResponse = deferred<{
      runs: Array<{
        runId: string;
        scalarTags: string[];
        histogramTags: string[];
        imageTags: string[];
        textTags: string[];
      }>;
    }>();
    const { logScalarRequests, logTagRequests } = setupLogsScenario({
      logRunsResponse: fixture.logRunsResponse,
      logExperimentsResponse: fixture.logExperimentsResponse,
      logTagsByRun: fixture.logTagsByRun,
      logScalarSeries: fixture.logScalarSeries,
      logTagsResponseFactory: (body) => {
        if (
          body.runIds.length === fixture.kaggleRunIds.length &&
          fixture.kaggleRunIds.every((runId) => body.runIds.includes(runId))
        ) {
          return kaggleTagsResponse.promise;
        }
        return undefined;
      },
    });
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await screen.findByText("Historical Scalars");
    await selectLogExperiments(user, ["normal_linear"]);
    expect(await screen.findByRole("img", { name: /validation\/accuracy_epoch scalar chart/i }))
      .toBeInTheDocument();

    await clickLogOption(user, "Experiments", "normal_linear");
    await clickLogOption(user, "Experiments", "kaggle_linear");

    await waitFor(() => {
      expect(logTagRequests.at(-1)).toEqual({ runIds: fixture.kaggleRunIds });
    });
    await user.click(
      within(logFilterSection("Scalar Tags")).getByRole("button", { name: /^all$/i }),
    );
    expect(screen.queryByText("No scalar points for selection")).not.toBeInTheDocument();

    kaggleTagsResponse.resolve({
      runs: fixture.kaggleRunIds.map((runId) => ({
        runId,
        scalarTags: fixture.kaggleTags,
        histogramTags: [],
        imageTags: [],
        textTags: [],
      })),
    });

    await waitFor(() => {
      expect(logScalarRequests).toContainEqual({
        runIds: fixture.kaggleRunIds,
        tags: ["train/kaggle_logloss"],
        maxPoints: 500,
        sampling: "tail",
      });
      expect(logScalarRequests).toContainEqual({
        runIds: fixture.kaggleRunIds,
        tags: ["validation/kaggle_auc"],
        maxPoints: 500,
        sampling: "tail",
      });
    });
    const staleKaggleRequests = logScalarRequests.filter(
      (request) =>
        request.runIds.length === fixture.kaggleRunIds.length &&
        fixture.kaggleRunIds.every((runId) => request.runIds.includes(runId)) &&
        request.tags.some((tag) => fixture.normalTags.includes(tag)),
    );
    expect(staleKaggleRequests).toEqual([]);
    expect(screen.queryByText("No scalar points for selection")).not.toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /train\/kaggle_logloss scalar chart/i }))
      .toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /validation\/kaggle_auc scalar chart/i }))
      .toBeInTheDocument();
  });

  it("shows the scalar empty state without scalar fetches when event runs have no tags", async () => {
    const { logScalarRequests } = setupLogsScenario({
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
    renderWorkbench();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);

    expect(await screen.findByText("No TensorBoard scalars")).toBeInTheDocument();
    expect(
      await screen.findByText("The selected runs do not contain scalar event data."),
    ).toBeInTheDocument();
    expect(logScalarRequests).toHaveLength(0);
  });

});
