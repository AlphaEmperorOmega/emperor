import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import {
  buildKaggleLinearLogFixture,
  buildLargeLogFixture,
  buildSubsetDeleteFixture,
  capabilitiesResponse,
  deferred,
  expectLogsChecklistRowSizing,
  installFetchMock,
  logTagsByRun,
  logMetricGroupToggle,
  logValidationExamplesToggle,
  logRunsResponse,
  logScalarSeries,
  renderViewer,
  resetViewerAppTestState,
  scalarChartGridFor,
} from "./support";

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function logOptionName(label: string) {
  return new RegExp(`^${escapeRegExp(label)}(?:\\b|$)`, "i");
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

function logScalarLegendButton(card: HTMLElement, runLabel: RegExp) {
  return within(card).getByRole("button", { name: runLabel });
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

async function openMetricPlotSelector(
  user: ReturnType<typeof userEvent.setup>,
  group: "Train" | "Validation",
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
  group: "Train" | "Validation",
  label: string,
) {
  const listbox = await openMetricPlotSelector(user, group);
  return within(listbox).findByRole("option", { name: logOptionName(label) });
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

describe("ViewerApp Logs Workspace", () => {
  beforeEach(resetViewerAppTestState);

  it("opens logs without selecting runs or loading TensorBoard data", async () => {
    const { logScalarRequests, logTagRequests } = installFetchMock();
    renderViewer();
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

    await user.click(screen.getByRole("button", { name: /all runs/i }));

    await expectLogFilterSelection(user, "Experiments", "test_model", false);
    await expectLogFilterSelection(user, "Experiments", "test_model_2", false);
    expect(screen.getByRole("combobox", { name: /^Datasets\b/i })).toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: /^Runs\b/i })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Datasets Mnist")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Datasets Cifar10")).not.toBeInTheDocument();
    expect(logTagRequests).toHaveLength(0);
    expect(logScalarRequests).toHaveLength(0);
  });

  it("keeps log scope controls inside Experiments without summary cards", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await screen.findByRole("combobox", { name: /^Experiments\b/i });

    const experimentSection = logFilterSection("Experiments");
    const currentTargetButton = within(experimentSection).getByRole("button", {
      name: /current target/i,
    });
    const allRunsButton = within(experimentSection).getByRole("button", {
      name: /all runs/i,
    });

    expect(currentTargetButton).toBeDisabled();
    expect(allRunsButton).toBeEnabled();
    expect(screen.queryByText(/^Runs$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^Tags$/)).not.toBeInTheDocument();
    expect(screen.queryByText(/^linear · BASELINE · Mnist$/)).not.toBeInTheDocument();

    await user.click(allRunsButton);

    await waitFor(() => {
      expect(
        within(logFilterSection("Experiments")).getByRole("button", {
          name: /current target/i,
        }),
      ).toBeEnabled();
    });
  });

  it("opens logs scoped to the current target dataset and broadens through All runs", async () => {
    const { logRunRequests, logScalarRequests } = installFetchMock();
    renderViewer();
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
            hasEventFiles: "true",
            limit: 5,
            offset: 0,
          },
        ]),
      );
    });
    const experimentSection = logFilterSection("Experiments");
    const currentTargetButton = within(experimentSection).getByRole("button", {
      name: /current target/i,
    });
    const allRunsButton = within(experimentSection).getByRole("button", {
      name: /all runs/i,
    });
    expect(currentTargetButton).toBeDisabled();
    expect(allRunsButton).toBeEnabled();
    await expectLogFilterSelection(user, "Experiments", "test_model", false);
    await selectLogExperiments(user, ["test_model"]);
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
    expect(logMetricGroupToggle("Train")).toHaveAttribute("aria-expanded", "true");
    expect(logMetricGroupToggle("Validation")).toHaveAttribute("aria-expanded", "true");
    expect(logMetricGroupToggle("Test")).toHaveAttribute("aria-expanded", "false");
    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(3);
      expect(logScalarRequests).toEqual(expect.arrayContaining([
        {
          runIds: ["log-mnist"],
          tags: ["train/loss"],
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
    expect(await screen.findByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /test\/accuracy scalar chart/i }))
      .not.toBeInTheDocument();

    await user.click(allRunsButton);
    await selectAllLogExperiments(user);
    await waitFor(() => {
      expect(logRunRequests).toEqual(
        expect.arrayContaining([
          expect.objectContaining({
            datasets: [],
            limit: 500,
            offset: 0,
          }),
        ]),
      );
    });
    await user.click(
      await screen.findByRole("button", { name: /^Test\s+\d+\s+metrics?$/i }),
    );

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
      screen.getAllByText(/test_model · Mnist · linear · linears · BASELINE · 2026-06-01 01:02:03/).length,
    ).toBeGreaterThan(0);
    const cifarLine = within(accuracyLeaderboard).getByRole("button", {
      name: /open run details for test_model_2 · Cifar10 · linear · linears · BASELINE · 2026-06-01 02:03:04/i,
    });

    await user.click(cifarLine);

    const detailsPanel = screen.getByRole("heading", { name: "Run Details" }).closest("aside");
    expect(detailsPanel).not.toBeNull();
    expect(within(detailsPanel as HTMLElement).getByText("Experiment")).toBeInTheDocument();
    expect(within(detailsPanel as HTMLElement).getByText("test_model_2")).toBeInTheDocument();
    expect(screen.getAllByText("No result.json").length).toBeGreaterThan(0);
    expect(screen.queryByRole("button", { name: /start training/i }))
      .not.toBeInTheDocument();
  });

  it("ranks best runs independently from selected scalar chart tags", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(screen.getByRole("button", { name: /all runs/i }));
    await selectAllLogExperiments(user);

    const bestRunHeading = await screen.findByRole("heading", { name: "Best Run" });
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
    expect(within(bestRunTable).getByText("Mnist")).toBeInTheDocument();
    expect(within(bestRunTable).getByText("Cifar10")).toBeInTheDocument();
    expect(within(bestRunTable).getByText("0.8")).toBeInTheDocument();
    expect(within(bestRunTable).getByText("aaa_20260601_010203"))
      .toBeInTheDocument();
    expect(within(bestRunTable).getByText("0.55")).toBeInTheDocument();
    expect(within(bestRunTable).getByText("bbb_20260601_020304"))
      .toBeInTheDocument();

    await clickLogOption(user, "Scalar Tags", "validation/accuracy");

    await waitFor(() => {
      expect(screen.queryByRole("img", { name: /validation\/accuracy scalar chart/i }))
        .not.toBeInTheDocument();
    });
    expect(
      within(panel).getByRole("table", {
        name: /validation\/accuracy best run leaderboard/i,
      }),
    ).toBeInTheDocument();

    await user.click(
      within(panel).getAllByRole("button", {
        name: /open details for test_model_2 · Cifar10/i,
      })[0],
    );

    const detailsPanel = screen.getByRole("heading", { name: "Run Details" }).closest("aside");
    expect(detailsPanel).not.toBeNull();
    expect(within(detailsPanel as HTMLElement).getByText("test_model_2"))
      .toBeInTheDocument();
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
    const { logScalarRequests, logTagRequests } = installFetchMock({
      logRunsResponse: { runs },
      logExperimentsResponse: {
        experiments: [
          { experiment: "multi_dataset", runCount: 2, relativePath: "multi_dataset" },
        ],
      },
      logTagsByRun: {
        "multi-mnist": ["train/loss", "validation/accuracy"],
        "multi-cifar": ["train/loss", "validation/accuracy"],
      },
      logScalarSeries: [
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
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(await screen.findByRole("button", { name: /all runs/i }));
    await selectAllLogExperiments(user);

    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
    expect(within(logFilterSection("Datasets")).getByText("2 / 2")).toBeInTheDocument();
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
        tags: ["validation/accuracy"],
        maxPoints: 500,
        sampling: "tail",
      });
      expect(logScalarRequests).toContainEqual({
        runIds: ["multi-cifar"],
        tags: ["train/loss"],
        maxPoints: 500,
        sampling: "tail",
      });
    });
    expect(within(logFilterSection("Datasets")).getByText("1 / 2")).toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
    await waitFor(() => {
      expect(screen.queryByText(/multi_mnist_20260601_010203/))
        .not.toBeInTheDocument();
    });
    expect(screen.getAllByText(/multi_dataset · Cifar10 · linear · linears · BASELINE/).length)
      .toBeGreaterThan(0);

    const detailsPanel = screen.getByRole("heading", { name: "Run Details" }).closest("aside");
    expect(detailsPanel).not.toBeNull();
    expect(within(detailsPanel as HTMLElement).getByTitle("multi_cifar_20260601_020304"))
      .toBeInTheDocument();
    expect(within(detailsPanel as HTMLElement).queryByTitle("multi_mnist_20260601_010203"))
      .not.toBeInTheDocument();
  });

  it("keeps validation examples collapsed until the accordion is opened", async () => {
    const { logMediaRequests } = installFetchMock({
      logTagsByRun: {
        "log-mnist": {
          scalarTags: ["validation/accuracy"],
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
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
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
    const { logScalarRequests } = installFetchMock({
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
    renderViewer();
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
    expect(
      await screen.findByRole("img", {
        name: /validation confusion matrix for aaa_20260601_010203/i,
      }),
    ).toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: /^Confusion Matrix\s+1 matrix$/i }),
    ).toHaveAttribute("aria-expanded", "true");
  });

  it("collapses logs metric groups without changing selected tags or refetching scalars", async () => {
    const { logScalarRequests } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);

    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();
    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(3);
    });

    await user.click(logMetricGroupToggle("Train"));

    await waitFor(() => {
      expect(logMetricGroupToggle("Train")).toHaveAttribute("aria-expanded", "false");
    });
    expect(screen.queryByRole("img", { name: /train\/loss scalar chart/i }))
      .not.toBeInTheDocument();
    await expectLogFilterSelection(user, "Scalar Tags", "train/loss", true);
    expect(logScalarRequests).toHaveLength(3);

    await user.click(logMetricGroupToggle("Test"));

    await waitFor(() => {
      expect(logMetricGroupToggle("Test")).toHaveAttribute("aria-expanded", "true");
    });
    expect(await screen.findByRole("table", { name: /test\/accuracy test leaderboard/i }))
      .toBeInTheDocument();
    await expectLogFilterSelection(user, "Scalar Tags", "test/accuracy", true);
    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(4);
    });
    expect(logScalarRequests[3]).toMatchObject({
      runIds: ["log-mnist"],
      tags: ["test/accuracy"],
    });

    await user.click(logMetricGroupToggle("Test"));

    await waitFor(() => {
      expect(logMetricGroupToggle("Test")).toHaveAttribute("aria-expanded", "false");
    });
    expect(screen.queryByRole("table", { name: /test\/accuracy test leaderboard/i }))
      .not.toBeInTheDocument();
    expect(logScalarRequests).toHaveLength(4);
  });

  it("filters train and validation plots locally from accordion selectors", async () => {
    const { logScalarRequests } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);

    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();
    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(3);
    });
    const scalarRequestCount = logScalarRequests.length;

    expect(
      screen.getByRole("combobox", { name: /^Train plots\s+1 \/ 1 selected$/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("combobox", { name: /^Validation plots\s+1 \/ 1 selected$/i }),
    ).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "Train plot controls" })).toHaveClass(
      "grid",
      "grid-cols-[minmax(0,1fr)_auto_auto]",
    );
    expect(screen.getByRole("group", { name: "Validation plot controls" })).toHaveClass(
      "grid",
      "grid-cols-[minmax(0,1fr)_auto_auto]",
    );
    expectLogOptionSelected(
      await findMetricPlotOption(user, "Train", "train/loss"),
      true,
    );
    expectLogOptionSelected(
      await findMetricPlotOption(user, "Validation", "validation/accuracy"),
      true,
    );

    await user.click(
      screen.getByRole("button", { name: "Select no Validation plots" }),
    );

    await waitFor(() => {
      expect(screen.queryByRole("img", { name: /validation\/accuracy scalar chart/i }))
        .not.toBeInTheDocument();
    });
    expect(screen.getByText("No plots selected in this group")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();
    await expectLogFilterSelection(user, "Scalar Tags", "validation/accuracy", true);
    expect(logScalarRequests).toHaveLength(scalarRequestCount);

    await user.click(
      screen.getByRole("button", { name: "Select all Validation plots" }),
    );

    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
    expect(logScalarRequests).toHaveLength(scalarRequestCount);
  });

  it("opens a metric plot selector without collapsing its accordion", async () => {
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);

    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
    const validationToggle = logMetricGroupToggle("Validation");
    expect(validationToggle).toHaveAttribute("aria-expanded", "true");

    await user.click(
      screen.getByRole("combobox", { name: /^Validation plots\s+1 \/ 1 selected$/i }),
    );

    expect(
      await screen.findByRole("listbox", { name: "Validation plots options" }),
    ).toBeInTheDocument();
    expect(validationToggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();

    await user.click(
      screen.getByRole("button", { name: "Select all Validation plots" }),
    );
    expect(validationToggle).toHaveAttribute("aria-expanded", "true");

    await user.click(
      screen.getByRole("button", { name: "Select no Validation plots" }),
    );
    expect(validationToggle).toHaveAttribute("aria-expanded", "true");
  });

  it("keeps loaded scalar groups visible while the default-open Train group loads", async () => {
    const trainScalarResponse = deferred<unknown>();
    const { logScalarRequests } = installFetchMock({
      logScalarResponseFactory: (body) => {
        if (body.tags.includes("train/loss")) {
          return trainScalarResponse.promise;
        }
        return undefined;
      },
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();

    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(3);
    });
    expect(screen.getByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
    expect(screen.queryByText(/^Loading scalar points$/i)).not.toBeInTheDocument();
    const trainBody = document.getElementById("logs-metric-group-train");
    expect(trainBody).toBeInstanceOf(HTMLElement);
    expect(within(trainBody as HTMLElement).getByText("Loading Train scalar points"))
      .toBeInTheDocument();

    trainScalarResponse.resolve({
      series: logScalarSeries.filter(
        (series) => series.runId === "log-mnist" && series.tag === "train/loss",
      ),
    });

    expect(await screen.findByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();
    expect(screen.queryByText("Loading Train scalar points")).not.toBeInTheDocument();
  });

  it("keeps existing charts visible while a later log tag chunk loads", async () => {
    const extraRuns = buildLargeLogFixture(55).logRunsResponse.runs.map(
      (run, index) => ({
        ...run,
        id: `extra-log-${index + 1}`,
        model: "wide-linear",
        dataset: "Cifar10",
        preset: "ALT",
        relativePath: run.relativePath
          .replace("/linear/", "/wide-linear/")
          .replace("/BASELINE/", "/ALT/")
          .replace("/Mnist/", "/Cifar10/"),
      }),
    );
    const runs = [...logRunsResponse.runs, ...extraRuns];
    const delayedTagChunk = deferred<null>();
    const { logTagRequests } = installFetchMock({
      logRunsResponse: { runs },
      logExperimentsResponse: {
        experiments: runs.map((run) => ({
          experiment: run.experiment,
          runCount: 1,
          relativePath: run.experiment,
        })),
      },
      logTagsByRun: {
        "log-mnist": ["train/loss", "validation/accuracy"],
        "log-cifar": ["validation/accuracy"],
        ...Object.fromEntries(
          extraRuns.map((run) => [run.id, ["validation/accuracy"]]),
        ),
      },
      logScalarSeries: [
        ...logScalarSeries,
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
            scalarTags: ["validation/accuracy"],
            histogramTags: [],
            imageTags: [],
            textTags: [],
          })),
        }));
      },
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await openLogFilter(user, "Experiments");
    await user.type(await screen.findByLabelText(/^search experiments$/i), "test_model");
    await selectLogExperiments(user, ["test_model"]);
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /all runs/i }));
    await selectAllLogExperiments(user);

    await waitFor(() => {
      expect(
        logTagRequests.some((request) => request.runIds.includes("extra-log-49")),
      ).toBe(true);
    });
    expect(screen.getByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
    expect(screen.getByText("Refreshing TensorBoard tags")).toBeInTheDocument();
    expect(screen.queryByText("Reading TensorBoard tags")).not.toBeInTheDocument();

    delayedTagChunk.resolve(null);

    await waitFor(() => {
      expect(screen.queryByText("Refreshing TensorBoard tags")).not.toBeInTheDocument();
    });
  });

  it("shows checkpoints, params, and artifacts in run details", async () => {
    const { logArtifactRequests, logCheckpointRequests } = installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);

    const detailsPanel = screen.getByRole("heading", { name: "Run Details" }).closest("aside");
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

    installFetchMock({
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
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(await screen.findByRole("button", { name: /all runs/i }));
    await selectAllLogExperiments(user);

    const detailsPanel = screen.getByRole("heading", { name: "Run Details" }).closest("aside");
    expect(detailsPanel).not.toBeNull();
    const panel = detailsPanel as HTMLElement;
    expect(panel).toHaveClass("min-w-0", "overflow-x-hidden");

    expect(await within(panel).findByTitle(longRunName)).toHaveClass("min-w-0", "truncate");
    const path = await within(panel).findByTitle(longRelativePath);
    expect(path).toHaveClass("min-w-0", "break-words");
    expect(path.classList.contains("[overflow-wrap:anywhere]")).toBe(true);

    for (const summaryValue of [
      longExperiment,
      longDataset,
      longModel,
      longPreset,
      longVersion,
    ]) {
      expect(within(panel).getByTitle(summaryValue)).toHaveClass("min-w-0", "truncate");
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
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
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
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await selectLogExperiments(user, ["test_model"]);
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
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
    await expectLogFilterSelection(user, "Scalar Tags", "train/loss", true);
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
    await user.click(screen.getByRole("button", { name: /all runs/i }));
    await selectAllLogExperiments(user);
    await user.click(logMetricGroupToggle("Test"));

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
    await selectLogExperiments(user, ["test_model"]);

    const chart = await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i });
    const chartGrid = scalarChartGridFor(chart);
    const layoutControl = screen.getByRole("radiogroup", { name: /scalar chart layout/i });
    const fullTab = within(layoutControl).getByRole("radio", { name: /^full$/i });
    const twoColumnTab = within(layoutControl).getByRole("radio", { name: /^2 col$/i });
    const threeColumnTab = within(layoutControl).getByRole("radio", { name: /^3 col$/i });

    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(3);
    });
    expect(fullTab).toHaveAttribute("aria-checked", "false");
    expect(twoColumnTab).toHaveAttribute("aria-checked", "true");
    expect(threeColumnTab).toHaveAttribute("aria-checked", "false");
    expect(chartGrid).toHaveClass("grid", "gap-4");
    expect(chartGrid).toHaveClass("xl:grid-cols-2");
    expect(chartGrid).not.toHaveClass("2xl:grid-cols-3");

    await user.click(fullTab);

    expect(fullTab).toHaveAttribute("aria-checked", "true");
    expect(twoColumnTab).toHaveAttribute("aria-checked", "false");
    expect(chartGrid).not.toHaveClass("xl:grid-cols-2");
    expect(chartGrid).not.toHaveClass("2xl:grid-cols-3");
    expect(logScalarRequests).toHaveLength(3);

    await user.click(twoColumnTab);

    expect(fullTab).toHaveAttribute("aria-checked", "false");
    expect(twoColumnTab).toHaveAttribute("aria-checked", "true");
    expect(chartGrid).toHaveClass("xl:grid-cols-2");
    expect(chartGrid).not.toHaveClass("2xl:grid-cols-3");
    expect(logScalarRequests).toHaveLength(3);

    await user.click(threeColumnTab);

    expect(twoColumnTab).toHaveAttribute("aria-checked", "false");
    expect(threeColumnTab).toHaveAttribute("aria-checked", "true");
    expect(chartGrid).toHaveClass("xl:grid-cols-2", "2xl:grid-cols-3");
    expect(logScalarRequests).toHaveLength(3);

    await user.click(screen.getByRole("button", { name: /refresh scalar charts/i }));

    await waitFor(() => {
      expect(logScalarRequests).toHaveLength(6);
    });
  });

  it("links scalar legend hover across charts in the same metric accordion", async () => {
    const trainAccuracySeries = logRunsResponse.runs.map((run, index) => ({
      runId: run.id,
      tag: "train/accuracy",
      points: [
        { step: 1, wallTime: 1780000000 + index, value: 0.51 + index / 10 },
        { step: 2, wallTime: 1780000001 + index, value: 0.71 + index / 10 },
      ],
    }));
    installFetchMock({
      logTagsByRun: Object.fromEntries(
        Object.entries(logTagsByRun).map(([runId, tags]) => [
          runId,
          Array.isArray(tags)
            ? [...tags, "train/accuracy"]
            : {
                ...tags,
                scalarTags: [...(tags.scalarTags ?? []), "train/accuracy"],
              },
        ]),
      ),
      logScalarSeries: [...logScalarSeries, ...trainAccuracySeries],
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(screen.getByRole("button", { name: /all runs/i }));
    await selectAllLogExperiments(user);

    expect(await screen.findByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /train\/accuracy scalar chart/i }))
      .toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();

    const mnistRun = /test_model · Mnist · linear · linears · BASELINE · 2026-06-01 01:02:03/i;
    const cifarRun = /test_model_2 · Cifar10 · linear · linears · BASELINE · 2026-06-01 02:03:04/i;
    const trainLossCard = scalarChartSection(/train\/loss scalar chart/i);
    const trainAccuracyCard = scalarChartSection(/train\/accuracy scalar chart/i);
    const validationCard = scalarChartSection(/validation\/accuracy scalar chart/i);
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
    installFetchMock();
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(screen.getByRole("button", { name: /all runs/i }));
    await selectAllLogExperiments(user);
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
    expect(await screen.findByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();
    await user.click(logMetricGroupToggle("Test"));
    expect(
      await screen.findByRole("table", { name: /test\/accuracy test leaderboard/i }),
    ).toBeInTheDocument();
    expect(
      screen.getAllByText(/test_model_2 · Cifar10 · linear · linears · BASELINE · 2026-06-01 02:03:04/).length,
    ).toBeGreaterThan(0);

    await clickLogOption(user, "Experiments", "test_model");

    await waitFor(() => {
      expect(screen.queryByText(/test_model · Mnist · linear · linears · BASELINE · 2026-06-01 01:02:03/))
        .not.toBeInTheDocument();
    });
    const datasetSection = logFilterSection("Datasets");
    expect(within(datasetSection).getByText("1 / 1")).toBeInTheDocument();
    await expectLogFilterSelection(user, "Datasets", "Cifar10", true);
    expect(queryOpenLogOption("Datasets", "Mnist")).not.toBeInTheDocument();
    expect(
      screen.getAllByText(/test_model_2 · Cifar10 · linear · linears · BASELINE · 2026-06-01 02:03:04/).length,
    ).toBeGreaterThan(0);

    await clickLogOption(user, "Scalar Tags", "validation/accuracy");

    await waitFor(() => {
      expect(screen.queryByRole("img", { name: /validation\/accuracy scalar chart/i }))
        .not.toBeInTheDocument();
    });
    expect(screen.getByRole("img", { name: /train\/loss scalar chart/i }))
      .toBeInTheDocument();

    await clickLogOption(user, "Scalar Tags", "test/accuracy");

    await waitFor(() => {
      expect(screen.queryByRole("table", { name: /test\/accuracy test leaderboard/i }))
        .not.toBeInTheDocument();
    });

    const experimentSection = logFilterSection("Experiments");
    await user.click(within(experimentSection).getByRole("button", { name: /^none$/i }));

    expect(await screen.findByText("No runs selected")).toBeInTheDocument();
  });

  it("requests checkpoint markers only for visible log runs", async () => {
    const { logCheckpointRequests } = installFetchMock({
      logRunsResponse: {
        runs: logRunsResponse.runs.map((run) =>
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
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(screen.getByRole("button", { name: /all runs/i }));
    await selectAllLogExperiments(user);

    await waitFor(() => {
      expect(logCheckpointRequests.at(-1)).toEqual({
        runIds: ["log-mnist", "log-cifar"],
      });
    });

    await clickLogOption(user, "Experiments", "test_model");

    await waitFor(() => {
      expect(logCheckpointRequests.at(-1)).toEqual({ runIds: ["log-cifar"] });
    });
    expect(screen.queryByText(/test_model · Mnist · linear · linears · BASELINE/))
      .not.toBeInTheDocument();
  });

  it("deletes a log experiment after exact-name confirmation", async () => {
    const { fetchMock, deleteExperimentRequests, logScalarRequests } = installFetchMock();
    renderViewer();
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
    expect(deleteRunPlanRequests).toHaveLength(0);
    expect(deleteRunRequests).toHaveLength(0);
  });

  it("deletes preset row runs using the target experiment and preset", async () => {
    const { deleteRunPlanRequests, deleteRunRequests } = installFetchMock(
      buildSubsetDeleteFixture(),
    );
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await clickLogOption(user, "Experiments", "test_model");

    await openLogFilter(user, "Presets");
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
        models: [{ modelType: "linears", model: "linear" }],
        presets: ["BASELINE"],
        runIds: ["log-cifar-baseline", "log-mnist-baseline"],
      },
    ]);
    expect(deleteRunRequests).toEqual(deleteRunPlanRequests);
  });

  it("blocks preset row deletion when an active training job uses an affected folder", async () => {
    const { deleteRunRequests } = installFetchMock({
      ...buildSubsetDeleteFixture(),
      deleteLogRunsBlockers: [
        { id: "job-1", logFolder: "test_model", status: "running" },
      ],
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await clickLogOption(user, "Experiments", "test_model");
    await openLogFilter(user, "Presets");
    await user.click(
      await screen.findByRole("button", {
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
    expect(deleteRunRequests).toHaveLength(0);
  });

  it("lazy-loads logs sidebar filters with many options", async () => {
    const fixture = buildLargeLogFixture();
    const { deleteExperimentRequests, logScalarRequests } = installFetchMock(fixture);
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(screen.getByRole("button", { name: /all runs/i }));
    await selectAllLogExperiments(user);

    let experimentsList = await openLogFilter(user, "Experiments");
    const firstExperiment = await within(experimentsList).findByRole("option", {
      name: logOptionName("experiment_01"),
    });
    expectLogOptionSelected(firstExperiment);
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
      within(experimentsList).getByRole("status", {
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
    await user.click(
      await screen.findByRole("button", { name: /^Other\s+1\s+metric$/i }),
    );

    const expectedRunIds = fixture.logRunsResponse.runs.map((run) => run.id);
    await waitFor(() => {
      const selectedTagRequests = logScalarRequests.filter(
        (request) =>
          request.tags.length === 1 && request.tags[0] === "custom/tag-42",
      );
      expect(selectedTagRequests.at(-2)).toEqual({
        runIds: expectedRunIds.slice(0, 50),
        tags: ["custom/tag-42"],
        maxPoints: 500,
        sampling: "tail",
      });
      expect(selectedTagRequests.at(-1)).toEqual({
        runIds: expectedRunIds.slice(50),
        tags: ["custom/tag-42"],
        maxPoints: 500,
        sampling: "tail",
      });
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
    await user.click(secondExperiment);
    expectLogOptionSelected(secondExperiment, false);
    expectLogOptionSelected(
      within(experimentsList).getByRole("option", {
        name: logOptionName("experiment_42"),
      }),
    );
  });

  it("does not delete a log experiment when the dialog is cancelled", async () => {
    const { deleteExperimentRequests } = installFetchMock();
    renderViewer();
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
    const { deleteExperimentRequests } = installFetchMock({
      deleteLogExperimentError: "Refusing to delete symlink log experiment: test_model",
    });
    renderViewer();
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
    const { logScalarRequests } = installFetchMock({
      logTagsByRun: {
        "log-mnist": ["train/loss", "validation/accuracy"],
        "log-cifar": ["train/loss"],
      },
    });
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await user.click(screen.getByRole("button", { name: /all runs/i }));
    await selectAllLogExperiments(user);

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
    const { logScalarRequests, logTagRequests } = installFetchMock({
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
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await screen.findByText("Historical Scalars");
    await user.click(await screen.findByRole("button", { name: /all runs/i }));
    await selectLogExperiments(user, ["prior_monitor"]);
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

  it("widens stale preset facets when selecting a newly loaded experiment", async () => {
    const priorBaselineRun = {
      ...logRunsResponse.runs[0],
      id: "prior-linear-baseline",
      group: "prior_linear",
      experiment: "prior_linear",
      preset: "BASELINE",
      dataset: "Mnist",
      runName: "prior_linear_baseline_20260601_010203",
      timestamp: "2026-06-01 01:02:03",
      relativePath:
        "prior_linear/linear/BASELINE/Mnist/prior_linear_baseline_20260601_010203/version_0",
      metrics: { "test/accuracy": 0.82 },
    };
    const priorLegacyRun = {
      ...priorBaselineRun,
      id: "prior-linear-legacy",
      preset: "LEGACY_ONLY",
      runName: "prior_linear_legacy_20260601_020304",
      timestamp: "2026-06-01 02:03:04",
      relativePath:
        "prior_linear/linear/LEGACY_ONLY/Mnist/prior_linear_legacy_20260601_020304/version_0",
      metrics: { "test/accuracy": 0.84 },
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
    const runs = [priorBaselineRun, priorLegacyRun, ...kaggleRuns];
    const { logScalarRequests, logTagRequests } = installFetchMock({
      logRunsResponse: { runs },
      logExperimentsResponse: {
        experiments: [
          { experiment: "prior_linear", runCount: 2, relativePath: "prior_linear" },
          {
            experiment: "kaggle_linear_all",
            runCount: kaggleRuns.length,
            relativePath: "kaggle_linear_all",
          },
        ],
      },
      logTagsByRun: Object.fromEntries(
        runs.map((run) => [run.id, ["train/loss", "validation/accuracy"]]),
      ),
      logScalarSeries: runs.flatMap((run, index) => [
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
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await screen.findByText("Historical Scalars");
    await user.click(await screen.findByRole("button", { name: /all runs/i }));
    await selectLogExperiments(user, ["prior_linear"]);
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();

    await clickLogOption(user, "Presets", "LEGACY_ONLY");
    await expectLogFilterSelection(user, "Presets", "BASELINE", true);
    await expectLogFilterSelection(user, "Presets", "LEGACY_ONLY", false);
    await clickLogOption(user, "Experiments", "prior_linear");
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
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
      .toBeInTheDocument();
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
    const { logScalarRequests, logTagRequests } = installFetchMock({
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
    renderViewer();
    const user = userEvent.setup();

    await user.click(await screen.findByRole("button", { name: /^logs$/i }));
    await screen.findByText("Historical Scalars");
    await user.click(await screen.findByRole("button", { name: /all runs/i }));
    await selectLogExperiments(user, ["normal_linear"]);
    expect(await screen.findByRole("img", { name: /validation\/accuracy scalar chart/i }))
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
    await selectLogExperiments(user, ["test_model"]);

    expect(await screen.findByText("No TensorBoard scalars")).toBeInTheDocument();
    expect(
      screen.getByText("The selected runs do not contain scalar event data."),
    ).toBeInTheDocument();
    expect(logScalarRequests).toHaveLength(0);
  });

});
